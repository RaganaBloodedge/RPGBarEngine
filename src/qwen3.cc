#include "qwen3.h"

#include <cmath>
#include <cstring>

#include "common.h"
#include "kernels.h"

namespace rpg {

void ForwardScratch::Ensure(const ModelConfig& c, int64_t n) {
  auto need = [](std::vector<float>& v, int64_t sz) {
    if (static_cast<int64_t>(v.size()) != sz) {
      v.assign(static_cast<size_t>(sz), 0.0f);
    }
  };
  need(x, n * c.hidden);
  need(h, n * c.hidden);
  need(q, n * c.q_dim());
  need(k, n * c.kv_dim());
  need(v, n * c.kv_dim());
  need(attn, n * c.q_dim());
  need(proj, n * c.hidden);
  need(gate, n * c.intermediate);
  need(up, n * c.intermediate);
  // **scores 不在这里管**。上面这些缓冲的索引是**本段内的局部下标**（0..n-1），
  // 但 scores 的索引是**绝对位置**（AttentionFromCache 里 `s <= pos0 + t`，见
  // qwen3.cc 的注释），长度得是 pos0 + n。放在这里按 n 缩容的话，一旦前缀复用
  // 时 pos0 > 0 就会写到缓冲区后面去 —— 那是堆越界，且症状是"过一会儿才崩"。
  // 它由 EnsureScores 独占管理（只增不减），调用点在 ForwardFull。
  seq = n;
}

void ForwardScratch::EnsureScores(int64_t n) {
  // 只增不减：增量解码每步都会调它，缩容再扩容纯属自找。
  //
  // n 是**绝对位置数**（pos0 + seq），不是本段的长度 —— scores 由
  // AttentionFromCache 按绝对位置索引。见 Ensure 里的说明。
  if (static_cast<int64_t>(scores.size()) < n) {
    scores.assign(static_cast<size_t>(n), 0.0f);
  }
}

void CausalAttention(const ModelConfig& c, const float* q, const float* k,
                     const float* v, int64_t seq, float* scores, float* out) {
  const int32_t nh = c.n_heads;
  const int32_t nkv = c.n_kv_heads;
  const int32_t hd = c.head_dim;
  const int32_t group = c.heads_per_kv();
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const int64_t q_stride = static_cast<int64_t>(nh) * hd;
  const int64_t kv_stride = static_cast<int64_t>(nkv) * hd;

  std::memset(out, 0, static_cast<size_t>(seq) * q_stride * sizeof(float));

  for (int32_t h = 0; h < nh; ++h) {
    const int32_t kh = h / group;  // GQA：多个 Q 头共享一个 KV 头
    for (int64_t t = 0; t < seq; ++t) {
      const float* qt = q + t * q_stride + static_cast<int64_t>(h) * hd;

      // 因果掩码：位置 t 只能看到 s <= t。直接不写 -inf 的格子，比先填 -inf
      // 再 masked_softmax 少一遍遍历，也少一次"把 -inf 变成 0"的机会。
      for (int64_t s = 0; s <= t; ++s) {
        const float* ks = k + s * kv_stride + static_cast<int64_t>(kh) * hd;
        scores[s] = Dot(qt, ks, hd) * scale;
      }
      SoftmaxInPlace(scores, t + 1);

      float* ot = out + t * q_stride + static_cast<int64_t>(h) * hd;
      for (int64_t s = 0; s <= t; ++s) {
        const float* vs = v + s * kv_stride + static_cast<int64_t>(kh) * hd;
        const float p = scores[s];
        for (int32_t d = 0; d < hd; ++d) ot[d] += p * vs[d];
      }
    }
  }
}

void AttentionFromCache(const ModelConfig& c, int32_t layer, int64_t pos0,
                        const float* q, int64_t seq, const KvCache& kv,
                        float* scores, float* out) {
  const int32_t nh = c.n_heads;
  const int32_t hd = c.head_dim;
  const int32_t group = c.heads_per_kv();
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const int64_t q_stride = static_cast<int64_t>(nh) * hd;

  std::memset(out, 0, static_cast<size_t>(seq) * q_stride * sizeof(float));

  for (int32_t h = 0; h < nh; ++h) {
    const int32_t kh = h / group;
    // 每头只取一次基址。cache 是紧凑布局（同一个头的连续位置连续），
    // 所以 s 的偏移就是 s * head_dim —— 与上面 kv_stride 的写法等价，
    // 但少一次乘法与一次越界检查。
    const float* kb = kv.KHead(layer, kh);
    const float* vb = kv.VHead(layer, kh);

    for (int64_t t = 0; t < seq; ++t) {
      const int64_t pos = pos0 + t;
      const float* qt = q + t * q_stride + static_cast<int64_t>(h) * hd;

      // 位置 pos 能看到 cache 里 [0, pos] 的全部历史 —— 包括本段里排在它
      // 前面的那些（它们已经写进 cache 了），因果性自然成立。
      for (int64_t s = 0; s <= pos; ++s) {
        scores[s] = Dot(qt, kb + s * hd, hd) * scale;
      }
      SoftmaxInPlace(scores, pos + 1);

      float* ot = out + t * q_stride + static_cast<int64_t>(h) * hd;
      for (int64_t s = 0; s <= pos; ++s) {
        const float* vs = vb + s * hd;
        const float p = scores[s];
        for (int32_t d = 0; d < hd; ++d) ot[d] += p * vs[d];
      }
    }
  }
}

void ForwardLayer(const Model& m, const LayerWeights& w, int32_t layer,
                  int64_t pos0, ForwardScratch* s, int64_t seq, KvCache* kv) {
  const ModelConfig& c = m.config();
  ThreadPool& pool = m.pool();
  const int64_t hidden = c.hidden;
  const int64_t inter = c.intermediate;
  const int64_t qd = c.q_dim();
  const int64_t kvd = c.kv_dim();
  const int32_t hd = c.head_dim;

  // ---- 注意力 ----
  // pre-norm：归一化在子层**输入**上做，残差加在原始 x 上（不是归一化后的 h）。
  for (int64_t t = 0; t < seq; ++t) {
    RmsNorm(s->x.data() + t * hidden, w.attn_norm, hidden, c.rms_eps,
            s->h.data() + t * hidden);
  }
  w.attn_q.MatMulBatch(s->h.data(), seq, s->q.data(), &pool);
  w.attn_k.MatMulBatch(s->h.data(), seq, s->k.data(), &pool);
  w.attn_v.MatMulBatch(s->h.data(), seq, s->v.data(), &pool);

  // QK-norm：**在 RoPE 之前**，对每个头在 head_dim 上各做一次 RMSNorm。
  // 顺序错了（放到 RoPE 之后）不会崩，但注意力会整体偏掉。
  for (int64_t t = 0; t < seq; ++t) {
    float* qt = s->q.data() + t * qd;
    for (int32_t h = 0; h < c.n_heads; ++h) {
      float* p = qt + static_cast<int64_t>(h) * hd;
      RmsNorm(p, w.q_norm, hd, c.rms_eps, p);
    }
    float* kt = s->k.data() + t * kvd;
    for (int32_t h = 0; h < c.n_kv_heads; ++h) {
      float* p = kt + static_cast<int64_t>(h) * hd;
      RmsNorm(p, w.k_norm, hd, c.rms_eps, p);
    }
  }

  // RoPE（半维配对）。K 不旋转 V —— V 不参与位置编码。
  // 位置要加 pos0：增量解码时 pos0 是 cache 已有长度，位置编码必须接续下去，
  // 从 0 重新开始会让模型以为每次都在句首。
  for (int64_t t = 0; t < seq; ++t) {
    const float* ct = m.rope_cos(pos0 + t);
    const float* st = m.rope_sin(pos0 + t);
    RopeInPlace(s->q.data() + t * qd, c.n_heads, hd, ct, st);
    RopeInPlace(s->k.data() + t * kvd, c.n_kv_heads, hd, ct, st);
  }

  if (kv != nullptr) {
    // 写 cache。注意 s->k 的布局就是 [n_kv_heads, head_dim] 紧密排列，
    // 与 cache 里"一个头一段"的布局一致，所以是逐头整段拷贝，不需要重排。
    for (int64_t t = 0; t < seq; ++t) {
      const int64_t pos = pos0 + t;
      const float* kt = s->k.data() + t * kvd;
      const float* vt = s->v.data() + t * kvd;
      for (int32_t h = 0; h < c.n_kv_heads; ++h) {
        std::memcpy(kv->K(layer, h, pos), kt + static_cast<int64_t>(h) * hd,
                    static_cast<size_t>(hd) * sizeof(float));
        std::memcpy(kv->V(layer, h, pos), vt + static_cast<int64_t>(h) * hd,
                    static_cast<size_t>(hd) * sizeof(float));
      }
    }
    AttentionFromCache(c, layer, pos0, s->q.data(), seq, *kv, s->scores.data(),
                       s->attn.data());
  } else {
    CausalAttention(c, s->q.data(), s->k.data(), s->v.data(), seq,
                    s->scores.data(), s->attn.data());
  }

  w.attn_o.MatMulBatch(s->attn.data(), seq, s->proj.data(), &pool);
  for (int64_t t = 0; t < seq; ++t) {
    VecAddInPlace(s->x.data() + t * hidden, s->proj.data() + t * hidden, hidden);
  }

  // ---- 前馈（SwiGLU）----
  for (int64_t t = 0; t < seq; ++t) {
    RmsNorm(s->x.data() + t * hidden, w.ffn_norm, hidden, c.rms_eps,
            s->h.data() + t * hidden);
  }
  w.ffn_gate.MatMulBatch(s->h.data(), seq, s->gate.data(), &pool);
  w.ffn_up.MatMulBatch(s->h.data(), seq, s->up.data(), &pool);
  for (int64_t t = 0; t < seq; ++t) {
    float* g = s->gate.data() + t * inter;
    SiluInPlace(g, inter);
    MulInPlace(g, s->up.data() + t * inter, inter);
  }
  w.ffn_down.MatMulBatch(s->gate.data(), seq, s->proj.data(), &pool);
  for (int64_t t = 0; t < seq; ++t) {
    VecAddInPlace(s->x.data() + t * hidden, s->proj.data() + t * hidden, hidden);
  }
}

void ForwardFull(const Model& m, const int32_t* tokens, int64_t seq,
                 ForwardScratch* s, std::vector<float>* logits,
                 const ForwardOptions& opt, KvCache* kv) {
  const ModelConfig& c = m.config();
  ThreadPool& pool = m.pool();
  const int64_t hidden = c.hidden;

  RPG_CHECK(seq > 0, "prompt 为空");
  RPG_CHECK(static_cast<int64_t>(c.vocab) == m.token_embd().rows(),
            "词表大小 %d 与 token_embd 行数 %lld 不一致", c.vocab,
            static_cast<long long>(m.token_embd().rows()));

  // 这一段在 cache 里的起始位置（不传 cache 就是 0）。
  const int64_t pos0 = (kv != nullptr) ? kv->len() : 0;
  RPG_CHECK(pos0 + seq <= m.rope_positions(),
            "位置 %lld 超出 RoPE 表覆盖的 %lld（构造 Model 时可用 max_pos 提高，"
            "或先 session_reset）",
            static_cast<long long>(pos0 + seq),
            static_cast<long long>(m.rope_positions()));
  if (kv != nullptr) {
    RPG_CHECK(pos0 + seq <= kv->max_seq(),
              "位置 %lld 超出 KV Cache 容量 %lld",
              static_cast<long long>(pos0 + seq),
              static_cast<long long>(kv->max_seq()));
  }

  const int32_t n_use =
      (opt.layer_limit > 0 && opt.layer_limit < c.n_layers) ? opt.layer_limit
                                                            : c.n_layers;
  if (opt.capture_hidden) {
    opt.capture_hidden->assign(
        static_cast<size_t>(n_use + 1) * seq * hidden, 0.0f);
  }
  s->Ensure(c, seq);
  // scores 按**绝对位置**备足：AttentionFromCache 会写到 [0, pos0 + seq)。
  // 这一句是必须的，不是保险 —— 少了它，前缀复用（pos0 > 0）时 attention 会写
  // 到 scores 缓冲区之后（堆越界）。症状极隐蔽：单轮 prompt 不崩，第二轮复用
  // 前缀时崩，且崩溃点在堆分配器里（0xC0000374），看不出与注意力有关。
  s->EnsureScores(pos0 + seq);

  // ---- embedding ----
  // Qwen3 **不**乘 sqrt(hidden)（Gemma 2 才乘）。这一点如果抄错，
  // 表现为整体 logits 缩放，softmax 后温度不对 —— 能出词，但不对。
  for (int64_t t = 0; t < seq; ++t) {
    const int32_t tok = tokens[t];
    RPG_CHECK(tok >= 0 && tok < c.vocab, "第 %lld 个 token id = %d 越界（词表 %d）",
              static_cast<long long>(t), tok, c.vocab);
    m.token_embd().Row(tok, s->x.data() + t * hidden);
  }
  if (opt.capture_hidden) {
    std::memcpy(opt.capture_hidden->data(), s->x.data(),
                static_cast<size_t>(seq) * hidden * sizeof(float));
  }

  // ---- 逐层 ----
  for (int32_t l = 0; l < n_use; ++l) {
    ForwardLayer(m, m.layers()[static_cast<size_t>(l)], l, pos0, s, seq, kv);
    if (opt.capture_hidden) {
      std::memcpy(opt.capture_hidden->data() +
                      static_cast<size_t>(l + 1) * seq * hidden,
                  s->x.data(), static_cast<size_t>(seq) * hidden * sizeof(float));
    }
  }

  // cache 的长度在**所有层都写完**之后才推进：中途任何一层抛错（越界、
  // 未实现的量化类型……）都保留原来的 len，调用方重试时不会多算一段空洞。
  if (kv != nullptr) kv->SetLen(pos0 + seq);

  if (n_use < c.n_layers) {
    // 截断模式：末端归一化与 lm_head 依赖更深的层，跳过并明确返回空 logits，
    // 而不是给一个"看起来能用"的错值。
    logits->clear();
    return;
  }

  // ---- 末端归一化 + 输出头 ----
  for (int64_t t = 0; t < seq; ++t) {
    RmsNorm(s->x.data() + t * hidden, m.output_norm(), hidden, c.rms_eps,
            s->h.data() + t * hidden);
  }
  logits->assign(static_cast<size_t>(seq) * c.vocab, 0.0f);
  m.lm_head().MatMulBatch(s->h.data(), seq, logits->data(), &pool);
}

void ForwardStep(const Model& m, int32_t token, int64_t pos, KvCache* kv,
                 ForwardScratch* s, std::vector<float>* logits) {
  RPG_CHECK(kv != nullptr, "ForwardStep 需要 KV Cache");
  RPG_CHECK(pos == kv->len(),
            "增量解码只支持顺序追加：pos = %lld，而 cache 长度 = %lld",
            static_cast<long long>(pos), static_cast<long long>(kv->len()));
  // scores 由 ForwardFull 按 pos0 + seq 备足，这里不必再管（pos0 就是 pos）。
  ForwardFull(m, &token, 1, s, logits, ForwardOptions(), kv);
}

void Generate(const Model& m, KvCache* kv, ForwardScratch* s,
              const float* first_logits, const GenerateOptions& opt,
              std::vector<int32_t>* out_ids) {
  const ModelConfig& c = m.config();
  RPG_CHECK(kv != nullptr, "Generate 需要 KV Cache");
  RPG_CHECK(first_logits != nullptr, "first_logits 为空（prefill 之后才有）");
  RPG_CHECK(opt.max_tokens > 0, "max_tokens 必须为正，给了 %d", opt.max_tokens);

  // 拷一份再改：调用方的 logits 缓冲区（prefill 的输出，可能是 [seq, vocab]）
  // 不该被这里覆盖。
  std::vector<float> logits(first_logits,
                            first_logits + static_cast<size_t>(c.vocab));
  // 贪心不消耗随机数。这条分支让"不传 temperature"的老路径与 M2 逐位一致。
  const bool greedy = !(opt.sample.temperature > 0.0f);

  for (int32_t i = 0; i < opt.max_tokens; ++i) {
    int32_t next;
    if (greedy) {
      next = ArgMax(logits.data(), c.vocab);
    } else {
      // 绝对位置 = 这个 token 将要落进 cache 的下标（此刻还没写进去）。
      // 用绝对下标而不是"本轮第几个"：宿主每轮重发完整 messages 时，同一段
      // 前缀的位置不变，于是暖缓存（续写）与冷缓存（从头）会拿到同一个 u。
      const uint64_t counter = static_cast<uint64_t>(kv->len());
      const float u = CounterUniform(opt.seed, counter);
      next = SampleToken(logits.data(), c.vocab, opt.sample, u);
    }
    out_ids->push_back(next);

    bool stop = false;
    for (const int32_t sid : opt.stop_ids) {
      if (sid == next) {
        stop = true;
        break;
      }
    }

    // 上下文满了就明确报错，不静默截断 —— 静默停下会让人以为"模型自己
    // 讲完了"，而实际是宿主把 max_seq 配小了。
    RPG_CHECK(kv->len() < kv->max_seq(),
              "上下文已满（%lld 个位置，已生成 %d 个 token）—— 需要更大的 "
              "max_seq，或先 session_reset",
              static_cast<long long>(kv->max_seq()), i + 1);

    // **每个**返回的 token 都写进 cache —— 包括最后那个与停止 token。
    // 代价是末尾多算一步前向（约 0.1 秒），换来的是这条不变量：
    //   cache.len() == prompt 长度 + 返回的 token 数
    // 它让调用方可以直接在 cache 上续下一轮（不必 reset 重来），并且可以
    // 被外部一句话验证。少了这一步，cache 会永远差最后一个 token ——
    // 那种"短一个"的错误不会崩，只会让下一轮的第一个 token 悄悄变味。
    ForwardStep(m, next, kv->len(), kv, s, &logits);
    if (stop) break;
  }
}

void GenerateGreedy(const Model& m, KvCache* kv, ForwardScratch* s,
                    const float* first_logits, const GenerateOptions& opt,
                    std::vector<int32_t>* out_ids) {
  GenerateOptions g = opt;
  g.sample.temperature = 0.0f;  // 贪心：与 M2 的行为逐位一致
  Generate(m, kv, s, first_logits, g, out_ids);
}

}  // namespace rpg
