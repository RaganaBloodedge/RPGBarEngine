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
  need(scores, n);
  seq = n;
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

void ForwardLayer(const Model& m, const LayerWeights& w, ForwardScratch* s,
                  int64_t seq) {
  const ModelConfig& c = m.config();
  ThreadPool& pool = m.pool();
  const int64_t hidden = c.hidden;
  const int64_t inter = c.intermediate;
  const int64_t qd = c.q_dim();
  const int64_t kvd = c.kv_dim();

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
      float* p = qt + static_cast<int64_t>(h) * c.head_dim;
      RmsNorm(p, w.q_norm, c.head_dim, c.rms_eps, p);
    }
    float* kt = s->k.data() + t * kvd;
    for (int32_t h = 0; h < c.n_kv_heads; ++h) {
      float* p = kt + static_cast<int64_t>(h) * c.head_dim;
      RmsNorm(p, w.k_norm, c.head_dim, c.rms_eps, p);
    }
  }

  // RoPE（半维配对）。K 不旋转 V —— V 不参与位置编码。
  for (int64_t t = 0; t < seq; ++t) {
    const float* ct = m.rope_cos(t);
    const float* st = m.rope_sin(t);
    RopeInPlace(s->q.data() + t * qd, c.n_heads, c.head_dim, ct, st);
    RopeInPlace(s->k.data() + t * kvd, c.n_kv_heads, c.head_dim, ct, st);
  }

  CausalAttention(c, s->q.data(), s->k.data(), s->v.data(), seq,
                  s->scores.data(), s->attn.data());

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
                 const ForwardOptions& opt) {
  const ModelConfig& c = m.config();
  ThreadPool& pool = m.pool();
  const int64_t hidden = c.hidden;

  RPG_CHECK(seq > 0, "prompt 为空");
  RPG_CHECK(seq <= m.rope_positions(),
            "序列长度 %lld 超过 RoPE 表覆盖的 %lld 个位置（构造 Model 时可用 "
            "max_pos 提高）",
            static_cast<long long>(seq),
            static_cast<long long>(m.rope_positions()));
  RPG_CHECK(static_cast<int64_t>(c.vocab) == m.token_embd().rows(),
            "词表大小 %d 与 token_embd 行数 %lld 不一致", c.vocab,
            static_cast<long long>(m.token_embd().rows()));

  const int32_t n_use =
      (opt.layer_limit > 0 && opt.layer_limit < c.n_layers) ? opt.layer_limit
                                                            : c.n_layers;
  if (opt.capture_hidden) {
    opt.capture_hidden->assign(
        static_cast<size_t>(n_use + 1) * seq * hidden, 0.0f);
  }
  s->Ensure(c, seq);

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
    ForwardLayer(m, m.layers()[static_cast<size_t>(l)], s, seq);
    if (opt.capture_hidden) {
      std::memcpy(opt.capture_hidden->data() +
                      static_cast<size_t>(l + 1) * seq * hidden,
                  s->x.data(), static_cast<size_t>(seq) * hidden * sizeof(float));
    }
  }

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

}  // namespace rpg
