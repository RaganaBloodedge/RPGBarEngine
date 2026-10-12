// RPGBarEngine —— Qwen3 前向
//
// 两条路径共用**同一份层实现**：
//   * ForwardFull —— 全序列（或一段）前向。传入 KV Cache 时顺便把 K/V 写进去
//                    （即 prefill）。
//   * ForwardStep —— 单 token 增量。读 cache 里的历史，只算新 token 对历史的
//                    注意力。
// 两者走同一个 ForwardLayer，区别只是 (seq, pos0)。于是"填 cache 的全序列"
// 与"不填 cache 的全序列"在数值上**必然逐位相同** —— 这条被 `forward_check
// --kv` 当成一个硬自检守着，因为它能一次性证明 cache 的写入/读取没有把
// 布局搞错（布局错了不会崩，只会让生成结果慢慢跑偏）。
//
// 这一版的首要目标仍是**可对拍**，不是快：
//   * 每个中间结果都留了命名的缓冲区（q/k/v/attn/gate/up/proj），出问题时
//     可以把任意一层导出来和参考实现对；
//   * GEMM 只在"输出行"这个维度并行，attention 走标量 —— 并行不改变每个
//     输出的累加顺序（浮点加法不满足结合律，改顺序 = 换答案）；
//   * 所有缓冲一次性按最长序列分配，不在热路径里 new。
//
// **引擎里没有有状态的随机数生成器。** 采样（M2.5）用 counter-based 随机源
// u = H(seed, 绝对位置)，它是纯函数（见 sampler.h）：同一个 seed + 同一段
// prompt 必得同一局剧情。温度的"创造力"与随机数的"掷骰子"是两条互不干涉的
// 链 —— 温度只缩放 logits，随机数只由 seed 决定 —— 见 docs/DESIGN.md §12 #6。

#ifndef RPGBAR_QWEN3_H_
#define RPGBAR_QWEN3_H_

#include <cstdint>
#include <vector>

#include "kv_cache.h"
#include "model.h"
#include "sampler.h"

namespace rpg {

// 一次前向的临时缓冲。复用它可以避免每个 token 重新分配几百 MB。
struct ForwardScratch {
  int64_t seq = 0;
  std::vector<float> x;       // [seq, hidden]   残差流
  std::vector<float> h;       // [seq, hidden]   归一化输出 = 矩阵乘的输入
  std::vector<float> q;       // [seq, q_dim]    多头拼在一起
  std::vector<float> k;       // [seq, kv_dim]
  std::vector<float> v;       // [seq, kv_dim]
  std::vector<float> attn;    // [seq, q_dim]    attention 输出（多头拼在一起）
  std::vector<float> proj;    // [seq, hidden]   子层输出（残差要加的那一份）
  std::vector<float> gate;    // [seq, intermediate]  SwiGLU 的 gate 与结果
  std::vector<float> up;      // [seq, intermediate]
  // scores 的索引是**绝对位置**（attention 里 `s <= pos0 + t`），所以长度要求是
  // **pos0 + seq**，不是 seq —— 前缀复用（pos0 > 0）时按 seq 备就是堆越界。
  // 由 Ensure（只增不减，见 .cc）独占管理，不在 Ensure 里缩容。
  std::vector<float> scores;

  // 本段缓冲：按 n 重设 x/h/q/k/v/attn/proj/gate/up（它们的索引都是段内局部下标）。
  void Ensure(const ModelConfig& c, int64_t n);
  // scores 单独一个入口：只增不减（每步重新 assign 一次纯属浪费），
  // 且 n 传的是**绝对位置数** pos0 + seq。
  void EnsureScores(int64_t n);
};

struct ForwardOptions {
  // 非空时逐层收集残差流，形状 [n_layers+1, seq, hidden]：
  //   [0]      = embedding 输出（第 0 层的输入）
  //   [l+1]    = 第 l 层输出（即第 l+1 层的输入）
  //   [n_layers] = 最后一层的输出，**尚未过 output_norm**
  // 需要这么切是因为 HF 的 output_hidden_states[-1] 是过完 final norm 的，
  // 对拍时要用 model.norm 换算一下才能逐层比。
  std::vector<float>* capture_hidden = nullptr;

  // 只跑前 N 层（0 = 全部）。用于把"误差从第几层开始"这件事定位到层。
  // 截断时不做末端归一化与 lm_head，logits 留空。
  int32_t layer_limit = 0;
};

// 全序列（或一段）前向。tokens 是 [seq] 的 token id，logits 输出 [seq, vocab]
// 行主。
//
// kv != nullptr 时：
//   * 把这一段的 K/V 追加写入 cache（位置从 cache 当前长度开始）；
//   * 注意力用 cache 里的**全部历史**（前缀 + 本段），等价于带前缀的因果注意力；
//   * 写完后 cache 长度 += seq。
// 这既是 prefill，也是"复用上一轮相同前缀"的手段。
void ForwardFull(const Model& m, const int32_t* tokens, int64_t seq,
                 ForwardScratch* s, std::vector<float>* logits,
                 const ForwardOptions& opt = ForwardOptions(),
                 KvCache* kv = nullptr);

// 单 token 增量前向。pos 必须等于 cache 的当前长度（只支持顺序追加）。
// logits 输出 [1, vocab]。
void ForwardStep(const Model& m, int32_t token, int64_t pos, KvCache* kv,
                 ForwardScratch* s, std::vector<float>* logits);

// 因果注意力（GQA，全因果，Qwen3 没有滑动窗口）—— 不读 cache 的版本。
// 单独拆出来是为了让"有 cache / 无 cache"两条路径可以互相印证。
//   q: [seq, n_heads*head_dim]      out: [seq, n_heads*head_dim]
//   k/v: [seq, n_kv_heads*head_dim] scores: [seq] 临时区
void CausalAttention(const ModelConfig& c, const float* q, const float* k,
                     const float* v, int64_t seq, float* scores, float* out);

// 因果注意力 —— 从 KV Cache 读历史的版本。q 的位置是 pos0 + t，可以看到
// [0, pos0 + t] 的 K/V（已在 cache 里）。scores 需容纳 pos0 + seq 个 float。
void AttentionFromCache(const ModelConfig& c, int32_t layer, int64_t pos0,
                        const float* q, int64_t seq, const KvCache& kv,
                        float* scores, float* out);

// 单层前向。x 就地更新（残差加在 x 上）。
// layer / pos0 / kv 只在 kv != nullptr 时被使用（layer 用于索引 cache 的层）。
void ForwardLayer(const Model& m, const LayerWeights& w, int32_t layer,
                  int64_t pos0, ForwardScratch* s, int64_t seq, KvCache* kv);

// ---------------------------------------------------------------- 生成

struct GenerateOptions {
  int32_t max_tokens = 64;
  // 命中任一 id 即停。停止 token 本身**计入**返回序列（与 HF generate 的
  // 行为一致 —— 它也会把 eos 放进输出）。空则只按 max_tokens 停。
  std::vector<int32_t> stop_ids;
  // 采样参数（M2.5）。sample.temperature <= 0 即贪心 —— 此时行为与 M2 的
  // 纯贪心完全一样，且**不消耗任何随机数**（向后兼容：不传 temperature 的
  // 老请求只会走贪心）。
  SampleOptions sample;
  // 采样种子。引擎按 u_i = CounterUniform(seed, 绝对位置) 展开成随机序列，
  // 是纯函数：同一 seed + 同一段 prompt → 同一串随机数 → 同一局剧情。
  // 位置取 KV Cache 的绝对下标，所以暖缓存（续写）与冷缓存（从头）逐位一致。
  uint64_t seed = 0;
};

// 生成。cache 里应已有 prompt 的 K/V（先调一次 ForwardFull 做 prefill），
// first_logits 是"下一个 token"的分布（[vocab]，即 prefill 最后一个位置那行）。
// 生成的 token 追加进 out_ids（不含 prompt）。
//
//   * sample.temperature <= 0 —— 贪心（argmax），与 GenerateGreedy 等价；
//   * 否则 —— 每个 token 用 counter-based 随机源抽一次样。
void Generate(const Model& m, KvCache* kv, ForwardScratch* s,
              const float* first_logits, const GenerateOptions& opt,
              std::vector<int32_t>* out_ids);

// 贪心解码。等价于把 sample.temperature 置 0 的 Generate。M2 的逐 token
// 对拍判据仍然走它。
void GenerateGreedy(const Model& m, KvCache* kv, ForwardScratch* s,
                    const float* first_logits, const GenerateOptions& opt,
                    std::vector<int32_t>* out_ids);

}  // namespace rpg

#endif  // RPGBAR_QWEN3_H_
