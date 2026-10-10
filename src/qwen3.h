// RPGBarEngine —— Qwen3 前向
//
// M1 阶段：**全序列**前向，不建 KV Cache。一条 prompt 进去，每个位置的
// logits 出来。生成（M2）会复用这里的算子，只把 attention 换成增量式的。
//
// 这一版的首要目标是**可对拍**，不是快。刻意做的事：
//   * 每个中间结果都留了缓冲区并命名（q/k/v/attn/gate/up/proj），
//     出问题时可以把任意一层导出来和参考实现对；
//   * GEMM 只在"输出行"这个维度并行，attention 走标量 —— 避免为了并行把
//     数值累加顺序改掉（浮点加法不满足结合律，改顺序 = 换答案）；
//   * 所有缓冲一次性按最长序列分配，不在热路径里 new。

#ifndef RPGBAR_QWEN3_H_
#define RPGBAR_QWEN3_H_

#include <cstdint>
#include <vector>

#include "model.h"

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
  std::vector<float> scores;  // [seq]           单头、单查询位置的注意力分数

  void Ensure(const ModelConfig& c, int64_t n);
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

// 全序列前向。tokens 是 [seq] 的 token id，logits 输出 [seq, vocab] 行主。
void ForwardFull(const Model& m, const int32_t* tokens, int64_t seq,
                 ForwardScratch* s, std::vector<float>* logits,
                 const ForwardOptions& opt = ForwardOptions());

// 因果注意力（GQA，全因果，Qwen3 没有滑动窗口）。
// 单独拆出来是为了 M2 能写一个带 KV Cache 的兄弟版本而不动其余部分。
//   q: [seq, n_heads*head_dim]      out: [seq, n_heads*head_dim]
//   k/v: [seq, n_kv_heads*head_dim] scores: [seq] 临时区
void CausalAttention(const ModelConfig& c, const float* q, const float* k,
                     const float* v, int64_t seq, float* scores, float* out);

// 单层前向。x 就地更新（残差加在 x 上）。
void ForwardLayer(const Model& m, const LayerWeights& w, ForwardScratch* s,
                  int64_t seq);

}  // namespace rpg

#endif  // RPGBAR_QWEN3_H_
