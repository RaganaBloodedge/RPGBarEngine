// RPGBarEngine —— 模型：配置、权重视图、加载
//
// 这个文件只负责"把 GGUF 变成一堆可用的权重 + 一份配置"，不含任何前向逻辑
// （前向在 qwen3.cc）。分开的原因是 M6 之后如果要加第二个架构，只多一个
// qwen3 之类的文件，模型加载这一层不用动。
//
// 两条贯穿始终的约定：
//   1. **配置只从文件读，不写死**。层数/维度/头数/rope_theta 全部来自 GGUF
//      元数据。1.7B 与 4B 共用这份代码，差别只是数字。任何形如
//      "如果是 1.7B 就……" 的分支都是设计错误。
//   2. **加载期就把形状校验做完**。张量名写错、形状对不上，必须在这里抛错，
//      而不是在几十层之后算出 NaN。见 CheckShape。

#ifndef RPGBAR_MODEL_H_
#define RPGBAR_MODEL_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"
#include "quant.h"
#include "threads.h"
#include "tokenizer.h"

namespace rpg {

struct ModelConfig {
  std::string arch;  // "qwen3"
  std::string name;

  int32_t n_layers = 0;
  int32_t hidden = 0;
  int32_t n_heads = 0;      // Q 头数
  int32_t n_kv_heads = 0;   // KV 头数（GQA）
  int32_t head_dim = 0;     // 只看 qwen3.attention.key_length，不许用 hidden/heads 推
  int32_t value_dim = 0;    // 通常 == head_dim
  int32_t intermediate = 0;
  int32_t vocab = 0;
  int32_t context_length = 0;

  float rope_theta = 10000.0f;
  float rms_eps = 1e-6f;

  // tied：没有 output.weight，lm_head 复用 token_embd。
  bool tied_embeddings = false;

  int32_t bos_id = -1;
  int32_t eos_id = -1;
  int32_t pad_id = -1;
  bool add_bos = false;

  int32_t q_dim() const { return n_heads * head_dim; }
  int32_t kv_dim() const { return n_kv_heads * head_dim; }
  int32_t heads_per_kv() const { return n_kv_heads > 0 ? n_heads / n_kv_heads : 0; }
};

// 一个量化矩阵的**视图**：只记指针与形状，不持有内存。
// 数据指向 GGUF 的 mmap 区（所以 Model 活着它才有效）。
class QuantMatrix {
 public:
  QuantMatrix() = default;
  QuantMatrix(const uint8_t* data, uint32_t type, int64_t rows, int64_t cols)
      : data_(data), type_(type), rows_(rows), cols_(cols) {}

  bool valid() const { return data_ != nullptr && rows_ > 0 && cols_ > 0; }
  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }
  uint32_t type() const { return type_; }
  const uint8_t* data() const { return data_; }
  const char* type_name() const { return TypeName(type_); }

  // out[i] = dot(W 的第 i 行, x)。out 需容纳 rows 个 float。
  void MatVec(const float* x, float* out) const;

  // 反量化第 r 行的 cols 个元素到 out。embedding 查表用它 —— 整张
  // [vocab, hidden] 表展开要 1.2 GB，只有用到的行才值得解出来。
  void Row(int64_t r, float* out) const;

  // X 是 [n_tokens, cols] 行主，OUT 是 [n_tokens, rows] 行主。
  // 一次并行调用搞定 n_tokens 次 MatVec —— 避免每行都唤醒一次线程池。
  void MatMulBatch(const float* X, int64_t n_tokens, float* OUT,
                   ThreadPool* pool) const;

 private:
  const uint8_t* data_ = nullptr;
  uint32_t type_ = kGgmlF32;
  int64_t rows_ = 0;
  int64_t cols_ = 0;
};

// 一层的全部权重。norm 权重指向 Model 里的 f32 副本（不是 mmap）。
struct LayerWeights {
  const float* attn_norm = nullptr;  // [hidden]
  const float* ffn_norm = nullptr;   // [hidden]
  const float* q_norm = nullptr;     // [head_dim]，RoPE 之前
  const float* k_norm = nullptr;     // [head_dim]，RoPE 之前

  QuantMatrix attn_q;   // [n_heads*head_dim, hidden]
  QuantMatrix attn_k;   // [n_kv_heads*head_dim, hidden]
  QuantMatrix attn_v;   // [n_kv_heads*head_dim, hidden]
  QuantMatrix attn_o;   // [hidden, n_heads*head_dim]

  QuantMatrix ffn_gate;  // [intermediate, hidden]
  QuantMatrix ffn_up;    // [intermediate, hidden]
  QuantMatrix ffn_down;  // [hidden, intermediate]
};

class Model {
 public:
  // n_threads <= 0 时用 DefaultThreads()。
  // max_pos：RoPE 表覆盖的最大位置数（超过 context_length 会被截到 context_length）。
  static Model Load(const std::string& path, int n_threads = 0,
                    int32_t max_pos = 4096);

  const ModelConfig& config() const { return cfg_; }
  const Tokenizer& tokenizer() const { return tokenizer_; }
  const GgufFile& gguf() const { return gguf_; }
  const std::string& path() const { return gguf_.path(); }

  ThreadPool& pool() const { return *pool_; }

  const QuantMatrix& token_embd() const { return token_embd_; }
  const QuantMatrix& lm_head() const { return lm_head_; }
  const float* output_norm() const { return output_norm_; }
  const std::vector<LayerWeights>& layers() const { return layers_; }

  int64_t rope_positions() const { return n_rope_pos_; }
  const float* rope_cos(int64_t pos) const {
    return rope_.data() + pos * (cfg_.head_dim / 2) * 2;
  }
  const float* rope_sin(int64_t pos) const {
    return rope_.data() + pos * (cfg_.head_dim / 2) * 2 + cfg_.head_dim / 2;
  }

  // 全部张量的裸字节数之和（不含 f32 副本与 rope 表）。
  int64_t weight_bytes() const {
    int64_t n = 0;
    for (const GgufTensor& t : gguf_.tensors()) n += t.n_bytes;
    return n;
  }

  std::string Describe() const;

 private:
  const GgufTensor& Require(const std::string& name) const;
  void CheckShape(const GgufTensor& t, int64_t rows, int64_t cols) const;
  // 把 F32 类型的 norm 权重拷进 norm_pool_，返回稳定指针。
  const float* ImportNorm(const GgufTensor& t);
  void BuildRopeTable(int32_t max_pos);

  GgufFile gguf_;
  ModelConfig cfg_;
  Tokenizer tokenizer_;
  std::unique_ptr<ThreadPool> pool_;

  QuantMatrix token_embd_;
  QuantMatrix lm_head_;
  const float* output_norm_ = nullptr;
  std::vector<LayerWeights> layers_;

  // norm 权重是 F32 的，但依然拷出来：一是避免依赖 mmap 区 4 字节对齐，
  // 二是不依赖宿主字节序。总量约 0.5 MB，可以忽略。
  std::deque<std::vector<float>> norm_pool_;

  // rope 表：每个位置 2 * (head_dim/2) 个 float，前半 cos 后半 sin。
  std::vector<float> rope_;
  int64_t n_rope_pos_ = 0;
};

}  // namespace rpg

#endif  // RPGBAR_MODEL_H_
