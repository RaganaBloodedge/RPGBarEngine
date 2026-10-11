#include "model.h"

#include <cstdio>
#include <cstring>

#include "common.h"
#include "dtypes.h"
#include "kernels.h"
#include "matmul.h"

namespace rpg {
namespace {

// GGUF 里一个张量的形状是"反着写"的：dims[0] 是最快维（连续维）。
// 所以对权重矩阵来说 dims = [cols, rows] —— 这一行注释值 100 个 bug。
int64_t Cols(const GgufTensor& t) { return t.cols(); }
int64_t Rows(const GgufTensor& t) { return t.rows(); }

}  // namespace

// ---------------------------------------------------------------- QuantMatrix

void QuantMatrix::MatVec(const float* x, float* out) const {
  RPG_CHECK(valid(), "对无效矩阵做 MatVec");
  const int64_t row_bytes = TypeRowBytes(type_, cols_);
  const uint8_t* p = data_;
  for (int64_t r = 0; r < rows_; ++r, p += row_bytes) {
    out[r] = DotRowQuant(type_, x, p, cols_);
  }
}

void QuantMatrix::Row(int64_t r, float* out) const {
  RPG_CHECK(valid(), "对无效矩阵取行");
  RPG_CHECK(r >= 0 && r < rows_, "行号 %lld 越界（共 %lld 行）",
            static_cast<long long>(r), static_cast<long long>(rows_));
  DequantizeRow(type_, data_ + r * TypeRowBytes(type_, cols_), cols_, out);
}

void QuantMatrix::MatMulBatch(const float* X, int64_t n_tokens, float* OUT,
                              ThreadPool* pool) const {
  RPG_CHECK(valid(), "对无效矩阵做 MatMulBatch");
  const int64_t row_bytes = TypeRowBytes(type_, cols_);

  // 展平成 n_tokens * rows 个独立的点积：第 i 个 = (t = i / rows, r = i % rows)。
  // 这样切分天然按行连续，且同一 token 的输出写在一起。
  auto body = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t t = i / rows_;
      const int64_t r = i - t * rows_;
      OUT[t * rows_ + r] =
          DotRowQuant(type_, X + t * cols_, data_ + r * row_bytes, cols_);
    }
  };

  const int64_t total = n_tokens * rows_;
  if (pool && total > 256) {
    pool->ParallelForRange(total, body);
  } else {
    body(0, total);
  }
}

// ---------------------------------------------------------------------- Model

const GgufTensor& Model::Require(const std::string& name) const {
  const GgufTensor* t = gguf_.Find(name);
  if (!t) {
    Throw("GGUF 里找不到张量 %s（架构 %s，共 %lld 个张量）", name.c_str(),
          cfg_.arch.c_str(),
          static_cast<long long>(gguf_.tensors().size()));
  }
  return *t;
}

void Model::CheckShape(const GgufTensor& t, int64_t rows, int64_t cols) const {
  if (t.dims.size() != 2 || Rows(t) != rows || Cols(t) != cols) {
    char got[64];
    std::snprintf(got, sizeof(got), "[%lld,%lld]",
                  static_cast<long long>(Rows(t)),
                  static_cast<long long>(Cols(t)));
    if (!t.is_matrix()) std::snprintf(got, sizeof(got), "%lld 维，不是矩阵",
                                     static_cast<long long>(t.dims.size()));
    Throw("张量 %s 形状不符：期望 [%lld,%lld]（行×列），实际 %s",
          t.name.c_str(), static_cast<long long>(rows),
          static_cast<long long>(cols), got);
  }
  RPG_CHECK(TypeSupported(t.type), "张量 %s 的类型 %s 未实现", t.name.c_str(),
            TypeName(t.type));
}

const float* Model::ImportNorm(const GgufTensor& t) {
  // norm 权重必须是 1 维且为 F32。GGUF 量化工具会刻意把 norm 留成 F32 —— 它们
  // 参与每次归一化，量化误差会被逐层放大。
  if (t.dims.size() != 1) {
    Throw("张量 %s 期望 1 维 norm 权重，实际 %lld 维", t.name.c_str(),
          static_cast<long long>(t.dims.size()));
  }
  norm_pool_.emplace_back(static_cast<size_t>(t.n_elements));
  std::vector<float>& buf = norm_pool_.back();
  if (t.type == kGgmlF32) {
    const uint8_t* src = gguf_.TensorBytes(t);
    for (int64_t i = 0; i < t.n_elements; ++i) {
      buf[i] = LoadF32(src + i * 4);
    }
  } else {
    // 少数 GGUF 会把 norm 也量化。支持它，但要留个日志。
    RPG_LOG_WARN("norm 权重 %s 是 %s，已反量化（精度略有损失）", t.name.c_str(),
                 TypeName(t.type));
    gguf_.DequantizeTo(t, buf.data());
  }
  return buf.data();
}

void Model::BuildRopeTable(int32_t max_pos) {
  const int32_t half = cfg_.head_dim / 2;
  n_rope_pos_ = max_pos > cfg_.context_length ? cfg_.context_length : max_pos;
  if (n_rope_pos_ <= 0) n_rope_pos_ = 1;
  rope_.assign(static_cast<size_t>(n_rope_pos_) * half * 2, 0.0f);

  // inv_freq[i] = theta^(-2i/head_dim)，i ∈ [0, head_dim/2)
  std::vector<double> inv(static_cast<size_t>(half));
  for (int32_t i = 0; i < half; ++i) {
    const double exp = static_cast<double>(2 * i) / static_cast<double>(cfg_.head_dim);
    inv[i] = 1.0 / std::pow(static_cast<double>(cfg_.rope_theta), exp);
  }

  // 用 double 算角度再降成 float：rope_theta = 1e6 时 pos * inv_freq 在大位置上
  // 能到几百弧度，float 里的角度误差会明显放大。
  for (int64_t p = 0; p < n_rope_pos_; ++p) {
    float* row = rope_.data() + p * half * 2;
    for (int32_t i = 0; i < half; ++i) {
      const double ang = static_cast<double>(p) * inv[i];
      row[i] = static_cast<float>(std::cos(ang));
      row[half + i] = static_cast<float>(std::sin(ang));
    }
  }
}

Model Model::Load(const std::string& path, int n_threads, int32_t max_pos) {
  Model m;
  m.gguf_ = GgufFile::Open(path);

  const std::string arch = m.gguf_.architecture();
  if (arch != "qwen3") {
    Throw("模型架构是 %s，本引擎（当前）只实现 qwen3", arch.c_str());
  }
  ModelConfig& c = m.cfg_;
  c.arch = arch;
  c.name = m.gguf_.MetaStr("general.name", "(无名)");

  c.n_layers = static_cast<int32_t>(m.gguf_.ArchInt("block_count", 0));
  c.hidden = static_cast<int32_t>(m.gguf_.ArchInt("embedding_length", 0));
  c.n_heads = static_cast<int32_t>(m.gguf_.ArchInt("attention.head_count", 0));
  c.n_kv_heads = static_cast<int32_t>(
      m.gguf_.ArchInt("attention.head_count_kv", c.n_heads));
  c.intermediate = static_cast<int32_t>(m.gguf_.ArchInt("feed_forward_length", 0));
  c.context_length = static_cast<int32_t>(m.gguf_.ArchInt("context_length", 4096));

  // head_dim 只能读 key_length。**不可以**退化成 hidden / n_heads：
  // 1.7B 上 2048/16 = 128 恰好相等，4B 上是 2560/32 = 80 ≠ 128。用推导的写法
  // 在 1.7B 上能跑通、换 4B 就静默算错。
  const int64_t key_len = m.gguf_.ArchInt("attention.key_length", 0);
  if (key_len > 0) {
    c.head_dim = static_cast<int32_t>(key_len);
  } else {
    RPG_CHECK(c.n_heads > 0 && c.hidden % c.n_heads == 0,
              "缺 attention.key_length 且 %d 不被 %d 整除，无法推 head_dim",
              c.hidden, c.n_heads);
    RPG_LOG_WARN("GGUF 缺 attention.key_length，按 hidden/heads 推得 %d", c.hidden / c.n_heads);
    c.head_dim = c.hidden / c.n_heads;
  }
  c.value_dim = static_cast<int32_t>(
      m.gguf_.ArchInt("attention.value_length", c.head_dim));

  c.rope_theta = m.gguf_.ArchFloat("rope.freq_base", 10000.0f);
  c.rms_eps = m.gguf_.ArchFloat("attention.layer_norm_rms_epsilon", 1e-6f);

  c.bos_id = static_cast<int32_t>(m.gguf_.MetaInt("tokenizer.ggml.bos_token_id", -1));
  c.eos_id = static_cast<int32_t>(m.gguf_.MetaInt("tokenizer.ggml.eos_token_id", -1));
  c.pad_id = static_cast<int32_t>(m.gguf_.MetaInt("tokenizer.ggml.padding_token_id", -1));
  c.add_bos = m.gguf_.MetaBool("tokenizer.ggml.add_bos_token", false);

  // ---- 基本一致性 ----
  RPG_CHECK(c.n_layers > 0 && c.hidden > 0 && c.n_heads > 0 &&
                c.n_kv_heads > 0 && c.head_dim > 0 && c.intermediate > 0,
            "GGUF 元数据不全：layers=%d hidden=%d heads=%d kv=%d head_dim=%d ffn=%d",
            c.n_layers, c.hidden, c.n_heads, c.n_kv_heads, c.head_dim,
            c.intermediate);
  RPG_CHECK(c.n_heads % c.n_kv_heads == 0,
            "Q 头数 %d 不是 KV 头数 %d 的整数倍，GQA 无法分组", c.n_heads,
            c.n_kv_heads);
  RPG_CHECK(c.value_dim == c.head_dim,
            "value_length=%d 与 key_length=%d 不同，当前实现假设两者相等",
            c.value_dim, c.head_dim);
  if (c.context_length <= 0) c.context_length = 4096;

  // ---- 词表与分词器 ----
  m.tokenizer_ = Tokenizer::FromGguf(m.gguf_);
  c.vocab = static_cast<int32_t>(m.tokenizer_.vocab_size());

  // ---- 权重 ----
  const GgufTensor& emb = m.Require("token_embd.weight");
  m.CheckShape(emb, c.vocab, c.hidden);
  m.token_embd_ = QuantMatrix(m.gguf_.TensorBytes(emb), emb.type, Rows(emb), Cols(emb));

  if (const GgufTensor* out = m.gguf_.Find("output.weight")) {
    m.CheckShape(*out, c.vocab, c.hidden);
    m.lm_head_ = QuantMatrix(m.gguf_.TensorBytes(*out), out->type, Rows(*out), Cols(*out));
    c.tied_embeddings = false;
  } else {
    // tied：lm_head 与 embedding 共享同一份量化权重，**不重复加载**。
    m.lm_head_ = m.token_embd_;
    c.tied_embeddings = true;
  }
  m.output_norm_ = m.ImportNorm(m.Require("output_norm.weight"));
  RPG_CHECK(m.Require("output_norm.weight").n_elements == c.hidden,
            "output_norm 长度 %lld != hidden %d",
            static_cast<long long>(m.Require("output_norm.weight").n_elements),
            c.hidden);

  m.layers_.resize(static_cast<size_t>(c.n_layers));
  for (int32_t l = 0; l < c.n_layers; ++l) {
    LayerWeights& w = m.layers_[static_cast<size_t>(l)];
    // 用 lambda 而不是宏：宏里塞语句，既不方便调试也不便移植
    // （GCC 的语句表达式 MSVC 不支持）。
    const int32_t layer = l;
    auto tensor = [&m, layer](const char* suffix) -> const GgufTensor& {
      char nm[64];
      std::snprintf(nm, sizeof(nm), "blk.%d.%s", layer, suffix);
      return m.Require(nm);
    };

    w.attn_norm = m.ImportNorm(tensor("attn_norm.weight"));
    w.ffn_norm = m.ImportNorm(tensor("ffn_norm.weight"));
    w.q_norm = m.ImportNorm(tensor("attn_q_norm.weight"));
    w.k_norm = m.ImportNorm(tensor("attn_k_norm.weight"));

    const GgufTensor& q = tensor("attn_q.weight");
    const GgufTensor& k = tensor("attn_k.weight");
    const GgufTensor& v = tensor("attn_v.weight");
    const GgufTensor& o = tensor("attn_output.weight");
    const GgufTensor& gate = tensor("ffn_gate.weight");
    const GgufTensor& up = tensor("ffn_up.weight");
    const GgufTensor& down = tensor("ffn_down.weight");

    m.CheckShape(q, c.q_dim(), c.hidden);
    m.CheckShape(k, c.kv_dim(), c.hidden);
    m.CheckShape(v, c.kv_dim(), c.hidden);
    m.CheckShape(o, c.hidden, c.q_dim());
    m.CheckShape(gate, c.intermediate, c.hidden);
    m.CheckShape(up, c.intermediate, c.hidden);
    m.CheckShape(down, c.hidden, c.intermediate);

    w.attn_q = QuantMatrix(m.gguf_.TensorBytes(q), q.type, Rows(q), Cols(q));
    w.attn_k = QuantMatrix(m.gguf_.TensorBytes(k), k.type, Rows(k), Cols(k));
    w.attn_v = QuantMatrix(m.gguf_.TensorBytes(v), v.type, Rows(v), Cols(v));
    w.attn_o = QuantMatrix(m.gguf_.TensorBytes(o), o.type, Rows(o), Cols(o));
    w.ffn_gate = QuantMatrix(m.gguf_.TensorBytes(gate), gate.type, Rows(gate), Cols(gate));
    w.ffn_up = QuantMatrix(m.gguf_.TensorBytes(up), up.type, Rows(up), Cols(up));
    w.ffn_down = QuantMatrix(m.gguf_.TensorBytes(down), down.type, Rows(down), Cols(down));
  }

  // ---- 线程池与 rope ----
  const int nt = n_threads > 0 ? n_threads : DefaultThreads();
  m.pool_ = std::make_unique<ThreadPool>(nt);
  m.BuildRopeTable(max_pos);

  RPG_LOG_INFO("已加载 %s：%d 层 / hidden %d / Q %d 头 / KV %d 头 / head_dim %d / "
               "词表 %d / tied=%s / 线程 %d",
               c.name.c_str(), c.n_layers, c.hidden, c.n_heads, c.n_kv_heads,
               c.head_dim, c.vocab, c.tied_embeddings ? "是" : "否", nt);
  return m;
}

std::string Model::Describe() const {
  std::string s;
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "%s（%s）\n"
                "  规模      : %d 层 / hidden %d / ffn %d\n"
                "  注意力    : %d Q 头 / %d KV 头 / head_dim %d（每组 %d 个 Q 头）\n"
                "  词表      : %d\n"
                "  位置编码  : RoPE theta=%.6g，表覆盖 %lld 个位置（上限 %d）\n"
                "  归一化    : RMSNorm eps=%.3g，norm 权重 F32\n"
                "  输出头    : %s\n"
                "  权重字节  : %lld（%.2f GiB）\n"
                "  线程      : %d\n",
                cfg_.name.c_str(), cfg_.arch.c_str(), cfg_.n_layers, cfg_.hidden,
                cfg_.intermediate, cfg_.n_heads, cfg_.n_kv_heads, cfg_.head_dim,
                cfg_.heads_per_kv(), cfg_.vocab, cfg_.rope_theta,
                static_cast<long long>(n_rope_pos_), cfg_.context_length,
                static_cast<double>(cfg_.rms_eps),
                cfg_.tied_embeddings ? "与 token_embd 共享（tied）" : "独立 output.weight",
                static_cast<long long>(weight_bytes()),
                static_cast<double>(weight_bytes()) / (1024.0 * 1024.0 * 1024.0),
                pool_ ? pool_->size() : 0);
  s += buf;
  return s;
}

}  // namespace rpg
