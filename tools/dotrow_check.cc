// dotrow_check —— M5 融合点积内核的导出探针
//
// 目的：把「SIMD 内核算出的点积」与「标量参考算出的点积」并排导出，交给
// tools/check_dotrow.py 判据（第 11 层）。
//
// 输入用**真实权重张量**，不用随机造的字节。理由：随机字节要造出"合法的
// 量化块"就得在工具里再实现一遍各类型的位布局 —— 那份实现一旦和 quant.cc
// 不一致，检查本身就失去意义（两个错的东西互相印证）。而从模型里取一行真的
// 权重，走的就是线上那条路。
//
// 激活向量是随机造的（这没问题：激活本来就是普通 f32，没有布局）。
//
// 用法:
//   dotrow_check <model.gguf> <n> <seed> <out_prefix> [--k <k>]
//
// 对模型里**每个有 SIMD 内核的类型**取第一张该类型的矩阵张量、第 0 行，
// 写四份 npy：
//   <前缀>.<类型>.a.npy       激活矩阵 (n, k)
//   <前缀>.<类型>.w.npy       该行反量化后的权重 (k,)（走标量 DequantizeRow）
//   <前缀>.<类型>.simd.npy    SIMD 入口算出的 n 个点积
//   <前缀>.<类型>.scalar.npy  标量参考算出的 n 个点积
//
// 第一行还会打印 kernel=<avx2|scalar>；Python 侧据此决定"要不要断言 SIMD
// 真的生效"——没有 AVX2 的机器上这个脚本应当照常通过（只是检查退化为恒等）。

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"
#include "matmul.h"
#include "npy.h"
#include "quant.h"

namespace {

// 固定种子的 xorshift64*。确定性 → 出问题时能原样重跑。
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
  uint64_t Next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 0x2545F4914F6CDD1Dull;
  }
  // [0,1)
  float Unit() {
    return static_cast<float>(Next() >> 11) * (1.0f / 9007199254740992.0f);
  }
};

int64_t ParseInt(const char* s) { return std::strtoll(s, nullptr, 10); }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "用法: dotrow_check <model.gguf> <n> <seed> <out_prefix> "
                 "[--k <k>]\n");
    return 2;
  }
  const std::string model = argv[1];
  const int64_t n = ParseInt(argv[2]);
  const uint64_t seed = static_cast<uint64_t>(ParseInt(argv[3]));
  const std::string prefix = argv[4];
  int64_t k_cap = 0;
  for (int i = 5; i < argc; ++i) {
    if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) k_cap = ParseInt(argv[++i]);
  }

  rpg::GgufFile f = rpg::GgufFile::Open(model);
  std::printf("kernel=%s\n", rpg::ActiveKernelName());

  // 每个类型挑**列数最大**的那张矩阵张量：k=6144 比 2048 更能暴露累加顺序的
  // 问题（4 路部分和的分段边界不同、尾巴处理不同），而且 Q4_K_M 里 ffn_* 就是
  // 6144，正是线上的热点形状。
  std::vector<const rpg::GgufTensor*> pick;
  for (const auto& t : f.tensors()) {
    if (!t.is_matrix() || !rpg::HasSimdKernel(t.type)) continue;
    bool placed = false;
    for (auto*& p : pick) {
      if (p->type == t.type) {
        if (t.cols() > p->cols()) p = &t;  // 同类型取列数更大的
        placed = true;
        break;
      }
    }
    if (!placed) pick.push_back(&t);
  }

  int n_type = 0;
  for (const rpg::GgufTensor* tp : pick) {
    const rpg::GgufTensor& t = *tp;
    const int64_t block = rpg::TypeInfo(t.type)->block_elems;
    int64_t k = t.cols();
    if (k_cap > 0 && k_cap < k) k = (k_cap / block) * block;
    if (k <= 0) continue;

    const auto* info = rpg::TypeInfo(t.type);
    const uint8_t* row = f.TensorBytes(t);  // 第 0 行
    std::vector<float> w(static_cast<size_t>(k));
    rpg::DequantizeRow(t.type, row, k, w.data());

    std::vector<float> a(static_cast<size_t>(n) * k);
    std::vector<float> simd(static_cast<size_t>(n));
    std::vector<float> scal(static_cast<size_t>(n));
    Rng rng(seed + t.type);
    const float scales[4] = {0.5f, 1.0f, 3.0f, 12.0f};
    for (int64_t i = 0; i < n; ++i) {
      float* ai = a.data() + i * k;
      const float sc = scales[i & 3];
      for (int64_t j = 0; j < k; ++j) ai[j] = (rng.Unit() * 2.0f - 1.0f) * sc;
      simd[i] = rpg::DotRowQuant(t.type, ai, row, k);
      scal[i] = rpg::DotRowQuantRef(t.type, ai, row, k);
    }

    const std::string tn = info->name;
    npy::WriteF32(prefix + "." + tn + ".a.npy", {n, k}, a.data(), a.size());
    npy::WriteF32(prefix + "." + tn + ".w.npy", {k}, w.data(), w.size());
    npy::WriteF32(prefix + "." + tn + ".simd.npy", {n}, simd.data(), simd.size());
    npy::WriteF32(prefix + "." + tn + ".scalar.npy", {n}, scal.data(), scal.size());

    std::printf("类型 %-6s 张量 %-28s k=%lld n=%lld\n", tn.c_str(), t.name.c_str(),
                static_cast<long long>(k), static_cast<long long>(n));
    ++n_type;
  }

  if (n_type == 0) {
    std::fprintf(stderr, "模型里没有任何带 SIMD 内核的矩阵张量\n");
    return 1;
  }
  std::printf("共 %d 个类型\n", n_type);
  return 0;
}
