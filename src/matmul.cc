// RPGBarEngine —— 融合点积的分派与标量参考实现（M5）
//
// 本文件做两件事：
//   1. 扮演"内核入口"：DotRowQuant 按 CPU 特性选 AVX2 或标量（选择在**静态
//      初始化期**做一次，之后只读函数指针 —— 于是 worker 线程并发调用不会有
//      竞态，也不需要锁）。
//   2. 留住**标量参考实现**：它原先是 quant.cc 里的 DotRowQuant，现在改名为
//      DotRowQuantRef。判据不变 —— SIMD 必须与它一致。

#include "matmul.h"

#include <cstdlib>

#include "common.h"
#include "matmul_internal.h"
#include "quant.h"

namespace rpg {
namespace {

using kernels::DotRowFn;

// CPU 是否支持 AVX2 + FMA。
//
// 只在"选内核"时问一次。MSVC 分支先返回 false：CPUID 探测留到 M6 与交叉
// 编译一起做（那条路上才有 MSVC 参与），当前 MSVC 构建会稳定走标量。
bool CpuHasAvx2() {
#if defined(__GNUC__) || defined(__clang__)
  // __builtin_cpu_init() 在静态初始化期调用是必须的：GCC 的文档明确说
  // __builtin_cpu_supports 若在构造函数里用要先 init，否则读到未初始化的表。
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

// 环境变量 RPGBAR_NO_SIMD=1 时强制标量。给对照基准用：同一台机器上跑
// "有 SIMD / 无 SIMD"两遍，收益才归因得干净（见 tools/dotrow_check.py）。
bool SimdDisabledByEnv() {
  const char* v = std::getenv("RPGBAR_NO_SIMD");
  return v != nullptr && v[0] != '\0' && v[0] != '0';
}

// 选一次，之后只读。静态初始化顺序问题在这里不存在：SelectKernel 只依赖本
// TU 的 CpuHasAvx2 / SimdDisabledByEnv，以及 matmul_avx2.cc 的两个函数（它们
// 不读任何全局状态）。
DotRowFn SelectKernel() {
  if (SimdDisabledByEnv()) return &DotRowQuantRef;
  if (!CpuHasAvx2()) return &DotRowQuantRef;
  DotRowFn f = kernels::Avx2DotRow();
  return f != nullptr ? f : &DotRowQuantRef;
}

const DotRowFn g_dot_row = SelectKernel();

}  // namespace

// ------------------------------------------------------------ 标量参考实现

float DotRowQuantRef(uint32_t id, const float* a, const uint8_t* w, int64_t k) {
  // 一次解一个块（块最大 256 元素），边解边点积。
  float tmp[kMaxBlockElems];
  const GgmlTypeInfo* t = TypeInfo(id);
  RPG_CHECK(t != nullptr, "未知的 ggml 类型 id=%u", id);
  RPG_CHECK(k % t->block_elems == 0, "点积长度 %lld 不是块大小 %lld 的整数倍",
            static_cast<long long>(k), static_cast<long long>(t->block_elems));

  // **多个累加器**，不是图省事，是为了精度。
  // 单个串行累加器的舍入误差随长度**线性**增长（n 项最多积累 n·u·Σ|term|），
  // 而 BLAS 的分块/向量化求和是对数级。k 到 6144 时这个差距实测是 4–5 倍 ——
  // 与 M1 对拍时"我们比参考离真值远 5 倍"的现象完全对得上。
  // 4 个独立部分和正好对应 SIMD 的 4 个 lane，所以这个结构可以直接留用。
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  const int64_t nb = k / t->block_elems;
  for (int64_t b = 0; b < nb; ++b) {
    DequantizeRow(id, w + b * t->block_bytes, t->block_elems, tmp);
    const float* ap = a + b * t->block_elems;
    int64_t j = 0;
    const int64_t lim = t->block_elems - 3;
    for (; j < lim; j += 4) {
      s0 += ap[j] * tmp[j];
      s1 += ap[j + 1] * tmp[j + 1];
      s2 += ap[j + 2] * tmp[j + 2];
      s3 += ap[j + 3] * tmp[j + 3];
    }
    for (; j < t->block_elems; ++j) s0 += ap[j] * tmp[j];
  }
  return (s0 + s1) + (s2 + s3);
}

// ---------------------------------------------------------------- 分派入口

float DotRowQuant(uint32_t type, const float* a, const uint8_t* w, int64_t k) {
  return g_dot_row(type, a, w, k);
}

const char* ActiveKernelName() {
  return g_dot_row == &DotRowQuantRef ? "scalar" : "avx2";
}

bool HasSimdKernel(uint32_t type) {
  if (g_dot_row == &DotRowQuantRef) return false;
  return kernels::Avx2SupportsType(type);
}

}  // namespace rpg
