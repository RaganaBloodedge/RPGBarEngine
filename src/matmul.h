// RPGBarEngine —— 量化权重的融合点积内核（M5）
//
// DESIGN §7 里写的是"移植 TinyGemmaCpp 的 matmul.cc"。真正动手时有一条必须
// 先说清楚：**能移植的是结构，不是 kernel 本体**。参考实现的内核针对 .sbs 的
// SFP/BF16/F32（gemma.cpp 的私有格式），本引擎的权重是 GGUF 的 ggml 量化类型
// （Q4_K/Q6_K/Q8_0…），位布局完全不同 —— 照抄 kernel 一行都用不上。
//
// 沿用过来的是三件事，它们才是性能的来源：
//
//   1. **融合反量化**：不把权重整行解成 f32 再点积，而是在寄存器里现解现 FMA。
//      M1 阶段慢的根因正是"读每个权重元素都要现反量化一次 + 落一趟内存"
//      （DESIGN §14.5），融合之后这趟内存往返就没了。
//   2. **四个独立累加器**：既打满 FMA 端口，也让舍入误差保持对数级增长 ——
//      与 kernels.h 的 Dot 同一条理由（单串行累加器的误差随长度线性增长）。
//   3. **按输出行并行**：行与行互不依赖，天然可切（MatMulBatch 已经在做）。
//
// 分派策略：**AVX2 + FMA 可用才走 SIMD**，否则退回标量。标量实现**不删**，
// 它有两个不可替代的用途 ——
//   * 非 AVX2 机器的兜底（库里没有 AVX2 指令在跑就不会非法指令崩溃）；
//   * 对拍基准：SIMD 内核必须与标量一致（tools/check_dotrow.py，第 11 层）。

#ifndef RPGBAR_MATMUL_H_
#define RPGBAR_MATMUL_H_

#include <cstdint>

namespace rpg {

// 点积：a[0..k) 与「ggml 量化类型 `type` 的一行权重」（w 指向该行起始字节）。
//
// 这是 MatVec / MatMulBatch 的唯一内核入口，内部按 CPU 特性分派。
// 前置条件与标量版一致：type 受支持、k 是该类型块大小的整数倍。
float DotRowQuant(uint32_t type, const float* a, const uint8_t* w, int64_t k);

// 标量参考实现。**不参与分派**，只给兜底与对拍用。
float DotRowQuantRef(uint32_t type, const float* a, const uint8_t* w, int64_t k);

// 当前实际生效的内核名："avx2" 或 "scalar"。给探针与日志用。
const char* ActiveKernelName();

// 该类型是否有 SIMD 内核（没有的类型走标量，这是正常降级，不是错误）。
bool HasSimdKernel(uint32_t type);

}  // namespace rpg

#endif  // RPGBAR_MATMUL_H_
