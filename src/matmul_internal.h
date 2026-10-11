// RPGBarEngine —— matmul 内核层内部接口（只给 matmul.cc / matmul_avx2.cc 用）
//
// 分开一个内部头的理由：AVX2 内核必须放在**单独一个编译单元**里（只有它开
// -mavx2 -mfma），而定义与分派又在不同 TU —— 两边需要一个共同声明的落点。
// 不要把这个头暴露给 model.cc 之类；上层只认 matmul.h。

#ifndef RPGBAR_MATMUL_INTERNAL_H_
#define RPGBAR_MATMUL_INTERNAL_H_

#include <cstdint>

namespace rpg {
namespace kernels {

using DotRowFn = float (*)(uint32_t, const float*, const uint8_t*, int64_t);

// 定义在 matmul_avx2.cc。
// 返回 AVX2 内核入口；**该编译单元没有真正编进 AVX2** 时返回 nullptr
// （编译器不认 -mavx2、或平台不支持），此时分派层退回标量。
//
// 入口内部对"没有 SIMD 内核的类型"自己退回 DotRowQuantRef —— 这样分派层只有
// "avx2 / scalar"两个状态，不必知道每个类型的能力表。
DotRowFn Avx2DotRow();

// 该类型是否真有 SIMD 内核（不是退回标量）。定义在 matmul_avx2.cc；
// 未编入 AVX2 时恒为 false。
bool Avx2SupportsType(uint32_t type);

}  // namespace kernels
}  // namespace rpg

#endif  // RPGBAR_MATMUL_INTERNAL_H_
