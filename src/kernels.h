// RPGBarEngine —— 标量算子
//
// 这里放"一条向量进、一条向量出"的小算子：归一化、softmax、激活、RoPE。
// 接口刻意**不暴露任何布局细节**（没有步长、没有对齐要求、没有 SIMD 宽度），
// 因为 M5 会把内部换成 AVX2 —— 到时候只改这一个文件，上层不动。
//
// 数值精度取舍：归一化的平方和与 softmax 的求和在 double 里累加。
// Torch 在 float32 里用分块求和，两者差 ~1e-7 相对量级 —— 远小于 M1 的
// 1e-4 判据，但换来的是"我们的答案更接近精确值"，出问题时更容易归因。
// 唯一例外是 attention 的加权求和：那是 fma 累加的浮点乘加，必须和 torch
// 一样用 float 累加，否则反而引入额外舍入。

#ifndef RPGBAR_KERNELS_H_
#define RPGBAR_KERNELS_H_

#include <cmath>
#include <cstdint>

namespace rpg {

// 点积。attention 的 QKᵀ 与逐头加权求和都用它。
//
// 4 个累加器而不是 1 个：单串行累加器的舍入误差随长度线性增长，分块则是对数
// 级。原因是实测出来的 —— M1 对拍时我们的误差是参考实现的 4–5 倍，换成多累加
// 器就回到同一水平。这个结构也正好是 M5 里 SIMD 的 4 个 lane。
inline float Dot(const float* a, const float* b, int64_t n) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  int64_t i = 0;
  const int64_t lim = n - 3;
  for (; i < lim; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// 最大值的下标。并列时取**最小下标**，与 torch.argmax 一致 ——
// 贪心解码的每一步都靠它，差一个并列规则就会让"逐 token 一致"的判据红。
inline int32_t ArgMax(const float* x, int64_t n) {
  int32_t best = 0;
  float bv = x[0];
  for (int64_t i = 1; i < n; ++i) {
    if (x[i] > bv) {  // 严格大于 → 并列保留最靠前的那个
      bv = x[i];
      best = static_cast<int32_t>(i);
    }
  }
  return best;
}

// RMSNorm：out = x * rsqrt(mean(x²) + eps) * w
// Qwen3 的权重就是 w 本身（不像 Gemma 2 那样用 1+w），也没有 bias。
inline void RmsNorm(const float* x, const float* w, int64_t n, float eps,
                    float* out) {
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(x[i]) * x[i];
  const double mean = sum / static_cast<double>(n);
  const float scale = static_cast<float>(1.0 / std::sqrt(mean + eps));
  for (int64_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

// 就地 softmax（数值稳定版：先减最大值）。
inline void SoftmaxInPlace(float* x, int64_t n) {
  if (n <= 0) return;
  float mx = x[0];
  for (int64_t i = 1; i < n; ++i)
    if (x[i] > mx) mx = x[i];
  double sum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const float e = std::exp(x[i] - mx);
    x[i] = e;
    sum += e;
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (int64_t i = 0; i < n; ++i) x[i] *= inv;
}

// SiLU：x / (1 + e^-x)。写成乘倒数而不是除，避免重复算指数。
inline void SiluInPlace(float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float v = x[i];
    x[i] = v / (1.0f + std::exp(-v));
  }
}

// 逐元素乘：a[i] *= b[i]。SwiGLU 的 silu(gate) * up 用它。
inline void MulInPlace(float* a, const float* b, int64_t n) {
  for (int64_t i = 0; i < n; ++i) a[i] *= b[i];
}

inline void VecAddInPlace(float* a, const float* b, int64_t n) {
  for (int64_t i = 0; i < n; ++i) a[i] += b[i];
}

inline void CopyVec(float* dst, const float* src, int64_t n) {
  for (int64_t i = 0; i < n; ++i) dst[i] = src[i];
}

// RoPE（**半维配对**，不是相邻配对）。
//
// 这一点必须记死：HF 的 rotate_half 把一条 head 向量从中间切开，
// 让 i 与 i + head_dim/2 组成一对 —— 而 Gemma 的 RoPE 是相邻配对 (2i, 2i+1)。
// 两种写法的 cos/sin 表长度一样、形状一样，错了不会崩，只会让注意力错位，
// 表现为"能出词但胡说"。cos/sin 各 head_dim/2 个，由 Model 预先算好。
inline void RopeInPlace(float* x, int32_t n_heads, int32_t head_dim,
                        const float* cos_t, const float* sin_t) {
  const int32_t half = head_dim / 2;
  for (int32_t h = 0; h < n_heads; ++h) {
    float* xh = x + static_cast<int64_t>(h) * head_dim;
    for (int32_t i = 0; i < half; ++i) {
      const float c = cos_t[i], s = sin_t[i];
      const float a = xh[i], b = xh[i + half];
      xh[i] = a * c - b * s;
      xh[i + half] = b * c + a * s;
    }
  }
}

}  // namespace rpg

#endif  // RPGBAR_KERNELS_H_
