// RPGBarEngine —— AVX2 融合反量化点积（M5）
//
// 本编译单元是**全项目唯一**开 -mavx2 -mfma 的地方（build.sh / CMakeLists 里
// 单独设的）。这样切分不是洁癖，是分发要求：库要丢进 Unity、发给别人的机器跑，
// 若整个工程都用 -march=native，库在没有 AVX2 的 CPU 上**加载即非法指令崩溃**。
// 把 AVX2 关进一个 TU，再由 matmul.cc 按 CPU 特性分派，旧机器只是走标量。
//
// 每个内核的共同形状：
//   * 在寄存器里把量化权重**现解成 f32**，立刻与激活做 FMA —— 不落内存。
//     这正是 M1 阶段慢的根因（DESIGN §14.5）：读一个权重元素要反量化一次，
//     还要落一趟栈。
//   * 4 个独立的向量累加器，最后一次性 HSum —— 打满 FMA 端口，同时让舍入
//     误差保持对数级（单串行累加器是线性，见 matmul.cc 的注释）。
//   * 循环按 32/64 元素整块展开，让乱序引擎把加载排开。
//
// 覆盖 Q4_K / Q6_K / Q8_0 / F32 —— 这是线上两份权重（Qwen3-1.7B 的 Q4_K_M 与
// Q8_0）用到的全部类型（Q4_K_M = Q4_K×168 + Q6_K×29 + F32×113；Q8_0 = Q8_0×197
// + F32×113）。其余类型（Q4_0/Q5_*/Q2_K/Q3_K/Q8_K/F16）**有意不做**：没有权重
// 用到它们，写了也是没测过的代码，走标量兜底更诚实。

#include "matmul.h"
#include "matmul_internal.h"

#include "common.h"
#include "dtypes.h"
#include "quant.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace rpg {
namespace kernels {

#if defined(__AVX2__)

namespace {

// ---- 小工具 ---------------------------------------------------------------

inline float HSum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  __m128 s = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(sh, s);
  s = _mm_add_ss(s, sh);
  return _mm_cvtss_f32(s);
}

// 8 个 uint8（取低 8 字节）→ 8 个 f32
inline __m256 U8x8ToF32(__m128i x) {
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(x));
}

// 8 个 int8 → 8 个 f32
inline __m256 I8x8ToF32(__m128i x) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(x));
}

// 8 个 uint8 → 8 个 int32（零扩展）
inline __m256i U8x8ToI32(__m128i x) { return _mm256_cvtepu8_epi32(x); }

// 融合反量化的一步：acc += a * (d*nib - mn)。
// 权重值 v = d*nib - mn 在寄存器里现算，不做内存往返。
inline __m256 FmaDeq(__m256 acc, const float* a, __m256 nib, __m256 d,
                     __m256 mn) {
  return _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_fmsub_ps(d, nib, mn), acc);
}

// ---- F32 ------------------------------------------------------------------
//
// 纯 fp32 权重（本模型的 F32 张量都是 norm，走不到这里；但 lm_head 若未量化
// 会用到）。任意长度都能处理，末尾走标量尾巴。
float DotF32Avx2(const float* a, const uint8_t* w, int64_t k) {
  const float* wp = reinterpret_cast<const float*>(w);
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 32 <= k; i += 32) {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(wp + i), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                         _mm256_loadu_ps(wp + i + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                         _mm256_loadu_ps(wp + i + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                         _mm256_loadu_ps(wp + i + 24), a3);
  }
  for (; i + 8 <= k; i += 8)
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(wp + i), a0);
  float s = HSum256(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                  _mm256_add_ps(a2, a3)));
  for (; i < k; ++i) s += a[i] * wp[i];
  return s;
}

// ---- Q8_0 -----------------------------------------------------------------
//
// 块 34 字节：d(fp16) | qs[32](int8)。值 = d * q。
// 一个块正好 32 个元素 = 4 条 AVX2 向量，展开成 4 路。
float DotQ8_0Avx2(const float* a, const uint8_t* w, int64_t k) {
  const int64_t nb = k / 32;
  __m256 o0 = _mm256_setzero_ps(), o1 = _mm256_setzero_ps();
  __m256 o2 = _mm256_setzero_ps(), o3 = _mm256_setzero_ps();
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = w + b * 34;
    const __m256 d = _mm256_set1_ps(F16ToF32(LoadU16(p)));
    const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
    const float* ap = a + b * 32;
    o0 = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 0),
                         _mm256_mul_ps(d, I8x8ToF32(_mm_loadl_epi64(
                                               reinterpret_cast<const __m128i*>(q)))),
                         o0);
    o1 = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 8),
                         _mm256_mul_ps(d, I8x8ToF32(_mm_loadl_epi64(
                                               reinterpret_cast<const __m128i*>(q + 8)))),
                         o1);
    o2 = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 16),
                         _mm256_mul_ps(d, I8x8ToF32(_mm_loadl_epi64(
                                               reinterpret_cast<const __m128i*>(q + 16)))),
                         o2);
    o3 = _mm256_fmadd_ps(_mm256_loadu_ps(ap + 24),
                         _mm256_mul_ps(d, I8x8ToF32(_mm_loadl_epi64(
                                               reinterpret_cast<const __m128i*>(q + 24)))),
                         o3);
  }
  return HSum256(_mm256_add_ps(_mm256_add_ps(o0, o1),
                              _mm256_add_ps(o2, o3)));
}

// ---- Q4_K -----------------------------------------------------------------
//
// 块 144 字节：d(fp16) | dmin(fp16) | scales[12](6-bit 打包) | qs[128]。
// 256 个元素分 4 组、每组 64：组内低半字节 → 前 32 个（scale/min = 第 2i 个），
// 高半字节 → 后 32 个（第 2i+1 个）。值 = d*sc*nib - dmin*mn。
//
// 6-bit 拆包直接调 quant.cc 里那份**标量** GetScaleMinK4Bytes —— 它每块只跑 8
// 次，相对 256 个元素的 FMA 完全不是瓶颈，没必要为了省这点开销再写一份容易错
// 的 SIMD 拆包（错了不崩、只静默算错）。
float DotQ4_KAvx2(const float* a, const uint8_t* w, int64_t k) {
  const int64_t nb = k / 256;
  __m256 o0 = _mm256_setzero_ps(), o1 = _mm256_setzero_ps();
  __m256 o2 = _mm256_setzero_ps(), o3 = _mm256_setzero_ps();
  const __m256i m4 = _mm256_set1_epi8(0x0F);
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = w + b * 144;
    const float d = F16ToF32(LoadU16(p + 0));
    const float dmin = F16ToF32(LoadU16(p + 2));
    const uint8_t* scales = p + 4;
    const uint8_t* qs = p + 16;
    const float* ap = a + b * 256;
    for (int g = 0; g < 4; ++g) {
      uint8_t sc0, mn0, sc1, mn1;
      GetScaleMinK4Bytes(2 * g + 0, scales, &sc0, &mn0);
      GetScaleMinK4Bytes(2 * g + 1, scales, &sc1, &mn1);
      const __m256 d0 = _mm256_set1_ps(d * static_cast<float>(sc0));
      const __m256 n0 = _mm256_set1_ps(dmin * static_cast<float>(mn0));
      const __m256 d1 = _mm256_set1_ps(d * static_cast<float>(sc1));
      const __m256 n1 = _mm256_set1_ps(dmin * static_cast<float>(mn1));

      const __m256i q = _mm256_loadu_si256(
          reinterpret_cast<const __m256i*>(qs + 32 * g));
      // 低半字节 / 高半字节：右移按 16 位做，再掩掉串过来的高位（ggml 同款）。
      const __m256i lo = _mm256_and_si256(q, m4);
      const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
      const __m128i lo0 = _mm256_castsi256_si128(lo);
      const __m128i lo1 = _mm256_extracti128_si256(lo, 1);
      const __m128i hi0 = _mm256_castsi256_si128(hi);
      const __m128i hi1 = _mm256_extracti128_si256(hi, 1);

      const float* alo = ap + 64 * g;
      const float* ahi = ap + 64 * g + 32;
      o0 = FmaDeq(o0, alo + 0, U8x8ToF32(lo0), d0, n0);
      o1 = FmaDeq(o1, alo + 8, U8x8ToF32(_mm_srli_si128(lo0, 8)), d0, n0);
      o2 = FmaDeq(o2, alo + 16, U8x8ToF32(lo1), d0, n0);
      o3 = FmaDeq(o3, alo + 24, U8x8ToF32(_mm_srli_si128(lo1, 8)), d0, n0);
      o0 = FmaDeq(o0, ahi + 0, U8x8ToF32(hi0), d1, n1);
      o1 = FmaDeq(o1, ahi + 8, U8x8ToF32(_mm_srli_si128(hi0, 8)), d1, n1);
      o2 = FmaDeq(o2, ahi + 16, U8x8ToF32(hi1), d1, n1);
      o3 = FmaDeq(o3, ahi + 24, U8x8ToF32(_mm_srli_si128(hi1, 8)), d1, n1);
    }
  }
  return HSum256(_mm256_add_ps(_mm256_add_ps(o0, o1),
                              _mm256_add_ps(o2, o3)));
}

// ---- Q6_K -----------------------------------------------------------------
//
// 块 210 字节：ql[128] | qh[64] | scales[16](int8) | d(fp16)。
// 256 个元素分两个 128 的半块；半块内 32 个 l 各自产 4 个值：
//   y[l+0]  = d*sc[is+0] * ((ql[l]&0xF) | ((qh[l]>>0&3)<<4) - 32)
//   y[l+32] = d*sc[is+2] * ((ql[l+32]&0xF) | ((qh[l]>>2&3)<<4) - 32)
//   y[l+64] = d*sc[is+4] * ((ql[l]>>4) | ((qh[l]>>4&3)<<4) - 32)
//   y[l+96] = d*sc[is+6] * ((ql[l+32]>>4) | ((qh[l]>>6&3)<<4) - 32)
// 其中 is = l/16（0 或 1）。
//
// 每个 8 元素向量需要：ql[l0..l0+7]、ql[l0+32..l0+39]、qh[l0..l0+7]。
void Q6Unpack8(__m128i qlb, __m128i qlb2, __m128i qhb, __m256* q1,
               __m256* q2, __m256* q3, __m256* q4) {
  const __m256i m4 = _mm256_set1_epi32(0x0F);
  const __m256i m2 = _mm256_set1_epi32(0x03);
  const __m256i b32 = _mm256_set1_epi32(32);
  const __m256i l = U8x8ToI32(qlb);
  const __m256i l2 = U8x8ToI32(qlb2);
  const __m256i h = U8x8ToI32(qhb);
  const __m256i i1 = _mm256_or_si256(
      _mm256_and_si256(l, m4), _mm256_slli_epi32(_mm256_and_si256(h, m2), 4));
  const __m256i i2 = _mm256_or_si256(
      _mm256_and_si256(l2, m4),
      _mm256_slli_epi32(
          _mm256_and_si256(_mm256_srli_epi32(h, 2), m2), 4));
  const __m256i i3 = _mm256_or_si256(
      _mm256_srli_epi32(l, 4),
      _mm256_slli_epi32(
          _mm256_and_si256(_mm256_srli_epi32(h, 4), m2), 4));
  const __m256i i4 = _mm256_or_si256(
      _mm256_srli_epi32(l2, 4),
      _mm256_slli_epi32(
          _mm256_and_si256(_mm256_srli_epi32(h, 6), m2), 4));
  *q1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(i1, b32));
  *q2 = _mm256_cvtepi32_ps(_mm256_sub_epi32(i2, b32));
  *q3 = _mm256_cvtepi32_ps(_mm256_sub_epi32(i3, b32));
  *q4 = _mm256_cvtepi32_ps(_mm256_sub_epi32(i4, b32));
}

float DotQ6_KAvx2(const float* a, const uint8_t* w, int64_t k) {
  const int64_t nb = k / 256;
  __m256 o0 = _mm256_setzero_ps(), o1 = _mm256_setzero_ps();
  __m256 o2 = _mm256_setzero_ps(), o3 = _mm256_setzero_ps();
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = w + b * 210;
    const uint8_t* ql = p;
    const uint8_t* qh = p + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
    const float d = F16ToF32(LoadU16(p + 208));
    const float* ap = a + b * 256;
    for (int half = 0; half < 2; ++half) {
      const uint8_t* qlh = ql + half * 64;
      const uint8_t* qhh = qh + half * 32;
      const int8_t* sch = sc + half * 8;
      const float* yh = ap + half * 128;
      for (int l0 = 0; l0 < 32; l0 += 8) {
        const __m128i qlb =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qlh + l0));
        const __m128i qlb2 =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qlh + l0 + 32));
        const __m128i qhb =
            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qhh + l0));
        __m256 q1, q2, q3, q4;
        Q6Unpack8(qlb, qlb2, qhb, &q1, &q2, &q3, &q4);
        const int si = l0 >> 4;  // 0（l0<16）或 1
        const __m256 s1 = _mm256_set1_ps(d * static_cast<float>(sch[si + 0]));
        const __m256 s2 = _mm256_set1_ps(d * static_cast<float>(sch[si + 2]));
        const __m256 s3 = _mm256_set1_ps(d * static_cast<float>(sch[si + 4]));
        const __m256 s4 = _mm256_set1_ps(d * static_cast<float>(sch[si + 6]));
        o0 = _mm256_fmadd_ps(_mm256_loadu_ps(yh + l0),
                             _mm256_mul_ps(s1, q1), o0);
        o1 = _mm256_fmadd_ps(_mm256_loadu_ps(yh + 32 + l0),
                             _mm256_mul_ps(s2, q2), o1);
        o2 = _mm256_fmadd_ps(_mm256_loadu_ps(yh + 64 + l0),
                             _mm256_mul_ps(s3, q3), o2);
        o3 = _mm256_fmadd_ps(_mm256_loadu_ps(yh + 96 + l0),
                             _mm256_mul_ps(s4, q4), o3);
      }
    }
  }
  return HSum256(_mm256_add_ps(_mm256_add_ps(o0, o1),
                              _mm256_add_ps(o2, o3)));
}

// ---- 分派 -----------------------------------------------------------------

float DotRowAvx2(uint32_t type, const float* a, const uint8_t* w, int64_t k) {
  switch (type) {
    case kGgmlQ4_K:
      RPG_CHECK(k % 256 == 0, "Q4_K 点积长度 %lld 不是 256 的整数倍",
                static_cast<long long>(k));
      return DotQ4_KAvx2(a, w, k);
    case kGgmlQ6_K:
      RPG_CHECK(k % 256 == 0, "Q6_K 点积长度 %lld 不是 256 的整数倍",
                static_cast<long long>(k));
      return DotQ6_KAvx2(a, w, k);
    case kGgmlQ8_0:
      RPG_CHECK(k % 32 == 0, "Q8_0 点积长度 %lld 不是 32 的整数倍",
                static_cast<long long>(k));
      return DotQ8_0Avx2(a, w, k);
    case kGgmlF32:
      return DotF32Avx2(a, w, k);
    default:
      // 没有 SIMD 内核的类型：在这里退回标量，而不是让调用方去判断。
      return DotRowQuantRef(type, a, w, k);
  }
}

}  // namespace

#endif  // __AVX2__

DotRowFn Avx2DotRow() {
#if defined(__AVX2__)
  return &DotRowAvx2;
#else
  return nullptr;
#endif
}

bool Avx2SupportsType(uint32_t type) {
#if defined(__AVX2__)
  switch (type) {
    case kGgmlQ4_K:
    case kGgmlQ6_K:
    case kGgmlQ8_0:
    case kGgmlF32:
      return true;
    default:
      return false;
  }
#else
  (void)type;
  return false;
#endif
}

}  // namespace kernels
}  // namespace rpg
