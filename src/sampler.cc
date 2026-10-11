#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "kernels.h"

namespace rpg {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace

uint64_t SplitMix64(uint64_t x) {
  // 三个奇数常量的乘法混合（原论文给的就是这三个），最后异或右移。
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

uint64_t CounterRng(uint64_t seed, uint64_t counter) {
  // 每个 counter 一条独立的"流"：乘一个奇常量把相邻 counter 拉开，再喂给
  // SplitMix64。这样 counter 差 1 的两个随机数之间没有可见的相关性。
  return SplitMix64(seed + 0x9E3779B97F4A7C15ull * counter);
}

float CounterUniform(uint64_t seed, uint64_t counter) {
  // 取高 24 位：float 只有 24 位有效位，取高位既用满精度又不碰质量较差的低位。
  const uint32_t bits = static_cast<uint32_t>(CounterRng(seed, counter) >> 40);
  return static_cast<float>(bits) * (1.0f / 16777216.0f);  // / 2^24
}

void FilterToProbs(const float* logits, int64_t n, const SampleOptions& opt,
                   std::vector<float>* probs) {
  probs->assign(logits, logits + n);
  float* s = probs->data();

  // 贪心：直接给 argmax 一个 one-hot，绕过下面所有筛选（也就不用碰 NaN）。
  if (!(opt.temperature > 0.0f)) {
    const int32_t best = ArgMax(s, n);
    std::fill(s, s + n, 0.0f);
    s[best] = 1.0f;
    return;
  }

  // 1) 温度：缩放 logits。温度 < 1 让分布变尖（更确定），> 1 让它变平（更跳）。
  if (opt.temperature != 1.0f) {
    const float inv = 1.0f / opt.temperature;
    for (int64_t i = 0; i < n; ++i) s[i] *= inv;
  }

  // 2) top_k：取第 k 大的值作阈值，严格小于它的全剔除（并列的一起保留，
  //    与 HF TopKLogitsWarper 的 `<` 语义一致）。
  if (opt.top_k > 1 && opt.top_k < n) {
    std::vector<float> tmp(s, s + n);
    std::nth_element(tmp.begin(), tmp.begin() + (opt.top_k - 1), tmp.end(),
                     std::greater<float>());
    const float kth = tmp[opt.top_k - 1];
    for (int64_t i = 0; i < n; ++i)
      if (s[i] < kth) s[i] = kNegInf;
  }

  // 3) softmax —— top_p 要在概率上算累积。
  SoftmaxInPlace(s, n);

  // 4) top_p（核采样）。与 HF TopPLogitsWarper 同语义：按概率**升序**累积，
  //    累积概率还没到 (1 - top_p) 的那些属于"尾巴"，整撮剔除；最大的一个
  //    强制保留（至少留 1 个 token，否则分布会空）。
  if (opt.top_p > 0.0f && opt.top_p < 1.0f) {
    std::vector<int32_t> asc(n);
    for (int64_t i = 0; i < n; ++i) asc[i] = static_cast<int32_t>(i);
    std::sort(asc.begin(), asc.end(),
              [s](int32_t a, int32_t b) { return s[a] < s[b]; });

    const double threshold = 1.0 - static_cast<double>(opt.top_p);
    double cum = 0.0;
    for (int64_t j = 0; j < n; ++j) {
      cum += static_cast<double>(s[asc[j]]);
      if (cum <= threshold) s[asc[j]] = 0.0f;
    }
    // 最后一个（概率最大）无论累积到哪都留着。
    if (n > 0 && s[asc[n - 1]] == 0.0f) s[asc[n - 1]] = 1.0f;

    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) sum += static_cast<double>(s[i]);
    if (sum > 0.0) {
      const float inv = static_cast<float>(1.0 / sum);
      for (int64_t i = 0; i < n; ++i) s[i] *= inv;
    }
  }
}

int32_t SampleToken(const float* logits, int64_t n, const SampleOptions& opt,
                    float u) {
  if (!(opt.temperature > 0.0f)) return ArgMax(logits, n);  // 贪心不耗随机数

  std::vector<float> probs;
  FilterToProbs(logits, n, opt, &probs);

  // 逆变换：累积概率第一次跨过 u 的那个 token 就是它。
  double cum = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    cum += static_cast<double>(probs[i]);
    if (cum > static_cast<double>(u)) return static_cast<int32_t>(i);
  }
  // 只可能被浮点误差推到这条分支（u 逼近 1）。返回概率最大的那个，
  // 保证函数永远给出一个合法 token。
  return ArgMax(probs.data(), n);
}

}  // namespace rpg
