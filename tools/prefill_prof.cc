// tools/prefill_prof.cc —— prefill 归因：算力上限 / matmul 占比 / 复用缺口
//
// 用法: prefill_prof <model.gguf> [threads] [reps]
//
// 输出四块：
//   1) 纯 FMA 上限（8 条独立链 + 汇编屏障）—— 后面所有百分比的"分母"
//   2) 端到端 prefill 随 token 数的变化（真实层序、真实 cache 压力）
//   3) matmul-only（28 层 × 7 个 MatMulBatch + lm_head 之和）→ 与端到端对比得占比
//   4) 权重流量推算式：每 token 有多少字节、实测等效带宽是多少
//
// **为什么必须用真实权重、真实层序**：本机 1.7B 一层的权重约 28 MB，而 L3 约
// 30 MB —— 已经贴着这条线。若合成一个 7 MB 的矩阵做基准，它整个装进 L3，
// 测出的"带宽"会比真实快好几倍，归因就全错了（见 cpu-kernel-perf-tuning §四）。
//
// 这个工具只测时间，不校验数值；数值正确性由第 11 层 check_dotrow.py 守。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "cli_util.h"
#include "common.h"
#include "model.h"
#include "qwen3.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define RPG_PROF_X86 1
#endif

namespace {

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#ifdef RPG_PROF_X86
// ---- 1. 纯算力上限 ----------------------------------------------------------
//
// 8 条**互不依赖**的链：FMA 延迟 4 周期、每核 2 个 FMA 端口，所以需要 8 条独立
// 链才能把两个端口喂满（只有 4 条时每周期最多发 4 条，吞吐正好卡一半）。
//
// 汇编屏障**必须有**：否则 GCC 的终结值替换（final value replacement）会把
// x = fma(x,b,c) 这样的线性递推折成闭式，整段循环直接消失 —— 症状是测出上百
// GHz 这种物理上不可能的读数。屏障是编译期的，不产生任何指令。
__attribute__((target("avx2,fma"))) double FmaThroughputGmacs(double seconds) {
  __m256 v0 = _mm256_set1_ps(1.0f), v1 = _mm256_set1_ps(1.1f);
  __m256 v2 = _mm256_set1_ps(1.2f), v3 = _mm256_set1_ps(1.3f);
  __m256 v4 = _mm256_set1_ps(1.4f), v5 = _mm256_set1_ps(1.5f);
  __m256 v6 = _mm256_set1_ps(1.6f), v7 = _mm256_set1_ps(1.7f);
  const __m256 b = _mm256_set1_ps(1.0000001f);
  const __m256 c = _mm256_set1_ps(1e-7f);

  long iters = 0;
  const double t0 = NowMs();
  double dt = 0.0;
  do {
    for (int k = 0; k < 2048; ++k) {
      v0 = _mm256_fmadd_ps(v0, b, c);
      v1 = _mm256_fmadd_ps(v1, b, c);
      v2 = _mm256_fmadd_ps(v2, b, c);
      v3 = _mm256_fmadd_ps(v3, b, c);
      v4 = _mm256_fmadd_ps(v4, b, c);
      v5 = _mm256_fmadd_ps(v5, b, c);
      v6 = _mm256_fmadd_ps(v6, b, c);
      v7 = _mm256_fmadd_ps(v7, b, c);
      __asm__ __volatile__("" : "+x"(v0), "+x"(v1), "+x"(v2), "+x"(v3), "+x"(v4),
                           "+x"(v5), "+x"(v6), "+x"(v7));
    }
    iters += 2048;
    dt = (NowMs() - t0) / 1000.0;
  } while (dt < seconds);
  // 每轮 8 条 FMA × 8 个 float = 64 个 MAC
  return 64.0 * static_cast<double>(iters) / dt / 1e9;
}

// 单链延迟 → 反推主频（FMA 延迟按 4 周期算）。只作参考，判据用吞吐。
__attribute__((target("avx2,fma"))) double FmaLatencyGhz(double seconds) {
  __m256 x = _mm256_set1_ps(1.0f);
  const __m256 y = _mm256_set1_ps(1.0000001f);
  const __m256 z = _mm256_set1_ps(1e-7f);
  long iters = 0;
  const double t0 = NowMs();
  double dt = 0.0;
  do {
    for (int k = 0; k < 4096; ++k) {
      x = _mm256_fmadd_ps(x, y, z);
      __asm__ __volatile__("" : "+x"(x));
    }
    iters += 4096;
    dt = (NowMs() - t0) / 1000.0;
  } while (dt < seconds);
  return 4.0 * static_cast<double>(iters) / dt / 1e9;
}

// 多线程 FMA 上限。**必须测这个**：单核 turbo（4.77 GHz）与全核频率（约 3.3 GHz）
// 差了 40%，拿单核上限去比一个 16 线程的内核，会得出"已用 89% 算力"的假结论。
double FmaPeakMultithreadGmacs(int n_threads, double seconds) {
  if (n_threads < 1) n_threads = 1;
  std::vector<double> per(static_cast<size_t>(n_threads), 0.0);
  std::vector<std::thread> ts;
  ts.reserve(static_cast<size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t) {
    ts.emplace_back([&per, t, seconds] {
      per[static_cast<size_t>(t)] = FmaThroughputGmacs(seconds);
    });
  }
  for (std::thread& th : ts) th.join();
  double sum = 0.0;
  for (const double v : per) sum += v;
  return sum;
}

// 顺序读带宽 —— 对应"扫权重"的访存模式（只读、顺序、不写）。
// 8 条独立累加链：单链 add 延迟 4 周期，链不够会被依赖链限速，测出来的是
// "加法吞吐"而不是"内存带宽"。
__attribute__((target("avx2,fma"))) double ReadSliceGbs(const float* p,
                                                        size_t n,
                                                        float* sink) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
  __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
  __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
  const double t0 = NowMs();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    a0 = _mm256_add_ps(a0, _mm256_loadu_ps(p + i + 0));
    a1 = _mm256_add_ps(a1, _mm256_loadu_ps(p + i + 8));
    a2 = _mm256_add_ps(a2, _mm256_loadu_ps(p + i + 16));
    a3 = _mm256_add_ps(a3, _mm256_loadu_ps(p + i + 24));
    a4 = _mm256_add_ps(a4, _mm256_loadu_ps(p + i + 32));
    a5 = _mm256_add_ps(a5, _mm256_loadu_ps(p + i + 40));
    a6 = _mm256_add_ps(a6, _mm256_loadu_ps(p + i + 48));
    a7 = _mm256_add_ps(a7, _mm256_loadu_ps(p + i + 56));
  }
  a0 = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  a4 = _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7));
  a0 = _mm256_add_ps(a0, a4);
  const double dt = (NowMs() - t0) / 1000.0;
  float lanes[8];
  _mm256_storeu_ps(lanes, a0);
  *sink = lanes[0] + lanes[7];  // 用掉结果，别让编译器把整段删掉
  return static_cast<double>(i) * 4.0 / dt / 1e9;
}

// 多线程读带宽：把 256 MB 缓冲（远超 30 MB L3）切成 n_threads 段并发读。
double ReadBandwidthGbs(int n_threads, size_t bytes) {
  if (n_threads < 1) n_threads = 1;
  std::vector<float> buf(bytes / sizeof(float));
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<float>(static_cast<int>(i % 97)) * 0.001f;
  }
  const size_t n = buf.size();
  const size_t per = n / static_cast<size_t>(n_threads);
  std::vector<float> sinks(static_cast<size_t>(n_threads), 0.0f);
  std::vector<std::thread> ts;
  const double t0 = NowMs();
  for (int t = 0; t < n_threads; ++t) {
    const size_t beg = static_cast<size_t>(t) * per;
    const size_t cnt = (t + 1 == n_threads) ? (n - beg) : per;
    ts.emplace_back([&buf, &sinks, beg, cnt, t] {
      ReadSliceGbs(buf.data() + beg, cnt, &sinks[static_cast<size_t>(t)]);
    });
  }
  for (std::thread& th : ts) th.join();
  const double dt = (NowMs() - t0) / 1000.0;
  return static_cast<double>(n) * 4.0 / dt / 1e9;
}
#endif  // RPG_PROF_X86

// 一次前向里所有 matmul 的 MAC 数（每 token）与权重字节数（每 token）。
struct PerToken {
  double macs = 0.0;
  double weight_bytes = 0.0;
  void Add(const rpg::QuantMatrix& w) {
    macs += static_cast<double>(w.rows()) * static_cast<double>(w.cols());
    weight_bytes +=
        static_cast<double>(rpg::TypeRowBytes(w.type(), w.cols())) * w.rows();
  }
};

PerToken MeasurePerToken(const rpg::Model& m) {
  PerToken p;
  for (const rpg::LayerWeights& w : m.layers()) {
    p.Add(w.attn_q);
    p.Add(w.attn_k);
    p.Add(w.attn_v);
    p.Add(w.attn_o);
    p.Add(w.ffn_gate);
    p.Add(w.ffn_up);
    p.Add(w.ffn_down);
  }
  p.Add(m.lm_head());
  return p;
}

// 只跑 matmul（真实层序、真实尺寸），返回毫秒。输入填确定性的小值 ——
// 别填 0（会短路反量化路径）、别填 NaN/Inf（会拖慢浮点），也别落到次正规数。
double MatMulOnlyMs(const rpg::Model& m, int n_tokens, rpg::ThreadPool* pool) {
  const rpg::ModelConfig& c = m.config();
  const size_t xsz = static_cast<size_t>(n_tokens) * c.intermediate;
  std::vector<float> X(xsz);
  for (size_t i = 0; i < xsz; ++i) {
    const int s = static_cast<int>(i % 2048) - 1024;
    X[i] = static_cast<float>(s) * 0.0007f;
  }
  std::vector<float> OUT(static_cast<size_t>(n_tokens) * c.vocab, 0.0f);

  const double t0 = NowMs();
  for (const rpg::LayerWeights& w : m.layers()) {
    w.attn_q.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.attn_k.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.attn_v.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.attn_o.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.ffn_gate.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.ffn_up.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
    w.ffn_down.MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
  }
  m.lm_head().MatMulBatch(X.data(), n_tokens, OUT.data(), pool);
  return NowMs() - t0;
}

double BestMatMulOnlyMs(const rpg::Model& m, int n_tokens, rpg::ThreadPool* pool,
                        int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    MatMulOnlyMs(m, n_tokens, pool);  // 预热
    const double dt = MatMulOnlyMs(m, n_tokens, pool);
    if (dt < best) best = dt;
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "用法: prefill_prof <model.gguf> [threads] [reps]\n"
                 "  threads 默认 0（= 硬件线程数），reps 默认 3（取最快一次）\n");
    return 2;
  }
  const std::string path = argv[1];
  const int threads = argc > 2 ? std::atoi(argv[2]) : 0;
  const int reps = argc > 3 ? std::atoi(argv[3]) : 3;

  rpg::Model m = rpg::Model::Load(path, threads, 4096);
  const rpg::ModelConfig& c = m.config();
  std::printf("%s\n", m.Describe().c_str());

  const PerToken per = MeasurePerToken(m);
  std::printf("每 token 的 matmul 规模: %.3f GMAC（= %.3f GFLOP）\n",
              per.macs / 1e9, per.macs * 2.0 / 1e9);
  std::printf("每 token 的权重字节  : %.1f MB\n\n",
              per.weight_bytes / (1024.0 * 1024.0));

#ifdef RPG_PROF_X86
  const int nt = static_cast<int>(m.pool().size());
  const double peak1 = FmaThroughputGmacs(0.30);
  const double ghz = FmaLatencyGhz(0.20);
  const double peaknt = FmaPeakMultithreadGmacs(nt, 0.30);
  const double bw1 = ReadBandwidthGbs(1, static_cast<size_t>(256) << 20);
  const double bwnt = ReadBandwidthGbs(nt, static_cast<size_t>(256) << 20);
  std::printf("纯 FMA 上限（单线程，8 条独立链）: %.1f GMAC/s（= %.1f GFLOP/s），"
              "延迟反推主频 ≈ %.2f GHz\n",
              peak1, peak1 * 2.0, ghz);
  std::printf("纯 FMA 上限（%d 线程）          : %.1f GMAC/s（= %.1f GFLOP/s）"
              "  ← 百分比的分母\n",
              nt, peaknt, peaknt * 2.0);
  std::printf("顺序读带宽（1 线程 / %d 线程）    : %.1f / %.1f GB/s（256 MB 缓冲）\n",
              nt, bw1, bwnt);
  std::printf("  —— 单核 turbo 与全核频率差得远（4.8 vs 3.3 GHz），拿单线程上限去比"
              "多线程内核会得出荒谬结论。\n\n");
  const double peak_gmacs = peaknt;
#else
  const double peak_gmacs = 0.0;
  std::printf("（非 x86 平台，跳过 FMA 上限测量；下面只报相对变化）\n\n");
#endif

  // ---- 预热：先把 1.1 GB 权重从页缓存拉出来（首次缺页跨文件系统很慢）----
  {
    rpg::KvCache kv(c, 256);
    rpg::ForwardScratch s;
    std::vector<float> logits;
    for (int i = 0; i < 3; ++i) {
      kv.Reset();
      const int32_t toks[4] = {100, 200, 300, 400};
      rpg::ForwardFull(m, toks, 4, &s, &logits, rpg::ForwardOptions(), &kv);
    }
  }

  const int ns[] = {8, 16, 32, 64, 128};
  std::printf("=== 端到端 prefill（含 attention / norm / rope）===\n");
  std::printf("%6s %12s %10s %10s %10s %10s %10s\n", "n", "ms", "tok/s",
              "GMAC/s", "峰值%", "权重GB/s", "matmul占比");
  for (const int n : ns) {
    std::vector<int32_t> toks(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) toks[static_cast<size_t>(i)] = 100 + (i * 37) % 1000;

    rpg::KvCache kv(c, 128);
    rpg::ForwardScratch s;
    std::vector<float> logits;
    double e2e = 1e30;
    for (int r = 0; r < reps; ++r) {
      kv.Reset();
      const double t0 = NowMs();
      rpg::ForwardFull(m, toks.data(), n, &s, &logits, rpg::ForwardOptions(), &kv);
      const double dt = NowMs() - t0;
      if (dt < e2e) e2e = dt;
    }

    // matmul-only：取最快一次
    const double mm = BestMatMulOnlyMs(m, n, &m.pool(), reps);

    const double macs = per.macs * n;
    const double gmacs = macs / (e2e / 1000.0) / 1e9;
    const double gbs = per.weight_bytes * n / (e2e / 1000.0) / 1e9;
    const double util = peak_gmacs > 0 ? 100.0 * gmacs / peak_gmacs : 0.0;
    std::printf("%6d %12.1f %10.1f %10.1f %10.1f %10.1f %9.1f%%\n", n, e2e,
                n / (e2e / 1000.0), gmacs, util, gbs, 100.0 * mm / e2e);
  }

  std::printf("\n=== matmul-only（真实层序，28 层 × 7 + lm_head）===\n");
  std::printf("%6s %12s %10s %10s %10s\n", "n", "ms", "GMAC/s", "峰值%", "权重GB/s");
  for (const int n : ns) {
    const double mm = BestMatMulOnlyMs(m, n, &m.pool(), reps);
    const double macs = per.macs * n;
    const double gmacs = macs / (mm / 1000.0) / 1e9;
    const double gbs = per.weight_bytes * n / (mm / 1000.0) / 1e9;
    const double util = peak_gmacs > 0 ? 100.0 * gmacs / peak_gmacs : 0.0;
    std::printf("%6d %12.1f %10.1f %10.1f %10.1f\n", n, mm, gmacs, util, gbs);
  }

  // 线程扩展：这是判定"访存受限 vs 结构受限"的关键实验。
  // 若加线程几乎不涨吞吐（曲线很快压平），说明内存带宽先饱和 —— 减少搬运量有效；
  // 若接近线性到 16 线程，说明是**每核结构效率**的问题 —— 那要先改内核本身。
  std::printf("\n=== matmul-only 线程扩展（n=32，固定工作量）===\n");
  std::printf("%8s %12s %10s %10s\n", "threads", "ms", "GMAC/s", "相对1线程");
  double base1 = 0.0;
  for (const int t : {1, 2, 4, 8, 16}) {
    rpg::ThreadPool pool(t);
    const double mm = BestMatMulOnlyMs(m, 32, &pool, 2);
    const double gmacs = per.macs * 32 / (mm / 1000.0) / 1e9;
    if (t == 1) base1 = gmacs;
    std::printf("%8d %12.1f %10.1f %9.2fx\n", t, mm, gmacs, gmacs / base1);
  }

  std::printf(
      "\n读法：\n"
      "  * matmul 的 GMAC/s 若随 n 基本不变 → 每个 token 都在重扫权重（无复用），\n"
      "    时间与 n 成正比；把 n 个 token 合成一块复用一次权重读，就是 M5 剩的那半。\n"
      "  * 权重GB/s 是「按需读一遍全部权重 × n」折算的等效带宽。若它已接近内存\n"
      "    实测带宽，说明瓶颈是**搬数据**；若离得远而峰值%% 又很低，说明瓶颈在别处。\n");
  return 0;
}
