// RPGBarEngine —— 线程与并行
//
// DetectPhysicalCores / DefaultThreads 从 TinyGemmaCpp 移植（同源同理由：
// GEMM 是访存受限的，超线程只会抢同一份 L2/L3 带宽）。
//
// 但并行后端**不沿用 OpenMP**，改成自带的常驻线程池。三个理由：
//   1. **零运行时依赖**：这个库要直接丢进 Unity 游戏里。libgomp / vcomp 是
//      额外一个 DLL，缺了就是"加载失败"，而且宿主自己也用 OpenMP 时会撞车。
//   2. **decode 是"很多次很小的并行区"**：每生成一个 token 要走几十次 GEMM，
//      OpenMP 每个 parallel region 的 fork/join 开销在此占比可观；常驻线程
//      + 条件变量唤醒没有线程创建成本。
//   3. **可控**：能自己确保"只有主线程发起并行"，避免嵌套并行死锁。

#ifndef RPGBAR_THREADS_H_
#define RPGBAR_THREADS_H_

#include <cstdint>
#include <functional>
#include <memory>

namespace rpg {

// 物理核数量。探测失败返回 0。
int DetectPhysicalCores();

// 默认并行度：物理核优先，探测不到退回 hardware_concurrency()。至少为 1。
int DefaultThreads();

// 常驻线程池。一个引擎实例持有一个。
//
// 用法约束（重要）：
//   * ParallelFor 只能由**创建它的线程**调用，不可重入。
//   * 若在 worker 线程内再调 ParallelFor，会自动退化成串行执行（不死锁）。
class ThreadPool {
 public:
  explicit ThreadPool(int n_threads);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const;

  // 把 [0, n) 切成 size() 段，交给各线程执行 body(begin, end)。
  // 主线程也参与（跑第 0 段）。返回时所有段都已完成。
  void ParallelForRange(int64_t n,
                        const std::function<void(int64_t, int64_t)>& body);

  // 便捷包装：按模板捕获，避免调用方手写 std::function。
  template <class F>
  void ParallelFor(int64_t n, F&& f) {
    if (n <= 0) return;
    ParallelForRange(n,
                     std::function<void(int64_t, int64_t)>(std::forward<F>(f)));
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// 当前线程是不是某个线程池的 worker。嵌套调用靠它兜底。
bool InPoolWorker();

}  // namespace rpg

#endif  // RPGBAR_THREADS_H_
