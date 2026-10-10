#include "threads.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "common.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cstdio>
#endif

namespace rpg {
namespace {

#if defined(_WIN32)
// 每个 RelationProcessorCore 记录恰好代表一个物理核。系统不会把 P 核和 E 核
// 分开报告，这对我们没影响：只要"物理核总数"。
int PhysicalCoresWin() {
  DWORD len = 0;
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len) &&
      GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    return 0;
  }
  if (len == 0) return 0;

  std::vector<char> buf(len);
  if (!GetLogicalProcessorInformationEx(
          RelationProcessorCore,
          reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
          &len)) {
    return 0;
  }

  // 只能用 off < len 判定：结构体因联合体 sizeof 是 80 字节，而
  // RelationProcessorCore 的记录只填 48 字节。拿 sizeof 当上界会漏掉最后一条。
  int cores = 0;
  DWORD off = 0;
  const char* base = buf.data();
  while (off < len) {
    const auto* rec =
        reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(base +
                                                                         off);
    if (rec->Size == 0) break;  // 防御：坏数据会死循环
    if (rec->Relationship == RelationProcessorCore) ++cores;
    off += rec->Size;
  }
  return cores;
}
#endif

#if defined(__linux__)
// /proc/cpuinfo 里同一物理核上的超线程共享 physical id / core id，去重即物理核数。
// 容器 / WSL 里这两个字段可能缺失，那时返回 0 走兜底，不瞎猜。
int PhysicalCoresLinux() {
  std::FILE* f = std::fopen("/proc/cpuinfo", "r");
  if (f == nullptr) return 0;

  std::set<std::pair<int, int>> seen;
  char line[512];
  int phys = -1;
  int core = -1;

  const auto flush = [&]() {
    if (phys >= 0 && core >= 0) seen.emplace(phys, core);
    phys = -1;
    core = -1;
  };
  const auto value_of = [](const char* s) -> int {
    const char* c = std::strchr(s, ':');
    return c ? std::atoi(c + 1) : 0;
  };

  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "physical id", 11) == 0) {
      phys = value_of(line);
    } else if (std::strncmp(line, "core id", 7) == 0) {
      core = value_of(line);
    } else if (line[0] == '\n') {
      flush();
    }
  }
  flush();
  std::fclose(f);
  return static_cast<int>(seen.size());
}
#endif

thread_local bool g_in_pool_worker = false;

}  // namespace

int DetectPhysicalCores() {
#if defined(_WIN32)
  return PhysicalCoresWin();
#elif defined(__linux__)
  return PhysicalCoresLinux();
#else
  return 0;
#endif
}

int DefaultThreads() {
  const int phys = DetectPhysicalCores();
  if (phys > 0) return phys;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int>(std::min<unsigned>(hw, 1024u)) : 1;
}

bool InPoolWorker() { return g_in_pool_worker; }

// ---------------------------------------------------------------------------
// ThreadPool

struct ThreadPool::Impl {
  int n_threads = 1;
  std::vector<std::thread> workers;

  std::mutex mu;
  std::condition_variable cv_start;
  std::condition_variable cv_done;

  uint64_t generation = 0;
  std::vector<uint64_t> my_generation;  // 每个 worker 上次见到的 generation
  std::vector<int64_t> chunk_begin;
  std::vector<int64_t> chunk_end;

  const std::function<void(int64_t, int64_t)>* body = nullptr;
  int active = 0;  // 尚未完成的 worker 数
  bool stop = false;
};

ThreadPool::ThreadPool(int n_threads) : impl_(std::make_unique<Impl>()) {
  impl_->n_threads = n_threads < 1 ? 1 : n_threads;
  // 先分配好所有 per-worker 数组，worker 线程起来后只读自己那一格。
  impl_->my_generation.assign(impl_->n_threads, 0);
  impl_->chunk_begin.assign(impl_->n_threads, 0);
  impl_->chunk_end.assign(impl_->n_threads, 0);

  impl_->workers.reserve(impl_->n_threads - 1);
  for (int i = 1; i < impl_->n_threads; ++i) {
    impl_->workers.emplace_back([this, i]() {
      g_in_pool_worker = true;
      for (;;) {
        int64_t begin = 0;
        int64_t end = 0;
        // body 指针也必须在锁内读：它是主线程写的，锁外读就是数据竞争。
        const std::function<void(int64_t, int64_t)>* body = nullptr;
        {
          std::unique_lock<std::mutex> lk(impl_->mu);
          impl_->cv_start.wait(lk, [&] {
            return impl_->stop || impl_->generation != impl_->my_generation[i];
          });
          if (impl_->stop) return;
          impl_->my_generation[i] = impl_->generation;
          begin = impl_->chunk_begin[i];
          end = impl_->chunk_end[i];
          body = impl_->body;
        }
        if (begin < end && body != nullptr) {
          (*body)(begin, end);
        }
        {
          std::lock_guard<std::mutex> lk(impl_->mu);
          if (--impl_->active == 0) impl_->cv_done.notify_all();
        }
      }
    });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->stop = true;
  }
  impl_->cv_start.notify_all();
  for (auto& t : impl_->workers) {
    if (t.joinable()) t.join();
  }
}

int ThreadPool::size() const { return impl_->n_threads; }

void ThreadPool::ParallelForRange(
    int64_t n, const std::function<void(int64_t, int64_t)>& body) {
  if (n <= 0) return;
  const int nt = impl_->n_threads;

  // 兜底：worker 内部再并行会破坏本实现的假设，直接串行跑。
  if (nt <= 1 || InPoolWorker()) {
    body(0, n);
    return;
  }

  const int64_t chunk = (n + nt - 1) / nt;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->body = &body;
    impl_->active = nt - 1;
    for (int i = 0; i < nt; ++i) {
      impl_->chunk_begin[i] = std::min<int64_t>(n, chunk * i);
      impl_->chunk_end[i] = std::min<int64_t>(n, chunk * (i + 1));
    }
    ++impl_->generation;
    impl_->cv_start.notify_all();
  }

  // 主线程负责第 0 段。
  const int64_t b0 = impl_->chunk_begin[0];
  const int64_t e0 = impl_->chunk_end[0];
  if (b0 < e0) body(b0, e0);

  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->cv_done.wait(lk, [&] { return impl_->active == 0; });
    impl_->body = nullptr;
  }
}

}  // namespace rpg
