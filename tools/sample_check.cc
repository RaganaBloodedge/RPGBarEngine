// tools/sample_check.cc —— 采样器的对拍入口
//
// 采样被拆成两半，这里各自留一个能独立验证的入口：
//
//   1. **确定性筛选**（`--out-probs`）：logits + temperature/top_k/top_p →
//      概率分布。Python 侧用 HF 的 Temperature/TopK/TopP warpers 算同一份
//      分布（`tools/check_sample.py`），逐元素比。
//   2. **counter-based 随机源**（`--rng`）：纯函数 u = H(seed, counter)。
//      验"同 seed 同位置必得同一个 u"，以及相邻 counter 不相关。
//   3. **采样落点**（`--u` 或 `--seed`）：给定 u 看选中哪个 token —— 用来
//      手验"同一个 u，温度不同 → 选中不同 token"（README 里那张图的数字）。
//
// 输入输出都用**裸 float32**（Python 侧 np.fromfile / tofile），不套 npy 头：
// 这里的数据都是一维的，裸数组更省事，也少一类"我算错了 header 偏移"的错误。
//
//   sample_check --logits L.bin --temperature 0.8 --out-probs P.bin
//   sample_check --rng --seed 42 --counter 0 --count 5
//   sample_check --logits L.bin --temperature 2.0 --u 0.95

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cli_util.h"
#include "sampler.h"

namespace {

std::string Arg(int argc, char** argv, const char* name, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return def == nullptr ? std::string() : std::string(def);
}

bool HasFlag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

std::vector<float> ReadF32(const std::string& path) {
  const std::string bytes = cli::ReadFile(path);
  RPG_CHECK(bytes.size() % sizeof(float) == 0,
            "%s 的字节数不是 4 的整数倍（" RPG_SZ " 字节）", path.c_str(),
            RPG_SZ_ARG(bytes.size()));
  std::vector<float> v(bytes.size() / sizeof(float));
  if (!v.empty()) std::memcpy(v.data(), bytes.data(), bytes.size());
  return v;
}

void WriteF32(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  RPG_CHECK(f != nullptr, "写不了 %s", path.c_str());
  if (!v.empty()) {
    RPG_CHECK(std::fwrite(v.data(), sizeof(float), v.size(), f) == v.size(),
              "写 %s 时中途失败", path.c_str());
  }
  std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
  rpg::SampleOptions opt;
  opt.temperature = std::strtof(Arg(argc, argv, "--temperature", "1.0").c_str(), nullptr);
  opt.top_k = std::atoi(Arg(argc, argv, "--top-k", "0").c_str());
  opt.top_p = std::strtof(Arg(argc, argv, "--top-p", "1.0").c_str(), nullptr);

  // ---- 模式 1：打印 counter-based 随机数（纯函数自证 + 相关性检查）
  if (HasFlag(argc, argv, "--rng")) {
    const uint64_t seed = std::strtoull(Arg(argc, argv, "--seed", "0").c_str(), nullptr, 10);
    const int64_t start = std::strtoll(Arg(argc, argv, "--counter", "0").c_str(), nullptr, 10);
    const int64_t count = std::strtoll(Arg(argc, argv, "--count", "1").c_str(), nullptr, 10);
    for (int64_t i = 0; i < count; ++i) {
      const double u = static_cast<double>(
          rpg::CounterUniform(seed, static_cast<uint64_t>(start + i)));
      std::printf("%.17g\n", u);
    }
    return 0;
  }

  // ---- 其余模式都要一份 logits
  const std::string path = Arg(argc, argv, "--logits", "");
  RPG_CHECK(!path.empty(), "--logits 必填（裸 float32 数组），或加 --rng 走随机源模式");
  const std::vector<float> logits = ReadF32(path);
  const int64_t n = static_cast<int64_t>(logits.size());
  RPG_CHECK(n > 0, "%s 是空的", path.c_str());

  // ---- 模式 2：导出筛选后的概率分布（对拍主体）
  const std::string out = Arg(argc, argv, "--out-probs", "");
  if (!out.empty()) {
    std::vector<float> probs;
    rpg::FilterToProbs(logits.data(), n, opt, &probs);
    WriteF32(out, probs);
    std::printf("probs -> %s（n=" RPG_SZ "）\n", out.c_str(), RPG_SZ_ARG(n));
  }

  // ---- 模式 3：给定 u（或 seed+counter）看选中哪个 token
  const std::string u_arg = Arg(argc, argv, "--u", "");
  const std::string seed_arg = Arg(argc, argv, "--seed", "");
  if (!u_arg.empty() || !seed_arg.empty()) {
    float u;
    if (!u_arg.empty()) {
      u = std::strtof(u_arg.c_str(), nullptr);
    } else {
      const uint64_t seed = std::strtoull(seed_arg.c_str(), nullptr, 10);
      const uint64_t counter =
          static_cast<uint64_t>(std::strtoll(Arg(argc, argv, "--counter", "0").c_str(), nullptr, 10));
      u = rpg::CounterUniform(seed, counter);
    }
    const int32_t tok = rpg::SampleToken(logits.data(), n, opt, u);
    std::printf("u=%.17g token=%d\n", static_cast<double>(u), tok);
  }

  return 0;
}
