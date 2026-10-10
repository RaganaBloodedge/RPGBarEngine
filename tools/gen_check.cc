// gen_check —— 贪心生成探针（M2 的验证工具）。
//
// 三件事：
//   1. prefill prompt（同时填 KV Cache），然后贪心生成 N 个 token；
//   2. **自检：带 cache 的 prefill 与不带 cache 的全序列，logits 必须逐位相同**。
//      这条能一次性证明 cache 的写入/读取没有把布局搞错 —— 布局错了不会崩，
//      只会让生成结果慢慢跑偏，那种最难查；
//   3. 把 prompt ids 与生成的 ids 写盘，交给 check_generate.py 与官方实现对拍。
//
// 用法：
//   gen_check model.gguf --prompt-file tests/prompts/zh_chatml.txt --max-tokens 32 --out runs/g1
//   gen_check model.gguf --tokens 151936,1234 --max-tokens 8 --no-kv-check
//
// 选项：
//   --prompt <文本>      用内置分词器编码
//   --prompt-file <文件>  从文件读提示词（Windows 下中文走命令行会被转码搞坏）
//   --tokens <a,b,c>     直接给 token id（对拍时两边必须用同一串 id）
//   --max-tokens <N>     生成多少个（默认 32）
//   --max-seq <N>        KV Cache 容量，默认按 prompt + max_tokens 自动取整到 256
//   --threads <N>        并行度，默认物理核
//   --max-pos <N>        RoPE 表覆盖的位置数，默认 4096
//   --stop <id,id>       停止 token（默认空 —— 对拍时两边都生成固定数量）
//   --stop-eos           用模型自带的 eos 当停止 token
//   --out <前缀>         写 <前缀>.tokens.txt 与 <前缀>.gen.txt
//   --no-kv-check        跳过 cache 一致性自检（省一半时间）
//   --info               只打印模型信息后退出

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "cli_util.h"
#include "common.h"
#include "model.h"
#include "qwen3.h"

namespace {

// 逐位比较两个 [n] 的 float 缓冲。返回第一个不同的下标，全同返回 -1。
int64_t FirstDiff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return static_cast<int64_t>(i);
  }
  return -1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: gen_check <model.gguf> [选项]\n"
        "  --prompt <文本>      用分词器编码（默认）\n"
        "  --prompt-file <f>    从文件读提示词（绕开 Windows argv 转码）\n"
        "  --tokens <a,b,c>     直接给 token id\n"
        "  --max-tokens <N>     生成数量（默认 32）\n"
        "  --max-seq <N>        KV Cache 容量（默认自动）\n"
        "  --threads <N>        并行度（默认物理核）\n"
        "  --max-pos <N>        RoPE 表覆盖的位置数（默认 4096）\n"
        "  --stop <id,id>       停止 token（默认无）\n"
        "  --stop-eos           用模型自带的 eos 当停止 token\n"
        "  --out <前缀>         写 <前缀>.tokens.txt / <前缀>.gen.txt\n"
        "  --no-kv-check        跳过 cache 一致性自检\n"
        "  --info               只打印模型信息\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "The capital of France is";
  std::string tokens_arg;
  std::string out_prefix;
  std::string stop_arg;
  int threads = 0;
  int32_t max_pos = 4096;
  int32_t max_tokens = 32;
  int64_t max_seq = 0;
  bool kv_check = true;
  bool info_only = false;
  bool stop_eos = false;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺少参数", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = cli::ReadFile(next());
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--max-tokens") max_tokens = std::atoi(next());
    else if (a == "--max-seq") max_seq = std::atoll(next());
    else if (a == "--threads") threads = std::atoi(next());
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--stop") stop_arg = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--no-kv-check") kv_check = false;
    else if (a == "--stop-eos") stop_eos = true;
    else if (a == "--info") info_only = true;
    else rpg::Throw("未知选项 %s", a.c_str());
  }

  try {
    rpg::Model model = rpg::Model::Load(model_path, threads, max_pos);
    std::printf("%s", model.Describe().c_str());
    if (info_only) return 0;

    const rpg::ModelConfig& c = model.config();

    // ---- 输入 ----
    std::vector<int32_t> ids;
    if (!tokens_arg.empty()) {
      ids = cli::ParseIds(tokens_arg);
      std::printf("\n[输入] 直接指定 " RPG_SZ " 个 token id\n",
                  RPG_SZ_ARG(ids.size()));
    } else {
      ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
      if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
      std::printf("\n[输入] " RPG_SZ " 个 token\n", RPG_SZ_ARG(ids.size()));
    }
    RPG_CHECK(!ids.empty(), "token 序列为空");
    const int64_t seq = static_cast<int64_t>(ids.size());

    // ---- KV Cache 容量 ----
    std::printf("[缓存] max_pos=%d\n", max_pos);
    RPG_CHECK(max_tokens > 0, "max_tokens 必须为正");
    const int64_t want = seq + static_cast<int64_t>(max_tokens) + 8;
    const int64_t cap =
        max_seq > 0 ? max_seq : ((want + 255) / 256) * 256;
    std::printf("[缓存] 序列容量 %lld（prompt %lld + 生成 %d + 余量）\n",
                static_cast<long long>(cap), static_cast<long long>(seq),
                max_tokens);
    std::printf("[缓存] 约 %.0f MiB（%d 层 × %d KV 头 × %lld × %d × 2 × 4 字节）\n",
                static_cast<double>(cap) * c.n_layers * c.n_kv_heads *
                    c.head_dim * 8 / 1048576.0,
                c.n_layers, c.n_kv_heads, static_cast<long long>(cap),
                c.head_dim);

    rpg::ForwardScratch scratch;
    std::vector<float> logits;

    // ---- 自检：带 cache 的 prefill vs 不带 cache 的全序列 ----
    if (kv_check) {
      std::vector<float> plain;
      rpg::ForwardFull(model, ids.data(), seq, &scratch, &plain);
      rpg::KvCache kv(c, cap);
      std::vector<float> cached;
      rpg::ForwardFull(model, ids.data(), seq, &scratch, &cached,
                       rpg::ForwardOptions(), &kv);
      const int64_t d = FirstDiff(plain, cached);
      if (d < 0) {
        std::printf("[自检] KV Cache 路径与全序列路径逐位相同（%lld 个数值）"
                    " —— cache 布局正确\n",
                    static_cast<long long>(plain.size()));
      } else {
        std::printf("[自检] **不一致**：第 %lld 个数值不同（%g vs %g）\n",
                    static_cast<long long>(d), static_cast<double>(plain[d]),
                    static_cast<double>(cached[d]));
        return 1;
      }
    }

    // ---- prefill + 贪心生成 ----
    rpg::KvCache kv(c, cap);
    rpg::ForwardOptions opt;
    const auto t0 = std::chrono::steady_clock::now();
    rpg::ForwardFull(model, ids.data(), seq, &scratch, &logits, opt, &kv);
    const auto t1 = std::chrono::steady_clock::now();
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\n[prefill] %lld 个位置 %.1f ms（%.1f tok/s）\n",
                static_cast<long long>(seq), prefill_ms,
                static_cast<double>(seq) / (prefill_ms / 1000.0));

    rpg::GenerateOptions gopt;
    gopt.max_tokens = max_tokens;
    gopt.stop_ids = cli::ParseIds(stop_arg);
    if (stop_eos && c.eos_id >= 0) gopt.stop_ids.push_back(c.eos_id);

    std::vector<int32_t> gen;
    const float* last = logits.data() + (seq - 1) * c.vocab;
    const auto t2 = std::chrono::steady_clock::now();
    rpg::GenerateGreedy(model, &kv, &scratch, last, gopt, &gen);
    const auto t3 = std::chrono::steady_clock::now();
    const double gen_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("[decode]  生成 " RPG_SZ " 个 token，%.1f ms（%.2f tok/s）\n",
                RPG_SZ_ARG(gen.size()), gen_ms,
                static_cast<double>(gen.size()) / (gen_ms / 1000.0));
    std::printf("[decode]  cache 现有 %lld 个位置\n",
                static_cast<long long>(kv.len()));

    std::printf("\n[生成] ");
    for (size_t i = 0; i < gen.size(); ++i) {
      std::printf(i ? ",%d" : "%d", gen[i]);
    }
    std::printf("\n[生成] 解码: \"%s\"\n", model.tokenizer().Decode(gen).c_str());
    std::printf("\n[全文] \"%s%s\"\n",
                model.tokenizer().Decode(ids).c_str(),
                model.tokenizer().Decode(gen).c_str());

    if (out_prefix.empty()) return 0;

    cli::WriteIds(out_prefix + ".tokens.txt", ids);
    cli::WriteIds(out_prefix + ".gen.txt", gen);
    std::printf("\n[输出] %s.tokens.txt / %s.gen.txt\n", out_prefix.c_str(),
                out_prefix.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
