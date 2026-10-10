// forward_check —— 跑一次全序列前向，把中间结果写成 npy 供 Python 参考实现对拍。
//
// 用法：
//   forward_check model.gguf --prompt "文本" --out runs/a
//   forward_check model.gguf --tokens 151936,1234 --out runs/a
//   forward_check model.gguf --info
//
// 选项：
//   --prompt <文本>      用内置分词器编码
//   --prompt-file <文件>  从文件读提示词（Windows 下中文走命令行会被转码搞坏）
//   --tokens <a,b,c>     直接给 token id（与参考脚本共用同一串 id 才谈得上对拍）
//   --out <前缀>         写 <前缀>.tokens.txt / .logits.npy / .hidden.npy
//   --threads <N>        并行度，默认物理核
//   --max-pos <N>        RoPE 表覆盖的位置数，默认 4096
//   --no-hidden          不导出逐层残差流（省 ~120 MB 磁盘）
//   --top <K>            打印最后位置 top-K，默认 5
//   --info               只打印模型信息后退出

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "model.h"
#include "npy.h"
#include "qwen3.h"

namespace {

std::vector<int32_t> ParseTokens(const std::string& s) {
  std::vector<int32_t> out;
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
    if (i >= s.size()) break;
    char* end = nullptr;
    const long v = std::strtol(s.c_str() + i, &end, 10);
    if (end == s.c_str() + i) rpg::Throw("--tokens 里有解析不了的内容：%s", s.c_str() + i);
    out.push_back(static_cast<int32_t>(v));
    i = static_cast<size_t>(end - s.c_str());
  }
  return out;
}

bool WriteText(const std::string& path, const std::string& text) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  std::fclose(f);
  return ok;
}

// 从文件读 UTF-8 文本。
// 存在的理由：Windows 上 `cmd`/Git Bash 传给 argv 的字符串会按控制台代码页转码，
// 中文提示词直接当命令行参数传会被搞坏（实测 "你好" 会变成一坨乱码字节）。
// 文件路径是 ASCII，内容按字节读，就没有这一层。
std::string ReadFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) rpg::Throw("读不了文件 %s", path.c_str());
  std::string s;
  char buf[4096];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf(
        "用法: forward_check <model.gguf> [选项]\n"
        "  --prompt <文本>    用分词器编码（默认）\n"
        "  --prompt-file <f>  从文件读提示词（绕开 Windows argv 转码）\n"
        "  --tokens <a,b,c>   直接给 token id\n"
        "  --out <前缀>       写 <前缀>.tokens.txt / .logits.npy / .hidden.npy\n"
        "  --threads <N>      并行度（默认物理核）\n"
        "  --max-pos <N>      RoPE 表覆盖的位置数（默认 4096）\n"
        "  --layers <N>       只跑前 N 层（定位误差起点用）\n"
        "  --no-hidden        不导出逐层残差流\n"
        "  --top <K>          打印最后位置 top-K（默认 5）\n"
        "  --info             只打印模型信息\n");
    return 2;
  }

  const std::string model_path = argv[1];
  std::string prompt = "你好，请用一句话介绍你自己。";
  std::string tokens_arg;
  std::string out_prefix;
  int threads = 0;
  int32_t max_pos = 4096;
  int32_t layers = 0;
  int top_k = 5;
  bool dump_hidden = true;
  bool info_only = false;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) rpg::Throw("选项 %s 后面缺少参数", a.c_str());
      return argv[++i];
    };
    if (a == "--prompt") prompt = next();
    else if (a == "--prompt-file") prompt = ReadFile(next());
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--out") out_prefix = next();
    else if (a == "--threads") threads = std::atoi(next());
    else if (a == "--max-pos") max_pos = std::atoi(next());
    else if (a == "--no-hidden") dump_hidden = false;
    else if (a == "--layers") layers = std::atoi(next());
    else if (a == "--top") top_k = std::atoi(next());
    else if (a == "--info") info_only = true;
    else rpg::Throw("未知选项 %s", a.c_str());
  }

  try {
    rpg::Model model = rpg::Model::Load(model_path, threads, max_pos);
    std::printf("%s", model.Describe().c_str());
    std::printf("  权重文件  : %s\n", model.path().c_str());
    if (info_only) return 0;

    const rpg::ModelConfig& c = model.config();

    std::vector<int32_t> ids;
    if (!tokens_arg.empty()) {
      ids = ParseTokens(tokens_arg);
      std::printf("\n[输入] 直接指定 " RPG_SZ " 个 token id\n", RPG_SZ_ARG(ids.size()));
    } else {
      ids = model.tokenizer().Encode(prompt, /*parse_special=*/true);
      // GGUF 里 tokenizer.ggml.add_bos_token 是 false（Qwen3 不加 BOS），
      // 但若某个转换版本设成 true 也要照做 —— 参考脚本读的是同一个键。
      if (c.add_bos && c.bos_id >= 0) ids.insert(ids.begin(), c.bos_id);
      std::printf("\n[输入] \"%s\" → " RPG_SZ " 个 token\n", prompt.c_str(),
                  RPG_SZ_ARG(ids.size()));
      std::printf("       id: ");
      for (size_t i = 0; i < ids.size() && i < 40; ++i) {
        std::printf(i ? ",%d" : "%d", ids[i]);
      }
      std::printf("%s\n", ids.size() > 40 ? " ..." : "");
      std::printf("       还原: \"%s\"\n", model.tokenizer().Decode(ids).c_str());
    }
    RPG_CHECK(!ids.empty(), "token 序列为空");

    rpg::ForwardScratch scratch;
    std::vector<float> logits;
    rpg::ForwardOptions opt;
    opt.layer_limit = layers;
    std::vector<float> hidden;
    if (dump_hidden) opt.capture_hidden = &hidden;

    const auto t0 = std::chrono::steady_clock::now();
    rpg::ForwardFull(model, ids.data(), static_cast<int64_t>(ids.size()), &scratch,
                     &logits, opt);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const int32_t used = layers > 0 && layers < c.n_layers ? layers : c.n_layers;
    std::printf("\n[前向] 跑 %d/%d 层，%lld 个位置，用时 %.1f ms\n", used,
                c.n_layers, static_cast<long long>(ids.size()), ms);

    const int64_t seq = static_cast<int64_t>(ids.size());

    // ---- 最后位置的 top-K，肉眼判个合理 ----
    if (!logits.empty()) {
      const float* last = logits.data() + (seq - 1) * c.vocab;
      std::vector<int32_t> order(static_cast<size_t>(c.vocab));
      for (int32_t i = 0; i < c.vocab; ++i) order[static_cast<size_t>(i)] = i;
      const int k = std::min<int>(top_k, c.vocab);
      std::partial_sort(order.begin(), order.begin() + k, order.end(),
                        [last](int32_t a, int32_t b) { return last[a] > last[b]; });
      std::printf("\n[最后位置 top-%d]  logits 范围 [%.3f, %.3f]\n", k,
                  *std::min_element(last, last + c.vocab),
                  *std::max_element(last, last + c.vocab));
      for (int i = 0; i < k; ++i) {
        const int32_t id = order[static_cast<size_t>(i)];
        std::printf("  #%d  id=%-7d logit=%9.4f  %s\n", i + 1, id, last[id],
                    model.tokenizer().TokenOf(id).c_str());
      }
    } else {
      std::printf("（截断模式：不计算末端归一化与 lm_head，无 logits）\n");
    }

    if (out_prefix.empty()) return 0;

    // ---- 落盘 ----
    std::string tok_txt = out_prefix + ".tokens.txt";
    std::string body;
    for (size_t i = 0; i < ids.size(); ++i) {
      body += std::to_string(ids[i]);
      body += (i + 1 == ids.size()) ? "\n" : ",";
    }
    RPG_CHECK(WriteText(tok_txt, body), "写不了 %s", tok_txt.c_str());

    if (!logits.empty()) {
      std::string logits_npy = out_prefix + ".logits.npy";
      RPG_CHECK(
          npy::WriteF32(logits_npy, {seq, c.vocab}, logits.data(), logits.size()),
          "写不了 %s", logits_npy.c_str());
      std::printf("\n[输出] %s  [%lld, %d]\n", logits_npy.c_str(),
                  static_cast<long long>(seq), c.vocab);
    }

    if (dump_hidden) {
      std::string hn = out_prefix + ".hidden.npy";
      const int64_t nl = static_cast<int64_t>(hidden.size()) / (seq * c.hidden);
      RPG_CHECK(npy::WriteF32(hn, {nl, seq, c.hidden}, hidden.data(),
                              hidden.size()),
                "写不了 %s", hn.c_str());
      std::printf("[输出] %s  [%lld, %lld, %d]\n", hn.c_str(),
                  static_cast<long long>(nl), static_cast<long long>(seq),
                  c.hidden);
    }
    std::printf("[输出] %s\n", tok_txt.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
