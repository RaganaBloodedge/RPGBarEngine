// tok_check —— 分词器对拍工具。
//
// 协议（避免 argv 的编码/长度问题，也避免每例一次进程启动）：
//   输入：stdin 的原始字节，多个用例用 **NUL（0x00）** 分隔。
//   输出：每个用例一行，空格分隔的 token id。
//   解码模式（--decode）：输入是用例行（每行空格分隔 id，用 NUL 分隔用例），
//                        输出是解码后的原始字节，同样用 NUL 分隔。
//
// 用法：
//   tok_check <model.gguf>            # 编码
//   tok_check <model.gguf> --plain    # 不做特殊 token 切分
//   tok_check <model.gguf> --pretok   # 打印预分词结果（用例内用 \x1F 分隔）
//   tok_check <model.gguf> --decode [--skip-special]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "tokenizer.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace {

std::string ReadAllStdin() {
  std::string data;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) {
    data.append(buf, n);
  }
  return data;
}

std::vector<std::string> SplitNul(const std::string& s) {
  std::vector<std::string> out;
  size_t start = 0;
  for (size_t i = 0; i <= s.size(); ++i) {
    if (i == s.size() || s[i] == '\0') {
      if (i > start) out.push_back(s.substr(start, i - start));
      start = i + 1;
    }
  }
  return out;
}

std::vector<int32_t> ParseIds(const std::string& line) {
  std::vector<int32_t> ids;
  const char* p = line.c_str();
  char* end = nullptr;
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (*p == '\0') break;
    const long v = std::strtol(p, &end, 10);
    if (end == p) break;
    ids.push_back(static_cast<int32_t>(v));
    p = end;
  }
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  // 必须二进制模式：文本模式会把 0x1A 当 EOF、把 \n 换成 \r\n，破坏协议。
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  if (argc < 2) {
    std::fprintf(stderr,
                 "用法: tok_check <model.gguf> [--plain] [--pretok] "
                 "[--decode [--skip-special]]\n");
    return 2;
  }
  const std::string path = argv[1];
  bool plain = false;
  bool pretok = false;
  bool decode = false;
  bool skip_special = false;
  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--plain") == 0) plain = true;
    else if (std::strcmp(argv[i], "--pretok") == 0) pretok = true;
    else if (std::strcmp(argv[i], "--decode") == 0) decode = true;
    else if (std::strcmp(argv[i], "--skip-special") == 0) skip_special = true;
  }

  try {
    const rpg::GgufFile g = rpg::GgufFile::Open(path);
    const rpg::Tokenizer tok = rpg::Tokenizer::FromGguf(g);
    const std::string input = ReadAllStdin();
    const std::vector<std::string> cases = SplitNul(input);

    if (decode) {
      for (size_t i = 0; i < cases.size(); ++i) {
        if (i) std::fputc('\0', stdout);
        const std::string s = tok.Decode(ParseIds(cases[i]), skip_special);
        std::fwrite(s.data(), 1, s.size(), stdout);
      }
      return 0;
    }

    for (const std::string& c : cases) {
      if (pretok) {
        std::string line;
        const std::vector<std::string> pieces = tok.PreTokenize(c);
        for (size_t i = 0; i < pieces.size(); ++i) {
          if (i) line.push_back('\x1F');
          line += pieces[i];
        }
        std::fwrite(line.data(), 1, line.size(), stdout);
        std::fputc('\n', stdout);
        continue;
      }
      const std::vector<int32_t> ids =
          plain ? tok.EncodePlain(c) : tok.Encode(c, true);
      for (size_t i = 0; i < ids.size(); ++i) {
        std::printf("%s%d", i ? " " : "", ids[i]);
      }
      std::fputc('\n', stdout);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误: %s\n", e.what());
    return 1;
  }
}
