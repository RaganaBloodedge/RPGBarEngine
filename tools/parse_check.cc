// parse_check —— 把一段模型输出文本喂进解析器，打印结构化结果。
//
// 它是 M4 里 rpg_engine_generate 返回值的预演：同一个 ParseModelOutput，
// 同一份 JSON 形状（text / reasoning / tool_calls）。放在 tools 下是为了能拿
// **真实生成结果**做端到端验证 —— selftest 里那些是构造用例。

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "cli_util.h"
#include "common.h"
#include "json.h"
#include "tool_parser.h"

namespace {

int Usage() {
  std::fprintf(stderr,
               "用法: parse_check --text-file <raw.txt> [--json]\n"
               "  不给 --text-file 就从 stdin 读（二进制）。\n"
               "  --json  只输出 JSON（供脚本消费）；否则人读格式。\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string text_path;
  bool json_out = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--text-file") == 0 && i + 1 < argc) {
      text_path = argv[++i];
    } else if (std::strcmp(argv[i], "--json") == 0) {
      json_out = true;
    } else {
      return Usage();
    }
  }

  cli::SetupBinaryStdio();
  try {
    std::string raw;
    if (text_path.empty()) {
      char buf[4096];
      size_t n;
      while ((n = std::fread(buf, 1, sizeof(buf), stdin)) > 0) raw.append(buf, n);
    } else {
      raw = cli::ReadFile(text_path);
    }

    const rpg::ParsedOutput o = rpg::ParseModelOutput(raw);

    rpg::Json calls = rpg::Json::Array();
    for (const rpg::ParsedToolCall& c : o.tool_calls) {
      rpg::Json one = rpg::Json::Object();
      one.Set("name", c.name);
      one.Set("ok", c.ok);
      one.Set("arguments_raw", c.arguments_raw);
      one.Set("arguments", c.arguments);
      calls.PushBack(std::move(one));
    }

    rpg::Json out = rpg::Json::Object();
    out.Set("text", o.text);
    out.Set("reasoning", o.reasoning);
    out.Set("tool_calls", std::move(calls));
    out.Set("unterminated_think", o.unterminated_think);
    out.Set("unterminated_tool_call", o.unterminated_tool_call);

    if (json_out) {
      std::printf("%s\n", out.DumpPython().c_str());
      return 0;
    }

    std::printf("text        : %s\n", o.text.c_str());
    std::printf("reasoning   : %s\n", o.reasoning.c_str());
    std::printf("tool_calls  : %d\n", static_cast<int>(o.tool_calls.size()));
    for (size_t i = 0; i < o.tool_calls.size(); ++i) {
      const rpg::ParsedToolCall& c = o.tool_calls[i];
      std::printf("  #%d  name=%s  ok=%s\n", static_cast<int>(i + 1),
                  c.name.c_str(), c.ok ? "true" : "false");
      std::printf("      arguments = %s\n", c.arguments.Dump().c_str());
    }
    if (o.unterminated_think || o.unterminated_tool_call) {
      std::printf("警告：输出被截断（think=%d tool_call=%d）\n",
                  o.unterminated_think ? 1 : 0,
                  o.unterminated_tool_call ? 1 : 0);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "parse_check: %s\n", e.what());
    return 2;
  }
}
