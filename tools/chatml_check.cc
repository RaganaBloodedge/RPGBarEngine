// chatml_check —— 渲染一份 ChatML case 并落盘，交给 check_chatml.py 与 HF 对拍。
//
// case 文件的格式（messages/tools 与 C ABI 的请求同构）：
//   {
//     "messages": [ {"role":..., "content":..., "reasoning_content":..., "tool_calls":[...]} ],
//     "tools": [ {"type":"function","function":{...}} ],   // 可选
//     "add_generation_prompt": true,                        // 可选，默认 true
//     "enable_thinking": false                              // 可选，缺省 = 思考模式
//   }
//
// 为什么走文件而不是命令行参数：Windows 下 `cmd` / Git Bash 会把 argv 按控制台
// 代码页转码，中文与特殊字符传进去就坏了（这个坑在 forward_check 上踩过一次）。

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "chat_template.h"
#include "cli_util.h"
#include "common.h"
#include "json.h"

namespace {

int Usage() {
  std::fprintf(stderr,
               "用法: chatml_check --case <case.json> [--out <rendered.txt>]\n"
               "  不给 --out 就写到 stdout（二进制，不做换行转换）。\n");
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string case_path;
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--case") == 0 && i + 1 < argc) {
      case_path = argv[++i];
    } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      return Usage();
    }
  }
  if (case_path.empty()) return Usage();

  cli::SetupBinaryStdio();
  try {
    std::string err;
    const rpg::Json spec = rpg::Json::Parse(cli::ReadFile(case_path), &err);
    RPG_CHECK(err.empty() && spec.IsObject(), "case 文件不是 JSON 对象：%s",
              err.c_str());

    const auto messages = rpg::ParseMessages(spec.Get("messages"));
    const rpg::Json tools = rpg::ParseTools(spec.Get("tools"));

    rpg::ChatTemplateOptions opt;
    opt.add_generation_prompt = spec.Get("add_generation_prompt").AsBool(true);
    // 缺省 = 思考模式（官方模板只在显式 false 时补空思考块）。
    opt.enable_thinking = spec.Get("enable_thinking").AsBool(true);

    const std::string rendered = rpg::RenderChatml(messages, tools, opt);
    if (out_path.empty()) {
      std::fwrite(rendered.data(), 1, rendered.size(), stdout);
    } else {
      RPG_CHECK(cli::WriteText(out_path, rendered), "写不了 %s", out_path.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "chatml_check: %s\n", e.what());
    return 2;
  }
}
