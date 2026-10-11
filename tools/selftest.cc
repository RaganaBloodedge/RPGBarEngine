// selftest —— 引擎基础件的进程内自检（不需要模型文件，不需服务器）。
//
// 覆盖：字节读写 / UTF-8 / 字节⇄码点映射 / Unicode 类别 / JSON / 简单量化解码。
//
// 为什么不把这些断言塞进 tok_ref.py 或 check_quant_layout.py：
// 那两个脚本验证的是"和外部参考对不对"，需要一个 1.8 GB 的模型和 Python 环境；
// 这里是"内部一致性"的快速回归 —— 改一行代码后 1 秒内就知道有没有把地基碰坏。
//
// K-quant 的**语义**正确性不在这里验（手搓 Q4_K 的 210 字节期望值收益太低）：
//   * 尺寸表由 tools/check_quant_layout.py 用 GGUF 写入器的 offset 独立验证；
//   * 语义由 M1 的端到端 logits 对拍验证（错了会直接表现为输出乱码）。
//
// 用法：selftest [--verbose]

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bytes.h"
#include "chat_template.h"
#include "common.h"
#include "dtypes.h"
#include "json.h"
#include "quant.h"
#include "tool_parser.h"
#include "unicode.h"

namespace {

int g_pass = 0;
int g_fail = 0;
bool g_verbose = false;

void Check(bool ok, const std::string& what) {
  if (ok) {
    ++g_pass;
    if (g_verbose) std::printf("  ok   %s\n", what.c_str());
  } else {
    ++g_fail;
    std::printf("  FAIL %s\n", what.c_str());
  }
}

void CheckEq(double got, double want, const std::string& what) {
  Check(std::fabs(got - want) < 1e-6,
        what + " (got " + std::to_string(got) + " want " +
            std::to_string(want) + ")");
}

// ---------------------------------------------------------------- bytes

void TestBytes() {
  std::printf("[bytes]\n");
  CheckEq(static_cast<double>(rpg::AlignUp(0, 32)), 0, "AlignUp(0,32)");
  CheckEq(static_cast<double>(rpg::AlignUp(1, 32)), 32, "AlignUp(1,32)");
  CheckEq(static_cast<double>(rpg::AlignUp(32, 32)), 32, "AlignUp(32,32)");
  CheckEq(static_cast<double>(rpg::AlignUp(33, 32)), 64, "AlignUp(33,32)");

  // 小端读：
  const uint8_t buf[] = {0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF};
  rpg::Reader r(buf, sizeof(buf));
  CheckEq(static_cast<double>(r.U32()), 1.0, "U32 小端");
  CheckEq(static_cast<double>(static_cast<int32_t>(r.U32())), -1.0, "I32 小端");

  // 越界必须抛（而不是静默读到垃圾）
  bool threw = false;
  try {
    rpg::Reader small(buf, 2);
    (void)small.U32();
  } catch (const std::exception&) {
    threw = true;
  }
  Check(threw, "越界读取抛异常");
}

// ---------------------------------------------------------------- utf8

void TestUtf8() {
  std::printf("[utf8]\n");
  const uint32_t cps[] = {0x0041, 0x00E9, 0x4E2D, 0x30A2, 0xD55C, 0x1F3B2,
                          0x10FFFF, 0x0000, 0x007F, 0x0800, 0xFFFF};
  bool all_ok = true;
  for (const uint32_t cp : cps) {
    char buf[4];
    const int n = rpg::Utf8Encode(cp, buf);
    const rpg::Utf8Char d = rpg::Utf8Decode(buf, static_cast<size_t>(n));
    if (d.cp != cp || d.len != n) {
      all_ok = false;
      std::printf("    往返失败 cp=U+%04X\n", cp);
    }
  }
  Check(all_ok, "UTF-8 编解码往返（含 4 字节与 U+0000）");

  // 非法序列应给出替换字符而不是崩
  const char bad[] = {(char)0x80, (char)0xC0, (char)0xAF, (char)0xE0,
                      (char)0x80};
  rpg::Utf8Char d = rpg::Utf8Decode(bad, 5);
  Check(d.cp == 0xFFFD && d.len == 1, "非法起始字节 -> U+FFFD");

  // 过长编码（0xC0 0xAF 是 '/' 的过长表示）必须拒绝
  const char overlong[] = {(char)0xC0, (char)0xAF};
  d = rpg::Utf8Decode(overlong, 2);
  Check(d.cp == 0xFFFD, "过长编码被拒绝");
}

// ---------------------------------------------------------------- 字节⇄码点

void TestByteMap() {
  std::printf("[byte map]\n");
  bool bijection = true;
  std::vector<int> seen(1024, 0);
  for (int b = 0; b < 256; ++b) {
    const uint32_t cp = rpg::ByteToCodepoint(static_cast<uint8_t>(b));
    if (cp > 323) bijection = false;
    if (seen[cp]++ != 0) bijection = false;  // 撞码点
    if (rpg::CodepointToByte(cp) != b) bijection = false;
  }
  Check(bijection, "256 个字节与码点构成双射且可逆");

  // 与 GPT-2 原始表对齐的几个锚点
  CheckEq(static_cast<double>(rpg::ByteToCodepoint(' ')), 0x120, "空格 -> 'Ġ'");
  CheckEq(static_cast<double>(rpg::ByteToCodepoint('A')), 'A', "'A' -> 'A'");
  CheckEq(static_cast<double>(rpg::ByteToCodepoint(0x00)), 0x100, "0x00 -> U+0100");
  CheckEq(static_cast<double>(rpg::ByteToCodepoint(0x0A)), 0x10A, "0x0A -> U+010A");
  CheckEq(static_cast<double>(rpg::ByteToCodepoint(0x7F)), 0x121, "0x7F -> U+0121");
  CheckEq(static_cast<double>(rpg::ByteToCodepoint(0xAD)), 0x143, "0xAD -> U+0143");
  Check(rpg::CodepointToByte(0x41) == 0x41, "反向 'A'");
  Check(rpg::CodepointToByte(0x120) == 0x20, "反向 'Ġ' -> 空格");
  Check(rpg::CodepointToByte(0x4E2D) == -1, "中文不是字节编码字符");
}

// ---------------------------------------------------------------- Unicode 类别

void TestUnicodeCategories() {
  std::printf("[unicode 类别]\n");
  Check(rpg::IsUnicodeLetter('a') && rpg::IsUnicodeLetter('Z'),
        "ASCII 字母属于 \\p{L}");
  Check(rpg::IsUnicodeLetter(0x4E2D), "中文'中'属于 \\p{L}");
  Check(rpg::IsUnicodeLetter(0xAC00), "韩文'가'属于 \\p{L}");
  Check(rpg::IsUnicodeLetter(0x0416), "西里尔'Ж'属于 \\p{L}");
  Check(!rpg::IsUnicodeLetter('5'), "数字不属于 \\p{L}");
  Check(!rpg::IsUnicodeLetter(0x3002), "句号不属于 \\p{L}");

  Check(rpg::IsUnicodeNumber('0') && rpg::IsUnicodeNumber('9'),
        "ASCII 数字属于 \\p{N}");
  Check(rpg::IsUnicodeNumber(0xFF10), "全角'０'属于 \\p{N}");
  Check(rpg::IsUnicodeNumber(0x2160), "罗马数字'Ⅰ'属于 \\p{N}");
  Check(rpg::IsUnicodeNumber(0x00B2), "上标'²'属于 \\p{N}");
  Check(!rpg::IsUnicodeNumber(0x4E00), "'一'是 \\p{L} 不是 \\p{N}");

  Check(rpg::IsUnicodeSpace(' ') && rpg::IsUnicodeSpace('\t') &&
            rpg::IsUnicodeSpace('\n') && rpg::IsUnicodeSpace('\r'),
        "ASCII 空白");
  Check(rpg::IsUnicodeSpace(0x3000), "表意空格 U+3000 是空白");
  Check(rpg::IsUnicodeSpace(0x00A0), "不换行空格 U+00A0 是空白");
  Check(!rpg::IsUnicodeSpace(0x200B), "零宽空格 U+200B **不是**空白");
  Check(!rpg::IsUnicodeSpace('a'), "字母不是空白");
}

// ---------------------------------------------------------------- JSON

void TestJson() {
  std::printf("[json]\n");
  const char* src =
      "{\"a\":1,\"b\":[true,false,null],\"c\":{\"d\":\"x\\ny\"},\"e\":1.5,"
      "\"f\":\"\\u4e2d\\u6587\",\"g\":[]}";
  std::string err;
  const rpg::Json j = rpg::Json::Parse(src, &err);
  Check(err.empty() && j.IsObject(), "解析对象");
  CheckEq(j["a"].AsDouble(), 1, "整数");
  Check(j["b"].IsArray() && j["b"].size() == 3, "数组长度");
  Check(j["b"].GetAt(0).AsBool(), "true");
  Check(j["b"].GetAt(2).IsNull(), "null");
  Check(j["c"]["d"].AsString() == "x\ny", "嵌套对象 + \\n 转义");
  CheckEq(j["e"].AsDouble(), 1.5, "浮点");
  Check(j["f"].AsString() == "\xE4\xB8\xAD\xE6\x96\x87", "\\u 转义 -> UTF-8");
  Check(j["g"].IsArray() && j["g"].size() == 0, "空数组");

  // 缺字段应落默认值，而不是崩
  Check(j["不存在"].IsNull(), "缺字段返回 null");
  CheckEq(rpg::Json().AsDouble(7.0), 7.0, "默认值");

  // 往返
  const std::string dumped = j.Dump();
  const rpg::Json j2 = rpg::Json::Parse(dumped, &err);
  Check(err.empty() && j2.Dump() == dumped, "解析/序列化往返稳定");

  // 整数值不应输出小数点
  Check(rpg::Json(static_cast<int64_t>(256)).Dump() == "256", "整数不带 .0");
  Check(rpg::Json(0.5).Dump() == "0.5", "小数正常");

  // 错误输入
  std::string e2;
  rpg::Json::Parse("{\"a\":}", &e2);
  Check(!e2.empty(), "非法 JSON 报错");
  std::string e3;
  rpg::Json::Parse("[1,2", &e3);
  Check(!e3.empty(), "未闭合数组报错");

  // 代理对
  const rpg::Json sp = rpg::Json::Parse("\"\\ud83c\\udfb2\"", &err);
  Check(sp.AsString() == "\xF0\x9F\x8E\xB2", "代理对合成 🎲");

  // 特殊字符转义往返
  rpg::Json esc = rpg::Json::Object();
  esc.Set("k", std::string("a\"b\\c\td\x01"));
  const rpg::Json back = rpg::Json::Parse(esc.Dump(), &err);
  Check(err.empty() && back["k"].AsString() == "a\"b\\c\td\x01",
        "控制字符与引号往返");
}

// ---------------------------------------------------------------- JSON（Python 样式）

void TestJsonPython() {
  std::printf("[json: Python 样式]\n");

  // 分隔符：官方模板的 tojson 过滤器就是 json.dumps 的默认样式。
  const rpg::Json obj =
      rpg::Json::Parse("{\"a\":1,\"b\":[1,2],\"c\":{\"d\":\"x\"}}");
  Check(obj.DumpPython() == "{\"a\": 1, \"b\": [1, 2], \"c\": {\"d\": \"x\"}}",
        "DumpPython 分隔符与 json.dumps 一致");
  Check(obj.Dump() == "{\"a\":1,\"b\":[1,2],\"c\":{\"d\":\"x\"}}",
        "紧凑 Dump 未被改变");

  // 整数字面量与浮点字面量：值一样、写法不同，回写时要保持原样。
  Check(rpg::Json::Parse("1").Dump() == "1", "\"1\" 保持整数写法");
  Check(rpg::Json::Parse("1.0").Dump() == "1.0", "\"1.0\" 保持浮点写法");
  Check(rpg::Json::Parse("[1,2.5]").DumpPython() == "[1, 2.5]", "混合数组");

  // 最短往返表示。"%.17g" 会把 0.1 写成 0.10000000000000001。
  Check(rpg::Json(0.1).Dump() == "0.1", "0.1 用最短表示");
  Check(rpg::Json(1e-5).Dump() == "1e-05", "1e-05 用最短表示");
  Check(rpg::Json(1.0).Dump() == "1.0", "整值浮点补 .0");
  Check(rpg::Json(123.456).Dump() == "123.456", "123.456 往返");
  Check(rpg::Json(-0.0).Dump() == "-0.0", "-0.0");

  std::string err;
  const rpg::Json rt = rpg::Json::Parse("[0.1,1e-05,1.0,-0.0]", &err);
  Check(err.empty() && rt.Dump() == "[0.1,1e-05,1.0,-0.0]", "浮点写法往返稳定");
}

// ---------------------------------------------------------------- ChatML 渲染

void TestChatTemplate() {
  std::printf("[ChatML 渲染]\n");
  using rpg::ChatMessage;
  using rpg::ChatTemplateOptions;

  auto mk = [](const char* role, const char* content) {
    ChatMessage m;
    m.role = role;
    m.content = content;
    return m;
  };

  // 非思考模式：末尾补一个空的 <think>…</think>。
  {
    std::vector<ChatMessage> msgs = {mk("system", "你是 DM。"), mk("user", "我撬箱子。")};
    ChatTemplateOptions opt;
    opt.enable_thinking = false;
    const std::string s = rpg::RenderChatml(msgs, rpg::Json(), opt);
    Check(s ==
              "<|im_start|>system\n你是 DM。<|im_end|>\n"
              "<|im_start|>user\n我撬箱子。<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "无工具 + 非思考：补空思考块");
  }

  // 思考模式（缺省）：不补。
  {
    std::vector<ChatMessage> msgs = {mk("user", "我撬箱子。")};
    const std::string s = rpg::RenderChatml(msgs, rpg::Json());
    Check(s == "<|im_start|>user\n我撬箱子。<|im_end|>\n<|im_start|>assistant\n",
          "思考模式不补空思考块");
  }

  // 有工具：system 内容与工具前言合并在同一条 system 里。
  {
    const rpg::Json tools = rpg::Json::Parse(
        "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"description\":\"d\"}}]");
    std::vector<ChatMessage> msgs = {mk("system", "你是 DM。"), mk("user", "来")};
    const std::string s = rpg::RenderChatml(msgs, tools);
    Check(s.find("<|im_start|>system\n你是 DM。\n\n# Tools") == 0,
          "有工具：system 与工具前言合并");
    Check(s.find("</tools>\n\nFor each function call") != std::string::npos,
          "有工具：前言结尾正确");
    Check(s.find("{\"type\": \"function\", \"function\": {\"name\": \"f\"") !=
              std::string::npos,
          "有工具：定义用 Python JSON 样式渲染");
  }

  // 连续的 tool 消息合并成一条 user，各自包一层 tool_response。
  {
    ChatTemplateOptions opt;
    opt.add_generation_prompt = false;
    ChatMessage a;
    a.role = "assistant";
    a.tool_calls.push_back({"f", "{}"});
    ChatMessage t1 = mk("tool", "{\"a\":1}");
    ChatMessage t2 = mk("tool", "{\"b\":2}");
    std::vector<ChatMessage> msgs = {a, t1, t2};
    const std::string s = rpg::RenderChatml(msgs, rpg::Json(), opt);
    Check(s.find("<|im_start|>user\n<tool_response>\n{\"a\":1}\n</tool_response>\n"
                 "<tool_response>\n{\"b\":2}\n</tool_response><|im_end|>\n") !=
              std::string::npos,
          "连续 tool 消息合并成一条 user");
    Check(s.find("<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>") !=
              std::string::npos,
          "assistant 的 tool_calls 按官方格式渲染");
  }

  // assistant 的思考块只在"最后一次真实提问之后"渲染。
  {
    ChatMessage a;
    a.role = "assistant";
    a.content = "箱子纹丝不动。";
    a.reasoning_content = "先看力量。";
    a.has_reasoning = true;  // "给了空串"与"没给"在官方模板里是两条分支
    std::vector<ChatMessage> msgs = {mk("user", "我撬箱子。"), a};
    const std::string s = rpg::RenderChatml(msgs, rpg::Json());
    Check(s.find("assistant\n<think>\n先看力量。\n</think>\n\n箱子纹丝不动。") !=
              std::string::npos,
          "最后一条 assistant 渲染思考块");
  }
}

// ---------------------------------------------------------------- 输出解析

void TestToolParser() {
  std::printf("[输出解析]\n");

  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput("  箱子纹丝不动。\n");
    Check(o.text == "箱子纹丝不动。" && o.tool_calls.empty() && o.reasoning.empty(),
          "纯文本：两端空白去掉");
  }
  {
    const rpg::ParsedOutput o =
        rpg::ParseModelOutput("<think>\n先看力量。\n</think>\n\n箱子纹丝不动。");
    Check(o.text == "箱子纹丝不动。" && o.reasoning == "\n先看力量。\n",
          "思考块剥离，内容单独给出");
    Check(!o.unterminated_think, "闭合的思考块不标记截断");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput("<think>还在想");
    Check(o.unterminated_think && o.text.empty() && o.reasoning == "还在想",
          "未闭合思考块：整段算思考并打标记");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<tool_call>\n{\"name\": \"read_scene\", \"arguments\": {\"scene\": \"temple\"}}\n"
        "</tool_call>");
    Check(o.tool_calls.size() == 1 && o.tool_calls[0].ok, "单工具调用");
    Check(o.tool_calls[0].name == "read_scene", "工具名");
    Check(o.tool_calls[0].arguments["scene"].AsString() == "temple", "参数对象");
    Check(o.text.empty(), "只有工具调用时文本为空");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "我先看看。\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n</tool_call>\n"
        "<tool_call>\n{\"name\":\"b\",\"arguments\":{\"x\":1}}\n</tool_call>\n结束");
    Check(o.tool_calls.size() == 2, "两个工具调用");
    Check(o.tool_calls[0].name == "a" && o.tool_calls[1].name == "b", "调用顺序");
    Check(o.tool_calls[1].arguments["x"].AsInt() == 1, "第二个调用的参数");
    Check(o.text.compare(0, 15, "我先看看。") == 0 &&
              o.text.compare(o.text.size() - 6, 6, "结束") == 0,
          "文本被拼回并去掉首尾空白");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<tool_call>{\"name\":\"a\",\"arguments\":\"{\\\"x\\\":2}\"}</tool_call>");
    Check(o.tool_calls.size() == 1 && o.tool_calls[0].ok &&
              o.tool_calls[0].arguments["x"].AsInt() == 2,
          "arguments 是字符串时也拆开");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<tool_call>{\"name\":\"a\",\"arguments\":{oops}</tool_call>");
    Check(o.tool_calls.size() == 1 && !o.tool_calls[0].ok, "非法参数标记为不可用");
  }
  {
    const rpg::ParsedOutput o =
        rpg::ParseModelOutput("<tool_call>{\"name\":\"a\",\"arguments\":{}}");
    Check(o.unterminated_tool_call && o.tool_calls.size() == 1 && o.tool_calls[0].ok,
          "未闭合工具调用：仍解析并打标记");
  }
  {
    // 思考块里的 <tool_call> 字样不该被当成真的调用（先剥思考块再抽调用）。
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<think>要不要调 <tool_call> 呢</think>答案是：不用。");
    Check(o.tool_calls.empty() && o.text == "答案是：不用。",
          "思考块内的 <tool_call> 字样不误判");
  }
  {
    // 控制类特殊 token 由解析器清掉。理由：<tool_call> 自己就是特殊 token，
    // 解码时没法用 skip_special 过滤（会把结构标记一起删掉）。
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<|im_start|>assistant\n你好。<|im_end|><|endoftext|>");
    Check(o.text == "你好。", "控制标记（含紧跟的 role 词）被清掉");
  }
  {
    const rpg::ParsedOutput o = rpg::ParseModelOutput(
        "<|im_start|>assistant\n<tool_call>\n{\"name\":\"a\",\"arguments\":{}}\n"
        "</tool_call><|im_end|>");
    Check(o.text.empty() && o.tool_calls.size() == 1 && o.tool_calls[0].ok,
          "带控制标记的真实输出形态可解析");
  }
}

// ---------------------------------------------------------------- 量化解码（简单类型）

void TestSimpleQuants() {
  std::printf("[量化：简单类型]\n");

  // Q8_0：d=1.0, 32 个 int8 = -16..15
  {
    std::vector<uint8_t> blk(34);
    const uint16_t d = rpg::F32ToF16(1.0f);
    std::memcpy(blk.data(), &d, 2);
    for (int i = 0; i < 32; ++i) {
      blk[2 + static_cast<size_t>(i)] = static_cast<uint8_t>(i - 16);
    }
    float out[32];
    rpg::DequantizeRow(rpg::kGgmlQ8_0, blk.data(), 32, out);
    bool ok = true;
    for (int i = 0; i < 32; ++i) {
      if (out[i] != static_cast<float>(i - 16)) ok = false;
    }
    Check(ok, "Q8_0：d × int8");
  }

  // Q4_0：低半字节 -> 元素 j，高半字节 -> 元素 j+16（最容易写反的地方）。
  // 用 low=j、high=15-j 两组不同的值，才能把"归谁"这件事测出来。
  {
    std::vector<uint8_t> blk(18);
    const uint16_t d = rpg::F32ToF16(2.0f);
    std::memcpy(blk.data(), &d, 2);
    for (int j = 0; j < 16; ++j) {
      const int lo = j;
      const int hi = 15 - j;
      blk[2 + static_cast<size_t>(j)] =
          static_cast<uint8_t>((hi << 4) | lo);
    }
    float out[32];
    rpg::DequantizeRow(rpg::kGgmlQ4_0, blk.data(), 32, out);
    bool ok = true;
    for (int j = 0; j < 16; ++j) {
      if (out[j] != 2.0f * static_cast<float>(j - 8)) ok = false;
      if (out[j + 16] != 2.0f * static_cast<float>((15 - j) - 8)) ok = false;
    }
    Check(ok, "Q4_0：低/高半字节的归属与 -8 偏置");
  }

  // Q4_1：value = d*q + m
  {
    std::vector<uint8_t> blk(20);
    const uint16_t d = rpg::F32ToF16(0.5f);
    const uint16_t m = rpg::F32ToF16(3.0f);
    std::memcpy(blk.data(), &d, 2);
    std::memcpy(blk.data() + 2, &m, 2);
    for (int j = 0; j < 16; ++j) {
      blk[4 + static_cast<size_t>(j)] = static_cast<uint8_t>((j << 4) | j);
    }
    float out[32];
    rpg::DequantizeRow(rpg::kGgmlQ4_1, blk.data(), 32, out);
    bool ok = true;
    for (int j = 0; j < 16; ++j) {
      if (out[j] != 0.5f * static_cast<float>(j) + 3.0f) ok = false;
      if (out[j + 16] != 0.5f * static_cast<float>(j) + 3.0f) ok = false;
    }
    Check(ok, "Q4_1：d×q + m");
  }

  // Q5_0：5 位、偏置 -16，高位来自 qh 的 32 个 bit
  {
    std::vector<uint8_t> blk(22);
    const uint16_t d = rpg::F32ToF16(1.0f);
    std::memcpy(blk.data(), &d, 2);
    // qh 全 0 => 高位均 0 => 取值 -16..-1
    std::memset(blk.data() + 2, 0, 4);
    for (int j = 0; j < 16; ++j) blk[6 + static_cast<size_t>(j)] = 0x11;
    float out[32];
    rpg::DequantizeRow(rpg::kGgmlQ5_0, blk.data(), 32, out);
    bool ok = true;
    for (int i = 0; i < 32; ++i) {
      if (out[i] != -15.0f) ok = false;  // (1 | 0) - 16 = -15
    }
    Check(ok, "Q5_0：高位缺失时按 -16 偏置");
  }

  // 类型表自洽：block_bytes 与 block_elems 的关系
  {
    const uint32_t ids[] = {rpg::kGgmlF32, rpg::kGgmlF16, rpg::kGgmlBF16,
                            rpg::kGgmlQ4_0, rpg::kGgmlQ4_1, rpg::kGgmlQ5_0,
                            rpg::kGgmlQ5_1, rpg::kGgmlQ8_0, rpg::kGgmlQ8_1,
                            rpg::kGgmlQ2_K, rpg::kGgmlQ3_K, rpg::kGgmlQ4_K,
                            rpg::kGgmlQ5_K, rpg::kGgmlQ6_K, rpg::kGgmlQ8_K};
    bool ok = true;
    for (const uint32_t id : ids) {
      const rpg::GgmlTypeInfo* t = rpg::TypeInfo(id);
      if (t == nullptr || !rpg::TypeSupported(id)) {
        ok = false;
        continue;
      }
      const int64_t bytes = rpg::TypeRowBytes(id, t->block_elems * 3);
      if (bytes != t->block_bytes * 3) ok = false;
    }
    Check(ok, "类型表：TypeRowBytes 与块大小一致");
  }

  // f16 转换的几个锚点（round-to-nearest-even）
  CheckEq(static_cast<double>(rpg::F16ToF32(rpg::F32ToF16(1.0f))), 1.0, "f16 1.0");
  CheckEq(static_cast<double>(rpg::F16ToF32(rpg::F32ToF16(0.5f))), 0.5, "f16 0.5");
  CheckEq(static_cast<double>(rpg::F16ToF32(rpg::F32ToF16(-2.5f))), -2.5, "f16 -2.5");
  CheckEq(static_cast<double>(rpg::F32ToF16(65504.0f)), 0x7BFF, "f16 最大值");
  // 注意：65505 **不该**溢出。f16 在 2^15 附近的间距是 32，65505 舍入到 65504。
  // 只有 >= 65520 才进位到 Inf —— 这里两个值都验一遍，防止把舍入写成了截断。
  CheckEq(static_cast<double>(rpg::F32ToF16(65505.0f)), 0x7BFF,
          "f16 65505 舍入到 65504");
  Check(rpg::F32ToF16(70000.0f) == 0x7C00, "f16 70000 -> Inf");
  Check(rpg::F32ToF16(1e30f) == 0x7C00, "f16 大数 -> Inf");
  CheckEq(static_cast<double>(rpg::F32ToF16(0.0f)), 0.0, "f16 0");
  // 次正规：2^-24 是 f16 能表示的最小正数
  CheckEq(static_cast<double>(rpg::F32ToF16(1.0f / 16777216.0f)), 1.0,
          "f16 最小次正规数");
  CheckEq(static_cast<double>(rpg::F32ToF16(1e-30f)), 0.0, "f16 下溢 -> 0");
}

}  // namespace

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--verbose") == 0) g_verbose = true;
  }

  TestBytes();
  TestUtf8();
  TestByteMap();
  TestUnicodeCategories();
  TestJson();
  TestJsonPython();
  TestChatTemplate();
  TestToolParser();
  TestSimpleQuants();

  std::printf("\n通过 %d，失败 %d\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
