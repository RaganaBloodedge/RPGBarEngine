#!/usr/bin/env python3
"""check_chatml.py —— M3 第 8 层：ChatML 渲染逐字节对拍。

判据是**逐字节相同**（不是"看起来差不多"）。参考侧直接用 transformers 的
apply_chat_template 跑官方 tokenizer_config.json 里的模板本身 —— 也就是说，
我们不是在和"某个人对模板的理解"比，而是在和官方模板原文比。

覆盖到官方模板里所有会走不同分支的情形：
  * 有 / 无 tools（两个头部分支）
  * 有 / 无 system 首条
  * assistant 显式 reasoning_content / content 内嵌 </think> / 两者都没有
  * last_query_index 前后（决定要不要渲染 <think> 块）
  * 连续 tool 消息合并成一条 user
  * 多条 tool_calls
  * add_generation_prompt 开 / 关，enable_thinking 开 / 关
  * 工具定义里的刁钻值（浮点、布尔、null、嵌套数组、转义、中文）

用法：python tools/check_chatml.py [--work runs/chatml] [--only 名字片段]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

TOOL_SCENE = {
    "type": "function",
    "function": {
        "name": "read_scene",
        "description": "读取当前场景的公开描述",
        "parameters": {
            "type": "object",
            "properties": {"scene": {"type": "string", "description": "场景 id"}},
            "required": ["scene"],
        },
    },
}

TOOL_TRICKY = {
    "type": "function",
    "function": {
        "name": "roll_check",
        "description": "暗骰：判定能否撬开箱子（含\"引号\"与\n换行）",
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "enum": ["力量", "敏捷"]},
                "dc": {"type": "integer", "minimum": 1, "maximum": 30},
                "advantage": {"type": "boolean"},
                # 0.1 / 1e-05 是刻意挑的：它们用 "%.17g" 会写成
                # 0.10000000000000001 / 1.0000000000000001e-05，
                # 用 Python 的最短往返表示才是 0.1 / 1e-05。
                # 拿 0.5 测不出来（两种写法结果一样）。
                "bonus": {"type": "number", "default": 0.1},
                "step": {"type": "number", "default": 1e-05},
                "tags": {"type": "array", "items": {"type": "string"}},
                "hint": {"type": "null"},
            },
            "required": ["skill"],
            "additionalProperties": False,
        },
    },
}

TOOL_ADVICE = {
    "type": "function",
    "function": {
        "name": "action_advice",
        "description": "给玩家一条行动建议",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def build_cases():
    cases = []

    cases.append(("01_user_only", {
        "messages": [{"role": "user", "content": "我撬开箱子。"}],
    }))

    cases.append(("02_system_user_nonthink", {
        "messages": [
            {"role": "system", "content": "你是一位严格但公正的 DM。"},
            {"role": "user", "content": "我撬开箱子。"},
        ],
        "enable_thinking": False,
    }))

    cases.append(("03_history", {
        "messages": [
            {"role": "system", "content": "你是 DM。"},
            {"role": "user", "content": "我走进神庙。"},
            {"role": "assistant", "content": "神庙里弥漫着尘土味。"},
            {"role": "user", "content": "我看看祭坛。"},
        ],
    }))

    cases.append(("04_tools_no_system", {
        "messages": [{"role": "user", "content": "现在场景里有什么？"}],
        "tools": [TOOL_SCENE],
        "enable_thinking": False,
    }))

    cases.append(("05_tools_with_system", {
        "messages": [
            {"role": "system", "content": "你是一位严格但公正的 DM。"},
            {"role": "user", "content": "现在场景里有什么？"},
        ],
        "tools": [TOOL_SCENE, TOOL_ADVICE],
        "enable_thinking": False,
    }))

    cases.append(("06_tool_result_roundtrip", {
        "messages": [
            {"role": "system", "content": "你是 DM。"},
            {"role": "user", "content": "现在场景里有什么？"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "read_scene", "arguments": {"scene": "temple"}},
            ]},
            {"role": "tool", "content": "{\"scene\":\"temple\",\"exits\":[\"north\"]}"},
        ],
        "tools": [TOOL_SCENE],
        "enable_thinking": False,
    }))

    cases.append(("07_two_tool_results", {
        "messages": [
            {"role": "user", "content": "同时读场景和检定。"},
            {"role": "assistant", "content": "稍等。", "tool_calls": [
                {"name": "read_scene", "arguments": {"scene": "temple"}},
                {"name": "action_advice", "arguments": {}},
            ]},
            {"role": "tool", "content": "{\"a\":1}"},
            {"role": "tool", "content": "{\"b\":2}"},
        ],
        "tools": [TOOL_SCENE, TOOL_ADVICE],
    }))

    cases.append(("08_assistant_reasoning", {
        "messages": [
            {"role": "user", "content": "我撬开箱子。"},
            {"role": "assistant", "content": "箱子纹丝不动。",
             "reasoning_content": "先看力量值……再决定难度。"},
        ],
    }))

    cases.append(("09_thinking_embedded_in_content", {
        "messages": [
            {"role": "user", "content": "我撬开箱子。"},
            {"role": "assistant",
             "content": "<think>\n先看力量值。\n</think>\n\n箱子纹丝不动。"},
        ],
    }))

    cases.append(("10_no_generation_prompt", {
        "messages": [{"role": "user", "content": "我撬开箱子。"}],
        "add_generation_prompt": False,
    }))

    cases.append(("11_multistep_tool_loop", {
        # 最后一次真实提问之后还有 assistant（带 reasoning），走
        # `loop.index0 > last_query_index` 分支。
        "messages": [
            {"role": "user", "content": "现在场景里有什么？"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"name": "read_scene", "arguments": {"scene": "temple"}}]},
            {"role": "tool", "content": "{\"scene\":\"temple\"}"},
            {"role": "assistant", "content": "神庙里有一座祭坛。",
             "reasoning_content": "工具结果到手，组织语言。"},
        ],
        "tools": [TOOL_SCENE],
    }))

    cases.append(("12_tricky_tool_json", {
        "messages": [
            {"role": "system", "content": "你是 DM。"},
            {"role": "user", "content": "投个暗骰。"},
        ],
        "tools": [TOOL_TRICKY],
        "enable_thinking": False,
    }))

    # 空 messages 不作为对拍用例：transformers 的 apply_chat_template 直接抛
    # "Cannot apply chat template to an empty conversation"，没有参考物可比。
    # 引擎侧会正常渲染出生成提示（不崩），但那属于我们自己的约定。

    cases.append(("14_assistant_only", {
        "messages": [{"role": "assistant", "content": "（上回说到神庙。）"}],
    }))

    cases.append(("15_system_only", {
        "messages": [{"role": "system", "content": "你是 DM。"}],
    }))

    cases.append(("16_tool_resp_role_user_literal", {
        # 用户"手动"写了一段 <tool_response>，最后一条真实提问要在它之前。
        "messages": [
            {"role": "user", "content": "我看看祭坛。"},
            {"role": "assistant", "content": "祭坛上刻着字。",
             "reasoning_content": "读完了。"},
        ],
    }))

    return cases


def first_diff(a: str, b: str):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(REPO, "runs/chatml"))
    ap.add_argument("--only", default="")
    ap.add_argument("--chatml-check", default=os.path.join(REPO, "build/chatml_check.exe"))
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)

    cases = build_cases()
    if args.only:
        cases = [c for c in cases if args.only in c[0]]
    if not cases:
        print("没有匹配的用例。", file=sys.stderr)
        return 1

    failed = []
    for name, spec in cases:
        case_path = os.path.join(args.work, name + ".case.json")
        ours_path = os.path.join(args.work, name + ".ours.txt")
        ref_path = os.path.join(args.work, name + ".ref.txt")

        with io.open(case_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

        r = subprocess.run(
            [args.chatml_check, "--case", case_path, "--out", ours_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name}: chatml_check 退出 {r.returncode}\n{r.stderr.strip()}")
            failed.append(name)
            continue

        r = subprocess.run(
            [sys.executable, os.path.join(REPO, "tools/chatml_ref.py"),
             "--case", case_path, "--out", ref_path],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL {name}: chatml_ref 退出 {r.returncode}\n{r.stderr.strip()[-600:]}")
            failed.append(name)
            continue

        ours = io.open(ours_path, encoding="utf-8", newline="").read()
        ref = io.open(ref_path, encoding="utf-8", newline="").read()

        if ours == ref:
            print(f"PASS {name:<34} {len(ours):>5} 字节")
        else:
            failed.append(name)
            i = first_diff(ours, ref)
            print(f"FAIL {name}: 首个差异在第 {i} 字节"
                  f"（我们 {len(ours)} 字节 / 官方 {len(ref)} 字节）")
            lo = max(0, i - 60)
            print(f"     我们: ...{ours[lo:i + 60]!r}")
            print(f"     官方: ...{ref[lo:i + 60]!r}")

    print()
    if failed:
        print(f"未通过 {len(failed)}/{len(cases)}：{', '.join(failed)}")
        return 1
    print(f"全部 {len(cases)} 个用例逐字节一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
