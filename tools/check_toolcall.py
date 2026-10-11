#!/usr/bin/env python3
"""M3 端到端验收：给工具定义 → 生成 → 解析出结构化 tool_calls。

这一条把三件事串起来验，对应 M3 的完成判据"给定工具定义，能稳定产出结构化
tool_calls"：

  1. **渲染**：prompt 由我们自己的 ChatML 渲染器产出。渲染本身在第 8 层已经
     与官方模板**逐字节**对拍过，所以这里不再重复比模板，而是把它当已知输入。
  2. **解码**：引擎贪心生成的结果与官方实现**逐 token** 一致。工具调用这种
     结构化输出比自由文本更严 —— 只要错一个 token，JSON 就坏了，解析会直接失败。
  3. **解析**：把引擎产出的 token 解码后喂进我们自己的解析器，应当取出
     name / arguments 与预期一致的结构化调用。

用法：
  python tools/check_toolcall.py                # Q8_0
  python tools/check_toolcall.py --model models/Qwen3-1.7B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

# 期望的调用：用例里的用户请求是"查看神庙大厅里有什么"。
EXPECT_NAME = "read_scene"


def read_ids(path):
    with open(path, "r", encoding="ascii") as f:
        return [int(x) for x in f.read().strip().replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(REPO, "models/Qwen3-1.7B-Q8_0.gguf"))
    ap.add_argument("--case", default=os.path.join(REPO, "tests/chatml/dm_toolcall.case.json"))
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--runs", default=os.path.join(REPO, "runs"))
    ap.add_argument("--gen-check", default=os.path.join(REPO, "build/gen_check.exe"))
    ap.add_argument("--chatml-check", default=os.path.join(REPO, "build/chatml_check.exe"))
    ap.add_argument("--parse-check", default=os.path.join(REPO, "build/parse_check.exe"))
    ap.add_argument("--tok-check", default=os.path.join(REPO, "build/tok_check.exe"))
    args = ap.parse_args()

    os.makedirs(args.runs, exist_ok=True)
    prompt_path = os.path.join(args.runs, "dm_toolcall.prompt.txt")
    prefix = os.path.join(args.runs, "toolcall_e2e")
    problems = []

    print("=" * 72)
    print("M3 端到端：工具定义 → 生成 → 结构化 tool_calls")
    print("=" * 72)

    # ---- 1. 渲染 prompt ----
    subprocess.run([args.chatml_check, "--case", args.case, "--out", prompt_path],
                   check=True)
    n_prompt = os.path.getsize(prompt_path)
    print(f"[渲染] {n_prompt} 字节的 ChatML prompt（含 2 个工具定义）")

    # ---- 2. 引擎贪心生成（遇到 <|im_end|> 停）----
    t0 = time.time()
    r = subprocess.run(
        [args.gen_check, args.model, "--prompt-file", prompt_path,
         "--max-tokens", str(args.max_tokens), "--stop-eos", "--stop", "151645",
         "--no-kv-check", "--out", prefix],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(f"[引擎] gen_check 失败：{(r.stderr or r.stdout).strip()[:600]}")
        return 1
    ours = read_ids(prefix + ".gen.txt")
    print(f"[引擎] 生成 {len(ours)} 个 token（{time.time() - t0:.0f}s）")

    # ---- 3. 解码 ----
    # **刻意不加 --skip-special**：`<tool_call>` / `</tool_call>` 本身是特殊
    # token（151657/151658），跳过特殊 token 会把结构标记一起删掉，解析器就
    # 只剩下一段裸 JSON。控制标记由解析器自己清（StripControlMarkers）。
    ids_line = " ".join(str(i) for i in ours)
    d = subprocess.run([args.tok_check, args.model, "--decode"],
                       input=ids_line.encode("utf-8"), capture_output=True)
    text = d.stdout.split(b"\x00")[0].decode("utf-8", errors="replace")
    raw_path = prefix + ".raw.txt"
    with io.open(raw_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"[解码] {raw_path}")
    print("-" * 72)
    print(text)
    print("-" * 72)

    # ---- 4. 我们自己的解析器 ----
    p = subprocess.run([args.parse_check, "--text-file", raw_path, "--json"],
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print(f"[解析] parse_check 失败：{p.stderr.strip()[:400]}")
        return 1
    parsed = json.loads(p.stdout)
    calls = parsed["tool_calls"]
    print(f"[解析] text={parsed['text']!r}")
    print(f"[解析] tool_calls={len(calls)}")
    for i, c in enumerate(calls):
        print(f"   #{i + 1} name={c['name']!r} ok={c['ok']} arguments={c['arguments']}")

    if not calls:
        problems.append("解析器没取出任何工具调用")
    else:
        if not calls[0]["ok"]:
            problems.append("第一个工具调用的参数不是合法对象")
        if calls[0]["name"] != EXPECT_NAME:
            problems.append(f"工具名是 {calls[0]['name']!r}，期望 {EXPECT_NAME!r}")
    if parsed["unterminated_tool_call"]:
        problems.append("工具调用被截断（max_tokens 可能太小）")

    # ---- 5. 与官方实现逐 token 对拍 ----
    import torch  # noqa: E402
    import qwen3_ref  # noqa: E402

    torch.set_grad_enabled(False)
    ref_model, _, _, _, _ = qwen3_ref.build_model(args.model)
    prompt_ids = read_ids(prefix + ".tokens.txt")
    t1 = time.time()
    ref = qwen3_ref.greedy_generate(ref_model, prompt_ids, args.max_tokens)
    del ref_model
    print(f"[参考] 生成 {len(ref)} 个 token（{time.time() - t1:.0f}s）")

    head = ref[:len(ours)]
    if head == ours:
        print(f"[对拍] 逐 token 一致（{len(ours)} 个）")
    else:
        n = min(len(head), len(ours))
        i = next((k for k in range(n) if head[k] != ours[k]), n)
        problems.append(f"逐 token 对拍失败：第 {i} 个 token 起不同")
        print(f"[对拍] FAIL 第 {i} 个 token 起不同："
              f"我们 {ours[i] if i < len(ours) else '-'} / 官方 {head[i] if i < len(head) else '-'}")

    print("=" * 72)
    if problems:
        for msg in problems:
            print(f"未通过：{msg}")
        return 1
    print("通过：渲染正确、解码与官方逐 token 一致、解析出结构化 tool_calls。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
