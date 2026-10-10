#!/usr/bin/env python3
"""M1 一次性跑完所有前向对拍组合。

一条命令覆盖"两个量化档位 × 中英两种提示词"，逐个跑
forward_check（我们）→ ref_forward（PyTorch 官方）→ cmp_npy（比）。

为什么要多个组合而不是只跑一个：
  * **Q8_0 / Q4_K_M** —— 覆盖不同的 ggml 量化类型（Q8_0 走 32 元素块，
    Q4_K/Q6_K 走 256 元素块 + 6-bit 打包的 scale/min），解码路径不同。
  * **英文短 / 中文长（ChatML，64 token）** —— 中文走 BBPE 的多字节合并，
    长序列走更多 RoPE 位置、更长的因果注意力。

用法：
  check_forward.py                       # 全部组合
  check_forward.py --tol 2e-4 --only q8
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmp_npy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-4, help="p99 相对误差上限")
    ap.add_argument("--only", choices=["q8", "q4"], help="只跑某一个量化档位")
    ap.add_argument("--forward-check", default="build/forward_check.exe")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--skip-ref", action="store_true",
                    help="复用已有的 .ref.*.npy（调试时省 30 秒）")
    args = ap.parse_args()

    py = sys.executable
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.chdir(root)
    os.makedirs(args.runs, exist_ok=True)

    zh = "tests/prompts/zh_chatml.txt"
    ascii_prompt = "The capital of France is"

    cases = []
    if args.only in (None, "q8"):
        q8 = "models/Qwen3-1.7B-Q8_0.gguf"
        cases.append(("q8_ascii", q8, ["--prompt", ascii_prompt]))
        cases.append(("q8_zh", q8, ["--prompt-file", zh]))
    if args.only in (None, "q4"):
        q4 = "models/Qwen3-1.7B-Q4_K_M.gguf"
        cases.append(("q4_zh", q4, ["--prompt-file", zh]))

    print("=" * 70)
    print("M1 前向对拍：C++ 实现 vs PyTorch 官方实现（同一份 GGUF 权重）")
    print("判据：逐元素相对误差 p99 <= {}".format(args.tol))
    print("=" * 70)

    results = []
    for tag, model, prompt_args in cases:
        if not os.path.exists(model):
            print(f"\n[{tag}] 跳过：缺 {model}")
            results.append((tag, None, 0.0, 0))
            continue

        prefix = os.path.join(args.runs, tag)
        print(f"\n{'-' * 70}\n[{tag}]  {model}")
        t0 = time.time()
        r = subprocess.run(
            [args.forward_check, model, *prompt_args, "--out", prefix, "--top", "3"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"  forward_check 失败：{r.stderr.strip()[:400]}")
            results.append((tag, False, 0.0, 0))
            continue
        # 只回显输入与 top-3，中间过程不刷屏
        for line in r.stdout.splitlines():
            if line.startswith(("[输入]", "       id:", "       还原:", "  #", "[前向]")):
                print("  " + line)

        if not args.skip_ref:
            r = subprocess.run(
                [py, os.path.join(here, "ref_forward.py"), model, prefix,
                 "--tokens-from", prefix + ".tokens.txt"],
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"  ref_forward 失败：{r.stderr.strip()[:600]}")
                results.append((tag, False, 0.0, 0))
                continue

        ok, worst, n = cmp_npy.compare(prefix, tol=args.tol, verbose=False)
        print(f"  → {n} 项比较，最差 p99 = {worst:.3g}  "
              f"{'PASS' if ok else 'FAIL'}   （{time.time() - t0:.0f}s）")
        results.append((tag, ok, worst, n))

    print("\n" + "=" * 70)
    print(f"{'组合':<14}{'最差 p99':>12}   结论")
    print("-" * 70)
    bad = 0
    for tag, ok, worst, n in results:
        if ok is None:
            state = "跳过（缺权重）"
        elif ok:
            state = f"PASS（{n} 项，argmax 全一致）"
        else:
            state = "FAIL"
            bad += 1
        print(f"{tag:<14}{worst:>12.3g}   {state}")

    ran = [r for r in results if r[1] is not None]
    print("=" * 70)
    if bad == 0 and ran:
        print(f"全部通过（{len(ran)} 个组合）。")
        return 0
    if not ran:
        print("一个组合都没跑 —— 先下权重：python scripts/fetch_model.py --preset qwen3-1.7b-q8")
        return 1
    print(f"{bad} 个组合未通过。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
