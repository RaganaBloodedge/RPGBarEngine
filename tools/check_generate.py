#!/usr/bin/env python3
"""M2 对拍：贪心解码**逐 token** 与官方实现一致。

为什么这条判据比 logits 对拍更硬：
  logits 对拍只要求"某个位置的分布差得足够小"，一个 1e-6 的偏差只要不改变
  argmax 就算过。而逐 token 一致要求**每一步的 argmax 都对** —— 一旦某一步
  选错，后面所有 token 都会跟着变（自回归里误差会被放大），所以"整串 N 个
  token 完全相同"是一句很重的断言。

参考侧：transformers 官方 Qwen3ForCausalLM + gguf.quants 反量化 + **同一份
GGUF 权重**（所以量化误差被消掉，残差只可能来自我们的实现：cache 布局、
位置编码、增量 attention）。参考的生成循环也是自己写的（见 qwen3_ref.
greedy_generate 里的理由：不用 model.generate，避开它的默认 logits processor）。

顺带说明：**两边都不需要随机数**。贪心就是 argmax，全程没有 RNG —— 这也是
这个判据能成立的前提（有随机性就没法逐 token 比）。

用法：
  check_generate.py                 # 全部组合
  check_generate.py --only q8       # 只跑 Q8_0
  check_generate.py --max-tokens 48
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_ids(path):
    with open(path, "r", encoding="ascii") as f:
        return [int(x) for x in f.read().strip().replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q8", "q4"], help="只跑某一个量化档位")
    ap.add_argument("--max-tokens", type=int, default=24, help="生成多少个 token")
    ap.add_argument("--gen-check", default="build/gen_check.exe")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--no-kv-check", action="store_true",
                    help="让 gen_check 跳过 cache 一致性自检（省一半 prefill 时间）")
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

    print("=" * 72)
    print("M2 生成对拍：C++ 贪心解码 vs PyTorch 官方实现（同一份 GGUF 权重）")
    print(f"判据：{args.max_tokens} 个 token 逐 token 相同")
    print("=" * 72)

    # 参考侧的重依赖只在这里 import —— 免得 --help 也要等 torch。
    import torch  # noqa: E402
    import qwen3_ref  # noqa: E402

    torch.set_grad_enabled(False)

    results = []
    for tag, model, prompt_args in cases:
        if not os.path.exists(model):
            print(f"\n[{tag}] 跳过：缺 {model}")
            results.append((tag, None, 0, ""))
            continue

        prefix = os.path.join(args.runs, tag)
        print(f"\n{'-' * 72}\n[{tag}]  {model}")
        t0 = time.time()
        cmd = [args.gen_check, model, *prompt_args,
               "--max-tokens", str(args.max_tokens), "--out", prefix]
        if args.no_kv_check:
            cmd.append("--no-kv-check")
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"  gen_check 失败：{(r.stderr or r.stdout).strip()[:500]}")
            results.append((tag, False, 0, ""))
            continue
        for line in r.stdout.splitlines():
            if line.startswith(("[输入]", "[缓存] 约", "[自检]", "[prefill]",
                                "[decode] 生成", "[decode]  cache", "[生成] 解码")):
                print("  " + line)
        if "[自检]" in r.stdout and "不一致" in r.stdout:
            results.append((tag, False, 0, "KV Cache 自检失败"))
            continue

        ours_prompt = read_ids(prefix + ".tokens.txt")
        ours_gen = read_ids(prefix + ".gen.txt")

        ref_model, _, cfg, vocab, tied = qwen3_ref.build_model(model)
        t1 = time.time()
        # 用**我们的 prompt ids** 喂参考侧：两边输入逐 id 相同，于是分词差异
        # 不参与这次比较（分词本身在 M0 已经单独对拍过）。
        ref_gen = qwen3_ref.greedy_generate(ref_model, ours_prompt, args.max_tokens)
        del ref_model
        print(f"  [参考] 生成 {len(ref_gen)} 个 token（{time.time() - t1:.0f}s）")

        same = (ours_gen == ref_gen)
        note = ""
        if not same:
            n = min(len(ours_gen), len(ref_gen))
            i = next((k for k in range(n) if ours_gen[k] != ref_gen[k]), n)
            note = (f"第 {i} 个 token 起不同：我们 {ours_gen[i] if i < len(ours_gen) else '-'}"
                    f" / 参考 {ref_gen[i] if i < len(ref_gen) else '-'}"
                    f"（长度 {len(ours_gen)} vs {len(ref_gen)}）")
            print(f"  → {note}")
        print(f"  → {len(ours_gen)}/{args.max_tokens} 个 token  "
              f"{'PASS' if same else 'FAIL'}   （{time.time() - t0:.0f}s）")
        results.append((tag, same, len(ours_gen), note))

    print("\n" + "=" * 72)
    print(f"{'组合':<12}{'token 数':>10}   结论")
    print("-" * 72)
    bad = 0
    for tag, ok, n, note in results:
        if ok is None:
            state = "跳过（缺权重）"
        elif ok:
            state = "PASS（逐 token 一致）"
        else:
            state = f"FAIL  {note}"
            bad += 1
        print(f"{tag:<12}{n:>10}   {state}")

    ran = [r for r in results if r[1] is not None]
    print("=" * 72)
    if bad == 0 and ran:
        print(f"全部通过（{len(ran)} 个组合）—— KV Cache 与贪心解码正确。")
        return 0
    if not ran:
        print("一个组合都没跑 —— 先下权重：python scripts/fetch_model.py --preset qwen3-1.7b-q8")
        return 1
    print(f"{bad} 个组合未通过。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
