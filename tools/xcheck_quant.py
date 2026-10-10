#!/usr/bin/env python3
"""xcheck_quant.py —— 跨量化档位的交叉校验。

同一份原始权重被量化成 Q8_0 与 Q4_K_M 两份 GGUF。两份的反量化结果必须
**高度相关**：它们逼近的是同一个张量，差异只应来自量化噪声。

这是一个"语义正确性"的独立检验 —— 只看文件头（check_quant_layout.py）能验证
块尺寸，但验证不了位布局的语义。如果 Q4_K / Q6_K 的 scale 拆包或 nibble 归属
写错了，反量化会读出量级完全不同的垃圾，相关系数会掉到 0 附近；写对了，相关系数
应在 0.99 以上，且 Q8 侧的幅度远大于差异。

F32 张量（各 norm 的权重）在两份文件里是同一份数据，必须**逐位相同**。

用法：
    python tools/xcheck_quant.py <Q8.gguf> <Q4KM.gguf> [--n 256]
"""
import argparse
import math
import re
import subprocess
import sys

TENSORS = [
    "blk.0.ffn_gate.weight",
    "blk.0.ffn_down.weight",
    "blk.13.attn_q.weight",
    "blk.13.attn_v.weight",
    "blk.0.attn_norm.weight",   # F32：必须逐位相同
    "output_norm.weight",       # F32：必须逐位相同
]


def sample(exe, gguf, tensor, n):
    out = subprocess.run([exe, gguf, "--sample", tensor, str(n)],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        raise SystemExit(f"dump_gguf 失败：{out.stderr}")
    vals = []
    for line in out.stdout.split("\n"):
        m = re.match(r"\s*\[\d+\]\s+(\S+)", line)
        if m:
            vals.append(float(m.group(1)))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("q8")
    ap.add_argument("q4")
    ap.add_argument("--exe", default="build/dump_gguf.exe")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    print(f"{'张量':<26}{'max|Δ|':>12}{'RMS(Δ)':>11}{'RMS(a)':>10}"
          f"{'相关系数':>11}  结论")
    bad = 0
    for name in TENSORS:
        a = sample(args.exe, args.q8, name, args.n)
        b = sample(args.exe, args.q4, name, args.n)
        if len(a) != len(b) or not a:
            print(f"{name:<26} 取值失败（{len(a)} vs {len(b)}）")
            bad += 1
            continue
        diffs = [x - y for x, y in zip(a, b)]
        mx = max(abs(d) for d in diffs)
        rms_d = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        rms_a = math.sqrt(sum(x * x for x in a) / len(a))
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        den = math.sqrt(sum((x - ma) ** 2 for x in a) *
                        sum((y - mb) ** 2 for y in b))
        corr = num / den if den else float("nan")

        # F32 张量必须逐位一致；量化张量要求高度相关。
        if name.endswith("norm.weight"):
            ok = (mx == 0.0)
            note = "逐位相同" if ok else "!! 应逐位相同却不同"
        else:
            ok = corr > 0.99
            note = "符合量化噪声" if ok else "!! 相关系数过低，疑似位布局错误"
        if not ok:
            bad += 1
        print(f"{name:<26}{mx:>12.6f}{rms_d:>11.6f}{rms_a:>10.4f}"
              f"{corr:>11.6f}  {note}")

    print()
    if bad:
        print(f"{bad} 个张量未通过。")
        return 1
    print("跨档位交叉校验通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
