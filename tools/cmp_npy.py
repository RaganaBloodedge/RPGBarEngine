#!/usr/bin/env python3
"""M1 主检查：C++ 前向 vs PyTorch 参考前向。

## 判据为什么不是"逐张量 max|Δ| < 1e-4"

M1 设计里原本写的是这个判据。跑出来发现逐层残差流的 max|Δ| 是 0.0117 ——
"超限 100 倍"，但 top-5 token 完全一致。原因是 **Qwen3 的残差流里有一个
量级 ~1.2e4 的离群维度**（第 2 层之后出现并一直保持）：

    |h| = 1.2e4  →  float32 的 1 ULP ≈ 0.001

也就是说在这个量级上，"绝对 1e-4" 比一个 ULP 还小 10 倍 —— 这个判据在
float32 下**不可能**满足，与实现好坏无关。

所以判据改为**逐元素相对误差的分位数**：

  * `中位`   —— 大部分元素算得准不准
  * `p99`    —— 尾部元素算得准不准（判据用这个）
  * `max|Δ|` —— 只作参考。它被两类元素主导而失去信息量：巨值维度（ULP 大）、
                以及近零坐标（绝对差极小但相对误差可以很大）。

用同一个统计量去量"float32 的噪声底"（check_noise_floor.py 干这个），
就能回答"我们的误差是正常舍入还是实现错误"：p99 比值 < 3 即为正常。

用法：
  cmp_npy.py <prefix> [--tol 2e-4]      # 单次，详细输出
  也可 `from cmp_npy import compare` 拿到 (ok, worst_p99)
"""

import argparse
import sys

import numpy as np

REL_FLOOR = 1e-3  # 只统计 |ref| > 此值的元素，剔除近零坐标导致的相对误差失真


def rel_stats(ours, ref):
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    a = np.abs(ref.astype(np.float64))
    m = a > REL_FLOOR
    if not m.any():
        return 0.0, 0.0, 0.0, float(d.max())
    rel = d[m] / a[m]
    return (float(np.median(rel)), float(np.percentile(rel, 99)), float(rel.max()),
            float(d.max()))


def compare(prefix, tol=2e-4, verbose=True, top=5):
    """比较 <prefix>.{logits,hidden}.npy 与 <prefix>.ref.{logits,hidden}.npy。

    返回 (是否通过, 最差 p99, 比较项数)。
    """
    say = print if verbose else (lambda *a, **k: None)

    ours_logits = np.load(prefix + ".logits.npy")
    ref_logits = np.load(prefix + ".ref.logits.npy")
    ours_hidden = np.load(prefix + ".hidden.npy")
    ref_hidden = np.load(prefix + ".ref.hidden.npy")

    seq, vocab = ours_logits.shape
    n_all, seq2, _ = ours_hidden.shape
    n_layers = ref_hidden.shape[0]
    assert seq == seq2 == ref_hidden.shape[1], (seq, seq2, ref_hidden.shape)
    assert n_all == n_layers + 1, (
        f"我们的 hidden 第一维应是 n_layers+1={n_layers + 1}，实际 {n_all}")

    say(f"[形状] logits {ours_logits.shape}  hidden 我们 {ours_hidden.shape}"
        f" / 参考 {ref_hidden.shape}")
    say(f"\n[逐层残差流] 相对误差（|ref| > {REL_FLOOR:g}），判据 p99 <= {tol:g}")
    hdr = f"  {'项':<16}{'中位':>11}{'p99':>11}{'max':>11}{'max|Δ|':>12}"
    say(hdr)

    ok_all = True
    worst_p99 = 0.0
    first_bad = None
    for k in range(n_layers):
        label = "embedding" if k == 0 else f"layer {k - 1} 输出"
        med, p99, mx, amax = rel_stats(ours_hidden[k], ref_hidden[k])
        worst_p99 = max(worst_p99, p99)
        ok = p99 <= tol
        say(f"  {label:<16}{med:>11.3g}{p99:>11.3g}{mx:>11.3g}{amax:>12.4g}"
            f"   {'OK' if ok else '**超限**'}")
        if not ok:
            ok_all = False
            if first_bad is None:
                first_bad = k

    say(f"\n[输出层] logits[{seq},{vocab}]（覆盖 output_norm + lm_head）")
    say(hdr)
    med, p99, mx, amax = rel_stats(ours_logits, ref_logits)
    worst_p99 = max(worst_p99, p99)
    ok = p99 <= tol
    say(f"  {'logits':<16}{med:>11.3g}{p99:>11.3g}{mx:>11.3g}{amax:>12.4g}"
        f"   {'OK' if ok else '**超限**'}")
    if not ok:
        ok_all = False
        d = np.abs(ours_logits.astype(np.float64) - ref_logits.astype(np.float64))
        idx = np.unravel_index(np.argmax(d), d.shape)
        say(f"      绝对差最大处做标 {idx}：我们 {ours_logits[idx]!r} vs 参考 {ref_logits[idx]!r}")

    say("\n[离散一致性] 每个位置的 argmax 必须相同（差一点就够生成跑偏）")
    n_bad = 0
    for t in range(seq):
        if int(np.argmax(ours_logits[t])) != int(np.argmax(ref_logits[t])):
            n_bad += 1
            say(f"  位置 {t}: argmax 不同 —— 我们 {int(np.argmax(ours_logits[t]))}"
                f" vs 参考 {int(np.argmax(ref_logits[t]))}")
    ok_all &= n_bad == 0
    if n_bad == 0:
        say(f"  {seq} 个位置的 argmax 全部一致")

    last_o, last_r = ours_logits[-1], ref_logits[-1]
    say(f"  最后位置 top-{top} 我们: " + ", ".join(
        f"{int(i)}({last_o[i]:.4f})" for i in np.argsort(-last_o)[:top]))
    say(f"  最后位置 top-{top} 参考: " + ", ".join(
        f"{int(i)}({last_r[i]:.4f})" for i in np.argsort(-last_r)[:top]))

    say("")
    if ok_all:
        say(f"PASS —— {n_layers + 1} 项全部满足 p99 相对误差 <= {tol:g}，argmax 全一致")
    else:
        say(f"FAIL —— 第一处超限在第 {first_bad} 项"
            f"（{'embedding' if first_bad == 0 else f'第 {first_bad - 1} 层之后'}）")
    return ok_all, worst_p99, n_layers + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="forward_check 写出的前缀，例如 runs/cpp_q8")
    ap.add_argument("--tol", type=float, default=2e-4, help="p99 相对误差上限")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()
    ok, _, _ = compare(args.prefix, tol=args.tol, verbose=True, top=args.top)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
