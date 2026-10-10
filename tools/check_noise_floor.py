#!/usr/bin/env python3
"""M1 的"判据该怎么定"实验：float32 的舍入噪声到底有多大。

## 为什么需要这个

M1 的原始判据是"逐张量误差 < 1e-4"。跑出来发现逐层残差流的 max|Δ| 是
0.0117 —— 超限 100 倍。但相对误差只有 ~1e-6，而且 top-5 token 完全一致。
看一眼量级就明白了：**Qwen3 的残差流里有一个巨值维度**（|h| ≈ 1.2e4，
第 2 层之后出现并一直保持）。在 1.2e4 上，float32 的 1 ULP ≈ 0.001 ——
"绝对 1e-4" 在这个量级上根本没有意义（比 ULP 还小 10 倍）。

所以问题变成："0.0117 是 float32 无法避免的舍入，还是我们哪里错了？"

## 怎么回答

把"我们 vs 真值"和"float32 自己 vs 真值"放在**同一深度**上比：

    ours32(D)          —— 我们跑前 D 层的 float32 结果
    ref32(D)           —— 官方实现跑前 D 层的 float32 结果
    ref64(D)           —— 官方实现跑前 D 层的 float64 结果（≈ 真值）

    e_ours = max|ours32 - ref64|        # 我们的总误差
    e_floor = max|ref32  - ref64|       # float32 自身的噪声底

若 `e_ours / e_floor ≈ 1`，说明我们的实现已经贴着 float32 的极限，
再优化也降不下去 —— 判据应当改成"相对误差"而不是绝对的 1e-4。

ref64 用 `use_exact_rmsnorm()` 打补丁：官方 RMSNorm 内部强制 float32，
不拿掉这个舍入，"float64 真值"就不真。

用法：
  check_noise_floor.py models/Qwen3-1.7B-Q8_0.gguf --layers 8 --tokens-from runs/x.tokens.txt
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref  # noqa: E402


def maxdiff(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def rel_stats(a, b, floor=1e-3):
    """逐元素相对误差的中位数 / p99 / 最大值。

    为什么要用相对误差的分位数，而不是 max|Δ|：
      * 残差流里有一个 ~1.2e4 的巨值维度（Qwen3 的离群通道），那里 1 ULP ≈ 1e-3；
      * 还有大量**接近零**的坐标，那里绝对差极小但相对误差可以很大。
    绝对差的最大值被第一类主导，相对误差的最大值被第二类主导 —— 两个极值
    都没信息量。中位数与 p99 才反映"整体算得准不准"。
    只统计 |b| > floor 的元素，把"接近零导致相对误差失真"这一类剔掉。
    """
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    m = np.abs(b.astype(np.float64)) > floor
    if not m.any():
        return 0.0, 0.0, 0.0
    rel = d[m] / np.abs(b.astype(np.float64))[m]
    return float(np.median(rel)), float(np.percentile(rel, 99)), float(rel.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--layers", type=int, default=8, help="对比到第几层")
    ap.add_argument("--tokens-from", required=True)
    ap.add_argument("--forward-check", default="build/forward_check.exe")
    ap.add_argument("--work", default="runs/floor")
    args = ap.parse_args()

    tokens = [int(x) for x in
              open(args.tokens_from, encoding="ascii").read().strip().replace(",", " ").split()]
    D = args.layers
    torch.set_grad_enabled(False)

    # ---- 我们的：只跑前 D 层 ----
    t0 = time.time()
    subprocess.run(
        [args.forward_check, args.model, "--tokens", ",".join(map(str, tokens)),
         "--layers", str(D), "--out", args.work, "--top", "1"],
        check=True, stdout=subprocess.DEVNULL,
    )
    ours = np.load(args.work + ".hidden.npy")
    print(f"[我们] 跑 {D} 层 → {ours.shape}  ({time.time() - t0:.1f}s)")

    # ---- 参考：float32 与 float64，各跑 D 层，final norm 换成恒等 ----
    # norm_identity 让 HF 的 hidden_states[D] 变成"第 D-1 层输出且没过 final norm"，
    # 与我们的 capture_hidden[D] 语义一致。
    refs = {}
    for tag, dtype in (("ref32", torch.float32), ("ref64", torch.float64)):
        t0 = time.time()
        if dtype == torch.float64:
            qwen3_ref.use_exact_rmsnorm()  # 真值：拿掉官方 norm 内部的 float32 舍入
        model, _, cfg, _, _ = qwen3_ref.build_model(
            args.model, n_layers=D, dtype=dtype, norm_identity=True)
        _, hidden = qwen3_ref.run_forward(model, tokens, hidden_count=D + 1)
        refs[tag] = hidden
        print(f"[{tag}] {hidden.shape}  ({time.time() - t0:.1f}s)")
        del model

    ref32, ref64 = refs["ref32"], refs["ref64"]
    assert ours.shape == ref32.shape == ref64.shape, (ours.shape, ref32.shape, ref64.shape)

    print("\n指标：逐元素相对误差（只统计 |ref64| > 1e-3 的元素），取中位数与 p99。")
    print("  '我们'  = |ours32 - ref64| / |ref64|   —— 我们的实现离真值多远")
    print("  '噪声底' = |ref32  - ref64| / |ref64|   —— float32 + 这套算子能做到多好")
    print("  绝对 max|Δ| 不作判据：残差流有 ~1.2e4 的离群维度，那里 1 ULP ≈ 1e-3。\n")

    hdr = (f"{'项':<18}{'中位 我们':>12}{'中位 噪声底':>13}"
           f"{'p99 我们':>12}{'p99 噪声底':>13}{'p99 比值':>10}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    for k in range(ours.shape[0]):
        label = "embedding" if k == 0 else f"layer {k - 1} 输出"
        mo, po, _ = rel_stats(ours[k], ref64[k])
        mf, pf, _ = rel_stats(ref32[k], ref64[k])
        ratio = po / pf if pf > 0 else float("inf")
        if np.isfinite(ratio):
            worst = max(worst, ratio)
        print(f"{label:<18}{mo:>12.3g}{mf:>13.3g}{po:>12.3g}{pf:>13.3g}{ratio:>10.2f}")

    print(f"\np99 比值的最大值 {worst:.2f}")
    if worst < 3.0:
        print("结论：我们的相对误差与 float32 噪声底同量级（p99 比值 < 3）。")
        print("      —— 实现没有算法问题；差异只是浮点舍入，且量级正常。")
        return 0
    print("结论：相对误差显著超过噪声底，说明实现里有真问题（不是精度问题）。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
