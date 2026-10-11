#!/usr/bin/env python3
"""采样对拍（M2.5）：引擎的"确定性筛选" vs HF 的 logits warpers。

采样被拆成两半（见 src/sampler.h）：
  * **确定性筛选** —— temperature / top_k / top_p 把 logits 变成分布。
    这一半直接对拍 `transformers` 的 warpers —— `generate(do_sample=True)`
    内部就是按 Temperature → TopK → TopP 的顺序处理 logits 的。
  * **随机性** —— counter-based 纯函数 u = H(seed, counter)。
    这一半没法和 torch.multinomial 逐数对齐（不同 RNG），所以验的是它的
    代数性质：同 seed 同位置必得同一个 u、相邻 counter 不重复、都落在 [0,1)。

判据：
  * 支持集一致（哪些 token 概率非零）；
  * 非零位置逐元素相对误差 p99 <= 1e-5。

另有一条"看得懂"的回归守卫：README 里那张图的数字 —— logits = [4,2,0]，
同一个 u = 0.95，在 T = 0.5 / 1.0 / 2.0 下分别选中 A / B / C。

用法：
    python tools/check_sample.py                 # 自动找 build/sample_check.exe
    python tools/check_sample.py --exe build/sample_check.exe

需要 numpy + torch + transformers。缺了会**跳过**并返回 0（不算失败）——
其它分层对拍脚本也是这个约定。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError as e:  # pragma: no cover
    print(f"跳过：需要 numpy（{e}）")
    sys.exit(0)

try:
    import torch
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )
except ImportError as e:  # pragma: no cover
    print(f"跳过：需要 torch + transformers（{e}）")
    sys.exit(0)


def find_exe(explicit: str) -> str:
    if explicit:
        return explicit
    for cand in ("build/sample_check.exe", "build/sample_check"):
        if os.path.exists(cand):
            return cand
    print("找不到 sample_check 可执行文件，请先跑 build.sh")
    sys.exit(2)


def run_engine_probs(exe: str, logits, temperature, top_k, top_p, tmp: str):
    lp = os.path.join(tmp, "logits.bin")
    op = os.path.join(tmp, "probs.bin")
    np.asarray(logits, dtype=np.float32).tofile(lp)
    r = subprocess.run(
        [exe, "--logits", lp, "--temperature", str(temperature),
         "--top-k", str(top_k), "--top-p", str(top_p), "--out-probs", op],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"sample_check 失败：{r.stderr.strip()}")
    return np.fromfile(op, dtype=np.float32)


def run_engine_sample(exe: str, logits, temperature, top_k, top_p, u, tmp: str) -> int:
    lp = os.path.join(tmp, "logits.bin")
    np.asarray(logits, dtype=np.float32).tofile(lp)
    r = subprocess.run(
        [exe, "--logits", lp, "--temperature", str(temperature),
         "--top-k", str(top_k), "--top-p", str(top_p), "--u", str(u)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"sample_check 失败：{r.stderr.strip()}")
    for line in r.stdout.splitlines():
        if line.startswith("u="):
            return int(line.split("token=")[1])
    raise RuntimeError(f"没解析到 token：{r.stdout!r}")


def run_engine_rng(exe: str, seed: int, counter: int, count: int) -> list[float]:
    r = subprocess.run(
        [exe, "--rng", "--seed", str(seed), "--counter", str(counter),
         "--count", str(count)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"sample_check --rng 失败：{r.stderr.strip()}")
    return [float(x) for x in r.stdout.split()]


def hf_probs(logits, temperature, top_k, top_p):
    scores = torch.tensor(np.asarray(logits, dtype=np.float32)).unsqueeze(0)
    if temperature and temperature > 0:
        if temperature != 1.0:
            scores = TemperatureLogitsWarper(temperature)(None, scores)
        if top_k and top_k > 1:
            scores = TopKLogitsWarper(top_k)(None, scores)
        if top_p and 0 < top_p < 1.0:
            scores = TopPLogitsWarper(top_p)(None, scores)
        return torch.softmax(scores, dim=-1).squeeze(0).numpy()
    # temperature <= 0：引擎退化为 one-hot(argmax)，HF 没有对应 warper。
    out = np.zeros_like(np.asarray(logits, dtype=np.float32))
    out[int(np.argmax(logits))] = 1.0
    return out


def compare(a, b, tol_rel=1e-5):
    sa = set(np.nonzero(a > 0)[0].tolist())
    sb = set(np.nonzero(b > 0)[0].tolist())

    # 强判据：支持集完全相同 → 逐元素比。
    if sa == sb:
        if not sa:
            return True, "双方都是全零"
        idx = sorted(sa)
        rel = np.abs(a[idx] - b[idx]) / np.maximum(np.abs(b[idx]), 1e-30)
        p99 = float(np.percentile(rel, 99))
        if p99 > tol_rel:
            return False, f"相对误差 p99 = {p99:.3e}（支持集 {len(sa)}）"
        return True, f"支持集 {len(sa)} 个，相对误差 p99 = {p99:.2e}"

    # 退化情形：logits 完全并列（例如全 0）时，top_p 在边界上保留哪几个是
    # **任意**的 —— 引擎和 torch.sort 的排序不稳定，取出的下标自然不同。
    # 这时比概率的**多重集**：数量与数值都对上就算一致。
    if len(sa) == len(sb):
        va = np.sort(a[sorted(sa)])
        vb = np.sort(b[sorted(sb)])
        rel = np.abs(va - vb) / np.maximum(np.abs(vb), 1e-30)
        p99 = float(np.percentile(rel, 99)) if len(va) else 0.0
        if p99 <= tol_rel:
            return True, f"并列重排（{len(sa)} 个，概率多重集一致，p99 = {p99:.2e}）"

    only_e = sorted(sa - sb)[:6]
    only_r = sorted(sb - sa)[:6]
    return False, f"支持集不同：引擎独有 {only_e}，参考独有 {only_r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="", help="sample_check 可执行文件路径")
    args = ap.parse_args()
    exe = find_exe(args.exe)

    passed = 0
    failed: list[str] = []

    def check(cond, what):
        nonlocal passed
        if cond:
            passed += 1
            print(f"  通过  {what}")
        else:
            failed.append(what)
            print(f"  失败  {what}")

    rng = np.random.default_rng(20261011)

    # ---------------------------------------------------------- 1. 确定性筛选
    print("== 确定性筛选：引擎 vs HF warpers ==")
    cases = {
        "均匀": np.zeros(64, dtype=np.float32),
        "尖峰": np.concatenate([[8.0], rng.normal(0, 1.0, 63)]).astype(np.float32),
        "平坦": rng.normal(0, 0.1, 64).astype(np.float32),
        "随机": rng.normal(0, 2.0, 128).astype(np.float32),
    }
    combos = [
        (t, k, p)
        for t in (0.5, 1.0, 2.0)
        for k in (0, 5, 40)
        for p in (1.0, 0.9, 0.5)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        n_case = 0
        n_bad = 0
        for name, logits in cases.items():
            for (t, k, p) in combos:
                if k >= len(logits):
                    continue
                got = run_engine_probs(exe, logits, t, k, p, tmp)
                ref = hf_probs(logits, t, k, p)
                ok, msg = compare(got, ref)
                n_case += 1
                if not ok:
                    n_bad += 1
                    print(f"  失败  {name} T={t} k={k} p={p}: {msg}")
        check(n_bad == 0, f"{n_case} 组 (温度×top_k×top_p) 与 HF 逐元素一致")

        # ------------------------------------------------------ 2. 随机源性质
        print("== counter-based 随机源 ==")
        check(run_engine_rng(exe, 42, 0, 1) == run_engine_rng(exe, 42, 0, 1),
              "纯函数：同 seed 同位置 → 同一个 u")
        seq = run_engine_rng(exe, 7, 0, 3000)
        # u 是 float32，只有 24 位有效位 —— 3000 个样本里按生日问题算期望碰撞
        # 约 0.27 个，所以判据是"几乎全不同"，不是"零重复"。
        uniq = len(set(seq))
        check(uniq >= len(seq) - 5,
              f"3000 个 u 的唯一值 {uniq}（24 位精度下允许极少量碰撞）")
        check(all(0.0 <= u < 1.0 for u in seq), "u 全部落在 [0,1)")
        check(run_engine_rng(exe, 7, 0, 8) != run_engine_rng(exe, 8, 0, 8),
              "不同 seed 给出不同序列")
        hist = [0] * 10
        for u in seq:
            hist[min(9, int(u * 10))] += 1
        dev = max(abs(h - len(seq) / 10) for h in hist) / (len(seq) / 10)
        check(dev < 0.15,
              f"10 个等宽桶的计数偏差 {dev:.1%} < 15%（分布均匀）")

        # ------------------------------------------------------ 3. 图里那条回归
        print("== 同一个 u、不同温度 → 不同 token（README 那张图）==")
        x = np.array([4.0, 2.0, 0.0], dtype=np.float32)
        for T, expect in ((0.5, 0), (1.0, 1), (2.0, 2)):
            got = run_engine_sample(exe, x, T, 0, 1.0, 0.95, tmp)
            check(got == expect,
                  f"T={T}：u=0.95 → token {got}（期望 {expect}）")

        # ------------------------------------------------------ 4. 贪心极限
        print("== T=0 退化贪心 ==")
        for name, logits in cases.items():
            g = run_engine_sample(exe, logits, 0.0, 0, 1.0, 0.9999, tmp)
            check(g == int(np.argmax(logits)), f"{name}：T=0 → argmax（{g}）")

    print("=" * 72)
    if failed:
        print(f"通过 {passed} 项，失败 {len(failed)} 项：")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"通过 {passed} 项，失败 0 项。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
