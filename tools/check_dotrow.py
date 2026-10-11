#!/usr/bin/env python3
# 第 11 层：SIMD 融合点积 vs 标量参考（M5）
#
# 为什么需要这一层：M5 把 DotRowQuant 换成了 AVX2 内核，**累加顺序变了**，
# 于是"与官方逐 token 一致"这种端到端判据仍然能过（它只要求 token 相同），
# 但一旦内核写错（比如 Q6_K 的 qh 位段取错两位），可能要好几十层之后才以
# "输出变味"的形式暴露。所以要有**一层专门的、贴身的**判据。
#
# 判据怎么定（沿用 §14.2 的教训：先量噪声底，再定阈值）：
#   * 参考物是 float64 的 `a @ w` —— 精度远高于两边的 float32 实现；
#   * 归一化用 `Σ|a_i·w_i|`（逐项绝对值之和），**不是** |ref| ——
#     点积有抵消，ref 可能接近 0 而逐项之和很大，用 |ref| 归一化会把一个
#     正常的 float32 舍入差放大成"看起来 1000% 的误差"。
#   * SIMD 与标量的差别只应来自"求和顺序"，量级应是几个 float32 ULP。
#
# 不做"逐位相同"的断言 —— 那等于要求 SIMD 与标量用同一条累加顺序，与
# "用 4 个独立累加器提速"的目的直接矛盾。

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "build", "dotrow_check.exe")
MODELS = [
    "models/Qwen3-1.7B-Q4_K_M.gguf",
    "models/Qwen3-1.7B-Q8_0.gguf",
]

# 阈值。**先量噪声再定**（脚本会把实测打出来）：
#   k=6144 上实测 max|Δ|/Σ|a·w| —— SIMD vs 标量 2.5e-8 ~ 4.5e-8，
#   两边各自 vs float64 参考同等量级。这组差异只来自"求和顺序不同"
#   （4 路部分和 vs 标量 4 路部分和，分段边界不一样），是 float32 的固有噪声。
# 取 1e-6 = 噪声底的约 20 倍：换个模型/换组随机激活也留足余量，
# 而真正写错位段（比如 Q6_K 的 qh 少取两位）会偏到 1e-2 量级，一撞就红。
TOL_PAIR = 1e-6   # SIMD vs 标量
TOL_SELF = 1e-6   # 各自 vs float64 参考


def run_one(model, exe):
    """返回 (通过数, 总数, 失败数, kernel)。"""
    tmp = tempfile.mkdtemp(prefix="rpgbar_dotrow_")
    prefix = os.path.join(tmp, "d")
    print(f"模型：{os.path.relpath(model, ROOT)}")

    out = subprocess.run([exe, model, "32", "20261011", prefix],
                         capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    if out.returncode != 0:
        sys.stderr.write(out.stderr)
        print(f"  dotrow_check 退出码 {out.returncode}")
        return 0, 0, 1, "?"

    kernel = "scalar"
    for line in out.stdout.splitlines():
        if line.startswith("kernel="):
            kernel = line.split("=", 1)[1].strip()
    if kernel != "avx2":
        print(f"  注意：当前内核是 {kernel}，SIMD 未生效 —— "
              "SIMD vs 标量 这条判据退化为恒等（仍会跑，作回归用）。")

    failed = 0
    n_case = 0
    types = sorted({os.path.basename(p)[2:-len(".scalar.npy")]
                    for p in glob.glob(prefix + ".*.scalar.npy")})
    if not types:
        print("  失败：没有产出任何类型的 npy")
        return 0, 0, 1, kernel

    print()
    print(f"{'类型':<8}{'k':>6}  {'SIMDvs标量':>12}  {'SIMDvsref':>12}  "
          f"{'标量vsref':>12}  判据")
    for t in types:
        a = np.load(f"{prefix}.{t}.a.npy").astype(np.float64)
        w = np.load(f"{prefix}.{t}.w.npy").astype(np.float64)
        simd = np.load(f"{prefix}.{t}.simd.npy").astype(np.float64)
        scal = np.load(f"{prefix}.{t}.scalar.npy").astype(np.float64)

        ref = a @ w
        # 归一化用 Σ|a_i·w_i| = |a| @ |w|。
        # **不是** a @ |w| —— 那个和会被正负相消，可能接近 0，把正常的 float32
        # 舍入差放大成天文数字（第一版就是这么写错的，实测报出 1e24）。
        scale = np.maximum(np.abs(a) @ np.abs(w), 1e-30)

        e_pair = float(np.max(np.abs(simd - scal) / scale))
        e_simd = float(np.max(np.abs(simd - ref) / scale))
        e_scal = float(np.max(np.abs(scal - ref) / scale))

        ok = (e_pair <= TOL_PAIR and e_simd <= TOL_SELF
              and e_scal <= TOL_SELF and np.all(np.isfinite(simd)))
        n_case += 1
        if not ok:
            failed += 1
        print(f"{t:<8}{w.shape[0]:>6}  {e_pair:>12.3e}  {e_simd:>12.3e}  "
              f"{e_scal:>12.3e}  {'通过' if ok else '失败'}")
    print()
    return n_case - failed, n_case, failed, kernel


def main():
    if not os.path.exists(EXE):
        print("  跳过：缺 build/dotrow_check.exe（先跑 bash build.sh）")
        return 0
    models = [os.path.join(ROOT, m) for m in MODELS
              if os.path.exists(os.path.join(ROOT, m))]
    if not models:
        print("  跳过：models/ 下没有 Q4_K_M 或 Q8_0 权重")
        return 0

    total_ok = total = total_bad = 0
    kernels = set()
    for m in models:
        ok, n, bad, kern = run_one(m, EXE)
        total_ok += ok
        total += n
        total_bad += bad
        kernels.add(kern)
    print(f"合计 {total_ok}/{total} 个类型通过（内核 {'/'.join(sorted(kernels))}）")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
