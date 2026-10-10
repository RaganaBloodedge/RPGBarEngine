#!/usr/bin/env python3
"""逐位对拍反量化：我们的 quant.cc vs gguf.quants（llama.cpp 官方算法的 Python 移植）。

## 为什么要单独做这件事

M1 对拍时发现"我们离 float64 真值比官方实现远几倍"。可能的来源有两类：

  (a) 求和顺序 —— 纯舍入，换不了根本
  (b) **反量化出的权重值本身就不一样** —— 每个权重差 1 ULP，点积后就系统性偏移

embedding 查表逐位相同（Δ=0）**不能**排除 (b)：取行不涉及算术。要有算术才能
暴露 1 ULP 的差异。所以这里把整张张量反量化出来逐位比。

这也是分层验证的延续：先把"同一份量化字节解出同一个数"钉死，上层的数值差异
才谈得上归因。

用法：
  xcheck_dequant.py <model.gguf> [--tensors 名字1,名字2] [--count 3]
                    [--forward-check build/forward_check.exe] [--dump build/dump_gguf.exe]
"""

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    from gguf import GGUFReader
    from gguf.quants import dequantize
except ImportError:
    print("需要 gguf 包：pip install gguf", file=sys.stderr)
    raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tensors", help="逗号分隔；默认自动挑代表性的几张")
    ap.add_argument("--count", type=int, default=4, help="自动挑选时取几张")
    ap.add_argument("--dump", default="build/dump_gguf.exe")
    args = ap.parse_args()

    reader = GGUFReader(args.model)
    by_name = {t.name: t for t in reader.tensors}

    if args.tensors:
        names = args.tensors.split(",")
    else:
        # 自动挑：优先覆盖不同量化类型 + 不同形状（1 维 norm / 2 维矩阵）
        want = [
            "blk.0.attn_norm.weight",   # 1 维，F32
            "blk.0.attn_q.weight",      # 2 维，与 Q 头相关
            "blk.0.ffn_gate.weight",    # 2 维，最宽的 ffn
            "blk.0.ffn_down.weight",    # 2 维，[hidden, intermediate]
            "token_embd.weight",        # 最大的一张
            "blk.13.attn_v.weight",
        ]
        names = [n for n in want if n in by_name][: args.count]

    print(f"对拍 {len(names)} 张张量：")
    all_ok = True
    types_seen = set()

    with tempfile.TemporaryDirectory() as td:
        for name in names:
            t = by_name[name]
            types_seen.add(int(t.tensor_type))

            npy_path = os.path.join(td, "ours.npy")
            subprocess.run([args.dump, args.model, "--dump-npy", name, npy_path],
                           check=True, stdout=subprocess.DEVNULL)

            ours = np.load(npy_path)
            theirs = dequantize(t.data, t.tensor_type)
            theirs = np.ascontiguousarray(theirs, dtype=np.float32)

            if ours.shape != theirs.shape:
                # 再看一眼是不是转置关系，给出更准确的报错
                hint = "（疑似转置）" if ours.T.shape == theirs.shape else ""
                print(f"  {name:<26} 形状不符 {ours.shape} vs {theirs.shape}{hint}")
                all_ok = False
                continue

            same_bits = np.array_equal(ours.view(np.uint32), theirs.view(np.uint32))
            if same_bits:
                print(f"  {name:<26} {str(ours.shape):<16} 逐位相同 ({ours.size} 个元素)")
                continue

            d = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
            ulp = np.spacing(np.abs(theirs).astype(np.float32)).astype(np.float64)
            ulp[ulp == 0] = np.finfo(np.float32).tiny
            n_ulp = d / ulp
            n_diff = int((d > 0).sum())
            print(f"  {name:<26} {str(ours.shape):<16} **不同**：{n_diff}/{ours.size} 个元素，"
                  f"max|Δ|={d.max():.4g}，max ULP={n_ulp.max():.4g}，中位 ULP={np.median(n_ulp[d > 0]) if n_diff else 0:.3g}")
            all_ok = False

    print()
    if all_ok:
        print(f"PASS —— 全部张量与 gguf.quants 逐位相同（覆盖量化类型 id {sorted(types_seen)}）")
        print("        于是上层前向的数值差异不可能来自反量化。")
        return 0
    print("FAIL —— 反量化与官方实现不一致，先修这里再看上层。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
