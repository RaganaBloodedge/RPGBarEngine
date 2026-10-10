#!/usr/bin/env python3
"""PyTorch 参考前向 —— M1 对拍的另一半。

三件事刻意都交给"官方实现"，只留一处是我们的：

| 环节 | 用谁 | 为什么 |
| --- | --- | --- |
| 反量化 | `gguf.quants.dequantize` | llama.cpp 官方算法的 Python 移植，与 `quant.cc` 是两份独立实现 |
| 前向   | `transformers.Qwen3ForCausalLM` | 官方实现 |
| 权重   | 同一份 GGUF | 两边之间**没有量化差异** |

所以残差只可能来自我们的实现：加载/形状、内存布局、RoPE、QK-norm、GQA 分组、
SwiGLU、tied 输出头、数值精度。这正是 M1 要的归因。

隐藏状态的对齐：默认模式下 HF 的 `hidden_states[k]` 是**第 k 层的输入**
（k=0 是 embedding 输出），最后一项是过完 final norm 的；所以只导出前
n_layers 项，正好与我们的 `capture_hidden[k]`（= 第 k 层的输入）逐索引直比。
`--norm-identity` 打开后改为导出 n_layers+1 项，多出的最后一项是"第 n_layers-1
层输出且没过 final norm"，与我们的 `capture_hidden[n_layers]` 对齐 ——
「噪声底」诊断要用这个模式。

用法：
  ref_forward.py <model.gguf> <out_prefix> --tokens-from <prefix>.tokens.txt
  ref_forward.py <model.gguf> <out_prefix> --tokens 785,6722,315,9625,374
  ref_forward.py <model.gguf> <prefix> --tokens-from <p>.tokens.txt \\
      --layers 8 --dtype float64 --norm-identity
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# 与 ref_forward.py 同目录，允许从任意 cwd 调用。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen3_ref  # noqa: E402

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out_prefix")
    ap.add_argument("--tokens-from", help="读 forward_check 写的 .tokens.txt")
    ap.add_argument("--tokens", help="逗号分隔的 token id")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="只用前 N 层（0=全部）")
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--norm-identity", action="store_true",
                    help="把 final norm 换成恒等映射（配合 --layers 做噪声底诊断）")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.tokens_from:
        raw = open(args.tokens_from, "r", encoding="ascii").read().strip()
        tokens = [int(x) for x in raw.replace(",", " ").split()]
    elif args.tokens:
        tokens = [int(x) for x in args.tokens.replace(",", " ").split()]
    else:
        ap.error("要 --tokens-from 或 --tokens")
    assert tokens, "token 序列为空"

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    t0 = time.time()
    model, reader, cfg, vocab, tied = qwen3_ref.build_model(
        args.model,
        n_layers=args.layers or None,
        dtype=DTYPES[args.dtype],
        norm_identity=args.norm_identity,
        verbose=True,
    )
    print(f"[参考] 权重就绪 {time.time() - t0:.1f}s")

    t0 = time.time()
    # 默认模式只导前 n_layers 项：HF 的 hidden_states[k] 是"第 k 层的输入"，
    # 正好与我们的 capture_hidden[k]（= 第 k 层的输入）逐索引对齐。
    # norm_identity 模式下多导一项，多出的最后一项是"最后一层的输出"。
    hidden_count = None if args.norm_identity else cfg.num_hidden_layers
    logits, hidden = qwen3_ref.run_forward(model, tokens, hidden_count=hidden_count)
    print(f"[参考] 前向 {len(tokens)} 个位置，{time.time() - t0:.1f}s")

    assert logits.shape == (len(tokens), vocab), logits.shape
    print(f"[参考] 导出 hidden_states {hidden.shape[0]} 项，形状 {hidden.shape}")
    if not args.norm_identity:
        assert hidden.shape[0] == cfg.num_hidden_layers, hidden.shape
    else:
        assert hidden.shape[0] == cfg.num_hidden_layers + 1, hidden.shape

    if not args.no_save:
        np.save(args.out_prefix + ".ref.logits.npy", logits)
        np.save(args.out_prefix + ".ref.hidden.npy", hidden)
        print(f"[输出] {args.out_prefix}.ref.logits.npy  {logits.shape}")
        print(f"[输出] {args.out_prefix}.ref.hidden.npy  {hidden.shape}")

    last = logits[-1].astype(np.float64)
    top = np.argsort(-last)[:5]
    print("[参考] 最后位置 top-5：" + ", ".join(f"id={int(i)} logit={last[i]:.4f}" for i in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
