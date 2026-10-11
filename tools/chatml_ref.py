#!/usr/bin/env python3
"""用 HF 官方 apply_chat_template 渲染一份 ChatML case（对拍参考侧）。

官方模板本身就是权威 —— 这里不做任何"重新实现"，直接把它交给 transformers
的渲染器跑一遍，输出就是我们要逐字节对齐的目标。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_tokenizer(tokenizer_json: str, tokenizer_config: str):
    from transformers import PreTrainedTokenizerFast

    with io.open(tokenizer_config, encoding="utf-8") as f:
        cfg = json.load(f)
    tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    tok.chat_template = cfg["chat_template"]
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-json", default=os.path.join(REPO, "models/qwen3-1.7b-tokenizer.json"))
    ap.add_argument("--tokenizer-config", default=os.path.join(REPO, "models/tokenizer_config.json"))
    args = ap.parse_args()

    with io.open(args.case, encoding="utf-8") as f:
        spec = json.load(f)

    tok = build_tokenizer(args.tokenizer_json, args.tokenizer_config)

    kw = {
        "tokenize": False,
        "add_generation_prompt": spec.get("add_generation_prompt", True),
    }
    # 只在 case 显式给了 enable_thinking 时才传 —— 官方模板用
    # `enable_thinking is defined and enable_thinking is false` 判断，
    # "不传" 与 "传 true" 虽然结果一样，但保持与引擎侧一致的输入更有意义。
    if "enable_thinking" in spec:
        kw["enable_thinking"] = spec["enable_thinking"]
    if spec.get("tools"):
        kw["tools"] = spec["tools"]

    text = tok.apply_chat_template(spec["messages"], **kw)
    with io.open(args.out, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
