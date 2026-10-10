#!/usr/bin/env python3
"""参考实现的可复用部分：从 GGUF 造一个官方 Qwen3 模型。

被两个脚本共用：
  * ref_forward.py      —— M1 主检查：C++ 前向 vs 参考前向
  * check_noise_floor.py —— 诊断：float32 的舍入噪声到底有多大

为什么权重必须来自**同一份 GGUF**：这样两边之间没有量化差异，残差只能来自
我们的实现。量化本身由 xcheck_dequant.py 单独负责。
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import Qwen3Config, Qwen3ForCausalLM

# `gguf.quants.dequantize` 返回的是只读 mmap 的视图；`torch.from_numpy` 会因此
# 警告"非可写数组"。我们只用 `load_state_dict(assign=True)` 把权重灌进去**读**，
# 从不回写，所以这个警告在这里是纯噪音 —— 而且不加这个过滤就得复制一份 6.8 GB
# 的 float32 权重（1.7B 模型），纯属浪费。
warnings.filterwarnings("ignore", message=".*NumPy array is not writable.*")

_exact_norm_patched = False


def use_exact_rmsnorm():
    """把 Qwen3RMSNorm 换成"完全按当前 dtype 计算"的版本。

    为什么需要：官方实现是

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        ...
        return self.weight * hidden_states.to(input_dtype)

    也就是**内部强制走 float32**。于是"把整个模型换成 float64"并不等于
    "float64 的真值" —— 每次归一化都会插一次 float32 舍入。

    做噪声底诊断时，我们要的是"数学上的真值"，所以把这个舍入拿掉。这是一个
    【只给诊断用】的补丁，默认不生效，正常对拍仍走官方实现。
    """
    global _exact_norm_patched
    if _exact_norm_patched:
        return
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

    Qwen3RMSNorm.forward = forward
    _exact_norm_patched = True

# GGUF 张量名 → HF 参数名
LAYER_SUFFIX = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}


def translate(name: str):
    """GGUF 张量名 → HF 参数名；认不出来返回 None。"""
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) != 3:
            return None
        hf = LAYER_SUFFIX.get(parts[2])
        return None if hf is None else f"model.layers.{parts[1]}.{hf}"
    return None


def meta(reader, key, default=None):
    try:
        field = reader.get_field(key)
    except KeyError:
        return default
    return default if field is None else field.contents()


def build_config(reader, vocab_size, tied, n_layers=None):
    # transformers v5 把 rope_theta 收进了 rope_parameters 字典。
    # rope_type 必须是 "default" —— Qwen3 的 GGUF 里没有 rope.scaling。
    rope_theta = float(meta(reader, "qwen3.rope.freq_base", 10000.0))
    layers = int(meta(reader, "qwen3.block_count"))
    if n_layers is not None and 0 < n_layers < layers:
        layers = n_layers
    cfg = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=int(meta(reader, "qwen3.embedding_length")),
        intermediate_size=int(meta(reader, "qwen3.feed_forward_length")),
        num_hidden_layers=layers,
        num_attention_heads=int(meta(reader, "qwen3.attention.head_count")),
        num_key_value_heads=int(meta(reader, "qwen3.attention.head_count_kv", 0)) or None,
        head_dim=int(meta(reader, "qwen3.attention.key_length")),
        max_position_embeddings=int(meta(reader, "qwen3.context_length", 4096)),
        rms_norm_eps=float(meta(reader, "qwen3.attention.layer_norm_rms_epsilon", 1e-6)),
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,  # Qwen3 全因果，无滑动窗口
        tie_word_embeddings=tied,
    )
    if cfg.num_key_value_heads is None:
        cfg.num_key_value_heads = cfg.num_attention_heads
    return cfg


def build_model(model_path, n_layers=None, dtype=torch.float32, norm_identity=False,
                verbose=False):
    """返回 (model, reader, cfg, vocab_size, tied)。

    n_layers     : 只建前 N 层（诊断用：把误差定位到深度）
    dtype        : torch.float32 / torch.float64
    norm_identity: 把 model.norm 换成恒等映射。这样 HF 的 hidden_states[N]
                   就是"第 N-1 层输出且没过 final norm"，与我们的
                   capture_hidden[N] 语义完全一致，可以直接逐位比。
    """
    reader = GGUFReader(model_path)
    by_name = {t.name: t for t in reader.tensors}
    vocab_size = int(by_name["token_embd.weight"].shape[1])  # GGUF dims = (hidden, vocab)
    tied = "output.weight" not in by_name
    cfg = build_config(reader, vocab_size, tied, n_layers=n_layers)

    # 只加载用到的层，省内存（float64 全量要 13.6 GB）
    keep_layers = set(range(cfg.num_hidden_layers))

    sd = {}
    unmapped = []
    for name, t in by_name.items():
        hf = translate(name)
        if hf is None:
            unmapped.append(name)
            continue
        if hf.startswith("model.layers."):
            li = int(hf.split(".")[2])
            if li not in keep_layers:
                continue
        w = dequantize(t.data, t.tensor_type)
        if dtype != torch.float32:
            w = w.astype({torch.float64: np.float64, torch.float16: np.float16}[dtype])
        else:
            w = np.ascontiguousarray(w, dtype=np.float32)
        sd[hf] = torch.from_numpy(w)

    model = Qwen3ForCausalLM(cfg)
    model.to(dtype).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if tied:
        extra = [k for k in missing if k != "lm_head.weight"]
        assert not extra, f"缺参数：{extra}"
        # assign=True 会打断 HF 的 tied 共享，显式接回去。
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        assert not missing, f"缺参数：{missing}"
    assert not unexpected, f"多余参数：{unexpected}"
    if unmapped:
        raise AssertionError(f"GGUF 里有未映射的张量：{unmapped}")
    if norm_identity:
        model.model.norm = torch.nn.Identity()

    if verbose:
        print(f"[参考] {meta(reader, 'general.name')}  layers={cfg.num_hidden_layers}"
              f" hidden={cfg.hidden_size} heads={cfg.num_attention_heads}"
              f"/{cfg.num_key_value_heads} head_dim={cfg.head_dim}"
              f" ffn={cfg.intermediate_size} vocab={vocab_size}"
              f" rope={cfg.rope_parameters} eps={cfg.rms_norm_eps:g}"
              f" tied={tied} dtype={dtype}")
    return model, reader, cfg, vocab_size, tied


def greedy_generate(model, tokens, max_new, stop_ids=None):
    """贪心解码 max_new 个 token，返回生成的 id 列表（不含输入）。

    刻意**不用** model.generate()：
      * generate 会挂上一串默认的 logits processor（重复惩罚、min_length、
        强制 EOS 的时机、多序列收尾……），那些行为随 transformers 版本改；
      * 我们要的是**最朴素的 argmax** —— 每一步只做"取最大"，与引擎的
        `GenerateGreedy` 是同一个东西。用 generate 对拍，等于把"引擎对不对"
        和"processor 配得对不对"混在一起，红了也不知道该怀疑哪边。

    自己走循环 + past_key_values 增量，语义只剩三行。
    stop_ids 命中时**计入**返回序列（与引擎的行为一致）。
    """
    stop = set(stop_ids or [])
    out = []
    cur = torch.tensor([list(tokens)], dtype=torch.long)
    past = None
    with torch.no_grad():
        for _ in range(max_new):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[0, -1]).item())
            out.append(nxt)
            if nxt in stop:
                break
            cur = torch.tensor([[nxt]], dtype=torch.long)
    return out


def run_forward(model, tokens, want_hidden=True, hidden_count=None):
    """跑一次前向。

    hidden_count 指定要导出的 hidden_states 个数（None = 全部）。
    注意 HF 的语义：hidden_states[k] 是**第 k 层的输入**，最后一项是过完
    final norm 的。要拿到"第 N-1 层输出"，请用 norm_identity=True 并取
    hidden_states[N]。
    """
    ids = torch.tensor([list(tokens)], dtype=torch.long)
    out = model(input_ids=ids, use_cache=False, output_hidden_states=want_hidden)
    logits = out.logits[0].to(torch.float32).numpy()
    if not want_hidden:
        return logits, None
    hs = out.hidden_states
    if hidden_count is not None:
        hs = hs[:hidden_count]
    hidden = np.stack([h[0].to(torch.float32).numpy() for h in hs])
    return logits, hidden
