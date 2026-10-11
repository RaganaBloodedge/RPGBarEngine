#!/usr/bin/env bash
# run_checks.sh —— 一条命令跑完引擎当前的**全部**验证。
#
# 分层，对应 DESIGN.md §11 的"分层对拍"思路：越靠前的层越便宜、越能快速定位。
#   1. 进程内自检（不需要模型）              —— 地基改动后 1 秒内发现问题
#   2. 量化布局校验（只需文件头）            —— 尺寸表的独立验证
#   3. 量化逐位对拍（vs gguf.quants）        —— "同一份量化字节解出同一个数"
#   4. 分词器对拍（需要 Python 参考）        —— 逐 id 与官方一致
#   5. 跨量化档位交叉校验                    —— 语义层面的独立验证
#   6. 前向对拍（M1，需要 torch/transformers）—— 与官方实现逐层比
#   7. 生成对拍（M2，需要 torch/transformers）—— 贪心解码逐 token 一致
#   8. ChatML 渲染对拍（M3，需要 transformers+jinja2）—— 逐【字节】与官方模板一致
#
# 缺什么就跳什么并明确说明，不静默跳过。
#
# 环境变量：
#   PY=...              指定 Python（要带 torch/transformers/gguf 才能跑第 6/7 层）
#   RPG_SKIP_FORWARD=1  跳过第 6 层（前向对拍要跑 2 分钟以上）
#   RPG_SKIP_GENERATE=1 跳过第 7 层（生成对拍比第 6 层更慢）
set -u
# 脚本在 tools/ 下，但所有路径都是相对项目根的。
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GGUF_Q8="models/Qwen3-1.7B-Q8_0.gguf"
GGUF_Q4="models/Qwen3-1.7B-Q4_K_M.gguf"
TOKREF="models/qwen3-1.7b-tokenizer.json"

fail=0
step() { echo; echo "=============================================================="
         echo "[$1] $2"; echo "=============================================================="; }

step "1/8" "进程内自检"
if [ -x build/selftest.exe ] || [ -x build/selftest ]; then
  ./build/selftest.exe 2>/dev/null || ./build/selftest || fail=1
else
  echo "  跳过：先跑 bash build.sh"
  fail=1
fi

step "2/8" "量化类型布局（用 GGUF 写入器的 offset 独立验证块尺寸）"
if [ -f "$GGUF_Q8" ]; then
  "$PY" tools/check_quant_layout.py "$GGUF_Q8" || fail=1
else
  echo "  跳过：缺 $GGUF_Q8（python scripts/fetch_model.py --preset qwen3-1.7b-q8）"
fi
if [ -f "$GGUF_Q4" ]; then
  "$PY" tools/check_quant_layout.py "$GGUF_Q4" || fail=1
else
  echo "  跳过：缺 $GGUF_Q4（python scripts/fetch_model.py --preset qwen3-1.7b-q4km）"
fi

step "3/8" "量化逐位对拍（我们的 quant.cc vs gguf.quants / llama.cpp 官方算法）"
if [ -f "$GGUF_Q8" ]; then
  "$PY" tools/xcheck_dequant.py "$GGUF_Q8" || fail=1
else
  echo "  跳过：缺 $GGUF_Q8"
fi

step "4/8" "分词器对拍（HF tokenizers 为权威参考）"
if [ -f "$GGUF_Q8" ] && [ -f "$TOKREF" ]; then
  "$PY" tools/tok_ref.py "$GGUF_Q8" "$TOKREF" --fuzz 3000 --dump-n 3 || fail=1
else
  echo "  跳过：缺权重或参考分词器定义"
  echo "  （python scripts/fetch_model.py --preset qwen3-1.7b-q8 会一并下参考物）"
fi

step "5/8" "跨量化档位交叉校验（Q8_0 vs Q4_K_M 的同一张量应高度相关）"
if [ -f "$GGUF_Q8" ] && [ -f "$GGUF_Q4" ]; then
  "$PY" tools/xcheck_quant.py "$GGUF_Q8" "$GGUF_Q4" || fail=1
else
  echo "  跳过：需要同时有 Q8_0 与 Q4_K_M 两份权重"
fi

step "6/8" "前向对拍（M1：C++ vs PyTorch 官方实现，同一份 GGUF）"
if [ "${RPG_SKIP_FORWARD:-0}" = "1" ]; then
  echo "  按 RPG_SKIP_FORWARD=1 跳过"
elif ! "$PY" -c "import torch, transformers, gguf" 2>/dev/null; then
  echo "  跳过：当前 Python 缺 torch / transformers / gguf"
  echo "  （pip install torch transformers gguf safetensors）"
elif [ -f "$GGUF_Q8" ] && [ -x build/forward_check.exe ]; then
  "$PY" tools/check_forward.py || fail=1
else
  echo "  跳过：缺权重或 build/forward_check.exe"
fi

step "7/8" "生成对拍（M2：贪心解码逐 token vs PyTorch 官方实现）"
if [ "${RPG_SKIP_GENERATE:-0}" = "1" ]; then
  echo "  按 RPG_SKIP_GENERATE=1 跳过"
elif ! "$PY" -c "import torch, transformers, gguf" 2>/dev/null; then
  echo "  跳过：当前 Python 缺 torch / transformers / gguf"
elif [ -f "$GGUF_Q8" ] && [ -x build/gen_check.exe ]; then
  # 第 7 层顺带跑 KV Cache 的"带 cache 与不带 cache 逐位相同"自检 ——
  # 那条断言能一次性证明 cache 的写入/读取没把布局搞错。
  "$PY" tools/check_generate.py || fail=1
else
  echo "  跳过：缺权重或 build/gen_check.exe"
fi

step "8/8" "ChatML 渲染对拍（M3：逐字节 vs HF 官方 apply_chat_template）"
if ! "$PY" -c "import transformers, jinja2" 2>/dev/null; then
  echo "  跳过：当前 Python 缺 transformers / jinja2"
  echo "  （pip install transformers jinja2）"
elif [ -x build/chatml_check.exe ] && [ -f "$TOKREF" ] && [ -f models/tokenizer_config.json ]; then
  # 第 8 层要的是官方 chat_template 原文（tokenizer_config.json 里的字段），
  # 不是"我们对模板的理解" —— 所以参考侧必须用官方模板本身。
  "$PY" tools/check_chatml.py || fail=1
else
  echo "  跳过：缺 build/chatml_check.exe 或参考物"
  echo "  （tokenizer_config.json 在 scripts/fetch_model.py 的 REFS 里）"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "全部检查通过。"
else
  echo "有检查未通过或未执行（见上）。"
fi
exit "$fail"
