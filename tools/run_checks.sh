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
#   9. C ABI（M4，只需要 build/ 与权重）       —— 宿主视角：句柄生命周期 + 前缀复用
#  10. 采样对拍（M2.5，需要 torch/transformers）—— 筛选结果 vs HF warpers；随机源代数性质
#  11. 融合点积对拍（M5，需要权重 + numpy）—— SIMD 内核 vs 标量参考 vs float64
#
# 缺什么就跳什么并明确说明，不静默跳过。
#
# 环境变量：
#   PY=...              指定 Python（要带 torch/transformers/gguf 才能跑第 6/7/8/10 层）
#   RPG_SKIP_FORWARD=1  跳过第 6 层（前向对拍要跑 2 分钟以上）
#   RPG_SKIP_GENERATE=1 跳过第 7 层（生成对拍比第 6 层更慢）
#   RPG_SKIP_ABI=1      跳过第 9 层（要加载完整权重并真的生成几次，约 1 分钟）
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

step "1/11" "进程内自检"
if [ -x build/selftest.exe ] || [ -x build/selftest ]; then
  ./build/selftest.exe 2>/dev/null || ./build/selftest || fail=1
else
  echo "  跳过：先跑 bash build.sh"
  fail=1
fi

step "2/11" "量化类型布局（用 GGUF 写入器的 offset 独立验证块尺寸）"
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

step "3/11" "量化逐位对拍（我们的 quant.cc vs gguf.quants / llama.cpp 官方算法）"
if [ -f "$GGUF_Q8" ]; then
  "$PY" tools/xcheck_dequant.py "$GGUF_Q8" || fail=1
else
  echo "  跳过：缺 $GGUF_Q8"
fi

step "4/11" "分词器对拍（HF tokenizers 为权威参考）"
if [ -f "$GGUF_Q8" ] && [ -f "$TOKREF" ]; then
  "$PY" tools/tok_ref.py "$GGUF_Q8" "$TOKREF" --fuzz 3000 --dump-n 3 || fail=1
else
  echo "  跳过：缺权重或参考分词器定义"
  echo "  （python scripts/fetch_model.py --preset qwen3-1.7b-q8 会一并下参考物）"
fi

step "5/11" "跨量化档位交叉校验（Q8_0 vs Q4_K_M 的同一张量应高度相关）"
if [ -f "$GGUF_Q8" ] && [ -f "$GGUF_Q4" ]; then
  "$PY" tools/xcheck_quant.py "$GGUF_Q8" "$GGUF_Q4" || fail=1
else
  echo "  跳过：需要同时有 Q8_0 与 Q4_K_M 两份权重"
fi

step "6/11" "前向对拍（M1：C++ vs PyTorch 官方实现，同一份 GGUF）"
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

step "7/11" "生成对拍（M2：贪心解码逐 token vs PyTorch 官方实现）"
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

step "8/11" "ChatML 渲染对拍（M3：逐字节 vs HF 官方 apply_chat_template）"
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

step "9/11" "C ABI（M4：宿主视角走 ctypes，验句柄生命周期与缓存前缀复用）"
if [ "${RPG_SKIP_ABI:-0}" = "1" ]; then
  echo "  按 RPG_SKIP_ABI=1 跳过"
elif [ ! -f build/rpg_engine.dll ] && [ ! -f build/rpg_engine.so ]; then
  echo "  跳过：缺 build/rpg_engine.dll（先跑 bash build.sh）"
elif [ -f "$GGUF_Q8" ] && [ -x build/gen_check.exe ]; then
  # 第 9 层与前几层的对象不同：它验的是"C 这一层"——宿主拿 dll 怎么用。
  # 它内部会拿工具链路径（chatml_check/gen_check/tok_check/parse_check）的结果
  # 做参照，所以两条路的产出必须一致。
  "$PY" tools/check_abi.py --max-tokens 32 || fail=1
else
  echo "  跳过：缺权重或 build/gen_check.exe"
fi

step "10/11" "采样对拍（M2.5：确定性筛选 vs HF warpers + counter-based 随机源）"
if ! "$PY" -c "import torch, transformers, numpy" 2>/dev/null; then
  echo "  跳过：当前 Python 缺 torch / transformers / numpy"
  echo "  （pip install torch transformers numpy）"
elif [ -x build/sample_check.exe ]; then
  # 第 10 层**不需要权重** —— 它验的是采样器本身：确定性筛选直接对 HF 的
  # Temperature/TopK/TopP warpers；随机源验其代数性质（纯函数 / 均匀性）。
  "$PY" tools/check_sample.py || fail=1
else
  echo "  跳过：缺 build/sample_check.exe（先跑 bash build.sh）"
fi

step "11/11" "融合点积对拍（M5：AVX2 内核 vs 标量参考 vs float64）"
if ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "  跳过：当前 Python 缺 numpy"
elif [ -x build/dotrow_check.exe ]; then
  # 第 11 层要权重（从真实张量里取一行），但**不要 torch**。
  # 判据：|SIMD - 标量| / Σ|a·w| ≤ 1e-6（实测噪声底 ~5e-8）。
  "$PY" tools/check_dotrow.py || fail=1
else
  echo "  跳过：缺 build/dotrow_check.exe（先跑 bash build.sh）"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "全部检查通过。"
else
  echo "有检查未通过或未执行（见上）。"
fi
exit "$fail"
