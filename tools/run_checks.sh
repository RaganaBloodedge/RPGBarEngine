#!/usr/bin/env bash
# run_checks.sh —— 一条命令跑完引擎当前的**全部**验证。
#
# 分三层，对应 DESIGN.md §11 的"三层对拍"思路：
#   1. 进程内自检（不需要模型）      —— 地基改动后 1 秒内发现问题
#   2. 量化布局校验（只需文件头）    —— 尺寸表的独立验证
#   3. 分词器对拍（需要 Python 参考）—— 逐 id 与官方一致
#   4. 跨量化档位交叉校验（需要权重）—— 语义层面的独立验证
#
# 缺什么就跳什么并明确说明，不静默跳过。
set -u
# 脚本在 tools/ 下，但所有路径都是相对项目根的。
cd "$(dirname "$0")/.."

PY="${PY:-python}"
GGUF_Q8="models/Qwen3-1.7B-Q8_0.gguf"
GGUF_Q4="models/Qwen3-1.7B-Q4_K_M.gguf"
TOKREF="models/qwen3-1.7b-tokenizer.json"

fail=0

echo "=============================================================="
echo "[1/4] 进程内自检"
echo "=============================================================="
if [ -x build/selftest.exe ] || [ -x build/selftest ]; then
  ./build/selftest.exe 2>/dev/null || ./build/selftest || fail=1
else
  echo "  跳过：先跑 bash build.sh"
  fail=1
fi

echo
echo "=============================================================="
echo "[2/4] 量化类型布局（用 GGUF 写入器的 offset 独立验证块尺寸）"
echo "=============================================================="
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

echo
echo "=============================================================="
echo "[3/4] 分词器对拍（HF tokenizers 为权威参考）"
echo "=============================================================="
if [ -f "$GGUF_Q8" ] && [ -f "$TOKREF" ]; then
  "$PY" tools/tok_ref.py "$GGUF_Q8" "$TOKREF" --fuzz 3000 --dump-n 3 || fail=1
else
  echo "  跳过：缺权重或参考分词器定义"
  echo "  （python scripts/fetch_model.py --preset qwen3-1.7b-q8 会一并下参考物）"
fi

echo
echo "=============================================================="
echo "[4/4] 跨量化档位交叉校验（Q8_0 vs Q4_K_M 的同一张量应高度相关）"
echo "=============================================================="
if [ -f "$GGUF_Q8" ] && [ -f "$GGUF_Q4" ]; then
  "$PY" tools/xcheck_quant.py "$GGUF_Q8" "$GGUF_Q4" || fail=1
else
  echo "  跳过：需要同时有 Q8_0 与 Q4_K_M 两份权重"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "全部检查通过。"
else
  echo "有检查未通过或未执行（见上）。"
fi
exit "$fail"
