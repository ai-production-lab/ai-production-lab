#!/usr/bin/env bash
# 套系 B：能力评测（GSM8K + C-Eval，OpenAI API 打已部署 vLLM）
# 用法：bash run_evalscope_capability.sh [quick|standard]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${2:-$ROOT/eval/config.env}"
MODE="${1:-quick}"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:50600/v1}"
MODEL="${VLLM_MODEL_ID:-/models/Qwen3.6-27B}"

case "$MODE" in
    quick)
        LIMIT="${CAP_EVAL_LIMIT_QUICK:-20}"
        ;;
    standard)
        LIMIT="${CAP_EVAL_LIMIT_STANDARD:-100}"
        ;;
    *)
        echo "usage: $0 [quick|standard] [config.env]"
        exit 1
        ;;
esac

OUT="$ROOT/eval/results/capability/evalscope"
mkdir -p "$OUT"
cd "$OUT"

echo "[evalscope-capability] mode=$MODE limit=$LIMIT api=$BASE_URL"

# gsm8k：推理；ceval：中文综合（EvalScope 数据集名以版本为准）
evalscope eval \
    --model "$MODEL" \
    --api-url "$BASE_URL" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k ceval \
    --limit "$LIMIT" \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 2048}'

echo "[evalscope-capability] done -> $OUT"
echo "[evalscope-capability] 完整榜请增大 limit 或去掉 --limit；耗时与 GPU 独占相关"
