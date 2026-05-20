#!/usr/bin/env bash
# 套系 A-2：通用 serving 压测（TTFT / TPOT / 吞吐，可与 vLLM bench 对照）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${1:-$ROOT/eval/config.env}"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

CHAT_URL="${VLLM_CHAT_URL:-http://127.0.0.1:50600/v1/chat/completions}"
MODEL="${VLLM_MODEL_ID:-/models/Qwen3.6-27B}"
TOKENIZER="${TOKENIZER_PATH:-Qwen/Qwen3.6-27B}"
OUT="$ROOT/eval/results/performance/evalscope"

mkdir -p "$OUT"
cd "$OUT"

echo "[evalscope-perf] url=$CHAT_URL parallel=1 number=20 max_tokens=512"

evalscope perf \
    --parallel 1 \
    --number 20 \
    --model "$MODEL" \
    --url "$CHAT_URL" \
    --api openai \
    --dataset random \
    --min-prompt-length 128 \
    --max-prompt-length 128 \
    --min-tokens 512 \
    --max-tokens 512 \
    --prefix-length 0 \
    --tokenizer-path "$TOKENIZER" \
    --extra-args '{"temperature": 1.0, "top_p": 0.95}'

echo "[evalscope-perf] done -> $OUT (see outputs/ subdir)"
