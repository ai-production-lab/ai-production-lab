#!/usr/bin/env bash
# 套系 A-1：部署性能快照（B1～B6，与调优文章/R2-A1 可比）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${1:-$ROOT/eval/config.env}"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

CHAT_URL="${VLLM_CHAT_URL:-http://127.0.0.1:50600/v1/chat/completions}"
MODEL="${VLLM_MODEL_ID:-/models/Qwen3.6-27B}"
LABEL="${BENCH_LABEL:-prod-local}-$(date +%Y%m%d)"
OUT="${BENCH_OUT_DIR:-$ROOT/eval/results/performance/deploy}"
SCRIPT="$ROOT/experiments/qwen36-27b-a100/round2-matrix/scripts/benchmark_r2.py"

mkdir -p "$OUT"

echo "[deploy-benchmark] label=$LABEL url=$CHAT_URL"
python3 "$SCRIPT" \
    --label "$LABEL" \
    --url "$CHAT_URL" \
    --model "$MODEL" \
    --out-dir "$OUT" \
    --run-b6

echo "[deploy-benchmark] done -> $OUT"
echo "[deploy-benchmark] compare with R2-A1: B2~85 tok/s, B3~6.96s, B4~52%"
