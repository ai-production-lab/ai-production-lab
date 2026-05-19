#!/usr/bin/env bash
# T1-A performance baseline (64K container, no rebuild).
set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:50600/v1/chat/completions}"
MODEL="${VLLM_MODEL_ID:-/models/Qwen3.6-27B}"
OUT_DIR="${TUNING_RESULTS_DIR:-$HOME/vllm-qwen36-tuning/results}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$OUT_DIR/t1a-${STAMP}.log"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "=== T1-A perf baseline ==="
echo "log: $LOG"
echo "api: $BASE_URL"

echo "--- A1 short request ---"
curl -sS "$BASE_URL" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}],\"max_tokens\":20,\"stream\":false}" \
  -w '\n总耗时:%{time_total}s\n'

echo "--- A2 512 token generation ---"
curl -sS "$BASE_URL" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"请写一篇500字的文章介绍人工智能的发展历史\"}],\"max_tokens\":512,\"stream\":false}" \
  -w '\n总耗时:%{time_total}s\n'

PAYLOAD="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"system\",\"content\":\"你是一个专业的AI助手，请认真回答用户的问题。\"},{\"role\":\"user\",\"content\":\"用一句话介绍人工智能\"}],\"max_tokens\":32,\"stream\":false}"

echo "--- A3 same system prompt #1 ---"
curl -sS "$BASE_URL" -H 'Content-Type: application/json' -d "$PAYLOAD" \
  -w '\n耗时:%{time_total}s\n' -o /dev/null

echo "--- A3 same system prompt #2 ---"
curl -sS "$BASE_URL" -H 'Content-Type: application/json' -d "$PAYLOAD" \
  -w '\n耗时:%{time_total}s\n' -o /dev/null

echo "--- A4 GPU memory ---"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

echo "=== done ==="
