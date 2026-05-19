#!/usr/bin/env bash
# Round-2 recommended production profile (R2-A1 family).
# 128K + BF16 KV + prefix cache + MTP(num=2).
# Based on results/r2/summary.csv: best B2/B3 vs T7b while keeping official MTP=2.
set -euo pipefail

docker rm -f vllm-qwen36-27b 2>/dev/null || true

docker run -d --name vllm-qwen36-27b \
  --restart unless-stopped \
  --runtime nvidia --gpus all \
  --shm-size 16g --ipc host \
  -p 50600:8000 \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  vllm/vllm-openai:v0.19.1 \
  --model /models/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

echo "R2-A1 profile started. Wait 3-8 min, then:"
echo "  curl -sS http://127.0.0.1:50600/v1/models | head -c 300"
