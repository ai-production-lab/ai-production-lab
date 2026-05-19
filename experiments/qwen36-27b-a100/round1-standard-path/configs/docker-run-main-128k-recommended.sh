#!/usr/bin/env bash
# Recommended production profile (T7b): 128K + FP8 KV + prefix cache + MTP.
# Source: tuning analysis 2026-05-19, see results/ANALYSIS-optimal-config.md
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
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'

echo "Wait 3-8 min, then: curl -sS http://127.0.0.1:50600/v1/models | head -c 300"
