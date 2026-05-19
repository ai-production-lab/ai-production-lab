#!/usr/bin/env bash
# Run one round-2 variant: rebuild 128K container + benchmark_r2.py
# Usage: bash run_round2_variant.sh R2-A1 [--b6]
set -euo pipefail

VARIANT="${1:?variant id required, e.g. R2-A0}"
RUN_B6="${2:-}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND2_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND2_ROOT}"
RESULTS="$ROOT/results/r2"
COMPLETED="$RESULTS/completed_variants.txt"
CONTAINER="vllm-qwen36-27b"
IMAGE="vllm/vllm-openai:v0.19.1"
MODEL_VOL="/data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro"
API_WAIT_SEC="${API_WAIT_SEC:-1200}"

case "$VARIANT" in
    R2-A0|R2-A2|R2-A4)
        if [[ -z "$RUN_B6" ]]; then
            RUN_B6="--b6"
        fi
        ;;
esac

mkdir -p "$RESULTS"
if grep -qx "$VARIANT" "$COMPLETED" 2>/dev/null; then
    if [[ "$RUN_B6" == "--b6" ]]; then
        latest_result="$(ls -t "$RESULTS/$VARIANT-"*.json 2>/dev/null | head -n 1 || true)"
        if [[ -n "$latest_result" ]] && grep -q '"b6_gen2048"' "$latest_result"; then
            echo "SKIP already completed with B6: $VARIANT"
            exit 0
        fi
        echo "RERUN completed variant without B6: $VARIANT"
    else
        echo "SKIP already completed: $VARIANT"
        exit 0
    fi
fi

# Defaults (T7b baseline)
MAX_LEN=131072
MAX_SEQS=2
GPU_UTIL=0.92
EXTRA_ARGS=()

case "$VARIANT" in
    R2-A0)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-A1)
        EXTRA_ARGS+=(--enable-prefix-caching
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-A2)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8)
        ;;
    R2-A3)
        EXTRA_ARGS+=(--enable-prefix-caching)
        ;;
    R2-A4)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}')
        ;;
    R2-A5)
        EXTRA_ARGS+=(--kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-B1)
        GPU_UTIL=0.88
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-B2)
        GPU_UTIL=0.94
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-B3)
        MAX_SEQS=1
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-B4)
        MAX_SEQS=3
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-B5)
        MAX_SEQS=4
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    R2-C1)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
            --enable-chunked-prefill)
        ;;
    R2-C2)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
            --max-num-batched-tokens 8192)
        ;;
    R2-C3)
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
            --max-num-batched-tokens 16384)
        ;;
    R2-D1)
        MAX_SEQS=1
        EXTRA_ARGS+=(--enable-prefix-caching)
        ;;
    R2-D2)
        EXTRA_ARGS+=(--enable-prefix-caching
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":4}')
        ;;
    R2-D3)
        GPU_UTIL=0.94
        EXTRA_ARGS+=(--enable-prefix-caching --kv-cache-dtype fp8
            --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}')
        ;;
    *)
        echo "Unknown variant: $VARIANT"
        exit 2
        ;;
esac

log() { echo "[$(date '+%F %T')] $*"; }

wait_api() {
    local waited=0
    while ((waited < API_WAIT_SEC)); do
        if ! docker ps --filter "name=$CONTAINER" --filter status=running -q | grep -q .; then
            log "Container not running"
            return 1
        fi
        if curl -sf http://127.0.0.1:50600/v1/models >/dev/null 2>&1; then
            log "API ready (${waited}s)"
            return 0
        fi
        sleep 15
        waited=$((waited + 15))
    done
    return 1
}

log "=== $VARIANT ==="
log "max_len=$MAX_LEN max_seqs=$MAX_SEQS gpu_util=$GPU_UTIL"
log "extra: ${EXTRA_ARGS[*]}"

docker rm -f "$CONTAINER" 2>/dev/null || true
if ! docker run -d --name "$CONTAINER" \
    --restart unless-stopped \
    --runtime nvidia --gpus all \
    --shm-size 16g --ipc host \
    -p 50600:8000 \
    -v "$MODEL_VOL" \
    "$IMAGE" \
    --model /models/Qwen3.6-27B \
    --tensor-parallel-size 2 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --language-model-only \
    --reasoning-parser qwen3 \
    "${EXTRA_ARGS[@]}"; then
    log "docker run failed"
    exit 1
fi

if ! wait_api; then
    docker logs --tail 50 "$CONTAINER" 2>&1 || true
    exit 1
fi

BENCH_ARGS=(python3 "$ROOT/scripts/benchmark_r2.py" --label "$VARIANT")
if [[ "$RUN_B6" == "--b6" ]]; then
    BENCH_ARGS+=(--run-b6)
fi
"${BENCH_ARGS[@]}"

echo "$VARIANT" >>"$COMPLETED"
log "DONE $VARIANT"
