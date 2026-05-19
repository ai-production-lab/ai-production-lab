#!/usr/bin/env bash
# Standard-path vLLM tuning orchestrator (A100). Run inside tmux.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND1_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND1_ROOT}"
RESULTS="${TUNING_RESULTS_DIR:-$ROOT/results}"
CONFIGS="$ROOT/configs"
STATE_FILE="$RESULTS/standard_path.state"
STAMP="$(date +%Y%m%d-%H%M%S)"
CONTAINER="vllm-qwen36-27b"
IMAGE="vllm/vllm-openai:v0.19.1"
MODEL_VOL="/data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro"
API_WAIT_SEC="${API_WAIT_SEC:-1200}"
CTX_SCRIPT="$ROOT/scripts/ctx_baseline_test.py"
T1A_SCRIPT="$ROOT/scripts/t1a_perf_baseline.sh"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"

mkdir -p "$RESULTS" "$CONFIGS"
touch "$STATE_FILE"
exec > >(tee -a "$RESULTS/standard_path-${STAMP}.log") 2>&1

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

step_done() {
    grep -qx "$1" "$STATE_FILE" 2>/dev/null
}

mark_done() {
    if ! step_done "$1"; then
        echo "$1" >>"$STATE_FILE"
    fi
    log "DONE: $1"
}

fail_step() {
    local step="$1"
    log "FAILED: $step"
    docker logs --tail 100 "$CONTAINER" 2>&1 || true
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
        exit 1
    fi
}

wait_api() {
    local waited=0
    while ((waited < API_WAIT_SEC)); do
        if ! docker ps --filter "name=$CONTAINER" --filter status=running -q | grep -q .; then
            log "Container exited (waited ${waited}s)"
            return 1
        fi
        if curl -sf "http://127.0.0.1:50600/v1/models" >/dev/null 2>&1; then
            log "API ready (${waited}s)"
            return 0
        fi
        sleep 15
        waited=$((waited + 15))
    done
    log "API timeout ${API_WAIT_SEC}s"
    return 1
}

docker_start() {
    local tag="$1"
    shift
    log "=== docker_start: $tag ==="
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "# $tag $STAMP extra: $*" >"$CONFIGS/run-${tag}-latest.meta"
    if ! docker run -d --name "$CONTAINER" \
        --restart unless-stopped \
        --runtime nvidia --gpus all \
        --shm-size 16g --ipc host \
        -p 50600:8000 \
        -v "$MODEL_VOL" \
        "$IMAGE" \
        --model /models/Qwen3.6-27B \
        --tensor-parallel-size 2 \
        --language-model-only \
        --reasoning-parser qwen3 \
        "$@"; then
        return 1
    fi
    if ! wait_api; then
        return 1
    fi
    curl -sS "http://127.0.0.1:50600/v1/models" | head -c 500 || true
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv || true
}

run_ctx() {
    local chars="$1"
    local label="$2"
    local timeout="${3:-900}"
    log "ctx: $label ($chars chars, timeout=${timeout}s)"
    python3 "$CTX_SCRIPT" "$chars" "$label" --timeout "$timeout"
}

do_step() {
    local id="$1"
    shift
    if step_done "$id"; then
        log "SKIP: $id"
        return 0
    fi
    log "BEGIN: $id"
    if "$@"; then
        mark_done "$id"
    else
        fail_step "$id"
    fi
}

# ---------- step implementations ----------

step_phase0() {
    python3 -c "import requests" 2>/dev/null || pip install --user requests
    docker ps --filter "name=$CONTAINER" --format '{{.Names}} {{.Status}}'
    test -x "$CTX_SCRIPT" && test -x "$T1A_SCRIPT"
}

step_t1a() {
    bash "$T1A_SCRIPT"
}

step_t1b_128k() {
    if docker_start "t1b-128k-bf16" \
        --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92; then
        :
    else
        log "128K BF16 failed, retry FP8"
        docker_start "t1b-128k-fp8" \
            --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
            --kv-cache-dtype fp8 || return 1
    fi
    run_ctx 60000 "128K-baseline" 900
}

step_t1b_200k() {
    docker_start "t1b-200k" \
        --max-model-len 204800 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --kv-cache-dtype fp8 || return 1
    run_ctx 100000 "200K-baseline" 1200
}

step_t1b_256k() {
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker run -d --name "$CONTAINER" \
        --restart unless-stopped \
        --runtime nvidia --gpus all \
        --shm-size 16g --ipc host \
        -p 50600:8000 \
        -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        -v "$MODEL_VOL" \
        "$IMAGE" \
        --model /models/Qwen3.6-27B \
        --tensor-parallel-size 2 \
        --max-model-len 262144 \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.95 \
        --language-model-only \
        --reasoning-parser qwen3 \
        --kv-cache-dtype fp8 || return 1
    wait_api || return 1
    run_ctx 130000 "256K-baseline" 1800
}

step_restore_64k() {
    docker_start "64k-restore" \
        --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.92
}

step_t3() {
    docker_start "t3-prefix" \
        --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching || return 1
    TUNING_RESULTS_DIR="$RESULTS" bash "$T1A_SCRIPT"
    docker logs --tail 30 "$CONTAINER" 2>&1 | grep -i "prefix cache" || true
}

step_t4() {
    docker_start "t4-mtp" \
        --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' || return 1
    curl -sS "http://127.0.0.1:50600/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"/models/Qwen3.6-27B","messages":[{"role":"user","content":"请写一篇500字的文章介绍人工智能的发展历史"}],"max_tokens":128,"stream":false}' \
        -w '\n总耗时:%{time_total}s\n' -o /dev/null
}

step_t6() {
    docker_start "t6-fp8-64k" \
        --max-model-len 65536 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
        --kv-cache-dtype fp8 || return 1
    curl -sS "http://127.0.0.1:50600/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"/models/Qwen3.6-27B","messages":[{"role":"user","content":"1+1等于几"}],"max_tokens":32,"stream":false}' \
        -w '\n总耗时:%{time_total}s\n'
}

step_t7() {
    if docker_start "t7-128k-prefix" \
        --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching; then
        :
    else
        docker_start "t7-128k-prefix-fp8" \
            --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
            --enable-prefix-caching --kv-cache-dtype fp8 || return 1
    fi
    run_ctx 60000 "128K-prefixcache" 900
}

step_t7b() {
    docker_start "t7b-main-128k" \
        --max-model-len 131072 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        --kv-cache-dtype fp8 \
        --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' || return 1
    curl -sS "http://127.0.0.1:50600/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"/models/Qwen3.6-27B","messages":[{"role":"user","content":"请写一篇500字的文章介绍人工智能的发展历史"}],"max_tokens":512,"stream":false}' \
        -w '\n总耗时:%{time_total}s\n' || true
    run_ctx 60000 "128K-main" 900
    echo "t7b-main-128k" >"$CONFIGS/active-profile.txt"
}

step_t8() {
    docker_start "t8-200k" \
        --max-model-len 204800 --max-num-seqs 2 --gpu-memory-utilization 0.92 \
        --enable-prefix-caching --kv-cache-dtype fp8 || return 1
    run_ctx 100000 "200K-prefixcache" 1200
}

step_t9() {
    docker rm -f "$CONTAINER" 2>/dev/null || true
    docker run -d --name "$CONTAINER" \
        --restart unless-stopped \
        --runtime nvidia --gpus all \
        --shm-size 16g --ipc host \
        -p 50600:8000 \
        -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        -v "$MODEL_VOL" \
        "$IMAGE" \
        --model /models/Qwen3.6-27B \
        --tensor-parallel-size 2 \
        --max-model-len 262144 \
        --max-num-seqs 1 \
        --gpu-memory-utilization 0.95 \
        --language-model-only \
        --reasoning-parser qwen3 \
        --enable-prefix-caching \
        --kv-cache-dtype fp8 || return 1
    wait_api || return 1
    run_ctx 130000 "256K-prefixcache" 1800
}

# ---------- main ----------

log "=== Standard path tuning start ==="
log "ROOT=$ROOT STATE=$STATE_FILE"

do_step "phase0" step_phase0
do_step "t1a" step_t1a
log "NOTE: T2 Cherry sampling is manual on Win11 (skipped)"
do_step "t1b-128k" step_t1b_128k
do_step "t1b-200k" step_t1b_200k
do_step "t1b-256k" step_t1b_256k
do_step "restore-64k" step_restore_64k
do_step "t3" step_t3
do_step "t4" step_t4
log "NOTE: T4 24h soak skipped in automation"
do_step "t6" step_t6
do_step "t7" step_t7
do_step "t7b" step_t7b
do_step "t8" step_t8
do_step "t9" step_t9

log "=== Standard path tuning complete ==="
log "Main config should be T7b (128K+FP8+prefix+MTP). Logs: $RESULTS/"
