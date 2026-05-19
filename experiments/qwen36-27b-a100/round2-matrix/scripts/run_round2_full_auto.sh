#!/usr/bin/env bash
# Run full round-2 matrix (P0+P1+P2, D gated separately) with logging.
set -euo pipefail

export TZ="${TZ:-Asia/Shanghai}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND2_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND2_ROOT}"
LOG_DIR="$ROOT/results/r2"
STAMP="$(date '+%Y%m%d-%H%M%S')"
MAIN_LOG="$LOG_DIR/full-auto-${STAMP}.log"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$MAIN_LOG") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

log "=== Round-2 full auto start (P0+P1+P2) ==="
cd "$ROOT"

for scope in P0 P1 P2; do
    log "--- matrix scope: $scope ---"
    if ! bash "$ROOT/scripts/run_round2_matrix.sh" "$scope"; then
        log "WARN: scope $scope had failures (STOP_ON_FAIL=0 continues)"
    fi
    python3 "$ROOT/scripts/summarize_r2.py" || true
    log "summary after $scope:"
    tail -n 20 "$LOG_DIR/summary.csv" 2>/dev/null || true
done

log "=== Round-2 full auto finished ==="
log "Next: review $LOG_DIR/summary.csv then run: bash scripts/run_round2_matrix.sh D"
