#!/usr/bin/env bash
# Start round-2 matrix in tmux. Usage: bash start_round2_tmux.sh [P0|P1|ALL]
set -euo pipefail

SESSION="${VLLM_R2_SESSION:-vllm-r2}"
SCOPE="${1:-P0}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND2_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND2_ROOT}"
LOG="$ROOT/results/r2/matrix-${SCOPE}.log"

mkdir -p "$ROOT/results/r2"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION exists. attach: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    "bash -lc 'cd $ROOT && bash scripts/run_round2_matrix.sh $SCOPE 2>&1 | tee -a $LOG'"

echo "Started tmux: $SESSION scope=$SCOPE"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
