#!/usr/bin/env bash
# Launch standard-path tuning in a detached tmux session (run on A100).
set -euo pipefail

SESSION="${VLLM_TUNING_SESSION:-vllm-tune}"
ROOT="${VLLM_TUNING_ROOT:-$HOME/vllm-qwen36-tuning}"
RUNNER="$ROOT/scripts/run_standard_path.sh"
LOG="$ROOT/results/standard_path-runner.log"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not installed. Run directly: bash $RUNNER"
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session $SESSION already exists. Attach: tmux attach -t $SESSION"
    exit 1
fi

mkdir -p "$ROOT/results"
tmux new-session -d -s "$SESSION" \
    "bash -lc 'cd $ROOT && exec bash $RUNNER 2>&1 | tee -a $LOG'"

echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    tail -f $LOG"
echo "  state:  cat $ROOT/results/standard_path.state"
