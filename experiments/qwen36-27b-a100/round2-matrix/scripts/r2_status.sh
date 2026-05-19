#!/usr/bin/env bash
# Print round-2 execution status snapshot.
set -euo pipefail

export TZ="${TZ:-Asia/Shanghai}"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND2_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND2_ROOT}"
R2="$ROOT/results/r2"

echo "===== R2 STATUS $(date '+%F %T') ====="
echo "[tmux]"
tmux ls 2>/dev/null || echo "  (no tmux sessions)"
echo "[docker]"
docker ps --filter name=vllm-qwen36-27b --format '  {{.Names}} {{.Status}}' 2>/dev/null || true
echo "[gpu]"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || true
echo "[completed variants]"
if [[ -f "$R2/completed_variants.txt" ]]; then
    cat "$R2/completed_variants.txt" | sed 's/^/  /'
else
    echo "  (none)"
fi
echo "[latest results]"
ls -lt "$R2"/R2-*.json 2>/dev/null | head -5 | sed 's/^/  /' || echo "  (none)"
echo "[summary.csv]"
if [[ -f "$R2/summary.csv" ]]; then
    column -t -s, "$R2/summary.csv" 2>/dev/null | tail -n 15 | sed 's/^/  /' \
        || tail -n 15 "$R2/summary.csv" | sed 's/^/  /'
else
    echo "  (not generated yet)"
fi
echo "[matrix log tail]"
for f in "$R2"/matrix-*.log "$R2"/full-auto-*.log; do
    [[ -f "$f" ]] || continue
    echo "  --- $(basename "$f") ---"
    tail -n 8 "$f" | sed 's/^/    /'
done
echo "===== END ====="
