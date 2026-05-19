#!/usr/bin/env bash
# Run round-2 variant matrix. Usage: bash run_round2_matrix.sh [P0|P1|P2|ALL|D]
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROUND2_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"
ROOT="${VLLM_TUNING_ROOT:-$_ROUND2_ROOT}"
RUNNER="$ROOT/scripts/run_round2_variant.sh"
SCOPE="${1:-P0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
export STOP_ON_FAIL

run_list() {
    local id
    for id in "$@"; do
        echo "======== $id ========"
        local extra_args=()
        case "$id" in
            R2-A0|R2-A2|R2-A4)
                extra_args+=(--b6)
                ;;
        esac
        if ! bash "$RUNNER" "$id" "${extra_args[@]}"; then
            echo "FAILED: $id"
            if [[ "$STOP_ON_FAIL" == "1" ]]; then
                exit 1
            fi
        fi
    done
}

P0_IDS=(R2-A0 R2-A1 R2-A2 R2-A3)
P1_IDS=(R2-A4 R2-A5 R2-B1 R2-B2 R2-B3)
P2_IDS=(R2-B4 R2-B5 R2-C1 R2-C2 R2-C3)
D_IDS=(R2-D1 R2-D2 R2-D3)

case "$SCOPE" in
    P0) run_list "${P0_IDS[@]}" ;;
    P1) run_list "${P1_IDS[@]}" ;;
    P2) run_list "${P2_IDS[@]}" ;;
    D) run_list "${D_IDS[@]}" ;;
    ALL)
        run_list "${P0_IDS[@]}" "${P1_IDS[@]}" "${P2_IDS[@]}"
        ;;
    *)
        echo "Usage: $0 [P0|P1|P2|D|ALL]"
        exit 2
        ;;
esac

python3 "$ROOT/scripts/summarize_r2.py" || true
echo "Matrix $SCOPE finished."
if [[ "$SCOPE" == "ALL" ]]; then
    echo "D variants are gated; inspect summary.csv before running: bash $0 D"
fi
