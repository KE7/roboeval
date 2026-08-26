#!/usr/bin/env bash
# run_sharded_v2.sh — convenience launcher for sharded roboeval runs.
#
# Usage:
#   bash scripts/run_sharded_v2.sh --config <yaml> --num-shards <N> [--output-dir <dir>]
#
# This script:
#   1. Backgrounds N shard processes (roboeval run --shard-id i --num-shards N).
#   2. Waits for all shards to finish.
#   3. Calls roboeval merge to combine shard results into final.json.
#
# Example:
#   bash scripts/run_sharded_v2.sh \
#       --config configs/libero_spatial_pi05_smoke.yaml \
#       --num-shards 4 \
#       --output-dir results/run_20260424
#
# Requirements:
#   - roboeval CLI installed (pip install -e .) or invoked as python -m roboeval.cli.main
#   - VLA server and sim_worker already running (use: roboeval serve --vla pi05 --sim libero)

set -euo pipefail

# ---- defaults ----
CONFIG=""
NUM_SHARDS=4
OUTPUT_DIR=""
LOG_DIR="logs/sharded"
EXTRA_ARGS=""

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)       CONFIG="$2"; shift 2 ;;
        --num-shards|-n)   NUM_SHARDS="$2"; shift 2 ;;
        --output-dir|-o)   OUTPUT_DIR="$2"; shift 2 ;;
        --log-dir)         LOG_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$CONFIG" ]]; then
    echo "Error: --config is required."
    echo "Usage: $0 --config <yaml> [--num-shards N] [--output-dir dir]"
    exit 1
fi

# ---- resolve roboeval command ----
if command -v roboeval &>/dev/null; then
    ROBOEVAL="roboeval"
elif python -m roboeval.cli.main --help &>/dev/null 2>&1; then
    ROBOEVAL="python -m roboeval.cli.main"
else
    echo "Error: roboeval not found. Install with: pip install -e ."
    exit 1
fi

# ---- derive output dir from config name if not specified ----
if [[ -z "$OUTPUT_DIR" ]]; then
    CONFIG_STEM=$(basename "$CONFIG" .yaml)
    TS=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="results/${CONFIG_STEM}_${TS}"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "================================================================"
echo "roboeval sharded run"
echo "  config:     $CONFIG"
echo "  shards:     $NUM_SHARDS"
echo "  output_dir: $OUTPUT_DIR"
echo "  logs:       $LOG_DIR"
echo "================================================================"

# ---- launch shards ----
PIDS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    LOG_FILE="$LOG_DIR/shard_${i}of${NUM_SHARDS}.log"
    echo "Starting shard $i/$NUM_SHARDS → $LOG_FILE"
    $ROBOEVAL run \
        --config "$CONFIG" \
        --shard-id "$i" \
        --num-shards "$NUM_SHARDS" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} shards. PIDs: ${PIDS[*]}"
echo "Waiting for all shards to complete..."

# ---- wait for all shards ----
FAILED=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        echo "  Shard pid=$pid completed OK."
    else
        echo "  Shard pid=$pid FAILED (exit $?)."
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "Warning: $FAILED shard(s) failed. Merge may be partial."
fi

# ---- merge shards ----
MERGE_OUTPUT="$OUTPUT_DIR/final.json"
SHARD_PATTERN="${OUTPUT_DIR}/*shard*.json"

echo ""
echo "Merging shards: $SHARD_PATTERN → $MERGE_OUTPUT"
$ROBOEVAL merge \
    --pattern "$SHARD_PATTERN" \
    --output "$MERGE_OUTPUT" \
    || true  # Don't fail the script if merge is partial

if [[ -f "$MERGE_OUTPUT" ]]; then
    echo ""
    echo "Merged result: $MERGE_OUTPUT"
    # Quick success rate summary
    python3 -c "
import json, sys
try:
    d = json.load(open('$MERGE_OUTPUT'))
    rate = d.get('mean_success', 0.0)
    total = d.get('merge_info', {}).get('total_episodes', '?')
    partial = ' (PARTIAL)' if d.get('partial') else ''
    print(f'Overall success: {rate:.1%} ({total} episodes){partial}')
except Exception as e:
    print(f'Could not parse merged JSON: {e}')
" 2>/dev/null || true
else
    echo "Warning: merged file not found at $MERGE_OUTPUT"
fi

echo ""
echo "All done. Results in: $OUTPUT_DIR"
exit $FAILED
