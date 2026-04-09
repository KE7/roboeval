#!/usr/bin/env bash
# LEGACY SCRIPT — for advanced use only. Prefer: robo-eval run --benchmark libero_pro
# =============================================================================
# run_parallel_p1p2.sh — Parallel LIBERO-PRO P1/P2 benchmark evaluation
# =============================================================================
#
# Runs multiple sim_worker instances simultaneously (one per task slot) to
# massively speed up the P1/P2 evaluation.  Each worker owns a unique port in
# the 6000-6099 range; the VLA policy server is shared and is NOT restarted.
#
# Strategy: hybrid Option C — N_WORKERS task slots run concurrently.
#   - All 10 tasks of each suite are dispatched into the shared job pool.
#   - Pool ensures at most N_WORKERS tasks run at the same time.
#   - Each slot has its own sim_worker process on a dedicated port.
#   - The sim_worker stays alive across tasks on the same slot (the /init
#     call inside SimWrapper reuses the running server, closing + reopening
#     the MuJoCo env for each new task).
#
# Usage:
#   bash scripts/run_parallel_p1p2.sh \
#     --vla-url http://localhost:5100 \
#     --results-dir results/my_run \
#     --episodes 50 \
#     --workers 8 \
#     [--p1-suites "libero_goal_task libero_spatial_task libero_10_task libero_object_task"] \
#     [--p2-suites "libero_goal_swap libero_spatial_swap libero_10_swap libero_object_swap"] \
#     [--suites "all_suites_space_separated"]
#
# Environment variables (all optional):
#   LIBERO_CONFIG_PATH    default: ~/.libero_pro
#   LD_LIBRARY_PATH       prepends ~/.micromamba/envs/libero_libs/lib automatically
#   MUJOCO_GL             default: egl
#   BASE_PORT             first port of the worker pool (default: 6000)
#   SIM_STARTUP_TIMEOUT   seconds to wait for sim /health (default: 60)
#
# Prerequisites:
#   - VLA policy server already running at --vla-url (not started here).
#   - Ports BASE_PORT .. BASE_PORT+N_WORKERS-1 must all be free.
#
# Output layout:
#   <results_dir>/
#     logs/<suite>_task<N>.log    — per-task eval stdout+stderr
#     logs/sim_worker_<port>.log  — sim_worker server log per slot
#     experience/<suite>/task<N>/ — planner experience dirs (one per task)
#     scores.json                 — aggregated scores (all suites, JSON)
#     summary.txt                 — human-readable summary
#
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VLA_URL="http://localhost:5100"
RESULTS_DIR=""
NUM_EPS=50
N_WORKERS=8
BASE_PORT="${BASE_PORT:-6000}"
SIM_STARTUP_TIMEOUT="${SIM_STARTUP_TIMEOUT:-60}"

# Default: run all 8 P1+P2 suites
P1_SUITES=(libero_goal_task libero_spatial_task libero_10_task libero_object_task)
P2_SUITES=(libero_goal_swap libero_spatial_swap libero_10_swap libero_object_swap)

P1_SUITES_OVERRIDE=()
P2_SUITES_OVERRIDE=()

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vla-url)       VLA_URL="$2";    shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --episodes)      NUM_EPS="$2";    shift 2 ;;
        --workers)       N_WORKERS="$2";  shift 2 ;;
        --suites)
            # Space-separated; split into P1 (_task suffix) and P2 (others)
            IFS=' ' read -r -a ALL_SUITES_ARG <<< "$2"
            for s in "${ALL_SUITES_ARG[@]}"; do
                if [[ "$s" == *_task ]]; then
                    P1_SUITES_OVERRIDE+=("$s")
                else
                    P2_SUITES_OVERRIDE+=("$s")
                fi
            done
            shift 2 ;;
        --p1-suites)
            IFS=' ' read -r -a P1_SUITES_OVERRIDE <<< "$2"; shift 2 ;;
        --p2-suites)
            IFS=' ' read -r -a P2_SUITES_OVERRIDE <<< "$2"; shift 2 ;;
        --base-port)     BASE_PORT="$2";  shift 2 ;;
        --help|-h)
            sed -n '3,52p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 1 ;;
    esac
done

# Apply suite overrides if provided
[[ ${#P1_SUITES_OVERRIDE[@]} -gt 0 ]] && P1_SUITES=("${P1_SUITES_OVERRIDE[@]}")
[[ ${#P2_SUITES_OVERRIDE[@]} -gt 0 ]] && P2_SUITES=("${P2_SUITES_OVERRIDE[@]}")

if [[ -z "$RESULTS_DIR" ]]; then
    echo "ERROR: --results-dir is required." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero_pro}"
export LD_LIBRARY_PATH="${HOME}/.micromamba/envs/libero_libs/lib:${LD_LIBRARY_PATH:-}"
export VLA_URL

# Project root is one level above the scripts/ directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SIM_PYTHON="$PROJECT_ROOT/.venvs/libero_pro/bin/python"
EVAL_PYTHON="$PROJECT_ROOT/.venvs/litellm/bin/python"
SIM_WORKER="$PROJECT_ROOT/sims/sim_worker.py"
EVAL_SCRIPT="$PROJECT_ROOT/run_sim_eval.py"

LOGS_DIR="$RESULTS_DIR/logs"
SCORES_FILE="$RESULTS_DIR/scores.json"
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

mkdir -p "$LOGS_DIR"

# ---------------------------------------------------------------------------
# Port budget check
# ---------------------------------------------------------------------------
PORT_BUDGET=100  # 6000-6099
if [[ "$N_WORKERS" -gt "$PORT_BUDGET" ]]; then
    echo "ERROR: --workers $N_WORKERS exceeds port budget ($PORT_BUDGET in range ${BASE_PORT}-$((BASE_PORT+PORT_BUDGET-1)))." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Slot → port mapping (fixed for the lifetime of the run)
# ---------------------------------------------------------------------------
declare -a WORKER_PIDS   # PID of sim_worker in each slot (0 = not running)
declare -a WORKER_PORTS  # port assigned to each slot

for (( slot=0; slot<N_WORKERS; slot++ )); do
    WORKER_PIDS[$slot]=0
    WORKER_PORTS[$slot]=$(( BASE_PORT + slot ))
done

# ---------------------------------------------------------------------------
# Cleanup on EXIT: kill all sim_workers we started
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "[parallel_p1p2] Shutting down sim_workers..."
    for (( slot=0; slot<N_WORKERS; slot++ )); do
        local_pid="${WORKER_PIDS[$slot]}"
        if [[ "$local_pid" -gt 0 ]] && kill -0 "$local_pid" 2>/dev/null; then
            kill "$local_pid" 2>/dev/null || true
            echo "  Killed sim_worker PID $local_pid (slot $slot, port ${WORKER_PORTS[$slot]})"
        fi
    done
    # Belt-and-suspenders: evict anything lingering on our port range
    for (( p=BASE_PORT; p<BASE_PORT+N_WORKERS; p++ )); do
        fuser -k "${p}/tcp" 2>/dev/null || true
    done
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# start_sim_worker SLOT
#   Launches a new sim_worker on the slot's port, replacing any existing one.
# ---------------------------------------------------------------------------
start_sim_worker() {
    local slot="$1"
    local port="${WORKER_PORTS[$slot]}"
    local logfile="$LOGS_DIR/sim_worker_${port}.log"

    # Terminate the previous worker in this slot if still alive
    local old_pid="${WORKER_PIDS[$slot]}"
    if [[ "$old_pid" -gt 0 ]] && kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid" 2>/dev/null || true
        wait "$old_pid" 2>/dev/null || true
    fi
    # Evict any other process holding the port (e.g. stale workers from a prior run)
    fuser -k "${port}/tcp" 2>/dev/null || true
    sleep 0.5

    MUJOCO_GL=egl \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$SIM_PYTHON" "$SIM_WORKER" \
            --sim libero_pro \
            --port "$port" \
            --headless \
        >> "$logfile" 2>&1 &

    WORKER_PIDS[$slot]=$!
    echo "  [slot $slot] sim_worker PID=${WORKER_PIDS[$slot]} port=$port"
}

# ---------------------------------------------------------------------------
# wait_for_sim PORT
#   Polls GET /health until the server responds or timeout expires.
# ---------------------------------------------------------------------------
wait_for_sim() {
    local port="$1"
    local elapsed=0
    local url="http://localhost:${port}/health"
    while [[ "$elapsed" -lt "$SIM_STARTUP_TIMEOUT" ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
        (( elapsed++ )) || true
    done
    echo "ERROR: sim_worker on port $port did not become healthy within ${SIM_STARTUP_TIMEOUT}s." >&2
    return 1
}

# ---------------------------------------------------------------------------
# run_task SLOT SUITE TASK_IDX NUM_EPS
#   Runs run_sim_eval.py for one task; called in a background subshell.
#   All output goes to the task's log file.
# ---------------------------------------------------------------------------
run_task() {
    local slot="$1"
    local suite="$2"
    local task_idx="$3"
    local num_eps="$4"
    local port=$(( BASE_PORT + slot ))
    local logfile="$LOGS_DIR/${suite}_task${task_idx}.log"
    local exp_dir="$RESULTS_DIR/experience/${suite}/task${task_idx}"

    MUJOCO_GL=egl \
    LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    VLA_URL="$VLA_URL" \
        "$EVAL_PYTHON" "$EVAL_SCRIPT" eval \
            --sim        libero_pro \
            --task       "$task_idx" \
            --suite      "$suite" \
            --sim-url    "http://localhost:${port}" \
            --headless \
            --max-episodes   "$num_eps" \
            --experience-dir "$exp_dir" \
            --delta-actions \
            --no-vlm \
        > "$logfile" 2>&1 || true
}

# ---------------------------------------------------------------------------
# Build job queue: "suite:task_idx:num_eps" for every (suite, task) pair
# ---------------------------------------------------------------------------
ALL_SUITES=("${P1_SUITES[@]}" "${P2_SUITES[@]}")

declare -a JOB_QUEUE

for suite in "${ALL_SUITES[@]}"; do
    for task_idx in $(seq 0 9); do
        JOB_QUEUE+=("${suite}:${task_idx}:${NUM_EPS}")
    done
done

TOTAL_TASKS=${#JOB_QUEUE[@]}

echo "[parallel_p1p2] ============================================"
echo "[parallel_p1p2] Job queue    : $TOTAL_TASKS tasks"
echo "[parallel_p1p2] Workers      : $N_WORKERS parallel slots"
echo "[parallel_p1p2] Episodes/task: $NUM_EPS"
echo "[parallel_p1p2] Port range   : ${BASE_PORT} - $(( BASE_PORT + N_WORKERS - 1 ))"
echo "[parallel_p1p2] VLA URL      : $VLA_URL"
echo "[parallel_p1p2] Results dir  : $RESULTS_DIR"
echo "[parallel_p1p2] ============================================"
echo ""

# ---------------------------------------------------------------------------
# Start all sim_workers upfront
# ---------------------------------------------------------------------------
echo "[parallel_p1p2] Starting $N_WORKERS sim_worker processes..."
for (( slot=0; slot<N_WORKERS; slot++ )); do
    start_sim_worker "$slot"
done

echo "[parallel_p1p2] Waiting for sim_workers to become healthy..."
for (( slot=0; slot<N_WORKERS; slot++ )); do
    port="${WORKER_PORTS[$slot]}"
    if ! wait_for_sim "$port"; then
        echo "ERROR: sim_worker slot $slot (port $port) failed to start." >&2
        exit 1
    fi
    echo "  Slot $slot port=$port: healthy"
done
echo "[parallel_p1p2] All sim_workers ready."
echo ""

# ---------------------------------------------------------------------------
# Job pool dispatch loop
#
# Invariant: SLOT_JOB_PID[slot] == 0  →  slot is idle
#            SLOT_JOB_PID[slot] > 0   →  slot is running that background PID
# ---------------------------------------------------------------------------
declare -a SLOT_JOB_PID
declare -a SLOT_SUITE
declare -a SLOT_TASK
declare -a SLOT_EPS

for (( slot=0; slot<N_WORKERS; slot++ )); do
    SLOT_JOB_PID[$slot]=0
    SLOT_SUITE[$slot]=""
    SLOT_TASK[$slot]=""
    SLOT_EPS[$slot]=0
done

JOB_INDEX=0
JOBS_COMPLETED=0

# Dispatch one job to the given slot.  Returns 1 if queue is empty.
dispatch_next_job() {
    local slot="$1"
    if [[ "$JOB_INDEX" -ge "$TOTAL_TASKS" ]]; then
        return 1
    fi

    local job="${JOB_QUEUE[$JOB_INDEX]}"
    (( JOB_INDEX++ )) || true

    local suite task_idx eps
    IFS=':' read -r suite task_idx eps <<< "$job"

    SLOT_SUITE[$slot]="$suite"
    SLOT_TASK[$slot]="$task_idx"
    SLOT_EPS[$slot]="$eps"

    echo "[parallel_p1p2] Dispatch slot=$slot suite=$suite task=$task_idx eps=$eps (job $JOB_INDEX/$TOTAL_TASKS)"

    # Background subshell — does not inherit SLOT_* arrays (no side effects needed)
    ( run_task "$slot" "$suite" "$task_idx" "$eps" ) &
    SLOT_JOB_PID[$slot]=$!
    return 0
}

# Fill all slots with the first batch of jobs
for (( slot=0; slot<N_WORKERS; slot++ )); do
    dispatch_next_job "$slot" || true   # 'true' handles "queue already empty" gracefully
done

# Poll loop: harvest finished jobs, dispatch replacements, until all done
while [[ "$JOBS_COMPLETED" -lt "$TOTAL_TASKS" ]]; do
    for (( slot=0; slot<N_WORKERS; slot++ )); do
        local_pid="${SLOT_JOB_PID[$slot]}"
        if [[ "$local_pid" -eq 0 ]]; then
            continue  # idle slot (fewer jobs than workers)
        fi

        # kill -0 returns non-zero when the PID no longer exists
        if ! kill -0 "$local_pid" 2>/dev/null; then
            wait "$local_pid" 2>/dev/null || true   # reap zombie

            local_suite="${SLOT_SUITE[$slot]}"
            local_task="${SLOT_TASK[$slot]}"
            local_logfile="$LOGS_DIR/${local_suite}_task${local_task}.log"
            local_successes=0
            if [[ -f "$local_logfile" ]]; then
                local_successes=$(awk '/Simulator reports success: True/{n++} END{print n+0}' \
                    "$local_logfile" 2>/dev/null || echo 0)
            fi

            echo "[parallel_p1p2] Done slot=$slot suite=$local_suite task=$local_task" \
                 "successes=${local_successes}/${SLOT_EPS[$slot]}"

            SLOT_JOB_PID[$slot]=0
            (( JOBS_COMPLETED++ )) || true

            dispatch_next_job "$slot" || true
        fi
    done
    sleep 2
done

# Reap any remaining background children
wait 2>/dev/null || true

echo ""
echo "[parallel_p1p2] All $TOTAL_TASKS tasks finished. Aggregating results..."
echo ""

# ---------------------------------------------------------------------------
# Aggregate: compute per-suite and overall success counts from log files.
# Must happen in the main shell (not a subshell) so the associative arrays
# are visible to the JSON + summary writers below.
# ---------------------------------------------------------------------------
declare -A SUITE_SUCCESS
declare -A SUITE_TOTAL

for suite in "${ALL_SUITES[@]}"; do
    SUITE_SUCCESS[$suite]=0
    SUITE_TOTAL[$suite]=0
done

OVERALL_SUCCESS=0
OVERALL_TOTAL=0

# Collect a flat ordered list of (suite, task_idx, successes, eps) for JSON
declare -a JSON_ENTRIES  # each entry: "suite|task|successes|eps"

for suite in "${ALL_SUITES[@]}"; do
    for task_idx in $(seq 0 9); do
        logfile="$LOGS_DIR/${suite}_task${task_idx}.log"
        successes=0
        if [[ -f "$logfile" ]]; then
            successes=$(awk '/Simulator reports success: True/{n++} END{print n+0}' \
                "$logfile" 2>/dev/null || echo 0)
        fi
        SUITE_SUCCESS[$suite]=$(( SUITE_SUCCESS[$suite] + successes ))
        SUITE_TOTAL[$suite]=$(( SUITE_TOTAL[$suite] + NUM_EPS ))
        OVERALL_SUCCESS=$(( OVERALL_SUCCESS + successes ))
        OVERALL_TOTAL=$(( OVERALL_TOTAL + NUM_EPS ))
        JSON_ENTRIES+=("${suite}|${task_idx}|${successes}|${NUM_EPS}")
    done
done

# ---------------------------------------------------------------------------
# Write scores.json
# ---------------------------------------------------------------------------
{
    echo '{'
    echo '  "tasks": ['
    first=true
    for entry in "${JSON_ENTRIES[@]}"; do
        IFS='|' read -r s t sc ep <<< "$entry"
        if [[ "$first" == true ]]; then first=false; else echo ','; fi
        printf '    {"suite": "%s", "task": %s, "successes": %s, "total": %s}' \
            "$s" "$t" "$sc" "$ep"
    done
    echo ""   # newline after last entry
    echo '  ],'
    echo '  "suites": {'
    first=true
    for suite in "${ALL_SUITES[@]}"; do
        rate=$(echo "scale=4; ${SUITE_SUCCESS[$suite]} / ${SUITE_TOTAL[$suite]}" | bc 2>/dev/null || echo "0.0000")
        if [[ "$first" == true ]]; then first=false; else echo ','; fi
        printf '    "%s": {"success": %d, "total": %d, "rate": %s}' \
            "$suite" "${SUITE_SUCCESS[$suite]}" "${SUITE_TOTAL[$suite]}" "$rate"
    done
    echo ""
    echo '  },'
    overall_rate=$(echo "scale=4; $OVERALL_SUCCESS / $OVERALL_TOTAL" | bc 2>/dev/null || echo "0.0000")
    echo "  \"overall\": {\"success\": $OVERALL_SUCCESS, \"total\": $OVERALL_TOTAL, \"rate\": $overall_rate}"
    echo '}'
} > "$SCORES_FILE"

# ---------------------------------------------------------------------------
# Write summary.txt (and tee to stdout)
# ---------------------------------------------------------------------------
{
    echo "Parallel LIBERO-PRO P1+P2 Benchmark — $(date)"
    echo "Results dir : $RESULTS_DIR"
    echo "VLA URL     : $VLA_URL"
    echo "Episodes    : $NUM_EPS per task"
    echo "Workers     : $N_WORKERS parallel slots"
    echo ""

    echo "### P1 SUITES (task perturbation, ~0% expected) ###"
    echo ""
    for suite in "${P1_SUITES[@]}"; do
        echo "=== Suite: $suite ($NUM_EPS ep/task) [P1] ==="
        for task_idx in $(seq 0 9); do
            logfile="$LOGS_DIR/${suite}_task${task_idx}.log"
            successes=0
            if [[ -f "$logfile" ]]; then
                successes=$(awk '/Simulator reports success: True/{n++} END{print n+0}' \
                    "$logfile" 2>/dev/null || echo 0)
            fi
            echo "  Task $task_idx: ${successes}/${NUM_EPS}"
        done
        rate=$(echo "scale=3; ${SUITE_SUCCESS[$suite]} / ${SUITE_TOTAL[$suite]}" | bc 2>/dev/null || echo "0.000")
        echo "  Suite $suite: ${SUITE_SUCCESS[$suite]}/${SUITE_TOTAL[$suite]} = $rate"
        echo ""
    done

    echo "### P2 SUITES (position swap, 10-40% expected) ###"
    echo ""
    for suite in "${P2_SUITES[@]}"; do
        echo "=== Suite: $suite ($NUM_EPS ep/task) [P2] ==="
        for task_idx in $(seq 0 9); do
            logfile="$LOGS_DIR/${suite}_task${task_idx}.log"
            successes=0
            if [[ -f "$logfile" ]]; then
                successes=$(awk '/Simulator reports success: True/{n++} END{print n+0}' \
                    "$logfile" 2>/dev/null || echo 0)
            fi
            echo "  Task $task_idx: ${successes}/${NUM_EPS}"
        done
        rate=$(echo "scale=3; ${SUITE_SUCCESS[$suite]} / ${SUITE_TOTAL[$suite]}" | bc 2>/dev/null || echo "0.000")
        echo "  Suite $suite: ${SUITE_SUCCESS[$suite]}/${SUITE_TOTAL[$suite]} = $rate"
        echo ""
    done

    overall_rate=$(echo "scale=4; $OVERALL_SUCCESS / $OVERALL_TOTAL" | bc 2>/dev/null || echo "0.0000")
    echo "=== OVERALL ==="
    echo "Total: $OVERALL_SUCCESS / $OVERALL_TOTAL (rate: $overall_rate)"
    echo ""
    echo "Completed at $(date)"
} | tee "$SUMMARY_FILE"

echo ""
echo "Done. Results written to $RESULTS_DIR"
echo "  Scores : $SCORES_FILE"
echo "  Summary: $SUMMARY_FILE"
echo "  Logs   : $LOGS_DIR/"
