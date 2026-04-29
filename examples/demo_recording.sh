#!/usr/bin/env bash
# =============================================================================
# examples/demo_recording.sh
#
# Paced, narrated walk-through of:
#   install → smoke-test → first eval
#
# Usage:
#   bash examples/demo_recording.sh           # full run
#   bash examples/demo_recording.sh --quick   # skip install (reuse existing venv)
#   bash examples/demo_recording.sh --no-color
#   bash examples/demo_recording.sh --simulate   # dry-run, prints commands only
#
# Asciinema recording:
#   asciinema rec --command 'bash examples/demo_recording.sh' demo.cast
#
# Requirements:
#   - Run from the repo root  (cd roboeval && bash examples/demo_recording.sh)
#   - uv installed, or setup.sh will install it
#   - NVIDIA GPU with CUDA (for pi05 model)
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
QUICK=0
NO_COLOR=0
SIMULATE=0

for arg in "$@"; do
  case "$arg" in
    --quick)    QUICK=1 ;;
    --no-color) NO_COLOR=1 ;;
    --simulate) SIMULATE=1 ;;
    --help|-h)
      sed -n '2,/^# ===/p' "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Usage: bash examples/demo_recording.sh [--quick] [--no-color] [--simulate]" >&2
      exit 1
      ;;
  esac
done

# ---------------------------------------------------------------------------
# Color helpers (gracefully degraded when NO_COLOR or not a tty)
# ---------------------------------------------------------------------------
if [[ "$NO_COLOR" -eq 0 ]] && command -v tput &>/dev/null && tput setaf 1 &>/dev/null 2>&1; then
  C_GREEN=$(tput setaf 2; tput bold)
  C_CYAN=$(tput setaf 6)
  C_YELLOW=$(tput setaf 3; tput bold)
  C_RED=$(tput setaf 1; tput bold)
  C_DIM=$(tput dim 2>/dev/null || tput setaf 7)
  C_RESET=$(tput sgr0)
else
  C_GREEN=""; C_CYAN=""; C_YELLOW=""; C_RED=""; C_DIM=""; C_RESET=""
fi

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
START=$(date +%s)

DEMO_DIR="/tmp/roboeval_demo_$(date +%Y%m%d_%H%M%S)"
FAILED_STEP=""

die() {
  local step="${FAILED_STEP:-unknown}"
  echo ""
  echo "${C_RED}✗ Demo failed at step: ${step}${C_RESET}"
  echo "${C_RED}  Error: $*${C_RESET}"
  echo ""
  elapsed
  exit 1
}

trap 'die "unexpected error (line $LINENO)"' ERR

elapsed() {
  local end
  end=$(date +%s)
  local secs=$(( end - START ))
  local m=$(( secs / 60 ))
  local s=$(( secs % 60 ))
  echo "${C_YELLOW}Demo complete in ${m}m ${s}s.${C_RESET}"
}

banner() {
  local title="$1"
  echo ""
  echo "${C_GREEN}# ============================================================${C_RESET}"
  echo "${C_GREEN}# $title${C_RESET}"
  echo "${C_GREEN}# ============================================================${C_RESET}"
  echo ""
}

narrate() {
  echo "${C_DIM}  ▸ $*${C_RESET}"
}

run_cmd() {
  # Print the command in cyan, then either execute or simulate it.
  echo "${C_CYAN}  \$ $*${C_RESET}"
  if [[ "$SIMULATE" -eq 1 ]]; then
    return 0
  fi
  eval "$@"
}

pause() {
  # Small breathing room for asciinema playback.
  local secs="${1:-1}"
  if [[ "$SIMULATE" -eq 0 ]]; then
    sleep "$secs"
  fi
}

# ---------------------------------------------------------------------------
# Sanity checks (not run in simulate mode)
# ---------------------------------------------------------------------------
if [[ "$SIMULATE" -eq 0 ]]; then
  if [[ ! -f "pyproject.toml" ]] || ! grep -q 'roboeval\|robo.eval\|robo_eval' pyproject.toml 2>/dev/null; then
    echo "${C_RED}Error: run this script from the roboeval repo root.${C_RESET}" >&2
    exit 1
  fi
fi

mkdir -p "$DEMO_DIR"

# =============================================================================
# STEP 0  —  intro
# =============================================================================
FAILED_STEP="intro"

banner "roboeval  ·  install → smoke → first eval"
narrate "This walk-through takes ~8–12 min on a fresh machine (GB10, A100-class)."
narrate "Output dir: $DEMO_DIR"
pause 2

# =============================================================================
# STEP 1  —  git clone (shown, not repeated if already inside the repo)
# =============================================================================
FAILED_STEP="git clone"

banner "STEP 1 / 6  —  Clone the repo"
narrate "In a real first-run you would start here:"
echo ""
echo "${C_CYAN}  \$ git clone https://github.com/KE7/roboeval.git${C_RESET}"
echo "${C_CYAN}  \$ cd roboeval${C_RESET}"
echo ""
narrate "We're already inside the repo, so we continue from here."
pause 2

# =============================================================================
# STEP 2  —  setup.sh
# =============================================================================
FAILED_STEP="setup"

banner "STEP 2 / 6  —  Install: Pi 0.5 policy server + LIBERO simulator"
narrate "setup.sh pi05 libero will:"
narrate "  • Install uv (if missing)"
narrate "  • Create .venvs/pi05  (Python 3.11, ~75s model download on first run)"
narrate "  • Create .venvs/libero (Python 3.8, clones LIBERO repo, patches missing __init__.py)"
narrate "  • Install CUDA-aware torch in each venv"
narrate "  • Run a quick import check on both"
echo ""

if [[ "$QUICK" -eq 1 ]]; then
  narrate "[--quick] Skipping setup — reusing existing .venvs."
  pause 1
else
  run_cmd "./scripts/setup.sh pi05 libero"
fi
pause 2

# =============================================================================
# STEP 3  —  robo-eval test
# =============================================================================
FAILED_STEP="robo-eval test"

banner "STEP 3 / 6  —  Validate the install  (robo-eval test)"
narrate "robo-eval test starts the VLA server and sim worker, calls /health on each,"
narrate "runs a 1-step dry-run, and verifies the ActionObsSpec contracts match."
echo ""
run_cmd ".venvs/roboeval/bin/robo-eval test"
pause 2

# =============================================================================
# STEP 4  —  robo-eval run (1-episode smoke)
# =============================================================================
FAILED_STEP="robo-eval run"

banner "STEP 4 / 6  —  First eval  (pi05 × libero_spatial, 1 episode)"
narrate "Running the CI smoke config: 1 task × 10 episodes."
narrate "The Pi 0.5 server and LIBERO worker start as subprocesses automatically."
narrate "Expected: ≥1/10 success  (historical baseline ~93%)."
narrate "Expected runtime: ~8 min (10 eps × ~50 s/ep on GB10)."
echo ""
run_cmd ".venvs/roboeval/bin/robo-eval run --config configs/ci/pi05_libero_spatial_smoke.yaml"
pause 2

# =============================================================================
# STEP 5  —  show result JSON
# =============================================================================
FAILED_STEP="show results"

banner "STEP 5 / 6  —  Inspect the result JSON"
narrate "Results are written to results/ci/  (configurable via output_dir in the YAML)."
echo ""

if [[ "$SIMULATE" -eq 1 ]]; then
  echo "${C_CYAN}  \$ find results/ci -name '*.json' | head -1 | xargs cat${C_RESET}"
  echo ""
  narrate "[--simulate] Showing expected output format:"
  echo ""
  cat <<'JSON'
  {
    "harness_version": "0.1.0",
    "config": "ci_pi05_libero_spatial_smoke",
    "vla": "pi05",
    "sim": "libero",
    "suite": "libero_spatial",
    "episodes_per_task": 10,
    "results": [
      {"task": 0, "successes": 9, "total": 10, "success_rate": 0.9}
    ],
    "overall_success_rate": 0.9
  }
JSON
else
  RESULT_JSON=$(find results/ci -name '*.json' 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$RESULT_JSON" ]]; then
    run_cmd "cat \"$RESULT_JSON\""
    echo ""
    HARNESS_VER=$(python3 -c "import json,sys; d=json.load(open('$RESULT_JSON')); print(d.get('harness_version','(key missing)'))" 2>/dev/null || echo "(could not parse)")
    echo "${C_GREEN}  harness_version = $HARNESS_VER${C_RESET}"
  else
    narrate "Result file not yet visible; check results/ci/ after the run."
  fi
fi
pause 2

# =============================================================================
# STEP 6  —  closing
# =============================================================================
FAILED_STEP="closing"

banner "STEP 6 / 6  —  Where to go next"
echo ""
echo "  ${C_GREEN}★ Quick start guide${C_RESET}"
echo "    ${C_CYAN}https://github.com/KE7/roboeval/blob/main/docs/quickstart.md${C_RESET}"
echo ""
echo "  ${C_GREEN}★ Full benchmark results${C_RESET}"
echo "    ${C_CYAN}https://github.com/KE7/roboeval/blob/main/docs/results.md${C_RESET}"
echo ""
echo "  ${C_GREEN}★ Head-to-head model comparisons${C_RESET}"
echo "    ${C_CYAN}https://github.com/KE7/roboeval/blob/main/docs/comparison.md${C_RESET}"
echo ""
echo "  ${C_GREEN}★ Add your own VLA or simulator${C_RESET}"
echo "    ${C_CYAN}https://github.com/KE7/roboeval/blob/main/docs/extending.md${C_RESET}"
echo ""
pause 1

# =============================================================================
elapsed
# =============================================================================
