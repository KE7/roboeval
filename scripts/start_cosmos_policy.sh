#!/usr/bin/env bash
# DEPRECATED: cosmos was dropped from the harness in 0056f01.
# Start the legacy Cosmos-Policy VLA server.
# Env vars: COSMOS_MODEL_ID, VLA_PORT are forwarded automatically.
#
# Usage:
#   bash scripts/start_cosmos_policy.sh [--port 5103]
#
# Or directly:
#   MUJOCO_GL=egl .venvs/cosmos/bin/python -m sims.vla_policies.cosmos_policy --port 5103

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${ROBO_EVAL_COSMOS_VENV:-$PROJECT_ROOT/.venvs/cosmos}"
PORT="${VLA_PORT:-5103}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "$VENV/bin/python" -m sims.vla_policies.cosmos_policy --port "$PORT" "$@"
