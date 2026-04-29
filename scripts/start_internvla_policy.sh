#!/usr/bin/env bash
# Start the InternVLA VLA server.
# Env vars: INTERNVLA_MODEL_ID, VLA_PORT are forwarded automatically.
#
# Usage:
#   bash scripts/start_internvla_policy.sh [--port 5104]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV="${ROBO_EVAL_INTERNVLA_VENV:-$PROJECT_ROOT/.venvs/internvla}"
PORT="${VLA_PORT:-5104}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "$VENV/bin/python" -m sims.vla_policies.internvla_policy --port "$PORT" "$@"
