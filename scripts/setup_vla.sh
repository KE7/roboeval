#!/usr/bin/env bash
#
# Setup script for the VLA (Vision-Language-Action) Python environment.
#
# Installs:
#   - lerobot[pi]   : LeRobot with Pi 0.5 support (custom transformers branch)
#   - OpenVLA deps  : transformers, torch, torchvision, accelerate
#   - Server deps   : fastapi, uvicorn[standard], pillow
#
# Note: lerobot[pi] installs a custom transformers branch (fix/lerobot_openpi).
#       OpenVLA uses standard transformers; the custom branch should be compatible.
#       If version conflicts arise the script will warn but continue.
#
# Prerequisites:
#   - uv >= 0.9 at ~/.local/bin/uv
#   - Python 3.11 available on the system
#
# Usage:
#   bash scripts/setup_vla.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venvs/vla"

UV="$HOME/.local/bin/uv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[setup_vla]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[setup_vla WARN]${NC} $*"; }
log_error() { echo -e "${RED}[setup_vla ERROR]${NC} $*"; }

# Check for uv
if [ ! -x "$UV" ]; then
    log_error "uv not found at $UV"
    log_error "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# -------------------------------------------------------------------
# Create venv
# -------------------------------------------------------------------

log_info "Creating VLA venv at $VENV_DIR (Python 3.11)..."
"$UV" venv --python 3.11 "$VENV_DIR"

PYTHON="$VENV_DIR/bin/python"

# -------------------------------------------------------------------
# Install lerobot with Pi 0.5 support
# -------------------------------------------------------------------

log_info "Installing lerobot with Pi 0.5 support..."
log_info "  (uses custom transformers branch: fix/lerobot_openpi)"
"$UV" pip install --python "$PYTHON" \
    "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git" || {
    log_warn "lerobot[pi] install encountered issues — check output above."
    log_warn "Continuing anyway; some features may be unavailable."
}

# -------------------------------------------------------------------
# Install OpenVLA dependencies
# -------------------------------------------------------------------

log_info "Installing OpenVLA dependencies (transformers, torch, torchvision, accelerate)..."
log_info "  Note: if transformers version conflicts with lerobot's custom branch, OpenVLA"
log_info "        inference will use whatever version was pinned by lerobot[pi]."
"$UV" pip install --python "$PYTHON" \
    transformers torch torchvision accelerate pillow || {
    log_warn "OpenVLA dependency install encountered issues — check output above."
    log_warn "Continuing anyway."
}

# -------------------------------------------------------------------
# Install server dependencies
# -------------------------------------------------------------------

log_info "Installing server dependencies (fastapi, uvicorn, pillow)..."
"$UV" pip install --python "$PYTHON" \
    fastapi "uvicorn[standard]" pillow

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------

log_info "Done! VLA venv ready at $VENV_DIR"
log_info "Start the VLA service with: bash scripts/start_vla.sh"
