#!/usr/bin/env bash
#
# Setup script for robo-eval simulator virtualenvs.
#
# Each simulator requires its own isolated venv because they have
# incompatible Python version and dependency requirements:
#   - LIBERO:      Python 3.8  (MuJoCo, robosuite 1.2)
#   - RoboCasa:    Python 3.11 (robosuite 1.5, newer deps)
#   - RoboTwin:    Python 3.10 (SAPIEN, PyTorch)
#   - LIBERO-Pro:  Python 3.8  (same as LIBERO, different repo)
#
# Prerequisites:
#   - uv (https://github.com/astral-sh/uv)
#   - System packages: libegl1-mesa-dev, libvulkan-dev, etc.
#
# Usage:
#   bash scripts/setup_envs.sh [--skip-system-deps] [--only <sim_name>]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENVS_DIR="$PROJECT_ROOT/.venvs"

# Vendor repos directory — shared across reboots (unlike /tmp/).
# Override via ROBO_EVAL_VENDORS_DIR env var.
VENDORS_DIR="${ROBO_EVAL_VENDORS_DIR:-$HOME/.local/share/robo-eval/vendors}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Parse arguments
SKIP_SYSTEM_DEPS=false
ONLY_SIM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-system-deps) SKIP_SYSTEM_DEPS=true; shift ;;
        --only) ONLY_SIM="$2"; shift 2 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

# Check for uv
if ! command -v uv &> /dev/null; then
    log_error "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

mkdir -p "$VENVS_DIR"
mkdir -p "$VENDORS_DIR"

# -------------------------------------------------------------------
# System dependencies
# -------------------------------------------------------------------

install_system_deps() {
    log_info "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            libegl1-mesa-dev \
            libgl1-mesa-dev \
            libgles2-mesa-dev \
            libvulkan-dev \
            libglfw3-dev \
            libosmesa6-dev \
            patchelf \
            cmake \
            build-essential \
            libx11-dev \
            libxrandr-dev \
            libxinerama-dev \
            libxcursor-dev \
            libxi-dev
        log_info "System dependencies installed."
    else
        log_warn "apt-get not found. Install these packages manually:"
        log_warn "  libegl1-mesa-dev libgl1-mesa-dev libvulkan-dev libosmesa6-dev patchelf cmake"
    fi
}

if [ "$SKIP_SYSTEM_DEPS" = false ]; then
    install_system_deps
fi

# -------------------------------------------------------------------
# LIBERO (Python 3.8)
# -------------------------------------------------------------------

setup_libero() {
    local VENV="$VENVS_DIR/libero"
    log_info "Setting up LIBERO venv at $VENV (Python 3.8)..."

    uv venv "$VENV" --python 3.8
    local PIP="$VENV/bin/pip"
    local PYTHON="$VENV/bin/python"

    # Install core dependencies
    "$PIP" install --upgrade pip setuptools wheel
    "$PIP" install numpy==1.24.3
    "$PIP" install torch torchvision
    "$PIP" install mujoco==2.3.7
    "$PIP" install robosuite==1.2.1

    # Install LIBERO from git
    "$PIP" install "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"

    # Install Pillow for image encoding in the worker
    "$PIP" install Pillow

    # HTTP server deps for sim_worker
    "$PIP" install fastapi "uvicorn[standard]"

    log_info "LIBERO venv ready at $VENV"
}

# -------------------------------------------------------------------
# RoboCasa (Python 3.11)
# -------------------------------------------------------------------

setup_robocasa() {
    local VENV="$VENVS_DIR/robocasa"
    log_info "Setting up RoboCasa venv at $VENV (Python 3.11)..."

    uv venv "$VENV" --python 3.11
    local PIP="$VENV/bin/pip"
    local PYTHON="$VENV/bin/python"

    "$PIP" install --upgrade pip setuptools wheel
    "$PIP" install numpy
    "$PIP" install torch torchvision
    "$PIP" install mujoco

    # Install robosuite (v1.5+ for RoboCasa compatibility)
    "$PIP" install "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.5"

    # Install RoboCasa
    "$PIP" install "robocasa @ git+https://github.com/robocasa/robocasa.git"

    # Download RoboCasa assets (kitchen models, textures, etc.)
    "$PYTHON" -c "import robocasa; robocasa.download_assets()" 2>/dev/null || \
        log_warn "RoboCasa asset download may need to be run manually."

    "$PIP" install Pillow

    # HTTP server deps for sim_worker
    "$PIP" install fastapi "uvicorn[standard]"

    log_info "RoboCasa venv ready at $VENV"
}

# -------------------------------------------------------------------
# RoboTwin (Python 3.10)
# -------------------------------------------------------------------

setup_robotwin() {
    local VENV="$VENVS_DIR/robotwin"
    log_info "Setting up RoboTwin venv at $VENV (Python 3.10)..."

    uv venv "$VENV" --python 3.10
    local PIP="$VENV/bin/pip"
    local PYTHON="$VENV/bin/python"

    "$PIP" install --upgrade pip setuptools wheel
    "$PIP" install numpy
    "$PIP" install torch torchvision
    "$PIP" install sapien==3.0.0b1
    "$PIP" install transforms3d
    "$PIP" install gymnasium

    # RoboTwin itself (clone separately, as it's not pip-installable)
    local ROBOTWIN_DIR="$VENDORS_DIR/RoboTwin"
    if [ ! -d "$ROBOTWIN_DIR" ]; then
        git clone https://github.com/TianxingChen/RoboTwin.git "$ROBOTWIN_DIR"
    fi
    # Add RoboTwin repo to the venv's path
    echo "$ROBOTWIN_DIR" > "$VENV/lib/python3.10/site-packages/robotwin.pth"

    "$PIP" install Pillow

    # HTTP server deps for sim_worker
    "$PIP" install fastapi "uvicorn[standard]"

    log_info "RoboTwin venv ready at $VENV"
}

# -------------------------------------------------------------------
# LIBERO-Pro (Python 3.8, separate repo)
# -------------------------------------------------------------------

setup_libero_pro() {
    local VENV="$VENVS_DIR/libero_pro"
    log_info "Setting up LIBERO-Pro venv at $VENV (Python 3.8)..."

    uv venv "$VENV" --python 3.8
    local PIP="$VENV/bin/pip"
    local PYTHON="$VENV/bin/python"

    "$PIP" install --upgrade pip setuptools wheel
    "$PIP" install numpy==1.24.3
    "$PIP" install torch torchvision
    "$PIP" install mujoco==2.3.7
    "$PIP" install robosuite==1.2.1

    # LIBERO-Pro uses a separate fork (Zxy-MLlab/LIBERO-PRO), NOT the upstream repo.
    # The upstream branch "libero-pro" does not exist; installing from there silently
    # falls back to plain LIBERO which is missing the PRO task/suite definitions.
    local LIBERO_PRO_DIR="${LIBERO_PRO_DIR:-$VENDORS_DIR/LIBERO-PRO}"
    if [ ! -d "$LIBERO_PRO_DIR" ]; then
        log_info "Cloning LIBERO-PRO into $LIBERO_PRO_DIR ..."
        git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$LIBERO_PRO_DIR"
    else
        log_info "LIBERO-PRO already cloned at $LIBERO_PRO_DIR, skipping git clone."
    fi
    "$PIP" install -e "$LIBERO_PRO_DIR"

    "$PIP" install Pillow

    # HTTP server deps for sim_worker
    "$PIP" install fastapi "uvicorn[standard]"

    log_info "LIBERO-Pro venv ready at $VENV"
    log_warn "IMPORTANT: LIBERO-Pro requires micromamba HDF5 native libs at runtime."
    log_warn "  Before starting the sim worker, set:"
    log_warn "    export LD_LIBRARY_PATH=~/.micromamba/envs/libero_libs/lib:\$LD_LIBRARY_PATH"
    log_warn "  Install micromamba and the libs with:"
    log_warn "    micromamba install -n libero_libs -c conda-forge hdf5 -y"
}

# -------------------------------------------------------------------
# LIBERO-Infinity (installed into the libero venv)
# -------------------------------------------------------------------

setup_libero_infinity() {
    # libero-infinity requires Python 3.11+ (Scenic 3 dependency), so it gets
    # its own venv rather than sharing the base LIBERO venv (Python 3.8).
    local VENV="$VENVS_DIR/libero_infinity"
    local PYTHON="${PYTHON311:-python3.11}"

    if [ ! -d "$VENV" ]; then
        log_info "Creating libero-infinity venv at $VENV (Python 3.11+) ..."
        "$PYTHON" -m venv "$VENV" || {
            log_error "Failed to create venv — is Python 3.11+ installed?"
            log_error "Set PYTHON311=/path/to/python3.11 if it's not on PATH."
            exit 1
        }
    fi

    local PIP="$VENV/bin/pip"
    log_info "Installing libero-infinity into $VENV ..."

    if [ "${DEV:-}" = "1" ] || [ "${DEV:-}" = "true" ]; then
        # Developer mode: clone and install editable
        local CLONE_DIR="${LIBERO_INFINITY_DEV_DIR:-$VENDORS_DIR/libero-infinity}"
        if [ ! -d "$CLONE_DIR" ]; then
            log_info "Cloning libero-infinity into $CLONE_DIR ..."
            git clone --recurse-submodules https://github.com/KE7/libero-infinity.git "$CLONE_DIR"
        else
            log_info "libero-infinity already cloned at $CLONE_DIR, pulling latest ..."
            git -C "$CLONE_DIR" pull --ff-only || true
            git -C "$CLONE_DIR" submodule update --init --recursive
        fi
        "$PIP" install -e "$CLONE_DIR"
        # Install vendored LIBERO (inside libero-infinity's vendor/ submodule)
        if [ -d "$CLONE_DIR/vendor/libero" ]; then
            "$PIP" install -e "$CLONE_DIR/vendor/libero"
        fi
    else
        # Standard mode: install from GitHub release
        "$PIP" install "libero-infinity @ git+https://github.com/KE7/libero-infinity.git"
        # Also install LIBERO (required runtime dependency)
        "$PIP" install "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
    fi

    log_info "libero-infinity installed into $VENV"
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if [ -n "$ONLY_SIM" ]; then
    case "$ONLY_SIM" in
        libero)            setup_libero ;;
        robocasa)          setup_robocasa ;;
        robotwin)          setup_robotwin ;;
        libero_pro)        setup_libero_pro ;;
        libero_infinity)   setup_libero_infinity ;;
        *) log_error "Unknown simulator: $ONLY_SIM. Valid: libero, robocasa, robotwin, libero_pro, libero_infinity"; exit 1 ;;
    esac
else
    setup_libero
    setup_robocasa
    setup_robotwin
    setup_libero_pro
    setup_libero_infinity
fi

log_info "All simulator environments set up in $VENVS_DIR"
log_info "Venv structure:"
ls -la "$VENVS_DIR"/*/bin/python 2>/dev/null || true
