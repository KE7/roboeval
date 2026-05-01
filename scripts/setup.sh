#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-command uv-based install for roboeval
#
# Usage:
#   ./scripts/setup.sh <component> [<component> ...]
#
# Examples:
#   ./scripts/setup.sh pi05 libero              # Pi 0.5 + LIBERO
#   ./scripts/setup.sh smolvla robocasa         # SmolVLA + RoboCasa
#   ./scripts/setup.sh openvla libero libero_pro
#   ./scripts/setup.sh all                      # every supported component
#
# Available components:
#   VLAs (Python varies): pi05  openvla  smolvla  groot  internvla
#   Sims (Python varies): libero  libero_pro  robocasa  robotwin  aloha_gym
#   Mode extras:          vlm
#
# The script:
#   1. Installs uv if missing.
#   2. Creates .venvs/roboeval (Python 3.13) for the orchestrator CLI.
#   3. For each requested component, creates .venvs/<name> with the right
#      Python version and installs all deps.
#   4. Validates each install with a quick import check.
#   5. Prints next-step instructions.
#
# Idempotent: re-running with the same args refreshes deps without destroying
# the existing venv.
#
# Environment variables:
#   ROBOEVAL_VENDORS_DIR   Where to clone non-PyPI repos (default: ~/.local/share/roboeval/vendors)
#   ROBOEVAL_SAPIEN_WHL    Local SAPIEN wheel override for RoboTwin
#   ROBOEVAL_SAPIEN_AARCH64_URL
#                           SAPIEN aarch64 wheel URL for RoboTwin
#   LIBERO_PRO_DIR          Override LIBERO-PRO clone path
#   SKIP_SYSTEM_DEPS        Set to 1 to skip apt-get installs
#   DRY_RUN                 Set to 1 to print the plan without executing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENVS_DIR="$PROJECT_ROOT/.venvs"
VENDORS_DIR="${ROBOEVAL_VENDORS_DIR:-$HOME/.local/share/roboeval/vendors}"
INSTALL_DOCS="$PROJECT_ROOT/docs/install.md"
ROBOEVAL_SAPIEN_AARCH64_URL="${ROBOEVAL_SAPIEN_AARCH64_URL:-https://github.com/haosulab/SAPIEN/releases/download/3.0.3/sapien-3.0.3-cp310-cp310-linux_aarch64.whl}"
ROBOEVAL_SAPIEN_AARCH64_SHA256="${ROBOEVAL_SAPIEN_AARCH64_SHA256:-9df50de3c2e018695313a41ba470b67b0a8f1c98b489e983672a011561d70598}"

SKIP_SYSTEM_DEPS="${SKIP_SYSTEM_DEPS:-0}"
DRY_RUN="${DRY_RUN:-0}"
# Path to a local cosmos-policy clone. Settable via:
#   --with-cosmos-policy=<path>   CLI flag
#   COSMOS_POLICY_PATH=<path>     env var
# If neither is set, setup_cosmos() auto-clones from GitHub.
COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-}"
# Path to a local Isaac-GR00T clone. Settable via:
#   --with-isaac-groot=<path>   CLI flag
#   ISAAC_GROOT_PATH=<path>     env var
# If neither is set, setup_groot() auto-clones from GitHub (~5 GB checkout).
ISAAC_GROOT_PATH="${ISAAC_GROOT_PATH:-}"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[roboeval]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[roboeval WARN]${NC} $*"; }
log_error() { echo -e "${RED}[roboeval ERROR]${NC} $*" >&2; }
log_step()  { echo -e "${CYAN}▶${NC} $*"; }
dry_echo()  { echo -e "${YELLOW}[DRY-RUN]${NC} $*"; }

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------
# Format: NAME:PYTHON_VERSION:INSTALL_METHOD
#   install method: "extra"  → uv pip install -e ".[<name>]"
#                   "reqs"   → use requirements/<name>.txt + requirements/sim-server.txt
#                   "extra+manual" → extra + post-install instructions
#                   "reqs+manual"  → reqs + post-install instructions

declare -A COMPONENT_PYTHON=(
    [pi05]="3.12"       [openvla]="3.11"    [smolvla]="3.11"
    [groot]="3.12"      [cosmos]="3.10"     [internvla]="3.11"
    [libero]="3.8"      [libero_pro]="3.8"
    [robocasa]="3.11"   [robotwin]="3.10"   [aloha_gym]="3.10"
    [vlm]="3.11"
    [diffusion_policy]="3.11"
    [gym_pusht]="3.11"
    [vqbet]="3.11"
    [act]="3.11"
    [tdmpc2]="3.11"
    [maniskill2]="3.10"
    [metaworld]="3.11"
    [libero_infinity]="3.11"
)

declare -A COMPONENT_METHOD=(
    [pi05]="extra"          [openvla]="extra"       [smolvla]="extra"
    [groot]="extra+manual"  [cosmos]="extra+manual" [internvla]="extra"
    [libero]="reqs"         [libero_pro]="reqs+manual"
    [robocasa]="extra"      [robotwin]="reqs+manual" [aloha_gym]="extra"
    [vlm]="extra"
    [diffusion_policy]="extra"
    [gym_pusht]="extra"
    [vqbet]="extra"
    [act]="extra"
    [tdmpc2]="extra"
    [maniskill2]="extra+manual"
    [metaworld]="extra"
    [libero_infinity]="reqs+manual"
)

# Key import to validate after install
declare -A COMPONENT_IMPORT=(
    [pi05]="lerobot"        [openvla]="transformers"    [smolvla]="lerobot"
    [groot]="gr00t"         [cosmos]="cosmos_policy"    [internvla]="lerobot"
    [libero]="libero"       [libero_pro]="bddl"
    [robocasa]="robocasa"   [robotwin]="torch"          [aloha_gym]="gym_aloha"
    [vlm]="litellm"
    [diffusion_policy]="lerobot"
    [gym_pusht]="gym_pusht"
    [vqbet]="lerobot"
    [act]="lerobot"
    [tdmpc2]="lerobot"
    [maniskill2]="numpy"
    [metaworld]="metaworld"
    [libero_infinity]="libero_infinity"
)

ALL_VLAS=(pi05 openvla smolvla groot cosmos internvla)
ALL_SIMS=(libero libero_pro robocasa robotwin aloha_gym)
ALL_COMPONENTS=("${ALL_VLAS[@]}" "${ALL_SIMS[@]}" vlm)
ALL_VLAS+=(diffusion_policy)
ALL_COMPONENTS+=(diffusion_policy)
ALL_SIMS+=(gym_pusht)
ALL_COMPONENTS+=(gym_pusht)
ALL_VLAS+=(vqbet)
ALL_COMPONENTS+=(vqbet)
ALL_VLAS+=(act)
ALL_COMPONENTS+=(act)
ALL_VLAS+=(tdmpc2)
ALL_COMPONENTS+=(tdmpc2)
ALL_SIMS+=(maniskill2)
ALL_COMPONENTS+=(maniskill2)
ALL_SIMS+=(metaworld)
ALL_COMPONENTS+=(metaworld)
ALL_SIMS+=(libero_infinity)
ALL_COMPONENTS+=(libero_infinity)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    log_error "No components specified."
    echo ""
    echo "Usage: $0 <component> [<component> ...]"
    echo ""
    echo "Components:"
    echo "  VLAs: pi05  openvla  smolvla  groot  cosmos  internvla"
    echo "  Sims: libero  libero_pro  robocasa  robotwin  aloha_gym"
    echo "  Modes: vlm"
    echo "  all   (installs everything)"
    echo ""
    echo "Examples:"
    echo "  $0 pi05 libero"
    echo "  $0 smolvla robocasa"
    echo "  $0 all"
    echo ""
    echo "Options:"
    echo "  --with-cosmos-policy=<path>   path to a local NVlabs/cosmos-policy clone"
    echo "                                (skips auto-clone; useful if you already have it)"
    echo "  --with-isaac-groot=<path>     path to a local NVIDIA/Isaac-GR00T clone"
    echo "                                (skips auto-clone; useful if you already have it)"
    exit 1
fi

REQUESTED=()
for arg in "$@"; do
    if [[ "$arg" == "all" ]]; then
        REQUESTED=("${ALL_COMPONENTS[@]}")
        break
    elif [[ "$arg" == --with-cosmos-policy=* ]]; then
        COSMOS_POLICY_PATH="${arg#--with-cosmos-policy=}"
        log_info "cosmos-policy path override: $COSMOS_POLICY_PATH"
    elif [[ "$arg" == --with-isaac-groot=* ]]; then
        ISAAC_GROOT_PATH="${arg#--with-isaac-groot=}"
        log_info "Isaac-GR00T path override: $ISAAC_GROOT_PATH"
    elif [[ -n "${COMPONENT_PYTHON[$arg]+_}" ]]; then
        REQUESTED+=("$arg")
    else
        log_error "Unknown component: '$arg'"
        log_error "Valid components: ${ALL_COMPONENTS[*]} all"
        log_error "  cosmos option: --with-cosmos-policy=<path>  (local clone of NVlabs/cosmos-policy)"
        log_error "  groot  option: --with-isaac-groot=<path>    (local clone of NVIDIA/Isaac-GR00T)"
        exit 1
    fi
done

# Deduplicate
REQUESTED=($(printf '%s\n' "${REQUESTED[@]}" | awk '!seen[$0]++'))

if [[ "$DRY_RUN" == "1" ]]; then
    dry_echo "Plan — would set up the following components:"
    for c in "${REQUESTED[@]}"; do
        dry_echo "  $c  (Python ${COMPONENT_PYTHON[$c]}, method=${COMPONENT_METHOD[$c]}, venv=.venvs/$c)"
    done
    dry_echo "  roboeval  (Python 3.13, venv=.venvs/roboeval)"
    exit 0
fi

# ---------------------------------------------------------------------------
# uv bootstrap
# ---------------------------------------------------------------------------

ensure_uv() {
    if command -v uv &>/dev/null; then
        log_info "uv $(uv --version 2>&1 | awk '{print $2}') found."
        return
    fi
    log_warn "uv not found — installing via official one-liner..."
    if command -v curl &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &>/dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        log_error "Neither curl nor wget found. Install uv manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    # The installer adds ~/.local/bin to PATH in shell rc files, but not this session
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        log_error "uv install appeared to succeed but 'uv' is still not on PATH."
        log_error "Run: export PATH=\"\$HOME/.local/bin:\$PATH\"  then re-run this script."
        exit 1
    fi
    log_info "uv installed successfully."
}

ensure_uv

UV="$(command -v uv)"

# ---------------------------------------------------------------------------
# micromamba bootstrap
# ---------------------------------------------------------------------------
# Only invoked by setup functions that need conda-forge CUDA packages (groot).
# uv is the default tool for all other envs (14+ pure-PyPI venvs).
#
# Why two tools:
#   PyPI does NOT ship aarch64 binary wheels for flash-attn (sdist only; source
#   build takes ~20 min on aarch64 and frequently fails on newer gcc).  conda-forge
#   distributes pre-built flash-attn 2.8.3 + pytorch 2.10.0 for aarch64+CUDA13 as
#   ordinary binary packages.  micromamba is the lightest path to conda-forge without
#   full conda/mamba overhead (~3 MB single binary, no base env, no solver GUI).
#   uv stays default for every other venv because it is 10× faster and conda solver
#   complexity is unnecessary for pure-PyPI closures.
# ---------------------------------------------------------------------------

ensure_micromamba() {
    # Populate MICROMAMBA with path to the micromamba binary.
    # Exits with a helpful error if micromamba is not installed.
    if command -v micromamba &>/dev/null; then
        MICROMAMBA="$(command -v micromamba)"
    elif [[ -x "$HOME/.local/bin/micromamba" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
        MICROMAMBA="$HOME/.local/bin/micromamba"
    else
        log_error "micromamba not found — it is required for this VLA (groot)."
        log_error "  Install: \"\${SHELL}\" <(curl -L micro.mamba.pm/install.sh)"
        log_error "  See: README.md § Prerequisites  or  $INSTALL_DOCS#prerequisites"
        exit 1
    fi
    log_info "micromamba $("$MICROMAMBA" --version 2>&1) found at $MICROMAMBA"
}

make_micromamba_env() {
    # make_micromamba_env <name> [conda packages...]
    # Creates a micromamba env at VENVS_DIR/<name> from conda-forge.
    # Exposes a standard bin/python — all uv_pip <name> calls work unchanged.
    #
    # Post-create: writes sitecustomize.py to remove user site-packages from
    # sys.path.  Python's site.py adds user site-packages BEFORE running
    # sitecustomize, so we remove it from sys.path there rather than
    # preventing the addition.  Without this, packages in ~/.local/lib/python*/
    # site-packages (e.g. a system-wide tokenizers) shadow the venv's pinned
    # versions and silently break pinned-dep installs.
    local name="$1"; shift
    local conda_pkgs=("$@")
    local venv="$VENVS_DIR/$name"
    if [[ -d "$venv" ]] && [[ -f "$venv/bin/python" ]]; then
        log_info ".venvs/$name already exists (micromamba env) — skipping create."
    else
        log_step "Creating .venvs/$name via micromamba (conda-forge: ${conda_pkgs[*]})..."
        "$MICROMAMBA" create -p "$venv" -c conda-forge -y "${conda_pkgs[@]}" || {
            log_error "micromamba create failed for .venvs/$name."
            log_error "  Packages requested: ${conda_pkgs[*]}"
            log_error "  Ensure conda-forge is reachable and CUDA is available."
            log_error "  See: $INSTALL_DOCS"
            exit 1
        }
        log_info ".venvs/$name created ($(du -sh "$venv" 2>/dev/null | cut -f1 || echo '?') on disk)."
    fi
    # Write sitecustomize.py to prevent user site-packages bleed-in.
    local _site_pkgs
    _site_pkgs="$("$venv/bin/python" -c \
        "import site; sp=[p for p in site.getsitepackages() if '$name' in p]; print(sp[0] if sp else '')" \
        2>/dev/null || true)"
    if [[ -n "$_site_pkgs" ]] && [[ ! -f "$_site_pkgs/sitecustomize.py" ]]; then
        cat > "$_site_pkgs/sitecustomize.py" << 'SITEPY'
# Auto-generated by make_micromamba_env() in scripts/setup.sh.
# Prevents ~/.local/lib/python*/site-packages from shadowing this
# micromamba-managed venv's pinned packages.  Python's site.py adds user
# site-packages BEFORE execsitecustomize(), so we remove it from sys.path here.
import sys as _sys
import site as _site
try:
    _user_site = _site.getusersitepackages()
    if _user_site in _sys.path:
        _sys.path.remove(_user_site)
    _site.ENABLE_USER_SITE = False
except Exception:
    pass
SITEPY
        log_info "  sitecustomize.py written — user site-packages isolated."
    fi
}

MICROMAMBA=""  # populated by ensure_micromamba() within setup functions that need it

# ---------------------------------------------------------------------------
# CUDA torch index detection
# ---------------------------------------------------------------------------
# PyPI's default `torch` wheel is CPU-only on aarch64 and on most platforms
# for newer torch versions. To get a CUDA-enabled build we have to point uv
# at PyTorch's CUDA index. Auto-detect the right one from `nvidia-smi`'s
# CUDA Version field; user can override via TORCH_CUDA_INDEX.
if [[ -z "${TORCH_CUDA_INDEX:-}" ]] && command -v nvidia-smi &>/dev/null; then
    _CUDA_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    _CUDA_MAJOR="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -1)"
    case "$_CUDA_MAJOR" in
        13.*) TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu130" ;;
        12.*) TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124" ;;
        11.*) TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu118" ;;
    esac
    [[ -n "${TORCH_CUDA_INDEX:-}" ]] && log_info "Detected CUDA $_CUDA_MAJOR; will install torch from $TORCH_CUDA_INDEX"
fi

install_cuda_torch() {
    # install_cuda_torch <venv_name>
    local name="$1"
    if [[ -z "${TORCH_CUDA_INDEX:-}" ]]; then
        log_warn "No TORCH_CUDA_INDEX set and nvidia-smi not detected; torch will be CPU-only in .venvs/$name."
        return 0
    fi
    log_step "Installing CUDA torch into .venvs/$name from $TORCH_CUDA_INDEX..."
    "$UV" pip install --python "$(venv_python "$name")" \
        --extra-index-url "$TORCH_CUDA_INDEX" \
        --reinstall-package torch \
        --reinstall-package torchvision \
        torch torchvision 2>&1 | tail -5 || \
        log_warn "CUDA torch install failed for $name — falling back to CPU torch."
}

# ---------------------------------------------------------------------------
# System dependency check
# ---------------------------------------------------------------------------

check_system_deps() {
    local missing=()

    # CUDA check — warn only (some sims are CPU-capable)
    if ! command -v nvidia-smi &>/dev/null; then
        log_warn "nvidia-smi not found — GPU-based VLAs (pi05, openvla, smolvla) require CUDA."
    fi

    # EGL — required for headless MuJoCo (LIBERO, RoboCasa)
    if ! ldconfig -p 2>/dev/null | grep -q libEGL; then
        log_warn "libEGL not found in ldconfig. Headless MuJoCo rendering may fail."
        log_warn "  Fix: sudo apt-get install libegl1-mesa-dev  (Ubuntu/Debian)"
        log_warn "  See: $INSTALL_DOCS"
    fi

    # HDF5 — required for LIBERO / LIBERO-Pro
    local needs_hdf5=0
    for c in "${REQUESTED[@]}"; do
        [[ "$c" == "libero" || "$c" == "libero_pro" ]] && needs_hdf5=1
    done
    if [[ "$needs_hdf5" == "1" ]] && ! (python3 -c "import h5py" 2>/dev/null || true); then
        if ! ldconfig -p 2>/dev/null | grep -q libhdf5; then
            log_warn "libhdf5 not found. h5py may fail to build on aarch64."
            log_warn "  Fix: sudo apt-get install libhdf5-dev  (or see docs/install.md)"
        fi
    fi

    if [[ "$SKIP_SYSTEM_DEPS" == "0" ]] && command -v apt-get &>/dev/null; then
        log_info "Installing system packages (EGL, patchelf, cmake)..."
        sudo apt-get install -y -qq \
            libegl1-mesa-dev \
            libgl1-mesa-dev \
            libgles2-mesa-dev \
            libosmesa6-dev \
            libvulkan-dev \
            libglfw3-dev \
            patchelf \
            cmake \
            build-essential 2>/dev/null || \
            log_warn "apt-get install encountered issues — some system deps may be missing."
    fi
}

check_system_deps

mkdir -p "$VENVS_DIR"
mkdir -p "$VENDORS_DIR"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

venv_python() {
    local name="$1"
    echo "$VENVS_DIR/$name/bin/python"
}

venv_pip() {
    local name="$1"
    echo "$VENVS_DIR/$name/bin/pip"
}

make_venv() {
    local name="$1"
    local pyver="$2"
    local venv="$VENVS_DIR/$name"
    if [[ -d "$venv" ]]; then
        log_info ".venvs/$name already exists — refreshing."
    else
        log_step "Creating .venvs/$name (Python $pyver)..."
        "$UV" venv "$venv" --python "$pyver" --seed || {
            log_error "Failed to create venv for $name with Python $pyver."
            log_error "Is Python $pyver installed?  Check: python$pyver --version"
            log_error "Install guide: $INSTALL_DOCS"
            exit 1
        }
    fi
}

uv_pip() {
    # uv_pip <venv_name> <pip args...>
    local name="$1"; shift
    "$UV" pip install --python "$(venv_python "$name")" "$@"
}

validate_import() {
    local name="$1"
    local pkg="$2"
    log_step "Validating $name: python -c 'import $pkg'..."
    if "$(venv_python "$name")" -c "import $pkg" 2>/dev/null; then
        log_info "  ✓  $name → $pkg"
    else
        log_warn "  ✗  $name → $pkg import failed."
        log_warn "     This may be expected for packages not yet downloaded (e.g. GR00T, Cosmos)."
        log_warn "     See docs/install.md for post-install steps."
    fi
}

download_file() {
    # download_file <url> <dest>
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    "$(venv_python robotwin)" - "$url" "$dest" <<'PY'
import pathlib
import sys
from urllib.request import Request, urlopen

url, dest = sys.argv[1], pathlib.Path(sys.argv[2])
tmp = dest.with_suffix(dest.suffix + ".tmp")
req = Request(url, headers={"User-Agent": "roboeval-setup"})
with urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
    while True:
        chunk = resp.read(1024 * 1024)
        if not chunk:
            break
        f.write(chunk)
tmp.replace(dest)
PY
}

repair_sapien_librt_if_needed() {
    # SAPIEN 3.0.3's aarch64 wheel bundles an auditwheel-repaired librt that
    # can reference GLIBC_PRIVATE symbols on Ubuntu 22.04/aarch64.  Replacing it
    # with the system librt matches SAPIEN's own Docker install guidance.
    local name="$1"
    [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]] || return 0

    local site_packages
    site_packages="$("$VENVS_DIR/$name/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
    local sapien_libs
    sapien_libs="$(find "$site_packages" -maxdepth 2 -type d -name sapien.libs | head -1)"
    [[ -n "$sapien_libs" && -e "$sapien_libs/librt-2-4cc790af.28.so" ]] || return 0

    log_step "Repairing SAPIEN bundled librt for Linux/aarch64..."
    rm -f "$sapien_libs/librt-2-4cc790af.28.so"
    ln -s /lib/aarch64-linux-gnu/librt.so.1 "$sapien_libs/librt-2-4cc790af.28.so"
}

# ---------------------------------------------------------------------------
# VLA setup functions (Python 3.11)
# ---------------------------------------------------------------------------

setup_pi05() {
    # lerobot>=0.5.0 requires Python>=3.12; use 3.12 venv.
    # lerobot[pi] is NOT in pyproject.toml extras (see note there) because it
    # conflicts with openvla's transformers<5.0 constraint in the `all` extra.
    # We install lerobot 0.4.5 from git (last version with Python>=3.10 support
    # and the fix/lerobot_openpi transformers branch for pi05).
    make_venv pi05 "3.12"
    log_step "Installing Pi 0.5 base deps into .venvs/pi05..."
    uv_pip pi05 -e "$PROJECT_ROOT[pi05]"

    log_step "Installing lerobot[pi] 0.4.4 into .venvs/pi05..."
    log_info "  (git+https://github.com/huggingface/lerobot.git@v0.4.4)"
    uv_pip pi05 \
        "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.4" || {
        log_warn "lerobot[pi] v0.4.4 tag not found — trying HEAD."
        log_warn "HEAD is lerobot>=0.5 and requires Python>=3.12 + transformers>=5.3."
        log_warn "The pi05 server may work, but action preprocessing was validated on 0.4.4."
        uv_pip pi05 \
            "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git"
    }

    install_cuda_torch pi05
    log_info "Pi 0.5 venv ready. Models downloaded on first use (~7.4 GB)."
    log_info "  lerobot/pi05_libero_finetuned  — LIBERO tasks"
    log_info "  lerobot/pi05_base              — base model"
}

setup_openvla() {
    make_venv openvla "3.11"
    log_step "Installing OpenVLA deps into .venvs/openvla..."
    uv_pip openvla -e "$PROJECT_ROOT[openvla]"
    install_cuda_torch openvla
    log_info "OpenVLA venv ready. Model downloaded on first use (~14 GB)."
    log_info "  openvla/openvla-7b"
}

setup_smolvla() {
    # lerobot[smolvla]==0.4.4 pinned — last PyPI version with Python>=3.10 support.
    make_venv smolvla "3.11"
    log_step "Installing SmolVLA deps into .venvs/smolvla..."
    uv_pip smolvla -e "$PROJECT_ROOT[smolvla]"
    install_cuda_torch smolvla
    log_info "SmolVLA venv ready."
}

setup_groot() {
    # GR00T (N1.6) uses a **micromamba env** (not a uv venv) because:
    #   - flash-attn 2.8.3 has no aarch64 binary wheel on PyPI — only an sdist
    #     whose source build takes ~20 min on aarch64 and often fails on gcc ≥ 13.
    #   - conda-forge ships pre-built binary wheels for both pytorch=2.10.0 and
    #     flash-attn=2.8.3 targeting aarch64+CUDA13 (sm_121/Blackwell PTX JIT OK).
    #
    # Recommended Blackwell package set for GB10 / sm_121 systems:
    #   pytorch=2.10.0=cuda130_generic_py312_h18d3ae0_203  (~5.5 GB)
    #   flash-attn=2.8.3=py312hb503c49_3                  (~1.2 GB)
    # Total env ~7 GB; ~45 s first install from warm conda-forge cache.
    # PTX JIT first-call: 516 ms; subsequent: 0.2 ms — non-fatal sm_121 warning.
    #
    # The env lives at .venvs/groot/ with a standard bin/python so all downstream
    # `uv pip install --python .venvs/groot/bin/python` calls work unchanged.
    #
    # NOTE: nvidia-cudss-cu13 is NOT installed separately here — conda-forge's
    # pytorch=2.10.0 pulls libcudss and other CUDA runtime deps as conda packages
    # automatically.  (That package was only needed for the old Jetson SBSA torch
    # build's baked-in RPATH; conda-forge bundles its own CUDA libs differently.)
    #
    # GR00T N1.6 pin (public main, 2026-02-03):
    local _GROOT_PIN="e29d8fc50b0e4745120ae3fb72447986fe638aa6"
    ensure_micromamba
    make_micromamba_env groot "python=3.12" "pytorch=2.10.0" "flash-attn=2.8.3"
    # On aarch64 (e.g. NVIDIA GB10 / DGX Spark), GR00T's scipy dependency
    # requires libnvpl_lapack_lp64_gomp.so.0.  The `nvpl` package on PyPI ships
    # the NVIDIA Performance Libraries as a Python wheel — no sudo, no apt, no
    # ~/.local/lib/nvpl/ directory required.  Installing it into the groot venv
    # puts the .so files in site-packages where the dynamic linker finds them
    # automatically.  On x86_64 the wheel is a no-op stub (~8 kB).
    # NOTE: scipy is still installed via uv pip (PyPI), not conda, so it needs
    # external BLAS/LAPACK on aarch64 — nvpl provides that.
    log_step "Installing nvpl (NVIDIA Performance Libraries) into .venvs/groot..."
    uv_pip groot "nvpl==25.11" || {
        log_warn "  nvpl==25.11 install failed — continuing anyway."
        log_warn "  On aarch64 this may cause libnvpl_lapack_lp64_gomp.so.0 errors."
        log_warn "  See: $INSTALL_DOCS#groot"
    }
    log_info "  nvpl==25.11 installed — libnvpl_*.so bundled in site-packages."
    # gr00t (NVIDIA Isaac-GR00T) is NOT on PyPI.
    # Auto-clone from https://github.com/NVIDIA/Isaac-GR00T and check out the
    # pinned N1.6 commit, then install editable with --no-deps (torch/flash-attn
    # already provided by the conda-forge micromamba env above).
    # Pass --with-isaac-groot=<path> (or ISAAC_GROOT_PATH=<path>) to reuse a
    # local clone (it will still be checked out at the pinned commit).
    local _IG_DIR="${ISAAC_GROOT_PATH:-$VENDORS_DIR/Isaac-GR00T}"

    if [[ -n "${ISAAC_GROOT_PATH:-}" ]]; then
        if [[ ! -d "$_IG_DIR" ]]; then
            log_error "--with-isaac-groot path does not exist: $_IG_DIR"
            log_error "  Provide a valid path to a local clone of:"
            log_error "    https://github.com/NVIDIA/Isaac-GR00T"
            exit 1
        fi
        log_info "Using user-supplied Isaac-GR00T clone: $_IG_DIR"
    else
        if [[ ! -d "$_IG_DIR/.git" ]]; then
            log_step "Cloning NVIDIA/Isaac-GR00T (full history, needed to checkout pinned commit) into $_IG_DIR..."
            git clone https://github.com/NVIDIA/Isaac-GR00T.git "$_IG_DIR" || {
                log_error "Isaac-GR00T git clone failed."
                log_error "  If the repo requires auth or you have a local copy, use:"
                log_error "    ./scripts/setup.sh groot --with-isaac-groot=<path>"
                log_error "  See: $INSTALL_DOCS#groot"
                exit 1
            }
        else
            log_info "Isaac-GR00T already at $_IG_DIR — fetching to ensure pin is reachable."
            (cd "$_IG_DIR" && git fetch --tags --quiet origin) || \
                log_warn "  git fetch failed — continuing with on-disk state."
        fi
    fi

    # Reset to a clean working tree before checking out the pin.  On re-runs the
    # vendor dir may be dirty: editable-install or previous patch steps can
    # modify files in-place (e.g. transformers-compat patches applied in an
    # earlier N1.7 era).  `git checkout` refuses to clobber modified tracked
    # files, causing setup to abort.  Because this is a build-only vendor clone
    # (patches are re-applied below) it is safe to hard-reset + scrub untracked
    # files before pinning.
    log_step "Cleaning Isaac-GR00T working tree before pin checkout (reset --hard + clean -fd)..."
    (cd "$_IG_DIR" && git reset --hard && git clean -fd) || \
        log_warn "  git reset/clean in $_IG_DIR failed — checkout may still succeed if tree is already clean."

    log_step "Checking out N1.6 pin: $_GROOT_PIN ..."
    (cd "$_IG_DIR" && git -c advice.detachedHead=false checkout "$_GROOT_PIN") || {
        log_error "Failed to checkout pinned N1.6 commit $_GROOT_PIN in $_IG_DIR"
        log_error "  This commit must be reachable in the clone.  If you supplied"
        log_error "  --with-isaac-groot, ensure the clone is not shallow."
        exit 1
    }

    log_step "Installing gr00t package (--no-deps, torch+flash-attn from micromamba) into .venvs/groot..."
    "$UV" pip install --python "$(venv_python groot)" --no-deps -e "$_IG_DIR" || {
        log_error "gr00t editable install failed."
        log_error "  Dir: $_IG_DIR (commit $_GROOT_PIN)"
        log_error "  See: $INSTALL_DOCS#groot"
        exit 1
    }

    # N1.6 inference-time Python deps (matches gr00t's pyproject at e29d8fc,
    # excluding torch/torchvision/flash-attn which are already provided by the
    # micromamba conda-forge env above).
    # transformers==4.51.3 is the critical pin: it is what N1.6 was built and
    # tested against, and it is the version that eliminates all 9
    # N1.7→transformers-5.x vendor patches.
    log_step "Installing N1.6 inference deps (transformers==4.51.3, diffusers, peft, ...) into .venvs/groot..."
    uv_pip groot \
        "transformers==4.51.3" \
        "diffusers==0.35.1" \
        "peft==0.17.1" \
        "albumentations==1.4.18" \
        "av==15.0.0" \
        "dm-tree==0.1.8" \
        "msgpack==1.1.0" \
        "msgpack-numpy==0.4.8" \
        "lmdb==1.7.5" \
        "pandas==2.2.3" \
        "termcolor==3.2.0" \
        "tyro==0.9.17" \
        "click==8.1.8" \
        "datasets==3.6.0" \
        "einops==0.8.1" \
        "gymnasium==1.2.2" \
        "matplotlib==3.10.1" \
        "numpy==1.26.4" \
        "omegaconf==2.3.0" \
        "scipy==1.15.3" \
        "wandb==0.23.0" \
        "pyzmq==27.0.1" \
        "imageio[ffmpeg]" \
        "opencv-python-headless>=4.9.0" \
        "tensorboardX" \
        "timm>=0.9.16" || {
        log_warn "Some gr00t N1.6 inference deps failed to install — check output above."
        log_warn "  See: $INSTALL_DOCS#groot"
    }

    # Roboeval's own [groot] extras (FastAPI server + aarch64 lib provisioners).
    log_step "Installing roboeval [groot] extras (server deps) into .venvs/groot..."
    uv_pip groot -e "$PROJECT_ROOT[groot]"

    # Validation: confirm the N1.6 GR00T model class is importable.
    log_step "Validating gr00t install (N1.6)..."
    if "$(venv_python groot)" -c \
        "from gr00t.policy.gr00t_policy import Gr00tPolicy; from gr00t.data.embodiment_tags import EmbodimentTag; print('gr00t N1.6 OK')" \
        2>/dev/null; then
        log_info "  ✓  groot → gr00t.policy.gr00t_policy.Gr00tPolicy (N1.6 @ $_GROOT_PIN)"
    else
        log_warn "  ✗  groot → gr00t import failed."
        log_warn "     Debug: $(venv_python groot) -c 'from gr00t.policy.gr00t_policy import Gr00tPolicy'"
        log_warn "     See: $INSTALL_DOCS#groot"
    fi

    log_info "GR00T (N1.6) venv ready. Model downloaded on first use."
    log_info "  LIBERO:   0xAnkitSingh/GR00T-N1.6-LIBERO  (94.9% LIBERO avg per community fine-tune)"
    log_info "  RoboCasa: nvidia/GR00T-N1.6-3B           (foundation, robocasa_panda_omron tag)"
}

setup_cosmos() {
    # cosmos-policy's config-loading scans ALL experiment configs at startup via
    # import_all_modules_from_package("...video2world.experiment").  One of those
    # configs (model_14b_reason_1p1.py) imports dataset_provider → video_decoder →
    # decord.  decord has no aarch64 PyPI wheel and no working source-build path on
    # Tegra GB10 (complex ffmpeg + cuda deps).  conda-forge ships a prebuilt
    # linux-aarch64 decord 0.6.0 (np2py310 build, verified: micromamba search -c
    # conda-forge decord → "decord 0.6.0 np2py310h4c95643_2 conda-forge linux-aarch64").
    #
    # Use a micromamba env so the required binary packages are available on aarch64.
    # python=3.10 is required — NVIDIA's custom index only ships aarch64 cp310 wheels
    # for transformer-engine (TE 2.8) and flash-attn (2.7.4.post1).  NVIDIA CUDA
    # wheels (torch 2.9+cu130, TE, flash-attn) are still installed via uv_pip into
    # the micromamba env — conda-forge does NOT have them.
    #
    # NOTE: if .venvs/cosmos exists as a legacy UV venv (no conda-meta/), remove it
    # first:  rm -rf .venvs/cosmos  then re-run this setup.

    # NVIDIA's custom index for aarch64 cp310 CUDA-13.0 wheels.
    # transformer-engine (TE 2.8) + flash-attn (2.7.4.post1) + torch 2.9+cu130.
    # These are required for GPU inference — te.pytorch.RMSNorm/Linear used in
    # model architecture, not training-only.  aarch64 cp311/cp312 TE wheels don't exist.
    _NVIDIA_COSMOS_INDEX="https://nvidia-cosmos.github.io/cosmos-dependencies/v1.2.0/cu130_torch29/simple"

    ensure_micromamba
    make_micromamba_env cosmos "python=3.10" "decord=0.6.0"
    log_step "Installing Cosmos base deps into .venvs/cosmos (Python 3.10)..."
    # Cannot use -e .[cosmos] — requires-python=>=3.11 blocks it in 3.10 venv.
    # Install the cosmos extra's deps directly (mirrors pyproject.toml cosmos = [...]).
    uv_pip cosmos \
        "pillow>=9.0" \
        "numpy>=1.24" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "huggingface-hub>=0.20" \
        "fvcore>=0.1.5.post20221221" \
        "iopath>=0.1.10" \
        "nvidia-ml-py>=13.580.82" \
        "wandb>=0.12" \
        "boto3>=1.20" \
        "multi-storage-client>=0.32" \
        "pandas>=1.3" || {
        log_warn "Some cosmos base deps failed — check output above."
    }

    # Install torch 2.9+cu130 from NVIDIA's index (TE/flash-attn compiled against torch29)
    log_step "Installing torch 2.9+cu130 (cosmos GPU stack) from NVIDIA index..."
    "$UV" pip install --python "$(venv_python cosmos)" \
        --extra-index-url "$_NVIDIA_COSMOS_INDEX" \
        "torch==2.9.0+cu130" "torchvision" 2>&1 | tail -5 || \
        log_warn "cosmos torch 2.9 install failed — GPU inference will not work."
    # cosmos_policy is NOT on PyPI. Auto-clone from https://github.com/NVlabs/cosmos-policy
    # (Apache-2.0, public) and install editable with --no-deps. The CUDA-accelerated extras
    # (flash-attn, transformer-engine, natten) require NVIDIA's custom index and are NOT
    # installed here. See docs/install.md#cosmos for the full GPU inference setup.
    local _CP_DIR="${COSMOS_POLICY_PATH:-$VENDORS_DIR/cosmos-policy}"

    if [[ -n "${COSMOS_POLICY_PATH:-}" ]]; then
        # User supplied a local clone via --with-cosmos-policy=<path> or COSMOS_POLICY_PATH
        if [[ ! -d "$_CP_DIR" ]]; then
            log_error "--with-cosmos-policy path does not exist: $_CP_DIR"
            log_error "  Provide a valid path to a local clone of:"
            log_error "    https://github.com/NVlabs/cosmos-policy"
            exit 1
        fi
        log_info "Using user-supplied cosmos-policy clone: $_CP_DIR"
    else
        # Auto-clone from GitHub
        if [[ ! -d "$_CP_DIR/.git" ]]; then
            log_step "Cloning NVlabs/cosmos-policy into $_CP_DIR..."
            git clone https://github.com/NVlabs/cosmos-policy.git "$_CP_DIR" || {
                log_error "cosmos-policy git clone failed."
                log_error "  If the repo requires auth or you have a local copy, use:"
                log_error "    ./scripts/setup.sh cosmos --with-cosmos-policy=<path>"
                log_error "  See: $INSTALL_DOCS#cosmos"
                exit 1
            }
        else
            log_info "cosmos-policy already at $_CP_DIR — skipping clone."
        fi
    fi

    log_step "Installing cosmos_policy package (no-deps) into .venvs/cosmos..."
    "$UV" pip install --python "$(venv_python cosmos)" -e "$_CP_DIR" --no-deps || {
        log_error "cosmos_policy editable install failed."
        log_error "  Dir: $_CP_DIR"
        log_error "  See: $INSTALL_DOCS#cosmos"
        exit 1
    }

    # Install the base (non-GPU) deps required for cosmos_policy submodule imports.
    # Heavy GPU extras (flash-attn==2.7.3, transformer-engine==2.2.0, natten==0.21.0)
    # need NVIDIA's custom wheel index — see docs/install.md#cosmos for those.
    log_step "Installing cosmos_policy base deps (non-GPU) into .venvs/cosmos..."
    uv_pip cosmos \
        "attrs>=25.4.0" \
        "av>=16.0.1" \
        "click>=8.3.0" \
        "diffusers>=0.35.2" \
        "einops>=0.8.1" \
        "ftfy>=6.3.1" \
        "h5py" \
        "huggingface-hub>=0.36.0" \
        "hydra-core>=1.3.2" \
        "imageio[ffmpeg]>=2.37.0" \
        "loguru>=0.7.3" \
        "omegaconf>=2.3.0" \
        "opencv-python-headless>=4.11.0.86" \
        "peft>=0.17.1" \
        "pillow>=11.0.0" \
        "pyyaml>=6.0.3" \
        "safetensors>=0.6.2" \
        "scikit-image>=0.25.0" \
        "scipy>=1.15.0" \
        "sentencepiece>=0.2.0" \
        "tqdm>=4.67.1" \
        "transformers>=4.57.1" \
        "tyro>=0.9.35" \
        "typing-extensions>=4.14.0" \
        "fvcore>=0.1.5.post20221221" \
        "iopath>=0.1.10" \
        "nvidia-ml-py>=13.580.82" || {
        log_warn "Some cosmos_policy base deps failed to install — check output above."
    }
    # Additional runtime dependencies reached by the model-loader import path.
    # Import chain (continued):
    #   callback.py → wandb
    #   easy_io/backends/boto3_client.py → boto3
    #   easy_io/backends/msc_backend.py → multistorageclient (from multi-storage-client)
    #   easy_io/handlers/pandas_handler.py → pandas
    uv_pip cosmos \
        "wandb>=0.12" \
        "boto3>=1.20" \
        "multi-storage-client>=0.32" \
        "pandas>=1.3" || {
        log_warn "Some cosmos_policy runtime_continued deps failed — check output above."
    }

    # cosmos_policy imports `megatron.core.parallel_state` in
    #   39 files. megatron-core ships aarch64 manylinux cp310 wheels on PyPI.
    log_step "Installing megatron-core into .venvs/cosmos (G3c fix)..."
    uv_pip cosmos "megatron-core>=0.14.0" || {
        log_warn "megatron-core install failed — cosmos GPU inference will be broken."
    }

    # megatron-core's import chain reaches setuptools via
    #   megatron.core → distributed_data_parallel → param_and_grad_buffer →
    #   nccl_allocator → torch.utils.cpp_extension → setuptools.
    # uv venvs do NOT include setuptools by default, so cosmos_policy fails at
    # runtime with: ModuleNotFoundError: No module named 'setuptools'.
    log_step "Installing setuptools into .venvs/cosmos (G3c-setuptools fix)..."
    uv_pip cosmos setuptools || {
        log_warn "setuptools install failed — cosmos_policy will fail to import megatron.core."
    }

    # transformer-engine + flash-attn GPU wheels (aarch64 cp310 only).
    # These are required for cosmos model inference — te.pytorch.RMSNorm etc. are
    # used in the model network architecture, not just training.
    # NVIDIA's custom index has cp310 aarch64 wheels for cu130/torch29.
    log_step "Installing transformer-engine + flash-attn from NVIDIA index (G4c fix)..."
    "$UV" pip install --python "$(venv_python cosmos)" \
        --extra-index-url "$_NVIDIA_COSMOS_INDEX" \
        "transformer-engine==2.8" \
        "flash-attn==2.7.4.post1" || {
        log_warn "transformer-engine / flash-attn GPU install failed."
        log_warn "  cosmos GPU inference requires these NVIDIA-compiled wheels."
        log_warn "  Check: CUDA 13.0 + Python 3.10 + aarch64 are required."
    }

    # Additional import-time dependencies.
    log_step "Installing cosmos import-cascade deps (matplotlib, webdataset, pytz)..."
    uv_pip cosmos matplotlib pytz webdataset || {
        log_warn "Some cosmos import-cascade deps failed — check output above."
    }

    log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_warn "cosmos_policy: GPU stack installed (Python 3.10 + CUDA 13.0)"
    log_warn "  Model weights (~10 GB each):"
    log_warn "    nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
    log_warn "    nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    log_warn "  See: $INSTALL_DOCS#cosmos"
    log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Validate: import cosmos_policy (namespace package — no __init__.py; __path__ check)
    log_step "Validating cosmos_policy install..."
    if "$(venv_python cosmos)" -c \
        "import cosmos_policy; print('cosmos_policy OK  path:', list(cosmos_policy.__path__))" \
        2>/dev/null; then
        log_info "  ✓  cosmos → cosmos_policy"
    else
        log_warn "  ✗  cosmos → cosmos_policy import failed."
        log_warn "     Debug: $(venv_python cosmos) -c 'import cosmos_policy'"
        log_warn "     See: $INSTALL_DOCS#cosmos"
    fi

    log_info "Cosmos venv ready. Checkpoint download required before inference."
    log_info "  See: $INSTALL_DOCS#cosmos"
}

setup_internvla() {
    # flash-attn tool decision: stays on uv (NOT micromamba).
    # InternVLA-A1 uses Qwen3VL as its vision-language backbone.  The model's
    # attention implementation is controlled by the transformers AutoModel config
    # field `_attn_implementation`.  InternVLA's inference pipeline (predict_action_chunk)
    # does NOT set _attn_implementation="flash_attention_2" — it relies on the
    # default SDPA path.  No code path in internvla_policy.py or the InternVLA-A1
    # modeling file imports flash_attn directly.  The model runs correctly on SDPA.
    # Switching to micromamba would add 7 GB overhead with no inference benefit.
    # Verdict: uv is sufficient; leave internvla on uv.
    make_venv internvla "3.11"
    log_step "Installing InternVLA base deps into .venvs/internvla..."
    uv_pip internvla -e "$PROJECT_ROOT[internvla]"
    # Auto-clone the InternVLA-A1 lerobot fork and install the full transitive
    # runtime closure without manual intervention.
    #
    # Root cause: InternVLA-A1/pyproject.toml pins huggingface-hub<0.36.0, but the
    # code imports from transformers.models.qwen3_vl (requires transformers>=4.52,
    # which in turn may pull huggingface-hub>=0.36 if using pip's resolver). When
    # `pip install -e /tmp/InternVLA-A1` ran manually it would either downgrade
    # huggingface-hub (purging transformers) or fail mid-install. Additionally,
    # several packages used by InternVLA-A1 source (mediapy, loguru, omegaconf)
    # are NOT listed in its pyproject.toml at all.
    #
    # Fix: clone into VENDORS_DIR, install with --no-deps to bypass the
    # huggingface-hub upper-bound constraint, then install the full runtime
    # closure explicitly including transformers>=4.52, mediapy, loguru, omegaconf.

    local INTERNVLA_DIR="$VENDORS_DIR/InternVLA-A1"
    if [[ ! -d "$INTERNVLA_DIR/.git" ]]; then
        log_step "Cloning InternVLA-A1 into $INTERNVLA_DIR..."
        git clone https://github.com/InternRobotics/InternVLA-A1.git "$INTERNVLA_DIR" || {
            log_error "InternVLA-A1 git clone failed."
            log_error "  Check network connectivity or set ROBOEVAL_VENDORS_DIR."
            log_error "  See: $INSTALL_DOCS#internvla"
            exit 1
        }
    else
        log_info "InternVLA-A1 already at $INTERNVLA_DIR — skipping clone."
    fi

    # Install the InternVLA-A1 lerobot fork WITHOUT its deps to avoid the
    # huggingface-hub<0.36.0 constraint from overriding what we need.
    log_step "Installing InternVLA-A1 lerobot fork (--no-deps) into .venvs/internvla..."
    uv_pip internvla --no-deps -e "$INTERNVLA_DIR" || {
        log_error "InternVLA-A1 editable install (--no-deps) failed."
        log_error "  Dir: $INTERNVLA_DIR"
        log_error "  See: $INSTALL_DOCS#internvla"
        exit 1
    }

    # Full runtime closure: InternVLA-A1 declared deps (with relaxed hf-hub bound) +
    # undeclared deps actually imported by the source: transformers (Qwen3VL),
    # mediapy, loguru, omegaconf.
    log_step "Installing InternVLA-A1 runtime closure into .venvs/internvla..."
    uv_pip internvla \
        "transformers>=4.52,<5.0" \
        "datasets>=4.0.0,<4.2.0" \
        "diffusers>=0.27.2,<0.36.0" \
        "huggingface-hub[hf-transfer,cli]>=0.20" \
        "accelerate>=1.10.0,<2.0.0" \
        "einops>=0.8.0,<0.9.0" \
        "opencv-python-headless>=4.9.0,<4.13.0" \
        "av>=15.0.0,<16.0.0" \
        "jsonlines>=4.0.0,<5.0.0" \
        "packaging>=24.2" \
        "pynput>=1.7.7,<1.9.0" \
        "pyserial>=3.5,<4.0" \
        "wandb>=0.20.0,<0.22.0" \
        "draccus==0.10.0" \
        "gymnasium>=1.1.1,<2.0.0" \
        "rerun-sdk>=0.24.0,<0.27.0" \
        "deepdiff>=7.0.1" \
        "imageio[ffmpeg]>=2.34.0,<3.0.0" \
        "termcolor>=2.4.0" \
        "mediapy" \
        "loguru" \
        "omegaconf" \
        "safetensors>=0.4.0" \
        "tqdm" \
        "torchvision>=0.21.0" || {
        log_warn "Some InternVLA-A1 runtime deps failed — check output above."
        log_warn "  See: $INSTALL_DOCS#internvla"
    }

    install_cuda_torch internvla

    # ---- Post-install DynamicCache.crop patch ----
    # Root cause: InternVLA's sample_actions() has a 10-step flow-matching
    # denoising loop.  Each call to denoise_step() internally calls
    # past_key_values.update() (inside Qwen3VLAttention.forward) which
    # appends suffix tokens to the DynamicCache in-place, even when
    # use_cache=False.  After step 1 the cache grows from 393→444 tokens;
    # step 2 tries to build a 444-token mask but sees 495 keys → crash.
    # Snapshot pref_kv_len before the loop and crop back after each step.
    local MVLA="$INTERNVLA_DIR/src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py"
    if [ -f "$MVLA" ] && ! grep -q "pref_kv_len" "$MVLA"; then
        log_step "Patching InternVLA DynamicCache.crop fix..."
        .venvs/internvla/bin/python - "$MVLA" <<'PYEOF'
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
src = p.read_text()
old = (
    "        x_t = noise\n"
    "        time = torch.tensor(1.0, dtype=torch.float32, device=device)\n"
    "        while time >= -dt / 2:\n"
    "            expanded_time = time.expand(bsize)\n"
    "            v_t = self.denoise_step(\n"
    "                state,\n"
    "                curr_pad_masks,\n"
    "                past_key_values,\n"
    "                max_position_ids, \n"
    "                x_t.to(dtype),\n"
    "                expanded_time.to(dtype),\n"
    "            )\n"
    "            x_t = x_t + dt * v_t\n"
    "            time += dt\n"
)
new = (
    "        x_t = noise\n"
    "        time = torch.tensor(1.0, dtype=torch.float32, device=device)\n"
    "        pref_kv_len = past_key_values.get_seq_length()  # Save pre-denoise cache length (prefix+middle)\n"
    "        while time >= -dt / 2:\n"
    "            expanded_time = time.expand(bsize)\n"
    "            v_t = self.denoise_step(\n"
    "                state,\n"
    "                curr_pad_masks,\n"
    "                past_key_values,\n"
    "                max_position_ids,\n"
    "                x_t.to(dtype),\n"
    "                expanded_time.to(dtype),\n"
    "            )\n"
    "            x_t = x_t + dt * v_t\n"
    "            time += dt\n"
    "            past_key_values.crop(pref_kv_len)  # Prevent KV cache growth between denoise steps\n"
)
if old in src:
    p.write_text(src.replace(old, new))
    print("DynamicCache.crop patch applied.")
else:
    print("WARNING: DynamicCache.crop patch pattern not found — may already be patched or source changed.")
PYEOF
    [[ $? -eq 0 ]] || log_warn "DynamicCache.crop patch script failed."
    else
        log_info "DynamicCache.crop patch already applied or model file not found."
    fi


    log_info "InternVLA venv ready. Model downloaded on first use (~7 GB)."
    log_info "  InternRobotics/InternVLA-A1-3B-RoboTwin"
}
setup_vqbet() {
    # VQ-BeT (Vector-Quantized Behavior Transformer) via lerobot.
    # Canonical checkpoint: lerobot/vqbet_pusht — original BeT paper's PushT
    # benchmark, enabling direct head-to-head against Diffusion Policy on the
    # identical gym_pusht observation/action space.
    #
    # Bundled with the base lerobot package (no [vqbet] extra needed beyond
    # lerobot==0.4.4 itself).  Pure-CPU friendly for the small PushT checkpoint.
    make_venv vqbet "3.11"
    log_step "Installing VQ-BeT base deps into .venvs/vqbet..."
    uv_pip vqbet -e "$PROJECT_ROOT[vqbet]"

    install_cuda_torch vqbet
    log_info "VQ-BeT venv ready. Model downloaded on first use (~50 MB)."
    log_info "  lerobot/vqbet_pusht  — canonical PushT benchmark"
}

# ---------------------------------------------------------------------------
# Sim setup functions
# ---------------------------------------------------------------------------

setup_libero() {
    make_venv libero "3.8"
    log_step "Installing LIBERO deps into .venvs/libero (Python 3.8)..."
    # Cannot use -e .[libero] — requires-python=>=3.11 blocks it in 3.8 venv.
    # Install from requirements files instead.
    uv_pip libero --upgrade pip setuptools wheel
    uv_pip libero \
        "numpy>=1.24,<2.0" \
        "pillow>=9.0" \
        "eval_type_backport>=0.2.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "h5py>=3.8" \
        "bddl>=3.0.0" \
        "mujoco>=2.3.7" \
        "torch>=2.0,<2.5" \
        "easydict" "future" "thop" "transforms3d" "termcolor" \
        "matplotlib" "opencv-python-headless" \
        "cloudpickle" "gym" "tianshou" "imageio"
    # === LIBERO transitive runtime deps — not declared in upstream LIBERO setup.py
    # but required by libero/libero/envs/__init__.py import chain (caught during
    # runtime validation). torch is needed because libero/libero/benchmark/
    # __init__.py imports it at module load. numpy<2.0 because gym/LIBERO break on
    # numpy 2.0. torch<2.5 because pytorch dropped Python 3.8 support in 2.5+. ===
    # Pin to v1.4.1_libero — robosuite master is 1.5.x and requires mujoco>=3.3.0
    # which requires Python>=3.9 (libero venv is 3.8). v1.4.1_libero is the LIBERO-
    # specific branch known to work with mujoco 3.2.3 and Python 3.8.
    uv_pip libero \
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.4.1_libero"

    # LIBERO upstream setup.py uses find_packages() but the repo is missing the
    # top-level libero/__init__.py and libero/configs/__init__.py shims that
    # find_packages() needs to descend. Without them, the wheel is empty.
    # Workaround: clone, add the shims, and install editable from the clone.
    local LIBERO_CLONE="$VENDORS_DIR/LIBERO"
    if [[ ! -d "$LIBERO_CLONE" ]]; then
        log_step "Cloning LIBERO into $LIBERO_CLONE..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_CLONE"
    else
        log_info "LIBERO already at $LIBERO_CLONE — skipping clone."
    fi
    touch "$LIBERO_CLONE/libero/__init__.py" \
          "$LIBERO_CLONE/libero/configs/__init__.py"
    uv_pip libero -e "$LIBERO_CLONE"
    # robosuite v1.4.1_libero fork has a circular import: robosuite/__init__.py
    # eagerly imports libero.libero.envs.problems.libero_kitchen_tabletop_manipulation
    # which itself imports robosuite.utils.transform_utils → ImportError. Patch the
    # robosuite __init__ to comment out the eager libero hook; LIBERO's own envs/
    # __init__.py imports robosuite-driven problems lazily later, so this is safe.
    local _RB_INIT="$VENVS_DIR/libero/lib/python3.8/site-packages/robosuite/__init__.py"
    if [[ -f "$_RB_INIT" ]] && grep -q '^from libero\.libero\.envs\.problems' "$_RB_INIT"; then
        sed -i 's|^from libero\.libero\.envs\.problems\.libero_kitchen_tabletop_manipulation|# from libero.libero.envs.problems.libero_kitchen_tabletop_manipulation|' "$_RB_INIT"
        log_info "Patched robosuite/__init__.py: lazy-skip libero kitchen import (breaks circular)."
    fi
    # LIBERO datasets (~5 GB) are required at runtime — sim_worker /init looks them
    # up under ~/.libero/<suite>/.  Auto-download via the package's `libero_get_assets`
    # helper if available; otherwise warn the user to fetch manually.
    log_step "Checking LIBERO dataset availability..."
    local _LIBERO_DATA="$HOME/.libero"
    if [[ ! -d "$_LIBERO_DATA" ]] || [[ -z "$(ls -A "$_LIBERO_DATA" 2>/dev/null || true)" ]]; then
        log_step "LIBERO datasets not found at $_LIBERO_DATA — attempting auto-download via libero.libero.utils..."
        if ! "$VENVS_DIR/libero/bin/python" -c "from libero.libero import get_libero_path; print(get_libero_path('init_states'))" 2>/dev/null; then
            log_warn "LIBERO dataset auto-download not available."
            log_warn "  Manual: download from https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
            log_warn "  Place under: $_LIBERO_DATA/{libero_spatial,libero_object,libero_goal,libero_10,libero_90}/"
            log_warn "  See: $INSTALL_DOCS#libero"
        fi
    else
        log_info "LIBERO datasets already present at $_LIBERO_DATA."
    fi
    # MuJoCo offscreen rendering on aarch64 / GB10 / headless servers requires
    # MUJOCO_GL=egl. Auto-set in the libero venv's activate script so users don't
    # have to remember (sim_worker also forces it via env at boot, but interactive
    # python invocations from the venv can otherwise default to GLFW and fail).
    local _ACTIVATE="$VENVS_DIR/libero/bin/activate"
    if [[ -f "$_ACTIVATE" ]] && ! grep -q "MUJOCO_GL" "$_ACTIVATE"; then
        cat >> "$_ACTIVATE" << 'ACTIVATE_EOF'

# === MUJOCO_GL=egl (auto-added by setup_libero — required for headless aarch64 / GB10) ===
export MUJOCO_GL="${MUJOCO_GL:-egl}"
ACTIVATE_EOF
        log_info "MUJOCO_GL=egl wired into $_ACTIVATE."
    fi

    log_info "LIBERO venv ready. Datasets must be symlinked under ~/.libero/ (see LIBERO docs)."
}

setup_libero_pro() {
    make_venv libero_pro "3.8"
    log_step "Installing LIBERO-Pro deps into .venvs/libero_pro (Python 3.8)..."
    uv_pip libero_pro --upgrade pip setuptools wheel
    uv_pip libero_pro \
        "numpy>=1.24,<2.0" \
        "pillow>=9.0" \
        "eval_type_backport>=0.2.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "h5py>=3.8" \
        "bddl>=3.0.0" \
        "mujoco>=2.3.7" \
        "torch>=2.0,<2.5" \
        "easydict" "future" "thop" "transforms3d" "termcolor" \
        "matplotlib" "opencv-python-headless" \
        "cloudpickle" "gym" "tianshou" "imageio"
    uv_pip libero_pro \
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.4.1_libero"
    # Apply the same robosuite circular-import workaround in the libero_pro venv.
    local _RB_INIT_PRO="$VENVS_DIR/libero_pro/lib/python3.8/site-packages/robosuite/__init__.py"
    if [[ -f "$_RB_INIT_PRO" ]] && grep -q '^from libero\.libero\.envs\.problems' "$_RB_INIT_PRO"; then
        sed -i 's|^from libero\.libero\.envs\.problems\.libero_kitchen_tabletop_manipulation|# from libero.libero.envs.problems.libero_kitchen_tabletop_manipulation|' "$_RB_INIT_PRO"
        log_info "Patched libero_pro robosuite/__init__.py: lazy-skip libero kitchen import."
    fi
    # MUJOCO_GL=egl in activate
    local _ACTIVATE_PRO="$VENVS_DIR/libero_pro/bin/activate"
    if [[ -f "$_ACTIVATE_PRO" ]] && ! grep -q "MUJOCO_GL" "$_ACTIVATE_PRO"; then
        echo -e "\n# === MUJOCO_GL=egl (auto-added by setup_libero_pro) ===\nexport MUJOCO_GL=\"\${MUJOCO_GL:-egl}\"" >> "$_ACTIVATE_PRO"
    fi

    # Install base LIBERO vendor package so `from libero.libero import ...` works in
    # the libero_pro venv.  Reuse the existing clone created by setup_libero if present.
    local _LIBERO_CLONE_PRO="${LIBERO_DIR:-$VENDORS_DIR/LIBERO}"
    if [[ ! -d "$_LIBERO_CLONE_PRO" ]]; then
        log_step "Cloning LIBERO into $_LIBERO_CLONE_PRO (needed by libero_pro venv)..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$_LIBERO_CLONE_PRO"
    else
        log_info "LIBERO already at $_LIBERO_CLONE_PRO — skipping clone."
    fi
    # Ensure find_packages() shims exist (same workaround as setup_libero).
    touch "$_LIBERO_CLONE_PRO/libero/__init__.py" \
          "$_LIBERO_CLONE_PRO/libero/configs/__init__.py"
    uv_pip libero_pro -e "$_LIBERO_CLONE_PRO" --no-deps

    # Clone LIBERO-PRO fork (not on PyPI)
    local LIBERO_PRO_CLONE="${LIBERO_PRO_DIR:-$VENDORS_DIR/LIBERO-PRO}"
    if [[ ! -d "$LIBERO_PRO_CLONE" ]]; then
        log_step "Cloning LIBERO-PRO into $LIBERO_PRO_CLONE..."
        git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git "$LIBERO_PRO_CLONE"
    else
        log_info "LIBERO-PRO already at $LIBERO_PRO_CLONE — skipping clone."
    fi
    # LIBERO-PRO mirrors upstream LIBERO's packaging gap: the source tree has
    # libero/libero/__init__.py but no top-level libero/__init__.py, so
    # setuptools.find_packages() can produce an empty editable install.
    touch "$LIBERO_PRO_CLONE/libero/__init__.py" \
          "$LIBERO_PRO_CLONE/libero/configs/__init__.py"
    uv_pip libero_pro -e "$LIBERO_PRO_CLONE"

    log_warn "LIBERO-Pro h5py may need native HDF5 libs on aarch64."
    log_warn "  export LD_LIBRARY_PATH=~/.micromamba/envs/libero_libs/lib:\$LD_LIBRARY_PATH"
    log_warn "  See: $INSTALL_DOCS#libero_pro"
}
# === START libero_infinity ===
setup_libero_infinity() {
    # libero-infinity (PyPI) requires Python >=3.11, so we need a separate
    # venv from the base LIBERO (3.8) venv.  We also bring in the LIBERO
    # upstream repo because LiberoInfinityBackend uses LIBERO's simulator
    # primitives (environments, BDDL runner, etc.).
    make_venv libero_infinity "3.11"
    log_step "Installing LIBERO-Infinity deps into .venvs/libero_infinity (Python 3.11)..."
    uv_pip libero_infinity --upgrade pip setuptools wheel
    uv_pip libero_infinity \
        "numpy>=1.24" \
        "pillow>=9.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "h5py>=3.8" \
        "bddl>=3.0.0" \
        "mujoco>=2.3.7" \
        "scenic>=3.0.0"

    # LIBERO requires robosuite.  On Python 3.11 we can use robosuite master
    # (1.5.x requires mujoco>=3.3.0 which is fine on 3.11).
    uv_pip libero_infinity \
        "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@master"

    # Clone LIBERO upstream (same repo as the libero venv uses) so
    # LiberoInfinityBackend can import libero.*  Reuse existing clone if present.
    local LIBERO_CLONE="${LIBERO_DIR:-$VENDORS_DIR/LIBERO}"
    if [[ ! -d "$LIBERO_CLONE" ]]; then
        log_step "Cloning LIBERO into $LIBERO_CLONE..."
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_CLONE"
    else
        log_info "LIBERO already at $LIBERO_CLONE — skipping clone."
    fi
    # Ensure find_packages() shims exist (same workaround as setup_libero).
    touch "$LIBERO_CLONE/libero/__init__.py" \
          "$LIBERO_CLONE/libero/configs/__init__.py"
    uv_pip libero_infinity -e "$LIBERO_CLONE"

    # Install libero-infinity. Prefer an editable local checkout when present;
    # otherwise install the upstream package directly so roboeval uses the real
    # compiler/generator for all perturbation axes.
    local _LI_SRC="${LIBERO_INFINITY_SRC:-$VENDORS_DIR/libero-infinity}"
    if [[ -f "$_LI_SRC/src/libero_infinity/compiler.py" ]]; then
        log_step "Installing libero-infinity from local dev source: $_LI_SRC"
        uv_pip libero_infinity -e "$_LI_SRC"
    else
        log_step "Installing libero-infinity from upstream git..."
        uv_pip libero_infinity \
            "libero-infinity @ git+https://github.com/KE7/libero-infinity.git"
    fi

    "$VENVS_DIR/libero_infinity/bin/python" - <<'PY'
import inspect
import sys

try:
    from libero_infinity.compiler import generate_scenic_file
except Exception as exc:
    raise SystemExit(
        "libero-infinity compiler support is unavailable. "
        "Install the real KE7/libero-infinity package; the legacy "
        "libero_infinity.scenic_generator fallback is not supported."
    ) from exc

if not callable(generate_scenic_file):
    raise SystemExit("libero_infinity.compiler.generate_scenic_file is not callable")

params = inspect.signature(generate_scenic_file).parameters
if "perturbation" not in params:
    raise SystemExit(
        "libero_infinity.compiler.generate_scenic_file does not accept "
        "the required perturbation argument"
    )

print("libero-infinity compiler import OK", file=sys.stderr)
PY

    log_info "LIBERO-Infinity venv ready."
    log_info "  Datasets must be symlinked under ~/.libero/ (see LIBERO docs)."
    log_warn "  Set MUJOCO_GL=egl for headless rendering."
    log_warn "  See: $INSTALL_DOCS#libero_infinity"
}
# === END libero_infinity ===

setup_robocasa() {
    # Pinned commit that matches the installed robocasa==1.0.0 dist-info.
    local ROBOCASA_COMMIT="56e355ccc64389dfc1b8a61a33b9127b975ba681"
    local ROBOCASA_VENDOR="${ROBOEVAL_ROBOCASA_DIR:-$VENDORS_DIR/robocasa}"

    make_venv robocasa "3.11"
    log_step "Installing RoboCasa deps into .venvs/robocasa..."

    # Clone robocasa at the pinned commit so we can install it WITH its asset
    # tree present in the wheel.
    #
    # Root cause: robocasa/setup.py declares include_package_data=True but ships
    # no MANIFEST.in, so when uv builds the wheel from the git URL the setuptools
    # VCS scanner silently omits models/assets/** (binary/ignored files).  This
    # leaves the installed package missing:
    #   • models/assets/arenas/empty_kitchen_arena.xml  → /init 500 on every eval
    #   • models/assets/box_links/box_links_assets.json → download_kitchen_assets.py
    #                                                      crashes before downloading
    # Fix: clone locally, inject a MANIFEST.in, then reinstall from the clone so
    # that models/assets/** is copied into site-packages.
    if [[ ! -d "$ROBOCASA_VENDOR/.git" ]]; then
        log_step "Cloning robocasa into $ROBOCASA_VENDOR (commit $ROBOCASA_COMMIT)..."
        git clone https://github.com/robocasa/robocasa.git "$ROBOCASA_VENDOR"
        git -C "$ROBOCASA_VENDOR" checkout "$ROBOCASA_COMMIT"
    else
        log_info "robocasa already at $ROBOCASA_VENDOR — skipping clone."
    fi
    # Inject MANIFEST.in so setuptools includes models/assets/** in the wheel.
    cat > "$ROBOCASA_VENDOR/MANIFEST.in" <<'MANIFEST_EOF'
recursive-include robocasa/models/assets *
MANIFEST_EOF

    # Install roboeval[robocasa] extras (mujoco, robosuite, etc.).  This also
    # pulls in robocasa from the git URL as a non-editable wheel — without assets.
    uv_pip robocasa -e "$PROJECT_ROOT[robocasa]"

    # Reinstall robocasa from the local vendor clone.  The MANIFEST.in above
    # ensures models/assets/** is included in the built wheel, so the files land
    # in site-packages where robocasa.__path__[0] and the /init endpoint expect them.
    log_step "Reinstalling robocasa from local clone to include asset files..."
    uv_pip robocasa --reinstall-package robocasa "$ROBOCASA_VENDOR"

    # -------------------------------------------------------------------------
    # Download kitchen assets (sub-set of the full ~10 GB pack).
    #
    # Root cause #2: robocasa ships NO 3-D object/fixture/texture assets inside
    # its Python package — these must be downloaded separately from Box.com.
    # The original download_kitchen_assets.py constructs direct-download URLs of
    # the form  https://utexas.box.com/shared/static/<ID>.zip  which no longer
    # fail with 404s.  The working URL is the
    # box_download_shared_file endpoint which requires a session cookie obtained
    # by first fetching the shared-link page.
    #
    # We download 3 of the 6 asset packs (total ~1.8 GB):
    #   • fixtures_lightwheel (505 MB) — sinks, stoves, fridges, cabinet_panels,
    #                                     handles: required for kitchen scene init
    #   • objects_lightwheel  (755 MB) — manipulable objects & wall accessories
    #                                     (utensil_rack, etc.): required for scene
    #   • textures            (518 MB) — floor/wall/counter textures: required by
    #                                     MuJoCo XML at load time
    # objaverse (~2 GB) and aigen_objs (~5.5 GB) are NOT needed because
    # The RoboCasa backend uses obj_registries=("lightwheel",).
    #
    # Each pack is idempotent: a sentinel file records a successful download so
    # re-runs skip already-present assets.
    # -------------------------------------------------------------------------
    local _ASSETS_ROOT
    _ASSETS_ROOT=$(
        PYTHONWARNINGS=ignore \
        .venvs/robocasa/bin/python 2>/dev/null -c \
            "import contextlib, io; _stdout = io.StringIO();
with contextlib.redirect_stdout(_stdout):
    import robocasa
print(robocasa.models.assets_root)"
    )
    if [[ -z "$_ASSETS_ROOT" ]]; then
        log_warn "Could not determine robocasa assets_root — skipping asset download."
    else
        log_step "Downloading robocasa kitchen assets to $_ASSETS_ROOT ..."

        # Helper: download a single Box-shared ZIP and extract it.
        # Args: <shared_name> <extract_parent_dir> <zip_basename>
        # The zip's top-level directory name must match the last path component of
        # the final install location (Box zips are structured that way).
        _dl_robocasa_asset() {
            local _sname="$1" _parent="$2" _zbase="$3"
            local _sentinel="$_parent/.downloaded_${_sname}"
            if [[ -f "$_sentinel" ]]; then
                log_info "  $_zbase already downloaded — skipping."
                return 0
            fi
            log_step "  Downloading $_zbase (~$(du -sh "$_parent" 2>/dev/null | cut -f1 || echo '?') already) ..."
            mkdir -p "$_parent"
            PYTHONWARNINGS=ignore \
            .venvs/robocasa/bin/python - "$_sname" "$_parent" "$_zbase" <<'PY_DOWNLOAD'
import sys, os, re, urllib.request, zipfile, pathlib

shared_name, parent_dir, zip_basename = sys.argv[1], sys.argv[2], sys.argv[3]
zip_path = os.path.join(parent_dir, zip_basename)

# Step 1 – get session cookie + itemID from shared-link page
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
with opener.open(f"https://utexas.box.com/s/{shared_name}") as r:
    page = r.read().decode("utf-8", errors="ignore")
m = re.search(r'"itemID":(\d+)', page)
if not m:
    print(f"  ERROR: could not find itemID for {shared_name}", flush=True)
    sys.exit(1)
file_id = m.group(1)

# Step 2 – follow box_download_shared_file redirect to get presigned URL
dl_url = (f"https://utexas.box.com/index.php"
           f"?rm=box_download_shared_file"
           f"&shared_name={shared_name}&file_id=f_{file_id}")
with opener.open(dl_url) as r:
    real_url = r.geturl()

# Step 3 – stream download
print(f"  Fetching {zip_basename}...", flush=True)
CHUNK = 4 * 1024 * 1024
done = 0
with urllib.request.urlopen(real_url) as resp:
    total = int(resp.headers.get("Content-Length", 0))
    with open(zip_path, "wb") as f:
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if total:
                print(f"\r  {done*100//total}% ({done//1024//1024}/{total//1024//1024} MB)",
                      end="", flush=True)
print(f"\n  Downloaded {done//1024//1024} MB", flush=True)

# Step 4 – extract (zip top-level dir merges into parent_dir)
print(f"  Extracting to {parent_dir}...", flush=True)
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(path=parent_dir)
os.remove(zip_path)
pathlib.Path(os.path.join(parent_dir, f".downloaded_{shared_name}")).touch()
print(f"  Done.", flush=True)
PY_DOWNLOAD
            if [[ $? -ne 0 ]]; then
                log_warn "  Asset download failed for $_zbase — eval may crash on init."
            fi
        }

        # fixtures_lightwheel: zip has top-level "fixtures/" — extract to assets root
        # so merged content lands in assets/fixtures/{sinks,stoves,...}
        _dl_robocasa_asset \
            "idbncsadpnaz1jfl4i6m8qejawk7p9pi" \
            "$_ASSETS_ROOT" \
            "fixtures_lightwheel.zip"

        # objects_lightwheel: zip has top-level "lightwheel/" — extract to objects/
        # so content lands in assets/objects/lightwheel/{aluminum_foil,...}
        mkdir -p "$_ASSETS_ROOT/objects"
        _dl_robocasa_asset \
            "vckqvvkh1z8t69k8qcpcmee6k66stii4" \
            "$_ASSETS_ROOT/objects" \
            "objects_lightwheel.zip"

        # textures: zip has top-level "textures/" — extract to assets root
        # so content lands in assets/textures/{bricks,flat,marble,...}
        _dl_robocasa_asset \
            "4i85ileasdvstmlln5sbvzptz7keuoy1" \
            "$_ASSETS_ROOT" \
            "textures.zip"
    fi

    log_info "RoboCasa venv + assets ready."
    log_info "  (objaverse/aigen_objs not downloaded; sim_worker limits to lightwheel objects)"
    log_info "  Set MUJOCO_GL=egl for headless rendering."
}

setup_robotwin() {
    make_venv robotwin "3.10"
    log_step "Installing RoboTwin base deps into .venvs/robotwin (Python 3.10)..."
    uv_pip robotwin --upgrade pip setuptools wheel
    uv_pip robotwin \
        "numpy>=1.24" \
        "torch>=2.0" \
        "pyyaml>=6.0" \
        "pillow>=9.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "huggingface-hub>=0.20" \
        "open3d>=0.18" \
        "h5py>=3.8" \
        "gymnasium>=0.29"

    # SAPIEN:
    #   - x86_64 and other supported platforms use the normal online package path.
    #   - Linux/aarch64 + cp310 is not reliably resolvable through PyPI/nightly
    #     indexes, so use SAPIEN's official GitHub release wheel URL.
    #   - ROBOEVAL_SAPIEN_WHL remains a local override for offline or custom builds.
    log_step "Installing SAPIEN into .venvs/robotwin..."
    local _SAPIEN_OK=0
    local _sapien_whl="${ROBOEVAL_SAPIEN_WHL:-}"
    if [[ -n "$_sapien_whl" ]]; then
        log_step "Installing SAPIEN from ROBOEVAL_SAPIEN_WHL=$_sapien_whl"
        uv_pip robotwin "numpy>=1.24,<2" "opencv-python<4.13" \
            "requests>=2.22" "transforms3d>=0.3" "lxml" "networkx" "pyperclip" "setuptools"
        uv_pip robotwin --no-deps "$_sapien_whl" && _SAPIEN_OK=1 || true
        repair_sapien_librt_if_needed robotwin
    elif [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "aarch64" ]]; then
        _sapien_whl="$VENDORS_DIR/$(basename "$ROBOEVAL_SAPIEN_AARCH64_URL")"
        if [[ ! -f "$_sapien_whl" ]] || ! echo "$ROBOEVAL_SAPIEN_AARCH64_SHA256  $_sapien_whl" | sha256sum -c - >/dev/null 2>&1; then
            log_step "Caching SAPIEN Linux/aarch64 wheel at $_sapien_whl"
            rm -f "$_sapien_whl"
            download_file "$ROBOEVAL_SAPIEN_AARCH64_URL" "$_sapien_whl"
            echo "$ROBOEVAL_SAPIEN_AARCH64_SHA256  $_sapien_whl" | sha256sum -c -
        else
            log_info "Using cached SAPIEN Linux/aarch64 wheel: $_sapien_whl"
        fi
        uv_pip robotwin "numpy>=1.24,<2" "opencv-python<4.13" \
            "requests>=2.22" "transforms3d>=0.3" "lxml" "networkx" "pyperclip" "setuptools"
        uv_pip robotwin --no-deps "$_sapien_whl" && _SAPIEN_OK=1 || true
        repair_sapien_librt_if_needed robotwin
    else
        if uv_pip robotwin sapien --pre \
                --extra-index-url "https://storage.googleapis.com/sapien-nightly/" 2>/dev/null; then
            _SAPIEN_OK=1
        fi
    fi
    if [[ "$_SAPIEN_OK" -eq 0 ]]; then
        log_warn "SAPIEN install failed — RoboTwin sim worker will not start."
        log_warn "  Override with ROBOEVAL_SAPIEN_WHL=/path/to/sapien*.whl or"
        log_warn "  ROBOEVAL_SAPIEN_AARCH64_URL=https://.../sapien-...linux_aarch64.whl"
    fi

    # ----- curobo + mplib: motion planning libs (full RoboTwin functionality) -----
    # x86_64 has prebuilt wheels (handled by `_install_robotwin_planners_wheels`);
    # aarch64 needs source builds against a conda-forge user-space env (no sudo).
    # mplib is pinned to 0.2.1 and patched post-install to remove the `or collide`
    # early-exit in planner.py — without that patch, TOPP returns early when
    # collide=True even for valid trajectories, causing topp_left_flag=False and
    # frozen arms.
    install_cuda_torch robotwin   # ensure torch+cu13 before curobo cpp build
    if [[ "$(uname -m)" == "aarch64" ]]; then
        _install_robotwin_planners_aarch64 || {
            log_warn "aarch64 curobo/mplib build failed — RoboTwin will fall back"
            log_warn "  to stub planners (success_rate=0%). See:"
            log_warn "  $INSTALL_DOCS#robotwin-aarch64"
            _install_robotwin_planners_stubs
        }
    else
        _install_robotwin_planners_wheels || {
            log_warn "x86_64 curobo/mplib wheels failed — falling back to stubs."
            _install_robotwin_planners_stubs
        }
    fi
    # Apply the `or collide` patch to mplib/planner.py (idempotent).
    # This is safe to run even after stub install — it no-ops if the pattern
    # is absent (stubs don't contain `or collide`) or already removed.
    _patch_mplib_or_collide robotwin

    # Clone RoboTwin repo
    local ROBOTWIN_DIR="$VENDORS_DIR/RoboTwin"
    if [[ ! -d "$ROBOTWIN_DIR" ]]; then
        log_step "Cloning RoboTwin into $ROBOTWIN_DIR..."
        git clone https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR"
    else
        log_info "RoboTwin already at $ROBOTWIN_DIR — skipping clone."
    fi

    # Download RoboTwin assets (~3.7 GB: embodiments.zip 220MB + objects.zip 3.5GB)
    # from HuggingFace TianxingChen/RoboTwin2.0
    echo "Downloading RoboTwin assets (~3.7GB) — this may take several minutes..."
    (cd "$ROBOTWIN_DIR/assets" && "$VENVS_DIR/robotwin/bin/python" _download.py)

    # Add RoboTwin to the venv's Python path
    local PYVER_SHORT
    PYVER_SHORT="$("$VENVS_DIR/robotwin/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local PTH_FILE="$VENVS_DIR/robotwin/lib/python${PYVER_SHORT}/site-packages/robotwin.pth"
    echo "$ROBOTWIN_DIR" > "$PTH_FILE"
    log_info "RoboTwin added to sys.path via $PTH_FILE"

    log_warn "RoboTwin CRITICAL: import torch BEFORE import sapien in the same process."
    log_warn "  See: $INSTALL_DOCS#robotwin"
}

# Pre-existing stub installer (factor out of any prior inline stubs)
_install_robotwin_planners_stubs() {
    log_warn "Installing curobo/mplib STUBS into .venvs/robotwin (degraded mode)..."
    local SP
    SP="$VENVS_DIR/robotwin/lib/python3.10/site-packages"
    install -d "$SP/mplib/sapien_utils"
    cat > "$SP/mplib/__init__.py" <<'PY'
"""mplib stub - real mplib unavailable on this platform"""
_IS_STUB = True
PY
    cat > "$SP/mplib/planner.py" <<'PY'
"""mplib.planner stub"""
class Planner: pass
PY
    cat > "$SP/mplib/sapien_utils/__init__.py" <<'PY'
def __getattr__(name):
    raise RuntimeError("mplib is not available on this platform.")
PY
    # curobo: leave uninstalled so RoboTwin's `try: import curobo`
    # falls into its CuroboPlanner stub branch.
}

# Patch mplib/planner.py to remove the `or collide` early-exit (idempotent).
# Without this patch TOPP() returns EARLY when the collision checker reports
# collide=True (common during arm motion), setting topp_left_flag=False and
# freezing the arms.
# Safe to run on stubs (pattern is absent) or if already patched (sed no-ops).
_patch_mplib_or_collide() {
    local venv_name="${1:-robotwin}"
    local PY="$VENVS_DIR/$venv_name/bin/python"
    local MPLIB_LOC
    MPLIB_LOC="$("$PY" -c "import mplib; print(mplib.__path__[0])" 2>/dev/null)" || {
        log_warn "_patch_mplib_or_collide: mplib not importable in .venvs/$venv_name — skipping patch."
        return 0
    }
    local PLANNER_PY="$MPLIB_LOC/planner.py"
    if [[ ! -f "$PLANNER_PY" ]]; then
        log_warn "_patch_mplib_or_collide: $PLANNER_PY not found — skipping patch."
        return 0
    fi
    # Idempotent: if `or collide` is already gone, sed makes no change.
    sed -i -E \
        's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
        "$PLANNER_PY"
    # grep -c exits 1 on zero matches (still outputs "0").  Use || remaining=0
    # so set -e doesn't fire and we don't accidentally double-append a "0".
    local remaining
    remaining=$(grep -c "or collide" "$PLANNER_PY" 2>/dev/null) || remaining=0
    if [[ "$remaining" -eq 0 ]]; then
        log_info "mplib or-collide patch applied (or was already absent): $PLANNER_PY"
    else
        log_warn "mplib or-collide patch: $remaining occurrence(s) of 'or collide' still remain in $PLANNER_PY"
    fi
}

# x86_64 path: PyPI wheels for both libs
# mplib pinned to 0.2.1; newer versions may change the MplibPlanner
# constructor signature or TOPP API.
_install_robotwin_planners_wheels() {
    log_step "Installing curobo + mplib==0.2.1 from PyPI wheels (x86_64)..."
    uv_pip robotwin "nvidia-curobo" "mplib==0.2.1"
}

# aarch64 path: micromamba-bootstrapped conda-forge C++ env + source builds
_install_robotwin_planners_aarch64() {
    local MM_BIN="$HOME/.local/bin/micromamba"
    local MM_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
    local MM_ENV="$MM_ROOT/envs/mplib_libs"
    local VENV="$VENVS_DIR/robotwin"
    local PY="$VENV/bin/python"
    local PIP="$VENV/bin/pip"

    # 1. Bootstrap user-space micromamba if missing
    if [[ ! -x "$MM_BIN" ]]; then
        log_step "Installing micromamba (~10s, no sudo) → $MM_BIN..."
        mkdir -p "$(dirname "$MM_BIN")"
        local _MM_TMP
        _MM_TMP="$(mktemp -d)"
        curl -sL "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest" \
            -o "$_MM_TMP/micromamba.tar.bz2"
        tar -xjf "$_MM_TMP/micromamba.tar.bz2" -C "$_MM_TMP" bin/micromamba
        mv "$_MM_TMP/bin/micromamba" "$MM_BIN"
        rm -rf "$_MM_TMP"
    fi
    export MAMBA_ROOT_PREFIX="$MM_ROOT"

    # 2. Create mplib_libs env (~4 min, ~3 GB) — idempotent
    if [[ ! -e "$MM_ENV/lib/libompl.so.17" ]]; then
        log_step "Creating $MM_ENV (conda-forge C++ deps, ~4 min, 3 GB)..."
        rm -rf "$MM_ENV"
        "$MM_BIN" create -y -n mplib_libs \
            -c conda-forge --platform linux-aarch64 \
            python=3.10 eigen=3.4 "ompl=1.6.0" "fcl<0.7.1" \
            "pinocchio=2.6.21" assimp orocos-kdl urdfdom urdfdom_headers \
            console_bridge tinyxml2 octomap \
            cmake ninja pkg-config libclang "clangdev>=11"
    else
        log_info "mplib_libs env already present — skipping."
    fi

    # 3. Build/re-link curobo v0.7.7
    # On a fresh venv, .so files exist from a previous build but the editable
    # link is missing — re-run pip install -e (fast if already compiled).
    local CUROBO_DIR="$VENDORS_DIR/curobo"
    if [[ ! -e "$CUROBO_DIR/src/curobo/curobolib/lbfgs_step_cu"*".so" ]]; then
        log_step "Building curobo v0.7.7 from source (~9 min)..."
        if [[ ! -d "$CUROBO_DIR" ]]; then
            git clone https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
        fi
        ( cd "$CUROBO_DIR" && git fetch --tags && git checkout v0.7.7 )
        ( cd "$CUROBO_DIR" && \
          CUDA_HOME=/usr/local/cuda \
          PATH=/usr/local/cuda/bin:$PATH \
          TORCH_CUDA_ARCH_LIST="9.0;10.0;11.0;12.0" \
          "$PIP" install -e . --no-build-isolation )
        # curobo's deps may have downgraded torch to CPU PyPI wheel — re-pin
        install_cuda_torch robotwin
    elif ! "$PY" -c "import curobo" >/dev/null 2>&1; then
        # Already compiled but not linked in this venv (fresh venv scenario).
        log_step "Re-linking curobo into venv (already compiled)..."
        ( cd "$CUROBO_DIR" && \
          CUDA_HOME=/usr/local/cuda \
          PATH=/usr/local/cuda/bin:$PATH \
          "$PIP" install -e . --no-build-isolation --no-build-isolation )
    else
        log_info "curobo already built and linked — skipping."
    fi

    # 4. Build mplib (~9 min)
    local MPLIB_DIR="$VENDORS_DIR/MPlib"
    if ! "$PY" -c "import mplib.pymp" >/dev/null 2>&1; then
        # Purge any pre-staged stub
        rm -rf "$VENV/lib/python3.10/site-packages/mplib" \
               "$VENV/lib/python3.10/site-packages/mplib.pth"
        log_step "Building mplib from source (~9 min)..."
        if [[ ! -d "$MPLIB_DIR" ]]; then
            git clone --recurse-submodules https://github.com/haosulab/MPlib.git \
                "$MPLIB_DIR"
        else
            ( cd "$MPLIB_DIR" && git submodule update --init --recursive )
        fi
        # Patch 1: relax -Werror (gcc 13 + Eigen + pinocchio templates)
        sed -i 's|"-O3 -g3 -Wall -Werror -fsized-deallocation -Wno-deprecated-declarations"|"-O3 -g3 -Wall -fsized-deallocation -Wno-deprecated-declarations -Wno-uninitialized -Wno-maybe-uninitialized"|' "$MPLIB_DIR/CMakeLists.txt"
        # Patch 2: mkdoc.sh must use venv python (which has libclang)
        sed -i 's|python3 "\$PY_SCRIPT_PATH"|"${PYTHON3:-python3}" "$PY_SCRIPT_PATH"|' "$MPLIB_DIR/dev/mkdoc.sh"
        "$PIP" install libclang
        local CLANG_VER
        CLANG_VER="$(ls "$MM_ENV/lib/clang/" | head -1)"
        ( cd "$MPLIB_DIR" && \
          PATH="$VENV/bin:$PATH" \
          PYTHON3="$VENV/bin/python3" \
          CMAKE_PREFIX_PATH="$MM_ENV" \
          LDFLAGS="-L$MM_ENV/lib -Wl,-rpath,$MM_ENV/lib" \
          LLVM_DIR_PATH="$MM_ENV/lib" \
          CLANG_INCLUDE_DIR="$MM_ENV/lib/clang/$CLANG_VER/include" \
          CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$MM_ENV \
                      -DCMAKE_INSTALL_RPATH=$MM_ENV/lib \
                      -DCMAKE_BUILD_RPATH=$MM_ENV/lib \
                      -DPython_EXECUTABLE=$VENV/bin/python3 \
                      -DCMAKE_EXE_LINKER_FLAGS=-L$MM_ENV/lib \
                      -DCMAKE_SHARED_LINKER_FLAGS=-L$MM_ENV/lib \
                      -DCMAKE_MODULE_LINKER_FLAGS=-L$MM_ENV/lib" \
          "$PIP" install -e . --no-build-isolation )
    else
        log_info "mplib already built — skipping."
    fi

    # 5. Validate: only check torch + mplib native extension.
    # curobo is not required by the eval fast-path (skipped via _EvalGripperPlanner).
    # sapien_utils validation is skipped here because sapien is installed
    # BEFORE this function runs; if it failed the overall install, a prior
    # warning has already been emitted.
    "$PY" -c "
import torch, mplib, mplib.pymp
assert torch.cuda.is_available(), 'torch.cuda lost during planner installs'
print('aarch64 mplib OK')
" || return 1
}

# =============================================================================
# gym-aloha — pure-uv aarch64-clean bimanual ALOHA simulator.
#
# Why a separate function (vs reusing the [aloha_gym] extra inline):
#   - python 3.10 (gym-aloha is 3.10+); cannot share venv with VLAs (3.11).
#   - keeps the bimanual aarch64-clean path entirely independent of RoboTwin's
#     SAPIEN / curobo / mplib build chain.
# =============================================================================
setup_aloha_gym() {
    make_venv aloha_gym "3.10"
    log_step "Installing gym-aloha into .venvs/aloha_gym (Python 3.10)..."
    uv_pip aloha_gym --upgrade pip setuptools wheel
    # requires-python>=3.11 in pyproject.toml prevents `-e .[aloha_gym]` from
    # resolving inside a 3.10 venv, so install the extra's contents directly.
    uv_pip aloha_gym \
        "gym-aloha>=0.1.3" \
        "gymnasium>=0.29" \
        "mujoco>=2.3.7" \
        "dm-control>=1.0.14" \
        "numpy>=1.24" \
        "pillow>=9.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22"
    log_info "gym-aloha venv ready (pure uv, no conda, aarch64-clean)."
    log_info "  Set MUJOCO_GL=egl for headless rendering."
    log_info "  Tasks: AlohaTransferCube-v0, AlohaInsertion-v0"
}
# gym-pusht — HuggingFace PushT pushing benchmark (Diffusion Policy canonical sim).
# Pure PyPI: gym-pusht + gymnasium + pygame + pymunk.
# Python 3.11, no MuJoCo, no conda, aarch64-clean.
setup_gym_pusht() {
    make_venv gym_pusht "3.11"
    log_step "Installing gym-pusht into .venvs/gym_pusht (Python 3.11)..."
    uv_pip gym_pusht --upgrade pip setuptools wheel
    # gym-pusht pulls pygame + pymunk transitively; pin gymnasium for stability.
    # Pin pymunk<7: pymunk 7.x renamed Space.add_collision_handler→on_collision,
    # breaking gym-pusht's envs/pusht.py._setup(). Sim-worker has a compatibility
    # shim but pinning 6.x avoids the patch path altogether on fresh installs.
    uv_pip gym_pusht \
        "gym-pusht>=0.1.5" \
        "pymunk<7" \
        "gymnasium>=0.29" \
        "numpy>=1.24" \
        "pillow>=9.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22"
    log_info "gym-pusht venv ready (pure uv, no conda, no MuJoCo, aarch64-clean)."
    log_info "  Task: PushT-v0 — push T-block into target zone."
    log_info "  Canonical companion to Diffusion Policy."
}
# =============================================================================
# ACT — Action Chunking Transformer via lerobot (Python 3.11, pure PyPI).
#
# Canonical checkpoints (gym-aloha bimanual sim):
#   lerobot/act_aloha_sim_transfer_cube_human  ← default
#   lerobot/act_aloha_sim_insertion_human
#
# Checkpoint size: ~300 MB. VRAM: ~2 GB (small CNN+Transformer model).
# =============================================================================
setup_act() {
    make_venv act "3.11"
    log_step "Installing ACT deps into .venvs/act (Python 3.11)..."
    uv_pip act -e "$PROJECT_ROOT[act]"
    install_cuda_torch act
    log_info "ACT venv ready. Checkpoints downloaded on first use (~300 MB)."
    log_info "  Default:   lerobot/act_aloha_sim_transfer_cube_human  (AlohaTransferCube-v0)"
    log_info "  Optional:  lerobot/act_aloha_sim_insertion_human      (AlohaInsertion-v0)"
    log_info "  Pair with: ./scripts/setup.sh aloha_gym"
}
# TDMPC2 (Hansen et al., model-based RL) — paired with metaworld.
# 4-dim Sawyer eef-delta exact match for metaworld.
#
# Install strategy
# ----------------
# nicklashansen/tdmpc2 (upstream) is research code — no setup.py / pyproject.toml,
# so it cannot be pip-installed directly.  We use a two-step approach:
#   1. Clone upstream to $VENDORS_DIR/tdmpc2-upstream (one-time; idempotent).
#   2. Assemble a minimal pyproject.toml shim + patched __init__.py at
#      $VENDORS_DIR/tdmpc2-pkg, then install with `uv pip install -e --no-deps`.
#   3. Install tensordict (required by the upstream Ensemble/TensorDictParams API).
#
# The vendor dir ($VENDORS_DIR = ~/.local/share/roboeval/vendors by default)
# persists across `rm -rf .venvs/tdmpc2` cycles so clone is done once.
setup_tdmpc2() {
    make_venv tdmpc2 "3.11"
    log_step "Installing TDMPC2 core deps into .venvs/tdmpc2 (Python 3.11)..."
    uv_pip tdmpc2 -e "$PROJECT_ROOT[tdmpc2]"
    install_cuda_torch tdmpc2

    # ── tensordict ────────────────────────────────────────────────────────────
    # The upstream TDMPC2 world-model uses tensordict's Ensemble/TensorDictParams
    # for vectorised Q-functions.  Install into the venv (compatible with torch
    # already installed above).
    log_step "Installing tensordict into .venvs/tdmpc2..."
    uv_pip tdmpc2 "tensordict>=0.6"

    # ── Vendor upstream nicklashansen/tdmpc2 ─────────────────────────────────
    local TDMPC2_UPSTREAM="$VENDORS_DIR/tdmpc2-upstream"
    local TDMPC2_PKG="$VENDORS_DIR/tdmpc2-pkg"

    # Step 1: clone upstream repo (one-time; idempotent on re-runs).
    if [[ ! -d "$TDMPC2_UPSTREAM/.git" ]]; then
        log_step "Cloning nicklashansen/tdmpc2 to $TDMPC2_UPSTREAM..."
        git clone --depth 1 https://github.com/nicklashansen/tdmpc2 \
            "$TDMPC2_UPSTREAM" || {
            log_error "Failed to clone nicklashansen/tdmpc2.  Check network access."
            return 1
        }
    else
        log_info "tdmpc2 upstream already cloned at $TDMPC2_UPSTREAM — skipping clone."
    fi

    # Step 2: assemble the pip-installable pkg dir (idempotent).
    if [[ ! -f "$TDMPC2_PKG/pyproject.toml" ]]; then
        log_step "Assembling vendored tdmpc2 pip package at $TDMPC2_PKG..."
        mkdir -p "$TDMPC2_PKG"

        # Copy the Python package (tdmpc2/ subdirectory from the upstream repo).
        cp -r "$TDMPC2_UPSTREAM/tdmpc2" "$TDMPC2_PKG/"

        # Patch __init__.py: add sys.path shim so that the upstream's
        # flat imports (from common import math, etc.) resolve correctly
        # even when tdmpc2 is installed as a proper package.
        cat > "$TDMPC2_PKG/tdmpc2/__init__.py" <<'PYEOF'
# TD-MPC2 vendor package — roboeval sys.path shim.
# The upstream repo uses flat imports (e.g. `from common import math`) which
# assume the tdmpc2/ directory is on sys.path.  We insert it here so that an
# installed `import tdmpc2` works identically to running from the repo root.
import sys as _sys
import os as _os

_pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
if _pkg_dir not in _sys.path:
    _sys.path.insert(0, _pkg_dir)

from tdmpc2.tdmpc2 import TDMPC2  # noqa: E402

__all__ = ["TDMPC2"]
PYEOF

        # Patch api_model_conversion to tolerate missing __batch_size/__device
        # metadata keys (no longer emitted by tensordict >=0.6).
        # Note: single-quoted heredoc so bash vars don't expand inside Python;
        # pass the path via env var instead.
        TDMPC2_PKG_PATH="$TDMPC2_PKG" python3 - <<'PATCHEOF'
import os, re, pathlib

p = pathlib.Path(os.environ["TDMPC2_PKG_PATH"] + "/tdmpc2/common/layers.py")
src = p.read_text()

old = """\t# add batch_size and device from target_state_dict to new_state_dict
\tfor prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
\t\tfor key in ('__batch_size', '__device'):
\t\t\tnew_key = prefix + 'params.' + key
\t\t\tnew_state_dict[new_key] = target_state_dict[new_key]

\t# check that every key in new_state_dict is in target_state_dict
\tfor key in new_state_dict.keys():
\t\tassert key in target_state_dict, f"key {key} not in target_state_dict"
\t# check that all Qs keys in target_state_dict are in new_state_dict
\tfor key in target_state_dict.keys():
\t\tif 'Qs' in key:
\t\t\tassert key in new_state_dict, f"key {key} not in new_state_dict" """

new = """\t# Older tensordict stored __batch_size/__device as metadata in the state dict;
\t# tensordict >=0.6 no longer does.  Skip those keys gracefully.
\tfor prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
\t\tfor key in ('__batch_size', '__device'):
\t\t\tnew_key = prefix + 'params.' + key
\t\t\tif new_key in target_state_dict:
\t\t\t\tnew_state_dict[new_key] = target_state_dict[new_key]

\t# check that all Qs weight/bias keys in target_state_dict are in new_state_dict
\tfor key in target_state_dict.keys():
\t\tif 'Qs' in key and not key.endswith(('__batch_size', '__device')):
\t\t\tassert key in new_state_dict, f"key {key} not in new_state_dict" """

if old in src:
    p.write_text(src.replace(old, new))
    print("layers.py patched OK")
else:
    print("layers.py already patched or pattern not found — skipping")
PATCHEOF

        # Patch tdmpc2.py: remove capturable=True from Adam optimizers.
        # capturable=True triggers CUDA-graph memory pre-allocation which can
        # OOM when other processes share the GPU.  For inference-only mode
        # this is unnecessary; we just need the optimizer objects to exist.
        TDMPC2_PKG_PATH="$TDMPC2_PKG" python3 - <<'CAPEOF'
import os, pathlib
p = pathlib.Path(os.environ["TDMPC2_PKG_PATH"] + "/tdmpc2/tdmpc2.py")
src = p.read_text()
patched = src.replace("capturable=True", "capturable=False")
if patched != src:
    p.write_text(patched)
    print("tdmpc2.py capturable patch OK")
else:
    print("tdmpc2.py capturable already patched or not found — skipping")
CAPEOF

        # Write the pyproject.toml.
        cat > "$TDMPC2_PKG/pyproject.toml" <<'TOMLEOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "tdmpc2"
version = "1.0.0.dev0"
description = "TD-MPC2 vendor shim (nicklashansen/tdmpc2 — inference only)"
license = { text = "MIT" }
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "tensordict>=0.6",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["tdmpc2*"]
TOMLEOF
        log_info "Vendored tdmpc2 package assembled at $TDMPC2_PKG"
    else
        log_info "Vendored tdmpc2 package already assembled at $TDMPC2_PKG — skipping."
    fi

    # Step 3: install the vendor package into the venv.
    log_step "Installing vendored tdmpc2 package into .venvs/tdmpc2..."
    uv_pip tdmpc2 -e "$TDMPC2_PKG" --no-deps

    log_info "TDMPC2 venv ready."
    log_info "  Upstream:  nicklashansen/tdmpc2  (MIT-licensed, model-based RL)"
    log_info "  Checkpts:  Downloaded on first use (~30 MB per task)"
    log_info "             Default: metaworld/mw-button-press-1.pt  (button-press-v2)"
    log_info "             Override: TDMPC2_CHECKPOINT=metaworld/mw-<task>-1.pt"
    log_info "  Pair with: ./scripts/setup.sh metaworld"
}
# ManiSkill2 (Hao Su lab, SAPIEN-based).
#
# aarch64 INSTALL BLOCKER
# ──────────────────────────────────────────────────────────────────────
# mani_skill2==0.5.3 requires sapien==2.2.2 which ships x86_64 wheels only.
# On aarch64 this function installs the non-SAPIEN deps only and prints a
# clear warning.  The ManiSkill2 backend raises RuntimeError at init().
#
# On x86_64 the full stack installs cleanly:
#   mani_skill2==0.5.3  +  sapien==2.2.2  +  numpy<1.24  +  gymnasium>=0.28.1
# SUPPORTED_PLATFORMS: x86_64 only for full ManiSkill2 execution. aarch64 is
# blocked by the sapien==2.2.2 pin; SAPIEN 3 now has aarch64 wheels, but using
# them would require migrating this backend/config to ManiSkill3 APIs.
# =============================================================================
setup_maniskill2() {
    make_venv maniskill2 "3.10"
    log_step "Installing ManiSkill2 deps into .venvs/maniskill2 (Python 3.10)..."
    uv_pip maniskill2 --upgrade pip setuptools wheel

    local ARCH
    ARCH="$(uname -m)"
    if [[ "$ARCH" == "aarch64" ]]; then
        log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log_warn "ManiSkill2 — INSTALL BLOCKER on aarch64"
        log_warn "  mani_skill2==0.5.3 requires sapien==2.2.2."
        log_warn "  sapien 2.2.2 ships ONLY manylinux2014_x86_64 wheels on PyPI."
        log_warn "  SAPIEN 3 has aarch64 wheels, but ManiSkill2 is not compatible"
        log_warn "  without a ManiSkill3 backend migration."
        log_warn ""
        log_warn "  Installing non-SAPIEN deps only (numpy, gymnasium, fastapi)."
        log_warn "  The ManiSkill2Backend in sim_worker.py will raise RuntimeError"
        log_warn "  at /init time with a pointer to the fix."
        log_warn ""
        log_warn "  Unblock requires a SAPIEN 2.2.2 aarch64 wheel or a ManiSkill3"
        log_warn "  migration; re-running setup.sh alone will not fix this."
        log_warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        # Install non-SAPIEN deps so shared imports still resolve cleanly.
        uv_pip maniskill2 \
            "numpy<1.24" \
            "scipy" \
            "gymnasium>=0.28.1" \
            "h5py" \
            "pyyaml>=6.0" \
            "pillow>=9.0" \
            "fastapi>=0.100" \
            "uvicorn[standard]>=0.22" \
            "transforms3d" \
            "trimesh"
        log_warn "ManiSkill2 venv provisioned (degraded — no SAPIEN on aarch64)."
        log_warn "  get_info() returns a valid spec; init() raises RuntimeError."
        return 0
    fi

    # x86_64: full install — sapien==2.2.2 is only on PyPI for x86_64.
    log_step "Installing sapien==2.2.2 (x86_64 only)..."
    uv_pip maniskill2 "sapien==2.2.2" || {
        log_warn "sapien==2.2.2 install failed.  Check network and Python version."
        log_warn "  Requires Python 3.10 (cp310) for sapien 2.2.2 wheel."
        log_warn "  Manual: .venvs/maniskill2/bin/pip install sapien==2.2.2"
    }

    log_step "Installing mani_skill2==0.5.3 and deps..."
    uv_pip maniskill2 \
        "mani_skill2==0.5.3" \
        "numpy<1.24" \
        "scipy" \
        "gymnasium>=0.28.1" \
        "h5py" \
        "pyyaml>=6.0" \
        "pillow>=9.0" \
        "fastapi>=0.100" \
        "uvicorn[standard]>=0.22" \
        "transforms3d" \
        "trimesh" \
        "imageio" \
        "imageio[ffmpeg]" || {
        log_warn "mani_skill2 install encountered errors — check output above."
    }

    log_info "ManiSkill2 venv ready (x86_64)."
    log_info "  Tasks: PickCube-v0  StackCube-v0  PegInsertionSide-v0"
    log_info "  Action: 7-dim pd_ee_delta_pose (compatible with pi05 / smolvla / openvla)"
    log_info "  Start server:"
    log_info "    .venvs/maniskill2/bin/python sims/sim_worker.py \\"
    log_info "      --sim maniskill2 --port 5306 --headless"
}

# =============================================================================
# Meta-World — pure-uv aarch64-clean single-arm manipulation benchmark.
#
# ~50 Sawyer single-arm tasks (push-button, pick-place, door-open, …) built on
# MuJoCo.  Python 3.11.  Pure PyPI install — no conda, no source builds, no
# SAPIEN, no curobo.  MuJoCo ships aarch64 wheels → fully uv-clean.
#
# Action space: 4-dim eef-delta [dx, dy, dz, gripper].  Current 7-dim VLAs
# (pi05, smolvla, openvla) will be blocked by the ActionObsSpec gate —
# that is expected when pairing these components.
# =============================================================================
setup_metaworld() {
    make_venv metaworld "3.11"
    log_step "Installing Meta-World into .venvs/metaworld (Python 3.11)..."
    uv_pip metaworld -e "$PROJECT_ROOT[metaworld]"
    log_info "Meta-World venv ready (pure uv, no conda, aarch64-clean)."
    log_info "  Set MUJOCO_GL=egl for headless rendering."
    log_info "  ~50 tasks: button-press-v2, pick-place-v2, door-open-v2, reach-v2, ..."
    log_warn "  NOTE: 4-dim action space — 7-dim VLAs (pi05, smolvla, openvla)"
    log_warn "        will be blocked by the ActionObsSpec gate (expected for v0.1)."
    log_info "  Start server:"
    log_info "    MUJOCO_GL=egl \\"
    log_info "    .venvs/metaworld/bin/python sims/sim_worker.py \\"
    log_info "      --sim metaworld --port 5307 --headless"
}

setup_vlm() {
    make_venv litellm "3.11"
    log_step "Installing LiteLLM/VLM deps into .venvs/litellm..."
    uv_pip litellm -e "$PROJECT_ROOT[vlm]"
    log_info "VLM proxy venv ready. Requires Ollama: https://ollama.com"
    log_info "  Pull a vision model: ollama pull qwen3-vl:latest"
}
# Diffusion Policy via lerobot — DDPM-based visuomotor policy.
# Canonical checkpoint: lerobot/diffusion_pusht (PushT 2D task).
# The base lerobot>=0.4.4 package ships DiffusionPolicy; no extra pip extras required.
# Python 3.11 venv (same as smolvla — both lerobot-based).
setup_diffusion_policy() {
    make_venv diffusion_policy "3.11"
    log_step "Installing Diffusion Policy deps into .venvs/diffusion_policy..."
    uv_pip diffusion_policy -e "$PROJECT_ROOT[diffusion_policy]"
    install_cuda_torch diffusion_policy
    log_info "Diffusion Policy venv ready."
    log_info "  Canonical checkpoint: lerobot/diffusion_pusht (PushT 2D, ~50 MB)"
    log_info "  Alternative:          lerobot/diffusion_aloha_sim_insertion_human (ALOHA sim)"
    log_info "  Start server:         .venvs/diffusion_policy/bin/python \\"
    log_info "                          -m sims.vla_policies.diffusion_policy_policy --port 5103"
    log_warn "  Note: diffusion_pusht checkpoint targets the PushT task environment."
    log_warn "  A LIBERO-compatible sim integration is planned for v0.2."
    log_warn "  See docs/install.md#diffusion_policy for details."
}

# ---------------------------------------------------------------------------
# Orchestrator venv
# ---------------------------------------------------------------------------

setup_roboeval() {
    make_venv roboeval "3.13"
    log_step "Installing roboeval orchestrator into .venvs/roboeval..."
    uv_pip roboeval -e "$PROJECT_ROOT"
    log_info "roboeval orchestrator ready."
    log_info "  Activate: source .venvs/roboeval/bin/activate"
    log_info "  CLI:      roboeval --help"
}

# ---------------------------------------------------------------------------
# Main install loop
# ---------------------------------------------------------------------------

log_info "Components to install: ${REQUESTED[*]}"
log_info "Venvs directory: $VENVS_DIR"
log_info "Vendors directory: $VENDORS_DIR"
echo ""

for component in "${REQUESTED[@]}"; do
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log_info "Setting up: $component (Python ${COMPONENT_PYTHON[$component]})"
    log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    case "$component" in
        pi05)       setup_pi05 ;;
        openvla)    setup_openvla ;;
        smolvla)    setup_smolvla ;;
        groot)      setup_groot ;;
        cosmos)     setup_cosmos ;;
        internvla)  setup_internvla ;;
        libero)     setup_libero ;;
        libero_pro) setup_libero_pro ;;
        robocasa)   setup_robocasa ;;
        robotwin)   setup_robotwin ;;
        aloha_gym)  setup_aloha_gym ;;
        vlm)        setup_vlm ;;
        diffusion_policy) setup_diffusion_policy ;;
        gym_pusht)  setup_gym_pusht ;;
        vqbet)      setup_vqbet ;;
        act)        setup_act ;;
        tdmpc2)     setup_tdmpc2 ;;
        maniskill2) setup_maniskill2 ;;
        metaworld) setup_metaworld ;;
        libero_infinity) setup_libero_infinity ;;
    esac
    echo ""
done

# Orchestrator always installed
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Setting up: roboeval orchestrator"
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
setup_roboeval
echo ""

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Validating installs..."
log_info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for component in "${REQUESTED[@]}"; do
    validate_import "$component" "${COMPONENT_IMPORT[$component]}"
done
validate_import roboeval roboeval
echo ""

# ---------------------------------------------------------------------------
# Next-step instructions
# ---------------------------------------------------------------------------

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!  Next steps:${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  1.  Activate the orchestrator environment:"
echo "        source .venvs/roboeval/bin/activate"
echo ""

# Print VLA-specific start commands
for component in "${REQUESTED[@]}"; do
    case "$component" in
        pi05)
            echo "  2a. Start the Pi 0.5 policy server (in a separate terminal):"
            echo "        roboeval serve --vla pi05"
            echo ""
            ;;
        openvla)
            echo "  2b. Start the OpenVLA policy server (in a separate terminal):"
            echo "        roboeval serve --vla openvla --vla-port 5101"
            echo ""
            ;;
        smolvla)
            echo "  2c. Start the SmolVLA policy server (in a separate terminal):"
            echo "        roboeval serve --vla smolvla --vla-port 5102"
            echo ""
            ;;
        diffusion_policy)
            echo "  2d. Start the Diffusion Policy server (in a separate terminal):"
            echo "        .venvs/diffusion_policy/bin/python -m sims.vla_policies.diffusion_policy_policy --port 5103"
            echo ""
            ;;
        vqbet)
            echo "  2e. Start the VQ-BeT policy server (in a separate terminal):"
            echo "        roboeval serve --vla vqbet --vla-port 5108"
            echo ""
            ;;
    esac
done

# Print sim-specific start commands
for component in "${REQUESTED[@]}"; do
    case "$component" in
        libero)
            echo "  3a. Start the LIBERO simulator worker (in a separate terminal):"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/libero/bin/python sims/sim_worker.py --sim libero --port 5001 --headless"
            echo ""
            ;;
        robocasa)
            echo "  3b. Start the RoboCasa simulator worker (in a separate terminal):"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/robocasa/bin/python sims/sim_worker.py --sim robocasa --port 5001 --headless"
            echo ""
            ;;
        robotwin)
            echo "  3c. Start the RoboTwin simulator worker (in a separate terminal):"
            echo "        # DISPLAY must be UNSET so SAPIEN uses EGL offscreen rendering."
            echo "        # With DISPLAY set, SAPIEN's Vulkan+X11 path triggers an XCB"
            echo "        # threading assertion crash before /init returns."
            echo "        unset DISPLAY \\"
            echo "        .venvs/robotwin/bin/python sims/sim_worker.py --sim robotwin --port 5001 --headless"
            echo ""
            ;;
        libero_pro)
            echo "  3d. Start the LIBERO-Pro simulator worker (in a separate terminal):"
            echo "        export LD_LIBRARY_PATH=~/.micromamba/envs/libero_libs/lib:\$LD_LIBRARY_PATH"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/libero_pro/bin/python sims/sim_worker.py --sim libero_pro --port 5001 --headless"
            echo ""
            ;;
        aloha_gym)
            echo "  3e. Start the gym-aloha simulator worker (in a separate terminal):"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/aloha_gym/bin/python sims/sim_worker.py --sim aloha_gym --port 5304 --headless"
            echo ""
            ;;
        gym_pusht)
            echo "  3f. Start the gym-pusht simulator worker (in a separate terminal):"
            echo "        .venvs/gym_pusht/bin/python sims/sim_worker.py --sim gym_pusht --port 5305 --headless"
            echo ""
            ;;
        metaworld)
            echo "  3g. Start the Meta-World simulator worker (in a separate terminal):"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/metaworld/bin/python sims/sim_worker.py --sim metaworld --port 5307 --headless"
            echo ""
            echo "        NOTE: 4-dim action space — pair with a 4-dim-native VLA."
            echo "              7-dim VLAs (pi05, smolvla, openvla) will be blocked"
            echo "              by the ActionObsSpec gate (expected in v0.1)."
            echo ""
            ;;
        libero_infinity)
            echo "  3h. Start the LIBERO-Infinity simulator worker (in a separate terminal):"
            echo "        MUJOCO_GL=egl \\"
            echo "        .venvs/libero_infinity/bin/python sims/sim_worker.py --sim libero_infinity --port 5308 --headless"
            echo ""
            echo "        NOTE: Datasets must be symlinked under ~/.libero/ (see LIBERO docs)."
            echo "              libero_infinity uses Scenic-based perturbation testing."
            echo ""
            ;;
    esac
done

echo "  4.  Run the smoke test:"
echo "        roboeval test"
echo ""
echo "  Full guide: $INSTALL_DOCS"
echo ""
