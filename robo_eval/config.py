"""Configuration constants for VLA servers, simulators, and suite definitions."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vendor repos directory.
# Default: ~/.local/share/robo-eval/vendors/
# Override via ROBO_EVAL_VENDORS_DIR environment variable.
# .resolve() ensures an absolute path even if the env var is relative.
VENDORS_DIR = Path(os.environ.get(
    "ROBO_EVAL_VENDORS_DIR",
    str(Path.home() / ".local" / "share" / "robo-eval" / "vendors"),
)).resolve()


def _resolve_libero_infinity_root() -> str | None:
    """Resolve the libero-infinity package root directory.

    Resolution order:
      1. ``LIBERO_INFINITY_ROOT`` environment variable (explicit override)
      2. Installed ``libero_infinity`` package (found via importlib)
      3. Adjacent checkout ``../libero-infinity`` (local development layout)

    Returns the resolved path as a string, or *None* if the package
    cannot be located by any method.
    """
    # 1. Explicit env var
    from_env = os.environ.get("LIBERO_INFINITY_ROOT")
    if from_env:
        return from_env

    # 2. Installed package (pip install / pip install -e)
    try:
        import importlib

        _li = importlib.import_module("libero_infinity")
        if hasattr(_li, "__path__") and _li.__path__:
            pkg_dir = Path(_li.__path__[0])
            # For a normal install the package root is one level above the
            # package directory; for an editable install __path__[0] *is*
            # the package directory inside the repo checkout.
            root = pkg_dir.parent
            _log.debug("libero-infinity found via importlib at %s", root)
            return str(root)
    except (ImportError, Exception):
        pass

    # 3. Adjacent checkout fallback (local development layout)
    adjacent = PROJECT_ROOT.parent / "libero-infinity"
    if adjacent.is_dir():
        _log.debug("libero-infinity found as adjacent checkout: %s", adjacent)
        return str(adjacent)

    return None


LIBERO_INFINITY_ROOT: str | None = _resolve_libero_infinity_root()


def validate_port(port: int, name: str = "port") -> int:
    """Validate that a port number is in a usable range.

    Args:
        port: The port number to validate.
        name: Human-readable name for error messages.

    Returns:
        The validated port number.

    Raises:
        ValueError: If the port is out of range.
    """
    if not isinstance(port, int):
        raise ValueError(f"{name} must be an integer, got {type(port).__name__}")
    if port < 1 or port > 65535:
        raise ValueError(
            f"{name} must be between 1 and 65535, got {port}"
        )
    if port < 1024:
        raise ValueError(
            f"{name}={port} is a privileged port (< 1024). "
            f"Use a port >= 1024."
        )
    return port


def is_port_available(port: int) -> bool:
    """Return True if the TCP port can be bound on this machine."""
    validate_port(port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def find_available_port(
    preferred_port: Optional[int] = None,
    *,
    search_start: int = 1024,
    search_end: int = 65535,
) -> int:
    """Find a single available TCP port, preferring the requested port."""
    if preferred_port is not None:
        validate_port(preferred_port)
        if is_port_available(preferred_port):
            return preferred_port
        search_start = max(search_start, preferred_port + 1)

    for port in range(search_start, search_end + 1):
        if is_port_available(port):
            return port

    raise RuntimeError(
        f"No available TCP port found in range {search_start}-{search_end}"
    )


def find_available_port_block(
    count: int,
    preferred_start: Optional[int] = None,
    *,
    search_start: int = 1024,
    search_end: int = 65535,
) -> int:
    """Find a contiguous block of available TCP ports."""
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")

    if preferred_start is not None:
        validate_port(preferred_start, "preferred_start")
        last = preferred_start + count - 1
        validate_port(last, "preferred_start + count - 1")
        if all(is_port_available(preferred_start + i) for i in range(count)):
            return preferred_start
        search_start = max(search_start, preferred_start + 1)

    max_start = search_end - count + 1
    for base_port in range(search_start, max_start + 1):
        if all(is_port_available(base_port + i) for i in range(count)):
            return base_port

    raise RuntimeError(
        f"No available block of {count} TCP ports found in range "
        f"{search_start}-{search_end}"
    )


# ---------------------------------------------------------------------------
# VLA server definitions
# ---------------------------------------------------------------------------
@dataclass
class VLAConfig:
    """Configuration for a VLA policy server."""

    name: str
    port: int
    venv: str  # relative to PROJECT_ROOT
    start_script: str  # relative to PROJECT_ROOT
    model_id: str
    startup_timeout: int = 300  # seconds

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def venv_python(self) -> Path:
        return PROJECT_ROOT / self.venv / "bin" / "python"

    @property
    def start_script_path(self) -> Path:
        return PROJECT_ROOT / self.start_script


# To add a new VLA: add a VLAConfig entry below, create a policy server in
# sims/vla_policies/, and a start script in scripts/. See docs/adding_a_vla.md.
VLA_CONFIGS: Dict[str, VLAConfig] = {
    "pi05": VLAConfig(
        name="pi05",
        port=5100,
        venv=os.environ.get("ROBO_EVAL_VLA_VENV", ".venvs/vla"),
        start_script="scripts/start_pi05_policy.sh",
        model_id="lerobot/pi05_libero_finetuned",
    ),
    "openvla": VLAConfig(
        name="openvla",
        port=5101,
        venv=os.environ.get("ROBO_EVAL_VLA_VENV", ".venvs/vla"),
        start_script="scripts/start_openvla_policy.sh",
        model_id="openvla/openvla-7b-finetuned-libero-spatial",
        startup_timeout=600,
    ),
    "smolvla": VLAConfig(
        name="smolvla",
        port=5102,
        venv=os.environ.get("ROBO_EVAL_SMOLVLA_VENV", ".venvs/smolvla"),
        start_script="scripts/start_smolvla_policy.sh",
        model_id="HuggingFaceVLA/smolvla_libero",
    ),
    "cosmos": VLAConfig(
        name="cosmos",
        port=5103,
        venv=os.environ.get("ROBO_EVAL_COSMOS_VENV", ".venvs/cosmos"),
        start_script="scripts/start_cosmos_policy.sh",
        model_id="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
        startup_timeout=600,
    ),
    "internvla": VLAConfig(
        name="internvla",
        port=5104,
        venv=os.environ.get("ROBO_EVAL_INTERNVLA_VENV", ".venvs/internvla"),
        start_script="scripts/start_internvla_policy.sh",
        model_id="InternRobotics/InternVLA-A1-3B-RoboTwin",
    ),
    "vqbet": VLAConfig(
        name="vqbet",
        port=5108,
        venv=os.environ.get("ROBO_EVAL_VQBET_VENV", ".venvs/vqbet"),
        start_script="scripts/start_vqbet_policy.sh",
        model_id="lerobot/vqbet_pusht",
    ),
    "tdmpc2": VLAConfig(
        name="tdmpc2",
        port=5109,
        venv=os.environ.get("ROBO_EVAL_TDMPC2_VENV", ".venvs/tdmpc2"),
        start_script="scripts/start_tdmpc2_policy.sh",
        model_id="nicklashansen/tdmpc2",
    ),
    "groot": VLAConfig(
        name="groot",
        port=5105,
        # GR00T requires a dedicated venv (nvidia/Isaac-GR00T deps conflict with lerobot).
        # Default: .venvs/groot  — override via ROBO_EVAL_GROOT_VENV.
        # Set ROBO_EVAL_GROOT_VENV explicitly if using a custom layout.
        venv=os.environ.get("ROBO_EVAL_GROOT_VENV", ".venvs/groot"),
        start_script="scripts/start_groot_policy.sh",
        model_id="nvidia/GR00T-N1.6-3B",
        startup_timeout=600,
    ),
}


# ---------------------------------------------------------------------------
# Per-suite model overrides for OpenVLA
# ---------------------------------------------------------------------------
# OpenVLA was fine-tuned independently on each LIBERO suite, producing
# separate checkpoints (unlike pi0.5 / SmolVLA which use a single checkpoint
# for all suites).  The eval framework must load the correct checkpoint before
# evaluating each suite.  Each entry maps a qualified suite name to the
# HuggingFace model_id and the unnorm_key used by predict_action().
#
# Reference: https://huggingface.co/openvla
OPENVLA_SUITE_MODELS: Dict[str, Dict[str, str]] = {
    "libero_spatial": {
        "model_id": "openvla/openvla-7b-finetuned-libero-spatial",
        "unnorm_key": "libero_spatial",
    },
    "libero_object": {
        "model_id": "openvla/openvla-7b-finetuned-libero-object",
        "unnorm_key": "libero_object",
    },
    "libero_goal": {
        "model_id": "openvla/openvla-7b-finetuned-libero-goal",
        "unnorm_key": "libero_goal",
    },
    "libero_10": {
        "model_id": "openvla/openvla-7b-finetuned-libero-10",
        "unnorm_key": "libero_10",
    },
}


def get_openvla_model_for_suite(suite: str) -> Optional[Dict[str, str]]:
    """Return the OpenVLA model_id and unnorm_key for a given suite.

    Returns None if no suite-specific override exists (the default model
    will be used).
    """
    return OPENVLA_SUITE_MODELS.get(suite)


# ---------------------------------------------------------------------------
# Simulator definitions
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Configuration for a simulator backend."""

    name: str  # "libero", "libero_pro", or "libero_infinity"
    venv: str  # relative to PROJECT_ROOT
    env_vars: Dict[str, str] = field(default_factory=dict)

    @property
    def venv_python(self) -> Path:
        return PROJECT_ROOT / self.venv / "bin" / "python"


# Compute LD_LIBRARY_PATH for libero_pro: prefer micromamba HDF5 libs if available.
# The micromamba path is optional and auto-detected — if the directory does not
# exist (e.g. HDF5 is installed system-wide), this is silently skipped.
_micromamba_lib = Path.home() / ".micromamba" / "envs" / "libero_libs" / "lib"
_ld_library_path = (
    str(_micromamba_lib) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    if _micromamba_lib.exists()
    else os.environ.get("LD_LIBRARY_PATH", "")
)

# To add a new simulator: add a SimConfig here, implement a backend class in
# sims/sim_worker.py, and add suites below. See docs/adding_a_benchmark.md.

SIM_CONFIGS: Dict[str, SimConfig] = {
    "libero": SimConfig(
        name="libero",
        venv=os.environ.get("ROBO_EVAL_LIBERO_VENV", ".venvs/libero"),
        env_vars={
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero")),
        },
    ),
    "libero_pro": SimConfig(
        name="libero_pro",
        venv=os.environ.get("ROBO_EVAL_LIBERO_PRO_VENV", ".venvs/libero_pro"),
        env_vars={
            "LIBERO_CONFIG_PATH": os.environ.get("LIBERO_PRO_CONFIG_PATH", str(Path.home() / ".libero_pro")),
            # HDF5 libs from micromamba (see docs/robo_eval.md § HDF5 Installation)
            "LD_LIBRARY_PATH": _ld_library_path,
        },
    ),
    "libero_infinity": SimConfig(
        name="libero_infinity",
        # libero-infinity requires Python 3.11+ (Scenic 3 dependency).
        # It has its own venv separate from the base LIBERO venv (Python 3.8).
        # Default: .venvs/libero_infinity.
        # Override via ROBO_EVAL_LIBERO_INFINITY_VENV for custom layouts (e.g.
        # pointing to the libero-infinity repo's own .venv/).
        venv=os.environ.get("ROBO_EVAL_LIBERO_INFINITY_VENV", ".venvs/libero_infinity"),
        env_vars={"MUJOCO_GL": "egl"},
    ),
    "robocasa": SimConfig(
        name="robocasa",
        venv=os.environ.get("ROBO_EVAL_ROBOCASA_VENV", ".venvs/robocasa"),
    ),
    "robotwin": SimConfig(
        name="robotwin",
        venv=os.environ.get("ROBO_EVAL_ROBOTWIN_VENV", ".venvs/robotwin"),
    ),
}


# ---------------------------------------------------------------------------
# Suite definitions
# ---------------------------------------------------------------------------
# P1 suites (task perturbation) — not benchmark-scoped, kept as-is
P1_SUITES = [
    "libero_goal_task",
    "libero_spatial_task",
    "libero_10_task",
    "libero_object_task",
]

# P2 suites (position swap) — not benchmark-scoped, kept as-is
P2_SUITES = [
    "libero_goal_swap",
    "libero_spatial_swap",
    "libero_10_swap",
    "libero_object_swap",
]

# ---------------------------------------------------------------------------
# Benchmark-scoped suite registry  (authoritative source of truth)
# ---------------------------------------------------------------------------
# Maps benchmark name -> list of short (unqualified) suite names.
# Use qualify_suite() to obtain the fully-qualified suite name.
BENCHMARK_SUITES: Dict[str, List[str]] = {
    "libero": ["spatial", "object", "goal", "10"],
    "libero_pro": ["spatial_object", "goal_swap", "spatial_with_mug"],
    "libero_infinity": ["spatial", "object", "goal", "10"],
    "robocasa": ["kitchen"],
    "robotwin": ["aloha_agilex"],
}


def qualify_suite(benchmark: str, short_suite: str) -> str:
    """Return the fully-qualified suite name for a benchmark + short suite name.

    Examples:
        qualify_suite('libero_infinity', 'spatial') -> 'libero_infinity_spatial'
        qualify_suite('libero', 'spatial')     -> 'libero_spatial'
    """
    return f"{benchmark}_{short_suite}"


def get_suites_for_benchmark(benchmark: str) -> List[str]:
    """Return the list of short suite names for a given benchmark.

    Raises:
        ValueError: If the benchmark name is not in BENCHMARK_SUITES.
    """
    if benchmark not in BENCHMARK_SUITES:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Valid benchmarks: {', '.join(sorted(BENCHMARK_SUITES))}"
        )
    return BENCHMARK_SUITES[benchmark]


def get_qualified_suites(benchmark: str) -> List[str]:
    """Return fully-qualified suite names for a benchmark.

    Equivalent to [qualify_suite(benchmark, s) for s in get_suites_for_benchmark(benchmark)].
    """
    return [qualify_suite(benchmark, s) for s in get_suites_for_benchmark(benchmark)]


# ---------------------------------------------------------------------------
# Derived suite lists (kept for compatibility with existing callers)
# ---------------------------------------------------------------------------
# New code should use get_qualified_suites(benchmark) instead.
#
# LIBERO_SUITES: re-derived from BENCHMARK_SUITES — values are unchanged
#   (qualify_suite("libero", "spatial") == "libero_spatial", etc.)
LIBERO_SUITES = get_qualified_suites("libero")

# LIBERO_PRO_SUITES: new benchmark-scoped names (libero_pro_*) generated by
#   qualify_suite(), consistent with what SUITE_PRESETS["libero_pro"] returns.
#   Used by the native-mode filter in cli.py to exclude PRO suites from
#   lerobot-eval (which doesn't support LIBERO-PRO).
#   Note: env_wrapper.py / sim_worker.py use libero_* names for compatibility;
#   those are kept in env_wrapper.py:SUITE_MAX_STEPS.
LIBERO_PRO_SUITES = get_qualified_suites("libero_pro")

LIBERO_INFINITY_SUITES = get_qualified_suites("libero_infinity")

# Reverse map: qualified suite name -> benchmark name (used for sim routing)
_SUITE_TO_BENCHMARK: Dict[str, str] = {
    qualify_suite(benchmark, short): benchmark
    for benchmark, shorts in BENCHMARK_SUITES.items()
    for short in shorts
}


def get_benchmark_for_suite(qualified_suite: str) -> Optional[str]:
    """Return the benchmark name for a fully-qualified suite name, or None.

    Examples::

        get_benchmark_for_suite('libero_spatial')       -> 'libero'
        get_benchmark_for_suite('libero_pro_goal_swap') -> 'libero_pro'
        get_benchmark_for_suite('unknown')              -> None
    """
    return _SUITE_TO_BENCHMARK.get(qualified_suite)


# Suite presets: name -> list of fully-qualified suite names.
# All entries are now derived from BENCHMARK_SUITES via qualify_suite.
SUITE_PRESETS: Dict[str, List[str]] = {
    "all": (
        get_qualified_suites("libero")
        + get_qualified_suites("libero_pro")
        + get_qualified_suites("robocasa")
        + get_qualified_suites("robotwin")
    ),
    "libero": get_qualified_suites("libero"),
    "libero_pro": get_qualified_suites("libero_pro"),
    "libero_infinity": get_qualified_suites("libero_infinity"),
    "robocasa": get_qualified_suites("robocasa"),
    "robotwin": get_qualified_suites("robotwin"),
    "p1": P1_SUITES,
    "p2": P2_SUITES,
    "p1p2": P1_SUITES + P2_SUITES,
}

# Tasks per suite (always 10 for all known suites)
TASKS_PER_SUITE = 10

# ---------------------------------------------------------------------------
# Suite-specific max steps (rollout horizon)
# ---------------------------------------------------------------------------
# These match the lerobot-eval defaults (lerobot/envs/libero.py:88-94),
# which are calibrated to ~1.1-1.4× the longest training demo per suite.
# Using tighter limits saves ~40% of wall time on failed episodes for
# spatial/object/goal suites without affecting success detection.
# NOTE: A copy also exists in sims/env_wrapper.py:SUITE_MAX_STEPS for the
# sim venv (Python 3.8 cannot import robo_eval). Keep both in sync.
# Max rollout steps per suite. Also duplicated in sims/env_wrapper.py — keep both in sync.
# To add a new suite: add an entry here AND in sims/env_wrapper.py:SUITE_MAX_STEPS.
SUITE_MAX_STEPS: Dict[str, int] = {
    # Standard LIBERO suites (from lerobot TASK_SUITE_MAX_STEPS)
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
    # LIBERO-PRO suites — new benchmark-scoped names (libero_pro_*)
    # Conservative default: no published training-demo stats.
    "libero_pro_spatial_object": 300,
    "libero_pro_goal_swap": 300,
    "libero_pro_spatial_with_mug": 300,
    # LIBERO-PRO compatibility names used by env_wrapper.py / sim_worker.py
    "libero_spatial_object": 300,
    "libero_goal_swap": 300,
    "libero_spatial_with_mug": 300,
    # LIBERO-INF suites
    "libero_infinity_spatial": 300,
    "libero_infinity_object": 300,
    "libero_infinity_goal": 300,
    "libero_infinity_10": 520,
    # RoboCasa suite (matches SIM_MAX_STEPS["robocasa"] = 500)
    "robocasa_kitchen": 500,
    # RoboTwin suite (matches SIM_MAX_STEPS["robotwin"] = 300)
    "robotwin_aloha_agilex": 300,
    # gym-aloha task suites (gymnasium IDs used as suite names)
    "AlohaTransferCube-v0": 400,
    "AlohaInsertion-v0": 400,
}

# Default for unknown suites (matches lerobot fallback)
DEFAULT_MAX_STEPS = 500


def get_suite_max_steps(suite: str) -> int:
    """Return the max rollout steps for a given suite name.

    Checks SUITE_MAX_STEPS first, then falls back to DEFAULT_MAX_STEPS (500).
    """
    return SUITE_MAX_STEPS.get(suite, DEFAULT_MAX_STEPS)


def resolve_suites(suite_spec: str) -> List[str]:
    """Resolve a comma-separated suite spec into a list of suite names.

    Supports preset names (all, libero, libero_pro, p1, p2, p1p2)
    and individual suite names.
    """
    suites = []
    for part in suite_spec.split(","):
        part = part.strip()
        if part in SUITE_PRESETS:
            suites.extend(SUITE_PRESETS[part])
        else:
            suites.append(part)
    # Deduplicate while preserving order
    seen = set()
    result = []
    for s in suites:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def get_sim_for_suite(suite: str) -> str:
    """Determine which simulator (benchmark) a suite belongs to.

    Uses the _SUITE_TO_BENCHMARK reverse map built from BENCHMARK_SUITES.
    Falls back to 'libero_pro' for unknown suites (P1, P2, custom).
    """
    return _SUITE_TO_BENCHMARK.get(suite, "libero_pro")


# ---------------------------------------------------------------------------
# Evaluation mode definitions
# ---------------------------------------------------------------------------
@dataclass
class ModeConfig:
    """Configuration for an evaluation mode."""

    name: str
    no_vlm: bool = False
    no_think: bool = False
    vlm_model: Optional[str] = None
    is_native: bool = False  # native mode uses lerobot-eval directly

    @property
    def description(self) -> str:
        if self.is_native:
            return "Native lerobot-eval (no planner)"
        elif self.no_vlm:
            return "Direct VLA (no VLM planner)"
        else:
            return f"Planner with VLM ({self.vlm_model or 'default'})"


DEFAULT_VLM_MODEL = "vertex_ai/gemini-3-flash-preview"


# Public evaluation mode aliases
PUBLIC_MODE_NAMES = {
    # Primary semantic names
    "direct": "direct",      # VLA only, no VLM planner
    "planner": "planner",    # Full planner pipeline with VLM
    # Aliases
    "no-vlm": "direct",      # alias for direct
    "vlm": "planner",        # alias for planner
}

# Additional evaluation modes used for parity/debugging.
INTERNAL_MODE_NAMES = {
    "native": "native",
}

MODE_NAMES = {**PUBLIC_MODE_NAMES, **INTERNAL_MODE_NAMES}

# Descriptions for each evaluation mode
MODE_DESCRIPTIONS = {
    "direct": "Direct VLA (no VLM planner)",
    "native": "Native lerobot-eval (no planner)",
    "planner": "Full planner pipeline with VLM",
}


def resolve_mode(mode_spec: str) -> str:
    """Resolve an evaluation mode name to canonical form.

    Returns one of: 'direct', 'native', 'planner'.
    Raises ValueError for unknown mode names.
    """
    canonical = MODE_NAMES.get(mode_spec.lower())
    if canonical is None:
        valid = ["direct", "no-vlm", "planner", "vlm"]
        raise ValueError(
            f"Unknown --mode value '{mode_spec}'. "
            f"Valid options: {', '.join(valid)}"
        )
    return canonical

# Eval python for running run_sim_eval.py (needs litellm)
_litellm_venv = os.environ.get("ROBO_EVAL_LITELLM_VENV", str(PROJECT_ROOT / ".venvs" / "litellm"))
EVAL_PYTHON = Path(_litellm_venv) / "bin" / "python"
EVAL_SCRIPT = PROJECT_ROOT / "run_sim_eval.py"

# VLM proxy
VLM_START_SCRIPT = PROJECT_ROOT / "scripts" / "start_vlm.sh"
DEFAULT_VLM_PORT = 4000

# Native eval configs (for --mode native)
NATIVE_EVAL_CONFIGS: Dict[str, Dict] = {
    "pi05": {
        "venv": os.environ.get("ROBO_EVAL_VLA_VENV", ".venvs/vla"),
        "policy_path": "lerobot/pi05_libero_finetuned",
    },
    "smolvla": {
        "venv": os.environ.get("ROBO_EVAL_SMOLVLA_VENV", ".venvs/smolvla"),
        "policy_path": "HuggingFaceVLA/smolvla_libero",
    },
    # OpenVLA native uses a custom script, not lerobot-eval
    "openvla": {
        "venv": os.environ.get("ROBO_EVAL_OPENVLA_NATIVE_VENV", ".venvs/openvla_native"),
        "script": "scripts/run_openvla_native_eval.py",
        "custom": True,
    },
}

# ---------------------------------------------------------------------------
# Resource cost estimates (GB RAM per component instance)
# ---------------------------------------------------------------------------
RAM_COSTS_GB: Dict[str, float] = {
    # VLA model servers (GPU/unified memory)
    "pi05": 15.0,       # ~15 GB for pi05_libero_finetuned
    "vqbet": 1.0,       # ~1 GB for lerobot/vqbet_pusht (small VQ-VAE + transformer head)
    "tdmpc2": 2.0,      # ~2 GB for tdmpc2 metaworld MT80 (small world-model + Q-fn)
    "openvla": 15.0,    # ~15 GB for openvla-7b
    "smolvla": 3.0,     # ~3 GB for smolvla (0.45B params)
    "cosmos": 5.0,      # ~5 GB for Cosmos-Policy-RoboCasa-Predict2-2B
    "internvla": 7.0,   # ~7 GB for InternVLA-A1-3B-RoboTwin
    # Simulators (CPU RAM)
    "sim_worker": 2.0,  # ~2 GB per sim worker process
    # VLM proxy (negligible — proxies to external service)
    "vlm_proxy": 0.5,
    # Orchestrator/eval process
    "eval_process": 0.5,
}


def estimate_ram_usage(
    vla_name: str,
    num_vla_servers: int = 1,
    num_sim_workers: int = 1,
    use_vlm: bool = False,
) -> Dict[str, float]:
    """Estimate total RAM usage for a given configuration.

    Returns dict with per-component and total estimates in GB.
    """
    vla_cost = RAM_COSTS_GB.get(vla_name, 10.0) * num_vla_servers
    sim_cost = RAM_COSTS_GB["sim_worker"] * num_sim_workers
    vlm_cost = RAM_COSTS_GB["vlm_proxy"] if use_vlm else 0.0
    eval_cost = RAM_COSTS_GB["eval_process"] * num_sim_workers

    return {
        "vla_servers": vla_cost,
        "sim_workers": sim_cost,
        "vlm_proxy": vlm_cost,
        "eval_processes": eval_cost,
        "total": vla_cost + sim_cost + vlm_cost + eval_cost,
    }
