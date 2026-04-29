"""Configuration constants for VLA servers, simulators, and suite definitions."""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Vendor repos directory.
# Default: ~/.local/share/roboeval/vendors/
# Override via ROBOEVAL_VENDORS_DIR environment variable.
# .resolve() ensures an absolute path even if the env var is relative.
VENDORS_DIR = Path(os.environ.get(
    "ROBOEVAL_VENDORS_DIR",
    str(Path.home() / ".local" / "share" / "roboeval" / "vendors"),
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
    model_id: str
    startup_timeout: int = 300  # seconds

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    @property
    def venv_python(self) -> Path:
        return PROJECT_ROOT / self.venv / "bin" / "python"

# To add a new VLA: add a VLAConfig entry below, create a policy server in
# sims/vla_policies/. See docs/extending.md.
VLA_CONFIGS: Dict[str, VLAConfig] = {
    "pi05": VLAConfig(
        name="pi05",
        port=5100,
        venv=os.environ.get("ROBOEVAL_VLA_VENV", ".venvs/pi05"),
        model_id="lerobot/pi05_libero_finetuned",
    ),
    "openvla": VLAConfig(
        name="openvla",
        port=5101,
        venv=os.environ.get("ROBOEVAL_OPENVLA_VENV", ".venvs/openvla"),
        model_id="openvla/openvla-7b-finetuned-libero-spatial",
        startup_timeout=600,
    ),
    "smolvla": VLAConfig(
        name="smolvla",
        port=5102,
        venv=os.environ.get("ROBOEVAL_SMOLVLA_VENV", ".venvs/smolvla"),
        model_id="HuggingFaceVLA/smolvla_libero",
    ),
    "cosmos": VLAConfig(
        name="cosmos",
        port=5103,
        venv=os.environ.get("ROBOEVAL_COSMOS_VENV", ".venvs/cosmos"),
        model_id="nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
        startup_timeout=600,
    ),
    "internvla": VLAConfig(
        name="internvla",
        port=5104,
        venv=os.environ.get("ROBOEVAL_INTERNVLA_VENV", ".venvs/internvla"),
        model_id="InternRobotics/InternVLA-A1-3B-RoboTwin",
    ),
    "vqbet": VLAConfig(
        name="vqbet",
        port=5108,
        venv=os.environ.get("ROBOEVAL_VQBET_VENV", ".venvs/vqbet"),
        model_id="lerobot/vqbet_pusht",
    ),
    "tdmpc2": VLAConfig(
        name="tdmpc2",
        port=5109,
        venv=os.environ.get("ROBOEVAL_TDMPC2_VENV", ".venvs/tdmpc2"),
        model_id="nicklashansen/tdmpc2",
    ),
    "groot": VLAConfig(
        name="groot",
        port=5105,
        # GR00T requires a dedicated venv (nvidia/Isaac-GR00T deps conflict with lerobot).
        # Default: .venvs/groot  — override via ROBOEVAL_GROOT_VENV.
        # Set ROBOEVAL_GROOT_VENV explicitly if using a custom layout.
        venv=os.environ.get("ROBOEVAL_GROOT_VENV", ".venvs/groot"),
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
# sim venv (Python 3.8 cannot import roboeval). Keep both in sync.
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
