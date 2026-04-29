"""Slim subprocess launcher for VLA policy servers and sim_worker.

Used by ``roboeval serve`` to start and manage the per-VLA host-venv process
and the sim_worker process.

Signal handling:
    - SIGINT / SIGTERM: graceful shutdown of all managed subprocesses.
    - atexit: best-effort cleanup in case signals are missed.
"""

from __future__ import annotations

import atexit
import errno
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Global registry of all processes we started so atexit can clean up.  We track
# the *process group* (pgid) too so cleanup can fan out to grandchildren
# (uvicorn workers, MuJoCo render threads, …) instead of orphaning them.
_MANAGED_PROCS: list[subprocess.Popen] = []
_GRACE_SECONDS = 5.0


def _cleanup_all() -> None:
    """Terminate all managed subprocesses and their entire process group.

    Sends SIGTERM, waits up to ``_GRACE_SECONDS`` seconds, then SIGKILLs any
    survivor.  This also cleans up child workers that may keep accelerator
    resources allocated after the parent CLI exits.
    """
    for proc in list(_MANAGED_PROCS):
        if proc.poll() is None:
            logger.info("Terminating process pid=%d (group)", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
    deadline = time.time() + _GRACE_SECONDS
    for proc in list(_MANAGED_PROCS):
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            logger.warning("pid=%d did not exit after SIGTERM; sending SIGKILL", proc.pid)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=2.0)
            except OSError:
                pass


def _signal_handler(signum: int, _frame: Any) -> None:
    logger.info("Caught signal %d — shutting down.", signum)
    _cleanup_all()
    sys.exit(0)


atexit.register(_cleanup_all)
# NOTE: signal handlers are intentionally NOT installed at module import time.
# Call install_signal_handlers() from CLI entry points that own the process
# lifecycle (e.g. roboeval serve).  Importing this module must be side-effect
# free so that orchestrators and test suites are unaffected.


def install_signal_handlers() -> None:
    """Install SIGINT / SIGTERM handlers that gracefully terminate managed processes.

    Must be called explicitly from CLI entry points (``roboeval serve``) that
    own the process lifecycle.  Do **not** call at module import time — callers
    that merely *use* ``start_vla`` / ``start_sim`` should not have their signal
    handling silently overridden.
    """
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# VLA server launcher
# ---------------------------------------------------------------------------

# Mapping: vla_name -> module path within the policy package
_VLA_MODULE_MAP: dict[str, str] = {
    "pi05": "sims.vla_policies.pi05_policy",
    "vqbet": "sims.vla_policies.vqbet_policy",
    "act": "sims.vla_policies.act_policy",
    "diffusion_policy": "sims.vla_policies.diffusion_policy_policy",
    "tdmpc2": "sims.vla_policies.tdmpc2_policy",
    "smolvla": "sims.vla_policies.smolvla_policy",
    "openvla": "sims.vla_policies.openvla_policy",
    "octo": "sims.vla_policies.octo_policy",
    "cosmos": "sims.vla_policies.cosmos_policy",
    "groot": "sims.vla_policies.groot_policy",
    "internvla": "sims.vla_policies.internvla_policy",
}

# Default venv paths per VLA (relative to project root; override via config)
_VLA_DEFAULT_VENVS: dict[str, str] = {
    "pi05": ".venvs/pi05",
    "vqbet": ".venvs/vqbet",
    "act": ".venvs/act",
    "diffusion_policy": ".venvs/diffusion_policy",
    "tdmpc2": ".venvs/tdmpc2",
    "smolvla": ".venvs/smolvla",
    "openvla": ".venvs/openvla",
    "octo": ".venvs/octo",
    "cosmos": ".venvs/cosmos",
    "groot": ".venvs/groot",
    "internvla": ".venvs/internvla",
}

# Default ports
_VLA_DEFAULT_PORTS: dict[str, int] = {
    "pi05": 5100,
    "vqbet": 5108,
    "act": 5106,
    "diffusion_policy": 5107,
    "tdmpc2": 5109,
    "smolvla": 5102,
    "openvla": 5101,
    "octo": 5110,
    "cosmos": 5103,
    "groot": 5104,
    "internvla": 5105,
}

_SIM_DEFAULT_PORTS: dict[str, int] = {
    "libero": 5300,
    "libero_pro": 5301,
    "robocasa": 5302,
    "robotwin": 5303,
    "aloha_gym": 5304,
    "gym_pusht": 5305,
    "metaworld": 5306,
    "libero_infinity": 5308,
}

_SIM_DEFAULT_VENVS: dict[str, str] = {
    "libero": ".venvs/libero",
    "libero_pro": ".venvs/libero_pro",
    "robocasa": ".venvs/robocasa",
    "robotwin": ".venvs/robotwin",
    "aloha_gym": ".venvs/aloha_gym",
    "gym_pusht": ".venvs/gym_pusht",
    "metaworld": ".venvs/metaworld",
    # libero_infinity requires Python 3.11 or newer.
    "libero_infinity": ".venvs/libero_infinity",
}

_KNOWN_SIMS = set(_SIM_DEFAULT_PORTS.keys())


def _assert_port_free(host: str, port: int) -> None:
    """Raise immediately if (host, port) is already bound (errno EADDRINUSE).

    A 1-line preflight beats a 120 s health-poll mystery.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"Port {port} on {host} is already in use. "
                    f"Another VLA/sim_worker may be running. "
                    f"Free it with `lsof -i :{port}` or pass --vla-port/--sim-port."
                ) from exc
            raise


def _tail_log(path: Path, n: int = 30) -> str:
    """Return the last *n* lines of *path*, or a placeholder if unreadable."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = min(size, 64 * 1024)
            f.seek(size - block)
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except OSError:
        return f"<could not read {path}>"


def _resolve_python(venv_path: str | None, project_root: Path, must_exist: bool = False) -> str:
    """Resolve the Python executable from a venv path or fall back to sys.executable."""
    if venv_path:
        venv = Path(venv_path)
        if not venv.is_absolute():
            venv = project_root / venv
        candidates = [
            venv / "bin" / "python",
            venv / "bin" / "python3",
            venv / "Scripts" / "python.exe",  # Windows
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        msg = (
            f"venv {venv} not found (looked for bin/python, bin/python3). "
            f"Create it: `python -m venv {venv} && {venv}/bin/pip install -r requirements.txt`. "
            f"To skip the venv, omit --vla-venv/--sim-venv from the command."
        )
        if must_exist:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        logger.warning("Falling back to sys.executable — most VLAs will fail with ModuleNotFoundError.")
    return sys.executable


def _open_log(name: str, logs_dir: Path) -> Any:
    """Open a log file for subprocess stdout/stderr."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"
    logger.info("Logging %s to %s", name, log_path)
    return open(log_path, "a")  # noqa: SIM115


def _poll_health(url: str, timeout: float = 60.0, interval: float = 2.0) -> tuple[bool, str]:
    """Poll GET /health until ready, error, or timeout.

    Returns ``(ready, last_error)`` — ``last_error`` is the most recent
    /health error string, useful for diagnosing OOM at load time.
    """
    last_error = ""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url + "/health", timeout=5.0)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            err = data.get("error")
            if err:
                last_error = str(err)
                # Persistent load errors are terminal for this launch.
                return False, last_error
            ready = data.get("ready")
            if ready is None:
                ready = data.get("status") in ("ok", "ready")
            if resp.ok and ready:
                return True, ""
        except (requests.RequestException, ValueError):
            pass
        time.sleep(interval)
    return False, last_error or "timeout"


def _launch_and_wait(
    *,
    log_name: str,
    display_kind: str,
    display_name: str,
    port: int,
    cmd: list[str],
    logs_dir: Path,
    project_root: Path,
    env: dict[str, str],
    health_timeout: float,
) -> subprocess.Popen:
    """Launch a managed subprocess and wait for its /health endpoint."""
    log_f = _open_log(log_name, logs_dir)
    logger.info("Starting %s: %s (port=%d)", display_kind, display_name, port)
    logger.debug("Command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=log_f,
        cwd=str(project_root),
        env=env,
        start_new_session=True,
    )
    _MANAGED_PROCS.append(proc)

    url = f"http://localhost:{port}"
    logger.info("Waiting for %s to be ready at %s/health ...", display_kind, url)
    ready, last_error = _poll_health(url, timeout=health_timeout)
    if not ready:
        log_path = logs_dir / f"{log_name}.log"
        tail = _tail_log(log_path, n=30)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except OSError:
            pass
        _MANAGED_PROCS.remove(proc)
        raise RuntimeError(
            f"{display_kind} {display_name!r} did not become ready within {health_timeout}s.\n"
            f"  /health last error: {last_error or 'no response'}\n"
            f"  log tail ({log_path}, last 30 lines):\n{tail}"
        )

    logger.info("%s %r is ready (port=%d)", display_kind, display_name, port)
    return proc


def start_vla(
    vla_name: str,
    port: int | None = None,
    venv_path: str | None = None,
    model_id: str | None = None,
    logs_dir: str | Path = "logs",
    health_timeout: float = 120.0,
    project_root: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    """Launch a VLA policy server subprocess and wait for it to become ready.

    Parameters
    ----------
    vla_name:
        One of the known VLA names (``pi05``, ``smolvla``, etc.).
    port:
        Port to listen on.  Defaults to the VLA's canonical port.
    venv_path:
        Path to the Python venv to use.  Defaults to ``_VLA_DEFAULT_VENVS[vla_name]``.
    model_id:
        Optional model ID to pass with ``--model-id``.
    logs_dir:
        Directory for stdout/stderr log files.
    health_timeout:
        Seconds to wait for /health to return ready.
    project_root:
        Project root directory.  Defaults to the roboeval repo root.
    extra_env:
        Extra env vars to set for the subprocess.

    Returns
    -------
    subprocess.Popen
        The running subprocess handle.

    Raises
    ------
    ValueError:
        If vla_name is unknown.
    RuntimeError:
        If the server does not become healthy within health_timeout.
    """
    if vla_name not in _VLA_MODULE_MAP:
        raise ValueError(
            f"Unknown VLA: {vla_name!r}. "
            f"Known VLAs: {sorted(_VLA_MODULE_MAP.keys())}"
        )

    port = port or _VLA_DEFAULT_PORTS.get(vla_name, 5100)
    _assert_port_free("0.0.0.0", port)

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    project_root = Path(project_root)

    venv = venv_path or _VLA_DEFAULT_VENVS.get(vla_name)
    python = _resolve_python(venv, project_root, must_exist=bool(venv_path))
    module = _VLA_MODULE_MAP[vla_name]
    logs_dir = Path(logs_dir)

    cmd = [python, "-m", module, "--port", str(port)]
    if model_id:
        cmd += ["--model-id", model_id]

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return _launch_and_wait(
        log_name=f"vla_{vla_name}",
        display_kind="VLA server",
        display_name=vla_name,
        port=port,
        cmd=cmd,
        logs_dir=logs_dir,
        project_root=project_root,
        env=env,
        health_timeout=health_timeout,
    )


def start_sim(
    backend: str,
    port: int | None = None,
    venv_path: str | None = None,
    headless: bool = True,
    logs_dir: str | Path = "logs",
    health_timeout: float = 120.0,
    project_root: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Launch a sim_worker subprocess and wait for it to become ready.

    Parameters
    ----------
    backend:
        One of ``libero``, ``libero_pro``, ``robocasa``, ``robotwin``.
    port:
        Port to listen on.  Defaults to the sim's canonical port.
    venv_path:
        Path to the Python venv.  Defaults to ``_SIM_DEFAULT_VENVS[backend]``.
    headless:
        Pass ``--headless`` flag to sim_worker.
    logs_dir:
        Directory for stdout/stderr log files.
    health_timeout:
        Seconds to wait for /health to return ready.
    project_root:
        Project root directory.
    extra_env:
        Extra env vars for the subprocess.
    extra_args:
        Extra command-line args passed to sim_worker.

    Returns
    -------
    subprocess.Popen
        The running subprocess handle.

    Raises
    ------
    RuntimeError:
        If the server does not become healthy within health_timeout.
    """
    if backend not in _KNOWN_SIMS:
        raise ValueError(
            f"Unknown sim backend: {backend!r}. "
            f"Known sims: {sorted(_KNOWN_SIMS)}"
        )

    port = port or _SIM_DEFAULT_PORTS.get(backend, 5300)
    _assert_port_free("0.0.0.0", port)

    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    project_root = Path(project_root)

    venv = venv_path or _SIM_DEFAULT_VENVS.get(backend, ".venvs/libero")
    python = _resolve_python(venv, project_root, must_exist=bool(venv_path))
    logs_dir = Path(logs_dir)

    cmd = [
        python, "-m", "sims.sim_worker",
        # sim_worker.py argparse expects --sim.
        "--sim", backend,
        "--port", str(port),
    ]
    if headless:
        cmd.append("--headless")
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    # RoboTwin uses SAPIEN (Vulkan renderer).  When DISPLAY is set, SAPIEN
    # tries to use the Vulkan+X11 path, which triggers an XCB threading
    # assertion crash ("Assertion '!xcb_xlib_unknown_seq_number' failed")
    # before /init ever returns.  Unsetting DISPLAY forces SAPIEN into EGL
    # offscreen mode, which is both stable and correct for eval workloads.
    if backend == "robotwin":
        env.pop("DISPLAY", None)

    return _launch_and_wait(
        log_name=f"sim_{backend}",
        display_kind="sim_worker",
        display_name=backend,
        port=port,
        cmd=cmd,
        logs_dir=logs_dir,
        project_root=project_root,
        env=env,
        health_timeout=health_timeout,
    )


def wait_for_exit(procs: list[subprocess.Popen]) -> None:
    """Block until all given processes exit or user sends SIGINT."""
    try:
        while True:
            time.sleep(1.0)
            dead = [p for p in procs if p.poll() is not None]
            if dead:
                logger.warning(
                    "%d process(es) exited unexpectedly: pids=%s",
                    len(dead),
                    [p.pid for p in dead],
                )
                for p in dead:
                    procs.remove(p)
            if not procs:
                break
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down servers.")
        _cleanup_all()
