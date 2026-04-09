"""
Server lifecycle management for VLA policy servers, sim workers, and VLM proxy.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from .config import (
    DEFAULT_PROXY_PORT,
    DEFAULT_VLM_MODEL,
    DOCKER_IMAGES,
    PROJECT_ROOT,
    SIM_CONFIGS,
    VLA_CONFIGS,
    VLA_DOCKER_IMAGES,
    VLAConfig,
    SimConfig,
    VLM_START_SCRIPT,
    DEFAULT_VLM_PORT,
    validate_port,
)

logger = logging.getLogger(__name__)


def check_health(url: str, timeout: float = 2.0) -> Tuple[bool, dict]:
    """Check if a server is healthy.

    Returns (is_healthy, response_dict).
    """
    try:
        resp = requests.get(f"{url}/health", timeout=timeout)
        data = resp.json()
        # VLA servers return {"ready": true/false}
        # Sim workers return {"status": "ok"}
        is_ready = data.get("ready", True) and data.get("status", "ok") == "ok"
        return is_ready, data
    except Exception:
        return False, {}


def check_vla_health(name: str, port: Optional[int] = None) -> Tuple[bool, dict]:
    """Check if a VLA server is healthy."""
    if name in VLA_CONFIGS:
        cfg = VLA_CONFIGS[name]
        p = port or cfg.port
    else:
        p = port or 5100
    return check_health(f"http://localhost:{p}")


def list_servers() -> List[Dict]:
    """List all known server endpoints and their status."""
    results = []

    # Check VLA servers
    for name, cfg in VLA_CONFIGS.items():
        healthy, data = check_health(cfg.url)
        results.append({
            "type": "vla",
            "name": name,
            "port": cfg.port,
            "url": cfg.url,
            "healthy": healthy,
            "model": cfg.model_id,
            "details": data,
        })

    # Check VLM proxy
    vlm_healthy, vlm_data = check_health(f"http://localhost:{DEFAULT_VLM_PORT}")
    results.append({
        "type": "vlm",
        "name": "litellm-proxy",
        "port": DEFAULT_VLM_PORT,
        "url": f"http://localhost:{DEFAULT_VLM_PORT}",
        "healthy": vlm_healthy,
        "model": DEFAULT_VLM_MODEL,
        "details": vlm_data,
    })

    return results


def start_vla_server(
    name: str,
    port: Optional[int] = None,
    wait: bool = True,
    log_file: Optional[str] = None,
    model_id: Optional[str] = None,
    runtime: str = "venv",
    gpu_id: Optional[str] = None,
    display: Optional[str] = None,
) -> Optional[int | str]:
    """Start a VLA policy server.

    Invokes the policy module directly via the venv Python binary — no bash
    script required.  The bash scripts in scripts/ delegate to this function
    and are kept only as thin convenience wrappers.

    When runtime='docker', starts a Docker container instead of a subprocess.

    Args:
        name: VLA name (pi05, openvla, smolvla)
        port: Override port (uses default from config if not specified)
        wait: Wait for server to become healthy
        log_file: Path for server log output
        model_id: Override HuggingFace model ID (uses config default if None)
        runtime: 'docker' or 'venv' (resolved from 'auto' by caller)
        gpu_id: GPU device ID (for Docker --gpus device=N)
        display: DISPLAY value for X11 forwarding (debug window)

    Returns:
        PID (int) for venv mode, or container ID (str) for docker mode,
        or None on failure.
    """
    if name not in VLA_CONFIGS:
        logger.error("Unknown VLA '%s'. Available: %s", name, list(VLA_CONFIGS.keys()))
        return None

    cfg = VLA_CONFIGS[name]
    actual_port = port or cfg.port

    # Check if already running
    healthy, _ = check_health(f"http://localhost:{actual_port}")
    if healthy:
        logger.info("%s server already running on port %d", name, actual_port)
        return None

    # ── Docker runtime path ──
    if runtime == "docker":
        from .docker_backend import docker_run
        image = VLA_DOCKER_IMAGES.get(name)
        if not image:
            logger.error("No Docker image for VLA '%s'. Available: %s", name, list(VLA_DOCKER_IMAGES.keys()))
            return None

        env_vars = {"MUJOCO_GL": "egl", "VLA_PORT": str(actual_port)}
        container_name = f"robo-eval-vla-{name}-{actual_port}"
        extra_args = ["--port", str(actual_port)]

        logger.info("Starting %s VLA in Docker container on port %d...", name, actual_port)
        try:
            container_id = docker_run(
                image=image,
                port=actual_port,
                env=env_vars,
                gpu_id=gpu_id,
                name=container_name,
                extra_args=extra_args,
                display=display,
            )
        except RuntimeError as e:
            logger.error("Failed to start %s Docker container: %s", name, e)
            return None

        if wait:
            logger.info("  Waiting for health check (timeout: %ds)...", cfg.startup_timeout)
            start_time = time.time()
            while time.time() - start_time < cfg.startup_timeout:
                healthy, data = check_health(f"http://localhost:{actual_port}")
                if healthy and data.get("ready", False):
                    elapsed = int(time.time() - start_time)
                    logger.info("  %s server ready after %ds", name, elapsed)
                    return container_id
                time.sleep(5)
            logger.warning("  %s server not ready after %ds", name, cfg.startup_timeout)

        return container_id

    # ── Venv runtime path (existing behavior) ──
    log_path = log_file or f"/tmp/{name}_policy.log"
    logger.info("Starting %s server on port %d...", name, actual_port)
    logger.info("  Log: %s", log_path)

    env = os.environ.copy()
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    env["VLA_PORT"] = str(actual_port)

    # Resolve model ID: explicit arg > env var > config default
    if name == "openvla":
        resolved_model_id = (
            model_id
            or os.environ.get("OPENVLA_MODEL_ID")
            or cfg.model_id
        )
    else:
        resolved_model_id = (
            model_id
            or os.environ.get("VLA_MODEL")
            or cfg.model_id
        )

    # Build command: invoke the policy module directly through the venv Python.
    # This is equivalent to what the bash scripts do after `source venv/activate`.
    policy_module = f"sims.vla_policies.{name}_policy"
    cmd: List[str] = [
        str(cfg.venv_python),
        "-m", policy_module,
        "--model-id", resolved_model_id,
        "--port", str(actual_port),
    ]
    # openvla requires an unnorm-key for action decoding
    if name == "openvla":
        unnorm_key = os.environ.get("OPENVLA_UNNORM_KEY", "libero_spatial")
        cmd += ["--unnorm-key", unnorm_key]

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            preexec_fn=os.setsid,
        )

    pid = proc.pid
    logger.info("  PID: %d", pid)

    # Save PID for later management
    pid_file = Path(f"/tmp/{name}_server.pid")
    pid_file.write_text(str(pid))

    if wait:
        logger.info("  Waiting for health check (timeout: %ds)...", cfg.startup_timeout)
        start_time = time.time()
        while time.time() - start_time < cfg.startup_timeout:
            healthy, data = check_health(f"http://localhost:{actual_port}")
            if healthy and data.get("ready", False):
                elapsed = int(time.time() - start_time)
                logger.info("  %s server ready after %ds", name, elapsed)
                return pid
            time.sleep(5)
        logger.warning("  %s server not ready after %ds", name, cfg.startup_timeout)

    return pid


def stop_vla_server(
    name: str,
    port: Optional[int] = None,
    runtime: str = "venv",
    container_id: Optional[str] = None,
) -> bool:
    """Stop a VLA policy server.

    Args:
        name: VLA name (pi05, openvla, smolvla)
        port: Override port
        runtime: 'docker' or 'venv'
        container_id: Container ID to stop (docker mode)

    Returns:
        True if server was stopped.
    """
    if name in VLA_CONFIGS:
        actual_port = port or VLA_CONFIGS[name].port
    else:
        actual_port = port
        if not actual_port:
            logger.error("Unknown VLA '%s' and no port specified.", name)
            return False

    # C1: In docker mode, use docker stop — never os.killpg()
    if runtime == "docker":
        from .docker_backend import docker_stop, docker_stop_by_name
        if container_id:
            return docker_stop(container_id)
        else:
            container_name = f"robo-eval-vla-{name}-{actual_port}"
            return docker_stop_by_name(container_name)

    # ── Venv mode (existing behavior) ──
    # Try PID file first
    pid_file = Path(f"/tmp/{name}_server.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Stopped %s server (PID %d)", name, pid)
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            pid_file.unlink(missing_ok=True)

    # Fall back to fuser
    try:
        subprocess.run(
            ["fuser", "-k", f"{actual_port}/tcp"],
            capture_output=True,
            timeout=10,
        )
        logger.info("Stopped process on port %d", actual_port)
        return True
    except Exception:
        pass

    logger.info("No %s server found on port %d", name, actual_port)
    return False


def start_vlm_proxy(
    port: int = DEFAULT_VLM_PORT,
    model: Optional[str] = None,
    wait: bool = True,
) -> Optional[int]:
    """Start the litellm VLM proxy server.

    Invokes the litellm binary directly from its venv — no bash script
    required.  Replicates scripts/start_vlm.sh exactly:
      - Model:    VLM_MODEL env var (default: DEFAULT_VLM_MODEL)
      - Port:     LITELLM_PORT env var (default: DEFAULT_VLM_PORT)
      - Vertex AI credentials set from gcloud ADC path
      - Flags:    --drop_params
    """
    healthy, _ = check_health(f"http://localhost:{port}")
    if healthy:
        logger.info("VLM proxy already running on port %d", port)
        return None

    log_path = "/tmp/vlm_proxy.log"

    # Resolve model: explicit arg > VLM_MODEL env var > DEFAULT_VLM_MODEL constant
    actual_model = model or os.environ.get("VLM_MODEL", DEFAULT_VLM_MODEL)

    logger.info("Starting VLM proxy (%s) on port %d...", actual_model, port)
    logger.info("  Log: %s", log_path)

    env = os.environ.copy()
    # Vertex AI credentials — required only when using vertex_ai/ models.
    if actual_model.startswith("vertex_ai/"):
        if "GOOGLE_APPLICATION_CREDENTIALS" not in env:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS env var required for Vertex AI VLM models. "
                "Set it to your GCP service account key path, or use a non-Vertex model."
            )
        if "VERTEXAI_PROJECT" not in env:
            raise RuntimeError(
                "VERTEXAI_PROJECT env var required for Vertex AI VLM models."
            )
        env.setdefault("VERTEXAI_LOCATION", "global")
    env["LITELLM_PORT"] = str(port)
    env["VLM_MODEL"] = actual_model

    # Invoke litellm binary directly from the dedicated venv
    _litellm_venv = os.environ.get("ROBO_EVAL_LITELLM_VENV", str(PROJECT_ROOT / ".venvs" / "litellm"))
    litellm_bin = Path(_litellm_venv) / "bin" / "litellm"
    cmd = [
        str(litellm_bin),
        "--model", actual_model,
        "--port", str(port),
        "--drop_params",
    ]

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            preexec_fn=os.setsid,
        )

    pid = proc.pid
    logger.info("  PID: %d", pid)
    Path("/tmp/vlm_proxy.pid").write_text(str(pid))

    if wait:
        logger.info("  Waiting for VLM proxy (timeout: 120s)...")
        for _ in range(60):
            healthy, _ = check_health(f"http://localhost:{port}")
            if healthy:
                logger.info("  VLM proxy ready on port %d", port)
                return pid
            time.sleep(2)
        logger.warning("  VLM proxy not ready after 120s")

    return pid


def stop_vlm_proxy(port: int = DEFAULT_VLM_PORT) -> bool:
    """Stop the VLM proxy."""
    pid_file = Path("/tmp/vlm_proxy.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Stopped VLM proxy (PID %d)", pid)
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            pid_file.unlink(missing_ok=True)

    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=10)
        logger.info("Stopped VLM proxy on port %d", port)
        return True
    except Exception:
        pass

    return False


def start_proxy(
    backend_urls: List[str],
    port: int = DEFAULT_PROXY_PORT,
    wait: bool = True,
    log_file: str = "/tmp/vla_proxy.log",
    health_interval: float = 10.0,
) -> Optional[int]:
    """Start the VLA round-robin proxy server.

    Args:
        backend_urls: URLs of VLA backend servers.
        port: Port for the proxy to listen on.
        wait: Wait for proxy to become healthy.
        log_file: Path for proxy log output.
        health_interval: Seconds between backend health checks.

    Returns:
        PID of the proxy process, or None if already running.
    """
    healthy, _ = check_health(f"http://localhost:{port}")
    if healthy:
        logger.info("VLA proxy already running on port %d", port)
        return None

    cmd = [
        sys.executable, "-m", "robo_eval.proxy",
        "--port", str(port),
        "--health-interval", str(health_interval),
        "--backends", *backend_urls,
    ]

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            preexec_fn=os.setsid,
        )

    pid = proc.pid
    pid_file = Path(f"/tmp/vla_proxy_{port}.pid")
    pid_file.write_text(str(pid))
    logger.info("Starting VLA proxy on port %d (PID %d)...", port, pid)

    if wait:
        # Proxy startup is fast (<2s), but backends might not be healthy yet
        # The proxy itself starts immediately; its /health returns ready=true
        # once at least one backend is healthy
        for _ in range(30):
            healthy, data = check_health(f"http://localhost:{port}")
            if healthy:
                n_backends = data.get("backends_healthy", "?")
                logger.info("VLA proxy ready on port %d (%s healthy backend(s))", port, n_backends)
                return pid
            time.sleep(2)
        logger.warning("VLA proxy not ready after 60s")

    return pid


def stop_proxy(port: int = DEFAULT_PROXY_PORT) -> bool:
    """Stop the VLA round-robin proxy.

    Args:
        port: Port the proxy is running on.

    Returns:
        True if the proxy was stopped.
    """
    pid_file = Path(f"/tmp/vla_proxy_{port}.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Stopped VLA proxy (PID %d)", pid)
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            pid_file.unlink(missing_ok=True)

    # Fall back to fuser
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=10,
        )
        logger.info("Stopped process on port %d", port)
        return True
    except Exception:
        pass

    return False


def _wait_port_free(port: int, timeout: float = 15.0) -> bool:
    """Wait until the given TCP port is free to bind, or timeout expires.

    Returns True if the port is free before the deadline, False otherwise.
    Used before launching a new sim worker to avoid EADDRINUSE races when
    the previous worker is still in TIME_WAIT.
    """
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return True
            except OSError:
                time.sleep(0.5)
    return False


class SimWorkerPool:
    """Manages a pool of sim_worker processes.

    Each worker gets its own port and runs in a separate process.
    Supports both venv and docker runtimes.
    """

    def __init__(
        self,
        sim_type: str,
        base_port: int,
        num_workers: int,
        headless: bool = True,
        logs_dir: Optional[Path] = None,
        runtime: str = "venv",
        display: Optional[str] = None,
    ):
        validate_port(base_port, "sim_base_port")
        self.sim_type = sim_type
        self.sim_config = SIM_CONFIGS[sim_type]
        self.base_port = base_port
        self.num_workers = num_workers
        self.headless = headless
        self.logs_dir = logs_dir or Path("/tmp")
        self.runtime = runtime
        self.display = display
        self.processes: Dict[int, subprocess.Popen] = {}  # port -> process (venv mode)
        self.containers: Dict[int, str] = {}  # port -> container_id (docker mode)
        self.log_paths: Dict[int, Path] = {}  # port -> log path

    def start_worker(self, slot: int) -> int:
        """Start a sim worker on the given slot. Returns port."""
        port = self.base_port + slot

        # ── Docker runtime ──
        if self.runtime == "docker":
            from .docker_backend import docker_run
            image = DOCKER_IMAGES.get(self.sim_type)
            if not image:
                raise RuntimeError(f"No Docker image for sim '{self.sim_type}'")

            env_vars = {"MUJOCO_GL": "egl" if self.headless else "glfw"}
            for k, v in self.sim_config.env_vars.items():
                env_vars[k] = v

            container_name = f"robo-eval-sim-{self.sim_type}-{port}"
            extra_args = ["--sim", self.sim_type, "--port", str(port)]
            if self.headless:
                extra_args.append("--headless")

            container_id = docker_run(
                image=image,
                port=port,
                env=env_vars,
                name=container_name,
                extra_args=extra_args,
                display=self.display,
                volumes=self.sim_config.volumes or [],
            )
            self.containers[port] = container_id
            self.processes[port] = None  # keep processes dict consistent for docker mode
            return port

        # ── Venv runtime (existing behavior) ──
        log_path = self.logs_dir / f"sim_worker_{port}.log"
        self.log_paths[port] = log_path

        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        for k, v in self.sim_config.env_vars.items():
            env[k] = v

        if not _wait_port_free(port):
            logger.warning("Port %d still in use after 15s; proceeding anyway", port)

        with open(log_path, "a") as lf:
            cmd = [
                str(self.sim_config.venv_python),
                "-m", "sims.sim_worker",
                "--sim", self.sim_type,
                "--port", str(port),
            ]
            if self.headless:
                cmd.append("--headless")
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                preexec_fn=os.setsid,
            )

        self.processes[port] = proc
        return port

    def _read_log_tail(self, port: int, max_lines: int = 20) -> str:
        """Return the last few lines of a worker log, if available."""
        log_path = self.log_paths.get(port)
        if not log_path or not log_path.exists():
            return ""
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-max_lines:])

    def start_all(self) -> List[int]:
        """Start all workers. Returns list of ports."""
        ports = []
        for slot in range(self.num_workers):
            port = self.start_worker(slot)
            ports.append(port)
        return ports

    def wait_for_health(self, port: int, timeout: int = 60) -> bool:
        """Wait for a sim worker to become healthy."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # In venv mode, check if the process died early
            if self.runtime == "venv":
                proc = self.processes.get(port)
                if proc is not None:
                    exit_code = proc.poll()
                    if exit_code is not None:
                        log_tail = self._read_log_tail(port)
                        logger.error(
                            "  sim worker on port %d exited early with code %d",
                            port,
                            exit_code,
                        )
                        if log_tail:
                            logger.error("  sim worker log tail (%s):\n%s", self.log_paths[port], log_tail)
                        return False
            # In docker mode, we could check container status, but health check suffices
            healthy, _ = check_health(f"http://localhost:{port}")
            if healthy:
                return True
            time.sleep(2)
        if self.runtime == "venv":
            log_tail = self._read_log_tail(port)
            if log_tail:
                logger.warning("  sim worker log tail (%s):\n%s", self.log_paths[port], log_tail)
        elif self.runtime == "docker" and port in self.containers:
            from .docker_backend import docker_logs
            logs = docker_logs(self.containers[port], tail=20)
            if logs:
                logger.warning("  Docker sim worker logs (port %d):\n%s", port, logs)
        return False

    def wait_all_healthy(self, timeout: int = 60) -> bool:
        """Wait for all workers to become healthy."""
        all_ok = True
        ports = self.containers if self.runtime == "docker" else self.processes
        for port in ports:
            if not self.wait_for_health(port, timeout):
                logger.warning("  sim worker on port %d not ready after %ds", port, timeout)
                all_ok = False
        return all_ok

    def kill_worker(self, port: int):
        """Kill a specific sim worker.

        C1: In docker mode, uses docker_stop — never os.killpg().
        """
        # ── Docker mode ──
        if self.runtime == "docker":
            container_id = self.containers.get(port)
            if container_id:
                from .docker_backend import docker_stop
                docker_stop(container_id)
                del self.containers[port]
            return

        # ── Venv mode (existing behavior) ──
        proc = self.processes.get(port)
        if proc:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            del self.processes[port]

        # Belt-and-suspenders
        try:
            subprocess.run(
                ["fuser", "-k", f"{port}/tcp"],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    def kill_all(self):
        """Kill all sim workers."""
        if self.runtime == "docker":
            for port in list(self.containers.keys()):
                self.kill_worker(port)
        else:
            for port in list(self.processes.keys()):
                self.kill_worker(port)


# ---------------------------------------------------------------------------
# Standalone sim worker lifecycle (for CLI `servers start/stop sim`)
# ---------------------------------------------------------------------------

def start_sim_server(
    sim_type: str,
    port: int = 5001,
    host: str = "0.0.0.0",
    headless: bool = True,
    wait: bool = True,
    log_file: Optional[str] = None,
    runtime: str = "venv",
    gpu_id: Optional[str] = None,
    display: Optional[str] = None,
) -> Optional[int | str]:
    """Start a standalone sim worker server.

    Invokes sims/sim_worker.py directly via the sim's own venv Python binary.
    Replicates scripts/start_sim.sh and scripts/start_libero_pro_sim.sh.

    When runtime='docker', starts a Docker container instead.

    Args:
        sim_type: Simulator backend name (libero, libero_pro, robocasa, robotwin)
        port: HTTP port to serve on (default: 5001)
        host: Interface to bind to (default: 0.0.0.0)
        headless: Use EGL headless rendering (sets MUJOCO_GL=egl)
        wait: Wait for the server to pass /health (up to 120s)
        log_file: Path for server log output
        runtime: 'docker' or 'venv'
        gpu_id: GPU device ID (for Docker --gpus device=N)
        display: DISPLAY value for X11 forwarding (debug window)

    Returns:
        PID (int) for venv mode, or container ID (str) for docker mode,
        or None on failure.
    """
    if sim_type not in SIM_CONFIGS:
        logger.error(
            "Unknown sim type '%s'. Available: %s", sim_type, list(SIM_CONFIGS.keys())
        )
        return None

    sim_cfg = SIM_CONFIGS[sim_type]

    # Check if already running
    healthy, _ = check_health(f"http://localhost:{port}")
    if healthy:
        logger.info("%s sim worker already running on port %d", sim_type, port)
        return None

    # ── Docker runtime path ──
    if runtime == "docker":
        from .docker_backend import docker_run
        image = DOCKER_IMAGES.get(sim_type)
        if not image:
            logger.error("No Docker image for sim '%s'. Available: %s", sim_type, list(DOCKER_IMAGES.keys()))
            return None

        env_vars = {"MUJOCO_GL": "egl" if headless else "glfw"}
        for k, v in sim_cfg.env_vars.items():
            env_vars[k] = v

        container_name = f"robo-eval-sim-{sim_type}-{port}"
        extra_args = ["--sim", sim_type, "--port", str(port)]
        if headless:
            extra_args.append("--headless")

        logger.info("Starting %s sim worker in Docker on port %d...", sim_type, port)
        try:
            container_id = docker_run(
                image=image,
                port=port,
                env=env_vars,
                gpu_id=gpu_id,
                name=container_name,
                extra_args=extra_args,
                display=display,
            )
        except RuntimeError as e:
            logger.error("Failed to start %s Docker container: %s", sim_type, e)
            return None

        if wait:
            logger.info("  Waiting for sim worker health check (timeout: 120s)...")
            for _ in range(60):
                healthy, _ = check_health(f"http://localhost:{port}")
                if healthy:
                    logger.info("  %s sim worker ready on port %d", sim_type, port)
                    return container_id
                time.sleep(2)
            logger.warning("  %s sim worker not ready after 120s on port %d", sim_type, port)

        return container_id

    # ── Venv runtime path (existing behavior) ──
    venv_python = sim_cfg.venv_python

    if not venv_python.exists():
        logger.error(
            "Venv Python not found: %s. Run 'bash scripts/setup_envs.sh --only %s' first.",
            venv_python,
            sim_type,
        )
        return None

    log_path = log_file or f"/tmp/{sim_type}_sim_{port}.log"
    logger.info("Starting %s sim worker on %s:%d...", sim_type, host, port)
    logger.info("  Log: %s", log_path)

    env = os.environ.copy()
    # Default to EGL headless rendering; honour pre-set MUJOCO_GL if already explicit
    if headless:
        env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")
    # Apply sim-specific env vars (e.g. LIBERO_CONFIG_PATH, LD_LIBRARY_PATH for libero_pro)
    for k, v in sim_cfg.env_vars.items():
        env.setdefault(k, v)

    cmd: List[str] = [
        str(venv_python),
        "-m", "sims.sim_worker",
        "--sim", sim_type,
        "--port", str(port),
        "--host", host,
    ]
    if headless:
        cmd.append("--headless")

    with open(log_path, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            env=env,
            preexec_fn=os.setsid,
        )

    pid = proc.pid
    logger.info("  PID: %d", pid)

    pid_file = Path(f"/tmp/{sim_type}_sim_{port}.pid")
    pid_file.write_text(str(pid))

    if wait:
        logger.info("  Waiting for sim worker health check (timeout: 120s)...")
        for _ in range(60):
            healthy, _ = check_health(f"http://localhost:{port}")
            if healthy:
                logger.info("  %s sim worker ready on port %d", sim_type, port)
                return pid
            time.sleep(2)
        logger.warning("  %s sim worker not ready after 120s on port %d", sim_type, port)

    return pid


def stop_sim_server(
    sim_type: str,
    port: int = 5001,
    runtime: str = "venv",
    container_id: Optional[str] = None,
) -> bool:
    """Stop a standalone sim worker server.

    Args:
        sim_type: Simulator backend name (for PID file lookup)
        port: Port the sim worker is running on
        runtime: 'docker' or 'venv'
        container_id: Container ID to stop (docker mode)

    Returns:
        True if the server was stopped.
    """
    # C1: In docker mode, use docker stop — never os.killpg()
    if runtime == "docker":
        from .docker_backend import docker_stop, docker_stop_by_name
        if container_id:
            return docker_stop(container_id)
        else:
            container_name = f"robo-eval-sim-{sim_type}-{port}"
            return docker_stop_by_name(container_name)

    # ── Venv mode (existing behavior) ──
    pid_file = Path(f"/tmp/{sim_type}_sim_{port}.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            logger.info("Stopped %s sim worker (PID %d)", sim_type, pid)
            pid_file.unlink(missing_ok=True)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            pid_file.unlink(missing_ok=True)

    # Fall back to fuser
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=10,
        )
        logger.info("Stopped process on port %d", port)
        return True
    except Exception:
        pass

    logger.info("No %s sim worker found on port %d", sim_type, port)
    return False
