"""
VLA server pool management: start N replicas, health check, graceful teardown.

Manages the lifecycle of VLA policy server processes. Each replica runs as a
separate subprocess with its own port and (optionally) GPU assignment.

Design for multi-machine extensibility:
    The VLAServerPool produces a list of backend URLs. Today those are all
    localhost:PORT, but the architecture supports mixing in remote URLs
    (e.g. from --remote-vlas ssh://gpu-box-2:5102) by simply appending them
    to the backend_urls list. The proxy doesn't care whether backends are
    local or remote.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import PROJECT_ROOT, VLA_CONFIGS, VLA_DOCKER_IMAGES, VLAConfig

logger = logging.getLogger(__name__)


@dataclass
class VLAReplica:
    """Tracks a single VLA server replica."""

    port: int
    gpu_id: Optional[str] = None  # CUDA_VISIBLE_DEVICES value (venv mode only)
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    container_id: Optional[str] = None  # Docker container ID (docker mode)
    log_path: Optional[Path] = None
    healthy: bool = False

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"


class VLAServerPool:
    """Manages N replicas of a VLA policy server.

    Lifecycle:
        1. pool = VLAServerPool("smolvla", num_replicas=4, gpus="0,1,2,3")
        2. pool.start_all()           # launches subprocesses
        3. pool.wait_all_healthy()     # blocks until all pass /health
        4. urls = pool.backend_urls    # pass to proxy
        5. pool.stop_all()             # graceful shutdown

    Multi-GPU support:
        When gpus="0,1,2,3", replica i gets CUDA_VISIBLE_DEVICES=gpus[i].
        When gpus is None, all replicas share the default GPU visibility.
        On GB10 (128GB unified), all replicas share GPU 0 but this is
        forward-compatible with multi-GPU systems.

    Multi-machine extensibility:
        Future: add remote_urls parameter that gets appended to backend_urls.
        The proxy treats all URLs identically regardless of origin.
    """

    def __init__(
        self,
        vla_name: str,
        num_replicas: int = 1,
        base_port: Optional[int] = None,
        gpus: Optional[str] = None,
        logs_dir: Optional[Path] = None,
        remote_urls: Optional[List[str]] = None,
        runtime: str = "venv",
    ):
        if vla_name not in VLA_CONFIGS:
            raise ValueError(
                f"Unknown VLA '{vla_name}'. Available: {list(VLA_CONFIGS.keys())}"
            )

        self.vla_name = vla_name
        self.config = VLA_CONFIGS[vla_name]
        self.num_replicas = num_replicas
        self.base_port = base_port or self.config.port
        self.logs_dir = logs_dir or Path("/tmp")
        self.remote_urls = remote_urls or []
        self.runtime = runtime

        # Parse GPU assignments
        if gpus:
            gpu_list = [g.strip() for g in gpus.split(",")]
            if len(gpu_list) < num_replicas:
                # Wrap around if fewer GPUs than replicas
                gpu_list = [gpu_list[i % len(gpu_list)] for i in range(num_replicas)]
        else:
            gpu_list = [None] * num_replicas

        # Create replica descriptors
        self.replicas: List[VLAReplica] = []
        for i in range(num_replicas):
            port = self.base_port + i
            self.replicas.append(
                VLAReplica(
                    port=port,
                    gpu_id=gpu_list[i],
                    log_path=self.logs_dir / f"vla_{vla_name}_{port}.log",
                )
            )

    @property
    def backend_urls(self) -> List[str]:
        """All backend URLs: local replicas + remote URLs."""
        local = [r.url for r in self.replicas]
        return local + self.remote_urls

    @property
    def local_ports(self) -> List[int]:
        """Ports of locally managed replicas."""
        return [r.port for r in self.replicas]

    def start_all(self) -> List[int]:
        """Start all VLA replica processes.

        Returns list of ports that were started.
        """
        started_ports = []

        for replica in self.replicas:
            self._start_replica(replica)
            started_ports.append(replica.port)

        n = len(started_ports)
        ports_str = f"{started_ports[0]}-{started_ports[-1]}" if n > 1 else str(started_ports[0])
        gpu_str = ""
        if any(r.gpu_id is not None for r in self.replicas):
            gpus = [r.gpu_id or "default" for r in self.replicas]
            gpu_str = f" (GPUs: {', '.join(gpus)})"

        logger.info(
            "Started %d %s replica(s) on port(s) %s%s",
            n, self.vla_name, ports_str, gpu_str,
        )

        return started_ports

    def _start_replica(self, replica: VLAReplica) -> None:
        """Start a single VLA replica process.

        C1: In docker mode, tracks container IDs not PIDs.
        C2: In docker mode, does NOT pass CUDA_VISIBLE_DEVICES.
        """
        config = self.config

        # ── Docker mode ──
        if self.runtime == "docker":
            from .docker_backend import docker_run
            image = VLA_DOCKER_IMAGES.get(self.vla_name)
            if not image:
                raise RuntimeError(
                    f"No Docker image for VLA '{self.vla_name}'. "
                    f"Available: {list(VLA_DOCKER_IMAGES.keys())}"
                )

            env_vars = {"MUJOCO_GL": "egl", "VLA_PORT": str(replica.port)}
            # Pass model_id and embodiment tag so the container uses the correct N1.6 config.
            if self.vla_name == "groot":
                env_vars["GROOT_MODEL_ID"] = config.model_id
                env_vars["GROOT_EMBODIMENT_TAG"] = "panda"
            # Inject VLA_MODULE for lerobot image (serves both pi05 and smolvla)
            if self.vla_name in ("pi05", "smolvla"):
                module_map = {
                    "pi05":    "sims.vla_policies.pi05_policy",
                    "smolvla": "sims.vla_policies.smolvla_policy",
                }
                env_vars["VLA_MODULE"] = module_map[self.vla_name]
            # C2: Do NOT pass CUDA_VISIBLE_DEVICES — Docker --gpus handles GPU isolation.

            container_name = f"robo-eval-vla-{self.vla_name}-{replica.port}"
            extra_args = ["--port", str(replica.port)]

            container_id = docker_run(
                image=image,
                port=replica.port,
                env=env_vars,
                gpu_id=replica.gpu_id,
                name=container_name,
                extra_args=extra_args,
                rw_cache=True,  # OpenVLA needs rw for transformers_modules cache
            )
            replica.container_id = container_id
            logger.debug(
                "Started %s replica in Docker on port %d (container %s, GPU %s)",
                self.vla_name, replica.port, container_id[:12],
                replica.gpu_id or "all",
            )
            return

        # ── Venv mode (existing behavior) ──
        # Build environment
        env = os.environ.copy()
        env["MUJOCO_GL"] = "egl"
        env["VLA_PORT"] = str(replica.port)

        if replica.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = replica.gpu_id

        # Use Python module invocation (avoids bash script dependency)
        # VLA module path from config
        vla_module = _get_vla_module(self.vla_name)
        python_path = str(config.venv_python)

        cmd = [
            python_path,
            "-m", vla_module,
            "--port", str(replica.port),
        ]

        # Ensure log directory exists
        replica.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(replica.log_path, "w") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                env=env,
                preexec_fn=os.setsid,
            )

        replica.process = proc
        replica.pid = proc.pid

        logger.debug(
            "Started %s replica on port %d (PID %d, GPU %s)",
            self.vla_name, replica.port, replica.pid,
            replica.gpu_id or "default",
        )

    def wait_all_healthy(
        self,
        timeout: Optional[int] = None,
        poll_interval: float = 5.0,
    ) -> bool:
        """Wait for all replicas to pass health checks.

        Args:
            timeout: Max seconds to wait. Defaults to VLA's startup_timeout.
            poll_interval: Seconds between health polls.

        Returns:
            True if all replicas are healthy within the timeout.
        """
        from .servers import check_health

        if timeout is None:
            timeout = self.config.startup_timeout

        start_time = time.time()
        pending = set(range(len(self.replicas)))

        while pending and (time.time() - start_time) < timeout:
            for i in list(pending):
                replica = self.replicas[i]

                # Check if process died (venv mode only)
                if self.runtime == "venv" and replica.process and replica.process.poll() is not None:
                    logger.error(
                        "%s replica on port %d exited with code %d",
                        self.vla_name, replica.port, replica.process.returncode,
                    )
                    pending.discard(i)
                    continue

                healthy, data = check_health(replica.url)
                if healthy and data.get("ready", False):
                    replica.healthy = True
                    pending.discard(i)
                    elapsed = int(time.time() - start_time)
                    logger.info(
                        "%s replica on port %d ready after %ds",
                        self.vla_name, replica.port, elapsed,
                    )

            if pending:
                time.sleep(poll_interval)

        # Report any that didn't come up
        for i in pending:
            replica = self.replicas[i]
            logger.warning(
                "%s replica on port %d not ready after %ds",
                self.vla_name, replica.port, timeout,
            )

        n_healthy = sum(1 for r in self.replicas if r.healthy)
        return n_healthy == len(self.replicas)

    def stop_all(self, timeout: float = 10.0) -> None:
        """Gracefully stop all VLA replica processes.

        C1: In docker mode, uses docker_stop — never os.killpg().
        In venv mode, sends SIGTERM, waits up to timeout, then SIGKILL.
        """
        # ── Docker mode ──
        if self.runtime == "docker":
            from .docker_backend import docker_stop
            for replica in self.replicas:
                if replica.container_id:
                    docker_stop(replica.container_id, timeout=int(timeout))
                    replica.container_id = None
            n = len(self.replicas)
            logger.info("Stopped %d %s Docker replica(s)", n, self.vla_name)
            return

        # ── Venv mode (existing behavior) ──
        # Send SIGTERM to all
        for replica in self.replicas:
            if replica.process and replica.process.poll() is None:
                try:
                    os.killpg(os.getpgid(replica.pid), signal.SIGTERM)
                    logger.debug(
                        "Sent SIGTERM to %s replica on port %d (PID %d)",
                        self.vla_name, replica.port, replica.pid,
                    )
                except (ProcessLookupError, PermissionError):
                    pass

        # Wait for graceful shutdown
        deadline = time.time() + timeout
        for replica in self.replicas:
            if replica.process and replica.process.poll() is None:
                remaining = max(0, deadline - time.time())
                try:
                    replica.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    # Force kill
                    try:
                        os.killpg(os.getpgid(replica.pid), signal.SIGKILL)
                        logger.warning(
                            "Force-killed %s replica on port %d (PID %d)",
                            self.vla_name, replica.port, replica.pid,
                        )
                    except (ProcessLookupError, PermissionError):
                        pass

        # Belt-and-suspenders: fuser kill on all ports
        for replica in self.replicas:
            try:
                subprocess.run(
                    ["fuser", "-k", f"{replica.port}/tcp"],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

        n = len(self.replicas)
        logger.info("Stopped %d %s replica(s)", n, self.vla_name)

    def get_status(self) -> List[Dict]:
        """Return status of all replicas (for display/debugging)."""
        from .servers import check_health

        statuses = []
        for replica in self.replicas:
            if self.runtime == "docker":
                alive = replica.container_id is not None
            else:
                alive = replica.process and replica.process.poll() is None
            healthy, data = check_health(replica.url) if alive else (False, {})

            statuses.append({
                "port": replica.port,
                "gpu": replica.gpu_id or "default",
                "pid": replica.pid,
                "container_id": replica.container_id,
                "alive": alive,
                "healthy": healthy,
                "log": str(replica.log_path),
                "runtime": self.runtime,
            })

        return statuses


def _get_vla_module(vla_name: str) -> str:
    """Get the Python module path for a VLA server.

    These map to the standalone policy servers in sims/vla_policies/.
    """
    modules = {
        "pi05": "sims.vla_policies.pi05_policy",
        "openvla": "sims.vla_policies.openvla_policy",
        "smolvla": "sims.vla_policies.smolvla_policy",
        "cosmos": "sims.vla_policies.cosmos_policy",
        "internvla": "sims.vla_policies.internvla_policy",
        "groot": "sims.vla_policies.groot_policy",
    }
    if vla_name not in modules:
        raise ValueError(f"No module mapping for VLA '{vla_name}'")
    return modules[vla_name]
