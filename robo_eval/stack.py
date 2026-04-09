"""
Full-stack lifecycle manager: VLA servers -> proxy -> sim workers -> evals.

The StackManager is the single orchestration point that ensures:
1. VLA replicas start and pass health checks
2. The round-robin proxy starts pointing at healthy backends
3. Sim workers start and pass health checks
4. Graceful teardown in reverse order on completion or Ctrl+C

Usage:
    stack = StackManager(
        vla_name="smolvla",
        vla_replicas=4,
        gpus="0,1,2,3",
        proxy_port=5200,
        sim_type="libero",
        sim_base_port=5300,
        num_sim_workers=10,
    )

    with stack:
        # stack.proxy_url is the single URL for all eval processes
        run_evals(vla_url=stack.proxy_url, ...)

    # On exit (or Ctrl+C): evals done -> sim workers killed -> proxy killed -> VLA killed
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from types import TracebackType
from typing import List, Optional, Type

from .config import DEFAULT_PROXY_PORT
from .servers import SimWorkerPool, check_health
from .vla_manager import VLAServerPool

logger = logging.getLogger(__name__)


class StackManager:
    """Manages the full eval stack lifecycle.

    Implements the context manager protocol for automatic cleanup.
    Can also be used imperatively via start() / stop().

    Teardown order (reverse of startup):
        1. Stop sim workers
        2. Stop proxy
        3. Stop VLA servers

    The proxy is ALWAYS used, even with 1 VLA replica. This is the
    architectural invariant from the design doc.
    """

    def __init__(
        self,
        vla_name: str,
        vla_replicas: int = 1,
        vla_base_port: Optional[int] = None,
        gpus: Optional[str] = None,
        proxy_port: int = DEFAULT_PROXY_PORT,
        sim_type: str = "libero",
        sim_base_port: int = 5300,
        num_sim_workers: int = 10,
        sim_headless: bool = True,
        logs_dir: Optional[Path] = None,
        remote_vlas: Optional[List[str]] = None,
        runtime: str = "venv",
    ):
        self.vla_name = vla_name
        self.proxy_port = proxy_port
        self.sim_type = sim_type
        self.runtime = runtime
        self.logs_dir = logs_dir or Path("/tmp/robo-eval-stack")

        # Create sub-managers
        self.vla_pool = VLAServerPool(
            vla_name=vla_name,
            num_replicas=vla_replicas,
            base_port=vla_base_port,
            gpus=gpus,
            logs_dir=self.logs_dir,
            remote_urls=remote_vlas,
            runtime=runtime,
        )

        self.sim_pool = SimWorkerPool(
            sim_type=sim_type,
            base_port=sim_base_port,
            num_workers=num_sim_workers,
            headless=sim_headless,
            logs_dir=self.logs_dir,
            runtime=runtime,
        )

        # State tracking
        self._proxy_pid: Optional[int] = None
        self._vlm_proxy_pid: Optional[int] = None  # Tracked for cleanup in stop()
        self._vlm_proxy_port: Optional[int] = None
        self._started = False
        self._original_sigint = None
        self._original_sigterm = None

    @property
    def proxy_url(self) -> str:
        """The single URL that all eval processes should use."""
        return f"http://localhost:{self.proxy_port}"

    @property
    def sim_ports(self) -> List[int]:
        """Ports of running sim workers."""
        return list(self.sim_pool.processes.keys())

    def start(self) -> None:
        """Start the full stack: VLA replicas -> proxy -> sim workers.

        Raises RuntimeError if any component fails to start.
        """
        if self._started:
            logger.warning("Stack already started")
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Install signal handlers for graceful shutdown
        self._install_signal_handlers()

        try:
            self._start_vla_servers()
            self._start_proxy()
            self._start_sim_workers()
            self._started = True
        except Exception:
            # If any stage fails, tear down what we've started
            logger.error("Stack startup failed, tearing down...")
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the full stack in reverse order.

        Safe to call multiple times. Safe to call even if start() wasn't
        called or partially completed.

        Ignores SIGINT during cleanup to prevent double Ctrl+C from
        leaving orphan processes.

        C1: In docker mode, uses docker_stop — never os.killpg().
        """
        # Ignore further SIGINT during cleanup to prevent orphans
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # 1. Stop sim workers
        try:
            self.sim_pool.kill_all()
            logger.info("Sim workers stopped")
        except Exception as e:
            logger.warning("Error stopping sim workers: %s", e)

        # 2. Stop VLA proxy
        try:
            self._stop_proxy()
            logger.info("Proxy stopped")
        except Exception as e:
            logger.warning("Error stopping proxy: %s", e)

        # 2b. Stop VLM proxy if we started it
        if self._vlm_proxy_pid is not None:
            try:
                self._stop_vlm_proxy()
                logger.info("VLM proxy stopped")
            except Exception as e:
                logger.warning("Error stopping VLM proxy: %s", e)

        # 3. Stop VLA servers
        try:
            self.vla_pool.stop_all()
            logger.info("VLA servers stopped")
        except Exception as e:
            logger.warning("Error stopping VLA servers: %s", e)

        # 4. Final docker cleanup for any orphaned containers
        if self.runtime == "docker":
            try:
                from .docker_backend import docker_stop_all
                docker_stop_all()
            except Exception as e:
                logger.warning("Error in final docker cleanup: %s", e)

        self._started = False

        # Restore original signal handlers after cleanup
        self._restore_signal_handlers()

    def __enter__(self) -> "StackManager":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()

    # ----- Internal startup stages -----

    def _start_vla_servers(self) -> None:
        """Stage 1: Start VLA replicas and wait for health."""
        n = self.vla_pool.num_replicas
        ports = self.vla_pool.local_ports
        ports_str = f"{ports[0]}-{ports[-1]}" if n > 1 else str(ports[0])

        logger.info("[stack] Starting %d %s replica(s) on port(s) %s...", n, self.vla_name, ports_str)

        self.vla_pool.start_all()

        logger.info("[stack] Waiting for VLA health checks (timeout: %ds)...", self.vla_pool.config.startup_timeout)

        if not self.vla_pool.wait_all_healthy():
            # Check how many are healthy
            n_healthy = sum(1 for r in self.vla_pool.replicas if r.healthy)
            if n_healthy == 0:
                raise RuntimeError(
                    f"No {self.vla_name} replicas became healthy. "
                    f"Check logs in {self.logs_dir}"
                )
            logger.warning(
                "[stack] WARNING: Only %d/%d replicas healthy. "
                "Continuing with healthy backends.", n_healthy, n
            )
        else:
            logger.info("[stack] All %d VLA replica(s) healthy", n)

    def _start_proxy(self) -> None:
        """Stage 2: Start the round-robin proxy pointing at healthy VLA backends."""
        from .servers import start_proxy

        backend_urls = [
            r.url for r in self.vla_pool.replicas if r.healthy
        ] + self.vla_pool.remote_urls

        if not backend_urls:
            raise RuntimeError("No healthy VLA backends to proxy")

        logger.info("[stack] Starting proxy on port %d -> %d backend(s)", self.proxy_port, len(backend_urls))

        self._proxy_pid = start_proxy(
            backend_urls=backend_urls,
            port=self.proxy_port,
            wait=True,
            log_file=str(self.logs_dir / "vla_proxy.log"),
        )

        if self._proxy_pid is None:
            # start_proxy returns None if already running — that's ok
            healthy, _ = check_health(self.proxy_url)
            if not healthy:
                raise RuntimeError(
                    f"Proxy failed to start on port {self.proxy_port}. "
                    f"Check {self.logs_dir / 'vla_proxy.log'}"
                )
            logger.info("[stack] Proxy already running on port %d", self.proxy_port)
        else:
            logger.info("[stack] Proxy ready on %s", self.proxy_url)

    def _start_sim_workers(self) -> None:
        """Stage 3: Start sim workers and wait for health."""
        n = self.sim_pool.num_workers
        base = self.sim_pool.base_port
        ports_str = f"{base}-{base + n - 1}" if n > 1 else str(base)

        logger.info("[stack] Starting %d sim worker(s) on port(s) %s...", n, ports_str)

        self.sim_pool.start_all()

        if not self.sim_pool.wait_all_healthy(timeout=60):
            logger.warning("[stack] WARNING: Some sim workers failed to start")
        else:
            logger.info("[stack] All %d sim worker(s) healthy", n)

    def start_vlm_proxy(self, port: int = 4000, model: Optional[str] = None) -> None:
        """Start the VLM proxy and track its PID for cleanup."""
        from .servers import start_vlm_proxy as _start_vlm_proxy
        pid = _start_vlm_proxy(port=port, model=model, wait=True)
        if pid is not None:
            self._vlm_proxy_pid = pid
            self._vlm_proxy_port = port

    def _stop_proxy(self) -> None:
        """Stop the VLA proxy server."""
        from .servers import stop_proxy
        stop_proxy(port=self.proxy_port)

    def _stop_vlm_proxy(self) -> None:
        """Stop the VLM proxy server if we started it."""
        from .servers import stop_vlm_proxy
        stop_vlm_proxy(self._vlm_proxy_port or 4000)
        self._vlm_proxy_pid = None
        self._vlm_proxy_port = None

    # ----- Signal handling -----

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

        def _shutdown_handler(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            logger.info("\n[stack] Received %s, shutting down gracefully...", sig_name)
            self.stop()
            sys.exit(128 + signum)

        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
            self._original_sigint = None
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
            self._original_sigterm = None

    # ----- Status / display -----

    def print_status(self) -> None:
        """Print current status of all stack components."""
        logger.info("\n=== Stack Status ===")

        # VLA replicas
        logger.info("\nVLA (%s):", self.vla_name)
        for status in self.vla_pool.get_status():
            health = "HEALTHY" if status["healthy"] else "DOWN"
            logger.info(
                "  port %s | GPU %s | PID %s | %s",
                status['port'], status['gpu'], status['pid'], health,
            )

        # Proxy
        proxy_healthy, proxy_data = check_health(self.proxy_url)
        proxy_status = "HEALTHY" if proxy_healthy else "DOWN"
        if proxy_healthy:
            n_backends = proxy_data.get("backends_healthy", "?")
            logger.info("\nProxy: port %d | %s | %s backend(s)", self.proxy_port, proxy_status, n_backends)
        else:
            logger.info("\nProxy: port %d | %s", self.proxy_port, proxy_status)

        # Sim workers
        logger.info("\nSim workers (%s):", self.sim_type)
        for port, proc in self.sim_pool.processes.items():
            alive = proc.poll() is None
            if alive:
                healthy, _ = check_health(f"http://localhost:{port}")
                status = "HEALTHY" if healthy else "STARTING"
            else:
                status = "DOWN"
            logger.info("  port %d | PID %d | %s", port, proc.pid, status)

        logger.info("")
