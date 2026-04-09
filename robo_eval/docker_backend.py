"""Docker backend for robo-eval server management.

Drop-in replacement for subprocess.Popen-based server management.
Same start/stop/health interface, but runs containers instead of local processes.

FM Critical Findings incorporated:
  C1: Track container IDs not PIDs. Never use os.killpg in docker mode.
  C2: Do NOT pass CUDA_VISIBLE_DEVICES when using --gpus device=N.
      Docker --gpus device=N remaps GPU N to device 0 inside the container,
      so setting CUDA_VISIBLE_DEVICES=N would reference a nonexistent device.
  W1: docker_cleanup_stale() removes robo-eval-* containers from crashed runs.
  W3: Mount HF cache as :ro by default to avoid concurrent-write races.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Registry of active container IDs (not PIDs) — used for teardown.
# Maps container_name -> container_id.
_active_containers: Dict[str, str] = {}


def is_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False



def docker_build(image: str) -> None:
    """Rebuild the Docker image to ensure we have the latest code."""
    # Map image names to their dockerfiles
    dockerfiles = {
        "robo-eval/sim-libero:latest": "docker/sim-libero.Dockerfile",
        "robo-eval/sim-robocasa:latest": "docker/sim-robocasa.Dockerfile",
        "robo-eval/sim-robotwin:latest": "docker/sim-robotwin.Dockerfile",
        "robo-eval/vla-openvla:latest": "docker/vla-openvla.Dockerfile",
        "robo-eval/vla-groot:latest": "docker/vla-groot.Dockerfile",
        "robo-eval/vla-internvla:latest": "docker/vla-internvla.Dockerfile",
        "robo-eval/vla-cosmos:latest": "docker/vla-cosmos.Dockerfile",
        "robo-eval/vla-lerobot:latest": "docker/vla-lerobot.Dockerfile",
        "robo-eval/proxy:latest": "docker/proxy.Dockerfile"
    }
    
    dockerfile = dockerfiles.get(image)
    if not dockerfile:
        logger.warning("No known Dockerfile for image %s, skipping build.", image)
        return
        
    logger.info("Rebuilding Docker image %s from %s...", image, dockerfile)
    
    # Root of the repo is expected to be cwd or we can resolve it
    project_root = Path(__file__).parent.parent.absolute()
    
    cmd = [
        "docker", "build",
        "-f", dockerfile,
        "-t", image,
        "."
    ]
    
    try:
        subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
            stdout=subprocess.DEVNULL, # Suppress build output unless it fails
        )
        logger.info("Successfully built %s", image)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to build %s: %s", image, e)
        raise RuntimeError(f"Failed to build Docker image {image}") from e

def _detect_gpu_mode() -> str:
    """Auto-detect the best GPU passthrough mode for the current hardware.

    Resolution order:
      1. ``ROBO_EVAL_DOCKER_GPU`` env var (explicit override: ``cdi``, ``dri``, ``gpus``)
      2. ``/dev/dri`` present  -> ``dri``  (Tegra/Thor/GB10 unified-memory ARM64)
      3. NVIDIA CDI spec file  -> ``cdi``  (discrete GPU with Container Toolkit CDI)
      4. Fallback              -> ``gpus`` (standard ``--gpus all``)

    Returns:
        One of ``"dri"``, ``"cdi"``, or ``"gpus"``.
    """
    explicit = os.environ.get("ROBO_EVAL_DOCKER_GPU", "").lower()
    if explicit in ("cdi", "dri", "gpus"):
        return explicit

    import pathlib as _pl
    # GB10 / Jetson / DGX Spark: /dev/dri is the reliable passthrough
    if _pl.Path("/dev/dri").exists():
        return "dri"
    # CDI spec installed by nvidia-container-toolkit
    if _pl.Path("/etc/cdi/nvidia.yaml").exists() or _pl.Path("/var/run/cdi/nvidia.yaml").exists():
        return "cdi"
    return "gpus"


def docker_run(
    image: str,
    port: int,
    env: Optional[Dict[str, str]] = None,
    gpu_id: Optional[str] = None,
    volumes: Optional[List[str]] = None,
    name: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    display: Optional[str] = None,
    rw_cache: bool = False,
) -> str:
    """Start a Docker container for a robo-eval service.

    Args:
        image: Docker image name (e.g. "robo-eval/sim-libero:latest").
        port: Host port to expose (mapped 1:1 into the container).
        env: Environment variables to pass into the container.
             NOTE: CUDA_VISIBLE_DEVICES is intentionally excluded — see C2.
        gpu_id: GPU device ID for --gpus device=<id>. If None, uses --gpus all.
        volumes: Additional volume mounts (["host:container", ...]).
        name: Container name. Defaults to "robo-eval-<port>".
        extra_args: Extra arguments appended after the image name (entrypoint args).
        display: DISPLAY value for X11 forwarding (debug window mode).
                 When set, mounts /tmp/.X11-unix and sets MUJOCO_GL=glfw.
        rw_cache: If True, mount HF cache as read-write instead of read-only.

    Returns:
        Container ID (short hash from docker run).

    Raises:
        RuntimeError: If docker run fails.
    """
    container_name = name or f"robo-eval-{port}"
    env = env or {}

    cmd: List[str] = ["docker", "run", "-d", "--rm"]
    cmd += ["-p", f"{port}:{port}"]
    cmd += ["--name", container_name]

    # GPU assignment — three modes depending on hardware:
    #
    #   dri:  --device=/dev/dri  (GB10, Jetson, DGX Spark — Tegra/Thor unified memory)
    #         These platforms lack NVML so --gpus all fails with "Unknown Error".
    #         /dev/dri provides DRM render nodes for EGL rendering + CUDA access.
    #
    #   cdi:  --device nvidia.com/gpu=<id|all>  (CDI spec installed)
    #         Modern Container Toolkit with CDI; works on discrete GPUs.
    #
    #   gpus: --gpus device=<id> | --gpus all   (classic NVIDIA runtime)
    #         Standard approach for discrete NVIDIA GPUs.
    #
    # C2: Never pass CUDA_VISIBLE_DEVICES in Docker mode.
    gpu_mode = _detect_gpu_mode()

    if gpu_mode == "dri":
        cmd += ["--device", "/dev/dri"]
        # Also need nvidia libs inside the container
        cmd += ["-e", "NVIDIA_VISIBLE_DEVICES=all"]
        cmd += ["-e", "NVIDIA_DRIVER_CAPABILITIES=all"]
    elif gpu_mode == "cdi":
        if gpu_id is not None:
            cmd += ["--device", f"nvidia.com/gpu={gpu_id}"]
        else:
            cmd += ["--device", "nvidia.com/gpu=all"]
    else:  # gpus
        if gpu_id is not None:
            cmd += ["--gpus", f"device={gpu_id}"]
        else:
            cmd += ["--gpus", "all"]

    logger.debug("Docker GPU mode: %s (override via ROBO_EVAL_DOCKER_GPU)", gpu_mode)

    # Filter out CUDA_VISIBLE_DEVICES from env — it must NOT be passed in
    # docker mode (C2). The GPU is isolated by Docker's device flags.
    filtered_env = {k: v for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"}
    # Ensure EGL environment is set for headless rendering
    filtered_env.setdefault("PYOPENGL_PLATFORM", "egl")
    for k, v in filtered_env.items():
        cmd += ["-e", f"{k}={v}"]

    # HuggingFace model cache — W3: mount as :ro by default to prevent
    # concurrent-write races between multiple VLA containers.
    hf_cache = Path.home() / ".cache" / "huggingface"
    cache_mode = "rw" if rw_cache else "ro"
    cmd += ["-v", f"{hf_cache}:/root/.cache/huggingface:{cache_mode}"]

    # Additional volumes
    for vol in volumes or []:
        cmd += ["-v", vol]

    # Display forwarding for --debug-window (X11)
    if display:
        cmd += ["-e", f"DISPLAY={display}"]
        cmd += ["-v", "/tmp/.X11-unix:/tmp/.X11-unix"]
        cmd += ["-e", "MUJOCO_GL=glfw"]

    cmd += [image] + (extra_args or [])

    logger.info("docker run: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"docker run failed for {container_name}: {stderr}"
        )

    container_id = result.stdout.strip()
    # C1: Track container IDs, not PIDs.
    _active_containers[container_name] = container_id
    logger.info(
        "Started container %s (id=%s) on port %d",
        container_name, container_id[:12], port,
    )
    return container_id


def docker_stop(container_id: str, timeout: int = 10) -> bool:
    """Stop a Docker container gracefully.

    C1: Uses `docker stop` exclusively — never os.killpg().

    Args:
        container_id: Container ID or name.
        timeout: Seconds to wait before SIGKILL.

    Returns:
        True if the container was stopped successfully.
    """
    logger.info("Stopping container %s (timeout=%ds)...", container_id[:12], timeout)
    result = subprocess.run(
        ["docker", "stop", "-t", str(timeout), container_id],
        capture_output=True,
        text=True,
    )

    # Remove from active tracking
    _active_containers.pop(container_id, None)
    # Also try removing by matching container_id in values
    for name, cid in list(_active_containers.items()):
        if cid == container_id or container_id.startswith(cid) or cid.startswith(container_id):
            del _active_containers[name]

    if result.returncode != 0:
        logger.warning("docker stop failed for %s: %s", container_id[:12], result.stderr.strip())
        return False

    logger.info("Stopped container %s", container_id[:12])
    return True


def docker_stop_by_name(container_name: str, timeout: int = 10) -> bool:
    """Stop a Docker container by its name.

    Convenience wrapper — looks up the container ID from the active registry
    or uses the name directly (docker stop accepts names too).

    Args:
        container_name: The --name used when starting the container.
        timeout: Seconds to wait before SIGKILL.

    Returns:
        True if the container was stopped successfully.
    """
    container_id = _active_containers.get(container_name, container_name)
    return docker_stop(container_id, timeout=timeout)


def docker_logs(container_id: str, tail: int = 50) -> str:
    """Get recent logs from a container.

    Args:
        container_id: Container ID or name.
        tail: Number of log lines to retrieve.

    Returns:
        Combined stdout + stderr log output.
    """
    result = subprocess.run(
        ["docker", "logs", "--tail", str(tail), container_id],
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def docker_cleanup_stale(prefix: str = "robo-eval-") -> int:
    """Remove stale robo-eval containers from previous crashed runs.

    W1: `fuser -k PORT/tcp` only handles stale venv processes. This function
    cleans up Docker containers left behind by crashed robo-eval runs.

    Args:
        prefix: Container name prefix to match (default: "robo-eval-").

    Returns:
        Number of containers removed.
    """
    # Find all containers (running or stopped) matching the prefix
    result = subprocess.run(
        [
            "docker", "ps", "-a",
            "--filter", f"name={prefix}",
            "--format", "{{.ID}} {{.Names}} {{.Status}}",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return 0

    lines = result.stdout.strip().splitlines()
    removed = 0

    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        container_id, container_name = parts[0], parts[1]
        status = parts[2] if len(parts) > 2 else ""
        # Skip running containers — only remove exited/dead/created ones.
        # Status for running containers starts with "Up".
        if status.startswith("Up"):
            logger.debug("Skipping running container: %s (%s)", container_name, container_id[:12])
            continue
        logger.info("Cleaning up stale container: %s (%s)", container_name, container_id[:12])
        rm_result = subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
        )
        if rm_result.returncode == 0:
            removed += 1
        else:
            logger.warning(
                "Failed to remove container %s: %s",
                container_id[:12], rm_result.stderr.strip(),
            )

    if removed:
        logger.info("Removed %d stale container(s)", removed)
    return removed


def docker_stop_all(timeout: int = 10) -> int:
    """Stop all active containers tracked by this module.

    C1: Uses docker_stop() exclusively — never os.killpg().

    Args:
        timeout: Seconds to wait per container before SIGKILL.

    Returns:
        Number of containers stopped.
    """
    stopped = 0
    for name, container_id in list(_active_containers.items()):
        if docker_stop(container_id, timeout=timeout):
            stopped += 1
    _active_containers.clear()
    return stopped


def get_active_containers() -> Dict[str, str]:
    """Return a copy of the active container registry.

    Returns:
        Dict mapping container_name -> container_id.
    """
    return dict(_active_containers)
