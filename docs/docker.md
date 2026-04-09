# Docker Runtime Guide

robo-eval supports running sim backends and VLA policy servers as Docker
containers instead of local virtualenvs. This provides reproducible
environments, dependency isolation, and simpler setup on new machines.

## Quick Start

```bash
# Build sim image
docker build -f docker/sim-libero.Dockerfile -t robo-eval/sim-libero:latest .

# Build VLA image
docker build -f docker/vla-lerobot.Dockerfile -t robo-eval/vla-lerobot:latest .

# Run with Docker backend
robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime docker

# Auto-detect: uses Docker if available, venvs otherwise
robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime auto

# Force venv mode (ignore Docker even if available)
robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime venv
```

## Runtime Selection

The `--runtime` flag controls how sim workers and VLA servers are launched:

| Flag | Behavior |
|------|----------|
| `auto` (default) | Uses Docker if available, falls back to venvs |
| `docker` | Requires Docker; errors if Docker is unavailable |
| `venv` | Uses local virtualenvs (original behavior) |

### Resolution Logic

The runtime is resolved to one of 5 explicit cases:

1. **auto + Docker available** -> uses Docker containers
2. **auto + Docker unavailable** -> falls back to venvs
3. **docker + Docker available** -> uses Docker containers
4. **docker + Docker unavailable** -> exits with error
5. **venv** -> uses venvs regardless of Docker availability

## Available Docker Images

### Simulator Backends

| Image | Dockerfile | Backends |
|-------|-----------|----------|
| `robo-eval/sim-libero` | `docker/sim-libero.Dockerfile` | libero, libero_pro, libero_infinity |
| `robo-eval/sim-robocasa` | `docker/sim-robocasa.Dockerfile` | robocasa |
| `robo-eval/sim-robotwin` | `docker/sim-robotwin.Dockerfile` | robotwin |

### VLA Policy Servers

| Image | Dockerfile | VLAs |
|-------|-----------|------|
| `robo-eval/vla-lerobot` | `docker/vla-lerobot.Dockerfile` | pi05, smolvla |
| `robo-eval/vla-openvla` | `docker/vla-openvla.Dockerfile` | openvla |
| `robo-eval/vla-cosmos` | `docker/vla-cosmos.Dockerfile` | cosmos |
| `robo-eval/vla-internvla` | `docker/vla-internvla.Dockerfile` | internvla |

### Utility

| Image | Dockerfile | Purpose |
|-------|-----------|---------|
| `robo-eval/proxy` | `docker/proxy.Dockerfile` | VLA round-robin proxy |

## Building Images

Build all images from the project root:

```bash
# Sim backends
docker build -f docker/sim-libero.Dockerfile -t robo-eval/sim-libero:latest .
docker build -f docker/sim-robocasa.Dockerfile -t robo-eval/sim-robocasa:latest .
docker build -f docker/sim-robotwin.Dockerfile -t robo-eval/sim-robotwin:latest .

# VLA servers
docker build -f docker/vla-lerobot.Dockerfile -t robo-eval/vla-lerobot:latest .
docker build -f docker/vla-openvla.Dockerfile -t robo-eval/vla-openvla:latest .
docker build -f docker/vla-cosmos.Dockerfile -t robo-eval/vla-cosmos:latest .
docker build -f docker/vla-internvla.Dockerfile -t robo-eval/vla-internvla:latest .

# Proxy
docker build -f docker/proxy.Dockerfile -t robo-eval/proxy:latest .
```

## GPU Support

Docker containers use NVIDIA Container Toolkit (or device passthrough) for GPU access.

### Prerequisites

```bash
# Install NVIDIA Container Toolkit
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### GPU Passthrough Modes

robo-eval auto-detects the best GPU passthrough mode for your hardware.
Override with `ROBO_EVAL_DOCKER_GPU=<mode>`:

| Mode | Flag | Hardware | How it works |
|------|------|----------|--------------|
| `dri` | `--device=/dev/dri` | GB10, Jetson, DGX Spark (Tegra/Thor, unified memory ARM64) | Uses DRM render nodes; avoids NVML which fails on unified-memory platforms |
| `cdi` | `--device nvidia.com/gpu=N` | Discrete GPUs with CDI spec installed | Modern Container Toolkit with CDI device injection |
| `gpus` | `--gpus device=N` | Discrete GPUs (classic) | Standard NVIDIA Container Runtime approach |

Auto-detection order: `/dev/dri` present -> `dri`; CDI spec found -> `cdi`; fallback -> `gpus`.

### GB10 / Jetson / DGX Spark (ARM64 Unified Memory)

These platforms use unified CPU/GPU memory and lack NVML, so `--gpus all` fails.
robo-eval auto-detects this and uses `--device=/dev/dri` instead:

```bash
# Auto-detected on GB10 hardware:
robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime docker

# Or force DRI mode explicitly:
ROBO_EVAL_DOCKER_GPU=dri robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime docker
```

Containers also set `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` automatically
for headless EGL rendering.

### Multi-GPU Assignment

When using `--gpus` with `--runtime docker`:

```bash
# Each VLA replica gets its own GPU
robo-eval run --benchmark libero --vla pi05 --vla-replicas 4 --gpus 0,1,2,3 --runtime docker
```

Docker's `--gpus device=N` remaps GPU N to device 0 inside the container.
**CUDA_VISIBLE_DEVICES is never passed to Docker containers** to avoid
referencing nonexistent devices.

## HuggingFace Model Cache

By default, `~/.cache/huggingface` is mounted **read-only** into VLA
containers to prevent concurrent-write races between multiple containers
downloading the same model (FM W3).

### Pre-populating the Cache

Before running Docker evaluations, pre-download models on the host:

```bash
# Download models to host cache
python -c "from transformers import AutoModel; AutoModel.from_pretrained('lerobot/pi05_libero_finetuned')"
```

## Debug Window (X11 Forwarding)

To open a live GLFW rendering window from a container:

```bash
# Allow X11 connections
xhost +local:docker

# Run with debug window
robo-eval run --benchmark libero --vla pi05 --episodes 1 --runtime docker \
    --debug-window --sequential --tasks-parallel 1 --suites spatial
```

This mounts `/tmp/.X11-unix` and sets `DISPLAY` and `MUJOCO_GL=glfw`
inside the container.

## Video Recording

Videos are recorded inside the container and written to the mounted
results volume:

```bash
robo-eval run --benchmark libero --vla pi05 --episodes 5 --runtime docker --record-video
```

## Stale Container Cleanup

If robo-eval crashes, containers may be left running. The CLI
automatically cleans up stale `robo-eval-*` containers on startup
(FM W1). You can also clean up manually:

```bash
# List stale containers
docker ps -a --filter name=robo-eval-

# Remove all stale containers
docker ps -a --filter name=robo-eval- -q | xargs docker rm -f
```

## Architecture

The orchestrator (CLI + runner) always runs on the host. Only sim backends
and VLA policy servers run in containers. This avoids Docker-in-Docker
complexity.

```
robo-eval run --runtime docker --benchmark libero --vla pi05
    |
    |-- docker run --gpus device=0 -p 5100:5100 robo-eval/vla-lerobot:latest
    |-- docker run -p 5200:5200 robo-eval/proxy:latest
    |-- docker run --gpus all -p 5300:5300 robo-eval/sim-libero:latest
    |-- (host) .venvs/litellm/bin/python run_sim_eval.py eval ...
```

Port allocation happens on the host via `PortAllocator`. Docker containers
receive already-reserved ports via `-p HOST_PORT:CONTAINER_PORT`.

## Standalone Container Usage

You can also run containers directly without the CLI:

```bash
# Start a sim worker (discrete GPU)
docker run --rm --gpus all -p 5300:5300 \
    -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    robo-eval/sim-libero:latest --sim libero --port 5300 --headless

# Start a sim worker (GB10/Jetson — use --device=/dev/dri)
docker run --rm --device=/dev/dri -p 5300:5300 \
    -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
    robo-eval/sim-libero:latest --sim libero --port 5300 --headless

# Health check
curl http://localhost:5300/health

# Start a VLA server
docker run --rm --gpus all -p 5100:5100 \
    -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
    robo-eval/vla-lerobot:latest --port 5100

# Check VLA health
curl http://localhost:5100/health
```

## Troubleshooting

### Docker daemon not running

```
Error: --runtime docker requested but Docker is not available.
```

Start the Docker daemon:
```bash
sudo systemctl start docker
```

### GPU not available in container

```bash
# On discrete GPUs: verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# On GB10/Jetson (unified memory): --gpus all will fail with "NVML Unknown Error"
# Use --device=/dev/dri instead:
docker run --rm --device=/dev/dri nvidia/cuda:12.4.0-base-ubuntu22.04 python3 -c "import torch; print(torch.cuda.is_available())"

# Force GPU mode:
ROBO_EVAL_DOCKER_GPU=dri robo-eval run --benchmark libero --vla pi05 --episodes 1 --runtime docker
```

### Container logs

```bash
# View logs for a running container
docker logs robo-eval-sim-libero-5300

# Follow logs
docker logs -f robo-eval-vla-pi05-5100
```

### Permission denied on X11

```bash
xhost +local:docker
```
