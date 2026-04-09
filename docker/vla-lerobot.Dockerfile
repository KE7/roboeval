# docker/vla-lerobot.Dockerfile
# ─────────────────────────────────────────────────────────────────────
# pi0.5 + SmolVLA policy server (shared lerobot base).
#
# Python 3.12 (required by lerobot >=1.0), PyTorch + CUDA 12.4.
# arm64/amd64 compatible via nvidia/cuda base.
#
# Build (from project root):
#   docker build -f docker/vla-lerobot.Dockerfile -t robo-eval/vla-lerobot:latest .
# ─────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="pi0.5 + SmolVLA VLA server (lerobot base)"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev \
        libxext6 libxrender1 \
        libglfw3-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        software-properties-common git linux-libc-dev build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install lerobot + deps first (will pull CPU torch)
RUN python3.12 -m pip install --no-cache-dir \
    "lerobot @ git+https://github.com/huggingface/lerobot.git"

# HTTP server + image handling
RUN python3.12 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow numpy opencv-python-headless

# CRITICAL: Replace CPU torch with CUDA version AFTER lerobot
RUN python3.12 -m pip uninstall -y torch torchvision \
    && python3.12 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# ── Copy VLA policy server code ──────────────────────────────────────
COPY sims/__init__.py /app/sims/
COPY sims/vla_policies/ /app/sims/vla_policies/

VOLUME ["/root/.cache/huggingface"]

# ── Environment setup for single-container pattern ──────────────────────────
ENV VLA_MODULE=sims.vla_policies.pi05_policy
ENV PYTHONPATH=/app
WORKDIR /app

# ── ENTRYPOINT/CMD to support dynamic VLA_MODULE injection ──────────────────
# vla_manager.py injects VLA_MODULE for pi05/smolvla variants
ENTRYPOINT ["/bin/sh", "-c", "exec python -m ${VLA_MODULE} \"$@\"", "--"]
CMD []
