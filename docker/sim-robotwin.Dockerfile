# docker/sim-robotwin.Dockerfile
# ─────────────────────────────────────────────────────────────────────
# RoboTwin 2.0 (SAPIEN-based bimanual) sim worker.
#
# Python 3.10, SAPIEN 3.0.0.dev (arm64 build) + Vulkan rendering.
# Headless EGL rendering by default. RoboTwin/SAPIEN also requires
# Vulkan for its physics/rendering pipeline.
#
# Build (from project root):
#   docker build -f docker/sim-robotwin.Dockerfile -t robo-eval/sim-robotwin:latest .
#
# Run (headless):
#   docker run --rm --gpus all -p 5300:5300 robo-eval/sim-robotwin:latest \
#       --sim robotwin --port 5300 --headless
#
# Run (debug window):
#   docker run --rm --gpus all -p 5300:5300 \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e MUJOCO_GL=glfw \
#       robo-eval/sim-robotwin:latest --sim robotwin --port 5300
# ─────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="RoboTwin 2.0 (SAPIEN + Vulkan) sim worker"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps: EGL + Vulkan (SAPIEN requirement) + GLFW ───────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # EGL headless rendering (libegl1-mesa-dev per FM finding)
        libegl1-mesa-dev \
        libgl1-mesa-glx \
        libgles2-mesa-dev \
        libosmesa6-dev \
        libxext6 \
        libxrender1 \
        # Vulkan — required by SAPIEN/RoboTwin
        libvulkan1 \
        libvulkan-dev \
        mesa-vulkan-drivers \
        vulkan-tools \
        # GLFW deps for windowed debug mode at runtime
        libglfw3-dev \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        # Python 3.10
        python3.10 \
        python3.10-venv \
        python3.10-dev \
        python3-pip \
        # Build tools
        git \
        curl \
        cmake \
        build-essential \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ── Vulkan ICD configuration for NVIDIA ──────────────────────────────
# The NVIDIA container runtime provides the ICD file, but we ensure the
# loader can find it. If running without nvidia-container-toolkit, mount
# the host's /usr/share/vulkan/icd.d/ into the container.
RUN mkdir -p /etc/vulkan/icd.d
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d

# ── NVIDIA container runtime env ─────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
# video capability needed for Vulkan rendering
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir \
        numpy \
        torch torchvision \
        pyyaml \
        transforms3d \
        gymnasium

# ── Install SAPIEN 3.0.0.dev (arm64 wheel) ──────────────────────────
# The official SAPIEN nightly index only has x86_64 wheels.
# We use the pre-built arm64 dev wheel instead.
COPY docker/sapien-3.0.0.dev20260202+5e6c676b-cp310-cp310-linux_aarch64.whl /tmp/
RUN python3.10 -m pip install --no-cache-dir /tmp/sapien-3.0.0.dev20260202+5e6c676b-cp310-cp310-linux_aarch64.whl \
    && rm /tmp/sapien-3.0.0.dev20260202+5e6c676b-cp310-cp310-linux_aarch64.whl

# HTTP server + image handling + video recording
RUN python3.10 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow opencv-python-headless

# ── Clone RoboTwin repo (relative-path asset loading) ────────────────
# RoboTwin uses relative paths to assets/objects/*.json, so the repo must
# be present and the working dir must be set to it at runtime.
RUN git clone --depth 1 https://github.com/TianxingChen/RoboTwin.git /app/vendors/RoboTwin

# Install CuRobo (RoboTwin dependency, editable from the repo)
RUN cd /app/vendors/RoboTwin && \
    python3.10 -m pip install --no-cache-dir -e envs/curobo 2>/dev/null || true

# Add /app (for sims module) and RoboTwin to Python path
ENV PYTHONPATH="/app:/app/vendors/RoboTwin:${PYTHONPATH}"

# ── Copy sim worker code ─────────────────────────────────────────────
COPY sims/ /app/sims/

# ── Volumes ──────────────────────────────────────────────────────────
VOLUME ["/results"]

# RoboTwin requires CWD to be the repo root for relative asset paths
WORKDIR /app/vendors/RoboTwin

EXPOSE 5300

# CRITICAL: `import torch` must come before `import sapien` to avoid CUDA segfault.
# The sim_worker already handles this import ordering.
ENTRYPOINT ["python3.10", "-m", "sims.sim_worker"]
CMD ["--sim", "robotwin", "--port", "5300", "--headless"]
