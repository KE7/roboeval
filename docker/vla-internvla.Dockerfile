# docker/vla-internvla.Dockerfile
# InternVLA-A1 FastAPI Server for Docker Container
# This container runs the FULL model server on port 5100
# Following the OFFICIAL InternVLA-A1 setup from https://github.com/InternRobotics/InternVLA-A1
#
# The host-side thin-client (sims/vla_policies/internvla_policy.py) proxies requests to this server.
#
# OFFICIAL SETUP (from tutorials/installation.md):
# 1. Clone the repo
# 2. pip install torch==2.7.1 torchvision==0.22.1 --index-url .../cu128
# 3. pip install transformers==4.57.1 [+ other deps]
# 4. pip install -e .
# 5. CRITICAL: Copy patched transformers model files over the installed package
#    cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models ${SITE_PACKAGES}/transformers/

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="InternVLA-A1 FastAPI model server (official setup)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies (FFmpeg, SVT-AV1, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenGL and graphics rendering
        libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev \
        libxext6 libxrender1 \
        # GLFW + X11 (for potential headless rendering, mujoco)
        libglfw3-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        # Build and Python
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git linux-libc-dev build-essential wget curl \
        # FFmpeg and video codecs (required by lerobot)
        ffmpeg libavformat-dev libavcodec-dev libavdevice-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3.10 as default (official setup uses Python 3.10)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Environment: CUDA, mujoco
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

WORKDIR /app

# Upgrade pip/setuptools/wheel first
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ────────────────────────────────────────────────────────────────────────────
# Clone the official InternVLA-A1 repository
# ────────────────────────────────────────────────────────────────────────────
RUN git clone https://github.com/InternRobotics/InternVLA-A1.git /internvla

WORKDIR /internvla

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Install PyTorch (EXACT VERSIONS, CUDA 12.8)
# From official tutorials/installation.md step 4
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Install core Python dependencies (EXACT VERSIONS from installation.md step 5)
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Install the InternVLA-A1 package (pip install -e .)
# This installs lerobot and all its dependencies
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir -e .

# ────────────────────────────────────────────────────────────────────────────
# Step 4: CRITICAL - Copy patched transformers model files
# From official tutorials/installation.md step 6
# This patches Qwen3-VL and other models to support custom architectures
# This is the OFFICIAL way to fix rope_theta, NOT runtime monkey-patches
# ────────────────────────────────────────────────────────────────────────────
RUN TRANSFORMERS_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")/transformers && \
    echo "Copying patched transformers to ${TRANSFORMERS_DIR}" && \
    cp -r src/lerobot/policies/pi0/transformers_replace/models ${TRANSFORMERS_DIR} && \
    cp -r src/lerobot/policies/InternVLA_A1_3B/transformers_replace/models ${TRANSFORMERS_DIR} && \
    cp -r src/lerobot/policies/InternVLA_A1_2B/transformers_replace/models ${TRANSFORMERS_DIR} && \
    echo "✓ Patched transformers files copied successfully"

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Install FastAPI + HTTP server dependencies
# ────────────────────────────────────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pydantic pillow opencv-python-headless

# ────────────────────────────────────────────────────────────────────────────
# Copy the policy module (now includes server code)
# ────────────────────────────────────────────────────────────────────────────
WORKDIR /app
COPY sims/__init__.py /app/sims/
COPY sims/vla_policies/ /app/sims/vla_policies/

# HuggingFace model cache
VOLUME ["/root/.cache/huggingface"]

# ────────────────────────────────────────────────────────────────────────────
# Run the policy module as a server
# ────────────────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENTRYPOINT ["python", "-m", "sims.vla_policies.internvla_policy"]
CMD []
