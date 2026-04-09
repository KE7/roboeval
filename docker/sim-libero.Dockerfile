# docker/sim-libero.Dockerfile
# ─────────────────────────────────────────────────────────────────────
# LIBERO / LIBERO-Pro / LIBERO-Infinity sim worker.
#
# Python 3.11, mujoco 3.2.3 (arm64-compatible, NOT mujoco-py).
# Headless EGL rendering by default; set MUJOCO_GL=glfw at runtime for
# windowed debug mode (requires X11 forwarding + GLFW deps below).
#
# Build (from project root):
#   docker build -f docker/sim-libero.Dockerfile -t robo-eval/sim-libero:latest .
#
# Run (headless):
#   docker run --rm --gpus all -p 5300:5300 robo-eval/sim-libero:latest \
#       --sim libero --port 5300 --headless
#
# Run (debug window):
#   docker run --rm --gpus all -p 5300:5300 \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e MUJOCO_GL=glfw \
#       robo-eval/sim-libero:latest --sim libero --port 5300
# ─────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="LIBERO/LIBERO-Pro/LIBERO-Infinity sim worker"

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# ── System deps: EGL headless rendering + GLFW for debug mode ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # EGL headless rendering (libegl1-mesa-dev includes headers + libs)
        libegl1-mesa-dev \
        libgl1-mesa-glx \
        libgles2-mesa-dev \
        libosmesa6-dev \
        libxext6 \
        libxrender1 \
        # GLFW deps for windowed debug mode at runtime
        libglfw3-dev \
        libx11-dev \
        libxrandr-dev \
        libxinerama-dev \
        libxcursor-dev \
        libxi-dev \
        # Python 3.11 (available in Ubuntu 22.04 repos)
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        # Build tools + h5py build deps (arm64 needs source build)
        git \
        curl \
        cmake \
        build-essential \
        patchelf \
        pkg-config \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# ── NVIDIA container runtime env ─────────────────────────────────────
ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────────────
RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.11 -m pip install --no-cache-dir \
        numpy==1.24.3 \
        torch torchvision \
        mujoco==3.2.3 \
        h5py>=3.8 \
        bddl>=3.0.0

# Install robosuite 1.4.0 (arm64-compatible, uses mujoco not mujoco-py)
RUN python3.11 -m pip install --no-cache-dir robosuite==1.4.0

# Install LIBERO from upstream
RUN python3.11 -m pip install --no-cache-dir \
    "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"

# Install libero-infinity (adds procedural task generation to LIBERO)
# --no-deps: upstream pins mujoco<3 and python-fcl>=0.7 (no aarch64 wheel);
# mujoco 3.x is backward-compatible and python-fcl is only needed for
# collision-checking utilities we don't use at eval time.
RUN python3.11 -m pip install --no-cache-dir --no-deps \
    "libero-infinity @ git+https://github.com/KE7/libero-infinity.git" \
    && python3.11 -m pip install --no-cache-dir \
        easydict "gym==0.25.2" "huggingface-hub>=0.20" "bddl==1.0.1"

# For LIBERO-Pro: clone the fork into vendors/ — it overlays the LIBERO package
# with additional task/suite definitions. Mounted or baked-in at build time.
# To build for LIBERO-Pro specifically, uncomment the following:
# RUN git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git /app/vendors/LIBERO-PRO \
#     && python3.11 -m pip install --no-cache-dir -e /app/vendors/LIBERO-PRO

# HTTP server + image handling
RUN python3.11 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow opencv-python-headless

# ── Copy sim worker code ─────────────────────────────────────────────
COPY sims/ /app/sims/

# ── Volumes for datasets and results ─────────────────────────────────
# LIBERO task datasets (HDF5 files): mount to /root/.libero or set LIBERO_DATASET_DIR
VOLUME ["/root/.libero", "/results"]

EXPOSE 5300

ENTRYPOINT ["python3.11", "-m", "sims.sim_worker"]
CMD ["--sim", "libero", "--port", "5300", "--headless"]
