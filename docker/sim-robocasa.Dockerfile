# docker/sim-robocasa.Dockerfile
# ─────────────────────────────────────────────────────────────────────
# RoboCasa (kitchen environments) sim worker.
#
# Python 3.11, robosuite v1.5+ MuJoCo backend.
# Headless EGL rendering by default; GLFW for debug mode at runtime.
#
# Build (from project root):
#   docker build -f docker/sim-robocasa.Dockerfile -t robo-eval/sim-robocasa:latest .
#
# Run (headless):
#   docker run --rm --gpus all -p 5300:5300 robo-eval/sim-robocasa:latest \
#       --sim robocasa --port 5300 --headless
#
# Run (debug window):
#   docker run --rm --gpus all -p 5300:5300 \
#       -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e MUJOCO_GL=glfw \
#       robo-eval/sim-robocasa:latest --sim robocasa --port 5300
# ─────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="RoboCasa kitchen environment sim worker"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps: EGL headless rendering + GLFW for debug mode ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # EGL headless rendering (libegl1-mesa-dev per FM finding — includes headers)
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
        # Python 3.11
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        # Build tools
        git \
        curl \
        cmake \
        build-essential \
        patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python3
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
        numpy \
        torch torchvision \
        mujoco

# Install robosuite v1.5+ (required by RoboCasa)
RUN python3.11 -m pip install --no-cache-dir \
    "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@master"

# Install RoboCasa
RUN python3.11 -m pip install --no-cache-dir \
    "robocasa @ git+https://github.com/robocasa/robocasa.git"

# HTTP server + image handling + video recording
RUN python3.11 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow opencv-python-headless

# ── Copy sim worker code ─────────────────────────────────────────────
COPY sims/ /app/sims/

# ── Volumes ──────────────────────────────────────────────────────────
# RoboCasa assets (~10 GB): mount host cache to avoid re-download
# Set ROBOCASA_ASSET_DIR if assets are in a non-default location.
VOLUME ["/root/.robocasa", "/results"]

EXPOSE 5300

ENTRYPOINT ["python3.11", "-m", "sims.sim_worker"]
CMD ["--sim", "robocasa", "--port", "5300", "--headless"]
