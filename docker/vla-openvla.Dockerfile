# docker/vla-openvla.Dockerfile
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

LABEL maintainer="robo-eval" \
      description="OpenVLA 7B VLA policy server"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1-mesa-dev libgl1-mesa-glx libgles2-mesa-dev \
        libxext6 libxrender1 \
        libglfw3-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
        python3.11 python3.11-venv python3.11-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility

WORKDIR /app

RUN python3.11 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# OpenVLA deps (pin transformers<5 for AutoModelForVision2Seq)
RUN python3.11 -m pip install --no-cache-dir \
    "transformers>=4.40,<5" accelerate>=0.28 "timm>=0.9.10,<1.0"

# HTTP server + image handling
RUN python3.11 -m pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow numpy opencv-python-headless

# CRITICAL: Replace CPU torch with CUDA version
RUN python3.11 -m pip uninstall -y torch torchvision \
    && python3.11 -m pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

COPY sims/__init__.py /app/sims/
COPY sims/vla_policies/ /app/sims/vla_policies/

VOLUME ["/root/.cache/huggingface"]

ENV PYTHONPATH=/app
WORKDIR /app
ENTRYPOINT ["python", "-m", "sims.vla_policies.openvla_policy"]
CMD []
