# docker/vla-groot.Dockerfile
# GR00T N1.6 VLA policy server for robo-eval
# Based on OFFICIAL NVIDIA Isaac-GR00T setup from:
#   https://github.com/NVIDIA/Isaac-GR00T/blob/main/docker/Dockerfile
#
# Build with:
#   docker build \
#     -f docker/vla-groot.Dockerfile \
#     -t robo-eval/vla-groot:latest \
#     .
#
# The official repo provides the exact setup procedure; this Dockerfile
# follows it exactly, with only minimal modifications for our policy server.

# Use official NVIDIA PyTorch container as base (includes CUDA, PyTorch, cuDNN)
FROM nvcr.io/nvidia/pytorch:25.04-py3

LABEL maintainer="robo-eval" \
      description="GR00T N1.6 3B VLA policy server (official NVIDIA Isaac-GR00T)"

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

# ── Install build dependencies ──────────────────────────────────────────────
# Following official Isaac-GR00T Dockerfile
RUN apt-get update && apt-get install -y \
    build-essential yasm cmake libtool git pkg-config \
    libass-dev libfreetype6-dev libvorbis-dev \
    autoconf automake texinfo tmux ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python packaging tools ─────────────────────────────────────────
# Following official Isaac-GR00T Dockerfile
RUN pip install --upgrade pip setuptools
RUN pip install uv

# ── Fix constraint.txt conflicts (documented in official Dockerfile) ──────────
# From official: these pins conflict with transformers dependencies
RUN sed -i s/dill==0.3.9/dill==0.3.8/g /etc/pip/constraint.txt
RUN sed -i s/scipy==1.15.2/scipy==1.15.3/g /etc/pip/constraint.txt

# ── Install core dependencies (following official Isaac-GR00T) ─────────────────
# These are required by GR00T N1.6 for model loading, inference, and utilities
# NOTE: decord and torchcodec will be installed by Isaac-GR00T via uv, skip here
#       to avoid ARM64 availability issues
RUN pip install \
    imageio h5py boto3 \
    transformers[torch] \
    deepspeed timm peft diffusers wandb tianshou dm_tree openai \
    albumentations==1.4.18

# ── Build PyTorch3D from source (required by GR00T) ────────────────────────────
RUN mkdir -p /opt
RUN cd /opt && git clone https://github.com/facebookresearch/pytorch3d.git
RUN cd /opt/pytorch3d && pip install .

WORKDIR /workspace

# ── Copy Isaac-GR00T source code ────────────────────────────────────────────────
# Following official: copy the entire GR00T source into workspace
# Isaac-GR00T is at ../Isaac-GR00T relative to liten-vla repo root
COPY ../Isaac-GR00T /workspace/gr00t

# ── Install Isaac-GR00T package and dependencies (via uv) ──────────────────────
# uv respects [tool.uv.sources] in pyproject.toml, which ensures correct wheels
RUN cd /workspace/gr00t && uv sync && uv pip install -e .

# ── Set up virtual environment paths ──────────────────────────────────────────
WORKDIR /workspace/gr00t
ENV VIRTUAL_ENV=/workspace/gr00t/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# ── Install FastAPI/Uvicorn via uv (respects venv) ─────────────────────────────
# Required by groot_server.py; use uv to install into the gr00t project venv
RUN cd /workspace/gr00t && uv pip install fastapi "uvicorn[standard]"

# ── Install EGL/OpenGL support (for rendering/vision tasks) ──────────────────
RUN apt-get update && apt-get install -y libegl1
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    cat >/usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

RUN mkdir -p /usr/share/vulkan/icd.d && \
    cat >/usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version": "1.2.140"
    }
}
EOF

# ── Copy policy module (now includes server code) ─────────────────────────
COPY sims/__init__.py /app/sims/
COPY sims/vla_policies/ /app/sims/vla_policies/

# ── HuggingFace model cache (mount at runtime) ─────────────────────────────
VOLUME ["/root/.cache/huggingface"]

# ── Set up server entrypoint ───────────────────────────────────────────────
ENV PYTHONPATH=/app
WORKDIR /app

# Use venv interpreter which has GR00T and FastAPI packages
ENTRYPOINT ["/workspace/gr00t/.venv/bin/python", "-m", "sims.vla_policies.groot_policy"]
CMD []
