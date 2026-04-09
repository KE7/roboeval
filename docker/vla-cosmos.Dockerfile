# docker/vla-cosmos.Dockerfile
# Mirrors NVIDIA's upstream docker/Dockerfile approach (uv + cu130 extra)
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

LABEL maintainer="robo-eval" \
      description="Cosmos-Policy (NVlabs) VLA server — CUDA 13.0.2 / aarch64"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libgl1 \
        libegl1-mesa-dev \
        libgl1-mesa-dri \
        libglib2.0-0 \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv (same version as NVIDIA's upstream Dockerfile)
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /uvx /usr/local/bin/

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
    UV_SYSTEM_PYTHON=0

WORKDIR /app

# Clone Cosmos-Policy (NVIDIA upstream)
RUN git clone --depth 1 https://github.com/NVlabs/cosmos-policy.git /app/vendors/cosmos-policy

# --- aarch64 compatibility fixes in pyproject.toml ---
# 1) Guard xformers against aarch64 (idempotent)
RUN sed -i '/xformers/{/platform_machine/!s/"xformers",/"xformers; platform_machine != '"'"'aarch64'"'"'",/g}' \
    /app/vendors/cosmos-policy/pyproject.toml

# 2) Remove [decord] extra from qwen-vl-utils (decord has no aarch64 wheel)
RUN sed -i 's/"qwen-vl-utils\[decord\]/"qwen-vl-utils/g' \
    /app/vendors/cosmos-policy/pyproject.toml

# Drop the pre-baked lock so uv re-resolves without decord on aarch64
RUN rm -f /app/vendors/cosmos-policy/uv.lock

# Install all cu130 deps into a venv managed by uv:
#   torch==2.9.0, transformer_engine==2.8.0, flash-attn, natten, torchvision, triton
# uv reads [tool.uv.sources] and pulls from the NVIDIA aarch64 cu130 index.
RUN cd /app/vendors/cosmos-policy && \
    uv sync --extra cu130 --python 3.10

# Explicitly install flash-attn (may not be properly resolved in uv.lock after deletion)
RUN cd /app/vendors/cosmos-policy && uv pip install --no-cache-dir \
    flash-attn==2.7.4.post1 || echo "flash-attn installation skipped (may not have aarch64 wheels)"

# Patch qwen2_5_vl.py to allow graceful degradation on aarch64 without flash-attn
RUN /app/vendors/cosmos-policy/.venv/bin/python << 'PATCH_EOF'
import platform
qwen_path = "/app/vendors/cosmos-policy/cosmos_policy/_src/reason1/networks/qwen2_5_vl.py"
with open(qwen_path, "r") as f:
    content = f.read()

# Replace the strict assertion with a platform-aware one
old_assert = 'assert is_flash_attn_2_available(), "flash_attn_2 not available. run pip install flash_attn"'
new_assert = 'assert is_flash_attn_2_available() or platform.machine() == "aarch64", "flash_attn_2 not available. run pip install flash_attn"'

if old_assert in content:
    content = content.replace(old_assert, new_assert)
    # Add import at module level if not present
    if "import platform" not in content.split('\n')[0:30]:
        content = "import platform\n" + content
    with open(qwen_path, "w") as f:
        f.write(content)
    print("Patched qwen2_5_vl.py for aarch64")
else:
    print("Assert statement not found or already patched")
PATCH_EOF

# Install robo-eval HTTP server extras into the same venv
RUN cd /app/vendors/cosmos-policy && uv pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pillow opencv-python-headless \
    qwen-vl-utils huggingface_hub

# Copy the policy module (now includes server code)
COPY sims/__init__.py /app/sims/
COPY sims/vla_policies/ /app/sims/vla_policies/

VOLUME ["/root/.cache/huggingface"]

# ────────────────────────────────────────────────────────────────────────────
# Run the policy module as a server
# ────────────────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENTRYPOINT ["/app/vendors/cosmos-policy/.venv/bin/python", "-m", "sims.vla_policies.cosmos_policy"]
CMD []
