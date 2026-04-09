# Adding a New VLA Policy Server

This guide walks through integrating a new Vision-Language-Action (VLA) model into robo-eval. By the end, you'll have a standalone FastAPI policy server that the robo-eval orchestrator can call for action predictions.

## Overview

robo-eval uses a **decoupled HTTP architecture**: each VLA model runs as an independent FastAPI server with three endpoints (`/health`, `/info`, `/predict`). The orchestrator discovers the server via `VLA_CONFIGS` in `robo_eval/config.py` and communicates over HTTP.

```
Orchestrator  --HTTP-->  VLA Policy Server  (your model)
     |
     +-----HTTP-->  Sim Worker  (LIBERO, RoboCasa, etc.)
```

## Step 1: Create the Policy Server

Copy the template to get started:

```bash
cp sims/vla_policies/template_policy.py sims/vla_policies/myvla_policy.py
```

The template has all three endpoints stubbed with `TODO` markers. You need to implement:

1. **Model loading** in `_load_model()` — load weights, tokenizer, preprocessor
2. **Inference** in `_predict()` — decode image, run forward pass, return actions

### Endpoint Contracts

#### `GET /health`

Returns whether the model is loaded and ready.

**Response:**
```json
{"ready": true, "model_id": "your-org/your-model"}
```

If loading failed:
```json
{"ready": false, "model_id": "your-org/your-model", "error": "RuntimeError: CUDA OOM"}
```

The orchestrator polls this endpoint on startup (up to `startup_timeout` seconds) before sending predictions.

#### `GET /info`

Returns model metadata so the orchestrator can adapt action translation and state encoding.

**Response:**
```json
{
  "name": "myvla",
  "model_id": "your-org/your-model",
  "action_space": {
    "type": "eef_delta",
    "dim": 7
  },
  "state_dim": 8,
  "action_chunk_size": 1
}
```

**Field details:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Short display name |
| `model_id` | str | HuggingFace model ID or path |
| `action_space.type` | str | `"eef_delta"` (end-effector delta) or `"joint_pos"` |
| `action_space.dim` | int | Action dimensionality (7 for LIBERO EEF, 12 for RoboCasa, 14 for RoboTwin) |
| `state_dim` | int | Expected proprioceptive state dim (8 for LIBERO: eef_pos×3 + axisangle×3 + gripper×2). Set to 0 if model doesn't use state. |
| `action_chunk_size` | int | Number of actions returned per `/predict` call (1 = replan every step, 50 = action chunking) |

#### `POST /predict`

Runs one inference step. The orchestrator sends the current observation and gets back action(s).

**Request:**
```json
{
  "obs": {
    "image": "<base64-encoded PNG, 256x256 RGB>",
    "image2": "<optional base64 PNG, wrist camera>",
    "instruction": "pick up the red bowl and place it on the plate",
    "state": [0.1, 0.2, 0.3, 3.14, 0.0, 0.0, 0.04, 0.04]
  }
}
```

**Response:**
```json
{
  "actions": [[0.01, -0.02, 0.005, 0.0, 0.0, 0.0, -1.0]],
  "chunk_size": 1,
  "model_id": "your-org/your-model"
}
```

**Field details:**

| Field | Type | Description |
|-------|------|-------------|
| `obs.image` | str | Base64-encoded PNG, primary camera (agentview), 256x256 RGB |
| `obs.image2` | str? | Optional base64 PNG, wrist camera. `null` if not available. |
| `obs.instruction` | str | Natural language task instruction |
| `obs.state` | list[float] | Proprioceptive state vector. For LIBERO: `[eef_x, eef_y, eef_z, ax_x, ax_y, ax_z, grip_l, grip_r]` |
| `actions` | list[list[float]] | List of action vectors. Length = `action_chunk_size`. |
| `chunk_size` | int | Number of actions returned (matches `action_chunk_size` from `/info`) |
| `model_id` | str | Echo back for logging/debugging |

### Example: Minimal Model Loading

```python
def _load_model(model_id: str, device: str) -> None:
    global _policy, _model_id, _device, _ready

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _model_id = model_id
    _device = device

    _policy = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    _policy.eval()
    _ready = True
```

### Example: Minimal Inference

```python
def _predict(image_b64: str, instruction: str, state: list | None) -> list[list[float]]:
    import torch
    from PIL import Image

    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")

    # Your model-specific preprocessing here
    inputs = preprocess(pil_img, instruction, state)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        actions = _policy.predict(inputs)

    # Return as list of action vectors
    return [actions.cpu().numpy().tolist()]
```

## Step 2: Register in VLA_CONFIGS

Open `robo_eval/config.py` and add your VLA to the `VLA_CONFIGS` dict:

```python
VLA_CONFIGS: Dict[str, VLAConfig] = {
    "pi05": VLAConfig(...),
    "openvla": VLAConfig(...),
    "smolvla": VLAConfig(...),
    # Add your VLA here:
    "myvla": VLAConfig(
        name="myvla",
        port=5103,                              # pick an unused port
        venv=".venvs/myvla",                    # path to your venv
        start_script="scripts/start_myvla_policy.sh",
        model_id="your-org/your-model",
        startup_timeout=300,                    # seconds to wait for /health ready=true
    ),
}
```

### VLAConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | — | Short identifier, must match the dict key |
| `port` | int | — | HTTP port for the policy server. Pick one that doesn't conflict (existing: 5100, 5101, 5102) |
| `venv` | str | — | Relative path from project root to the Python venv containing your model's dependencies |
| `start_script` | str | — | Relative path to the shell script that launches the server |
| `model_id` | str | — | HuggingFace model ID (used for logging and passed to the server) |
| `startup_timeout` | int | 300 | Max seconds to wait for `/health` to return `ready: true` before giving up |

**Derived properties** (auto-computed):
- `url` → `http://localhost:{port}`
- `venv_python` → `{PROJECT_ROOT}/{venv}/bin/python`
- `start_script_path` → `{PROJECT_ROOT}/{start_script}`

## Step 3: Create a Start Script

Create `scripts/start_myvla_policy.sh`:

```bash
#!/usr/bin/env bash
# Starts the MyVLA policy server.
# Usage: ./scripts/start_myvla_policy.sh [--port PORT]
#
# Environment variables:
#   VLA_MODEL  - HuggingFace model ID (default: your-org/your-model)
#   VLA_PORT   - Port to serve on     (default: 5103)

set -euo pipefail

VENV=.venvs/myvla
MODEL_ID="${VLA_MODEL:-your-org/your-model}"
PORT="${VLA_PORT:-5103}"

export MUJOCO_GL=egl

source "$VENV/bin/activate"
exec python -m sims.vla_policies.myvla_policy \
    --model-id "$MODEL_ID" \
    --port "$PORT" \
    "$@"
```

Make it executable:

```bash
chmod +x scripts/start_myvla_policy.sh
```

## Step 4: Set Up the Virtual Environment

Create and populate a venv with your model's dependencies:

```bash
python3.11 -m venv .venvs/myvla
source .venvs/myvla/bin/activate
pip install torch torchvision  # or your specific versions
pip install transformers fastapi uvicorn[standard] pillow numpy pydantic
pip install your-model-package  # if applicable
```

## Step 5: Test

### Manual testing

```bash
# 1. Start the server
./scripts/start_myvla_policy.sh

# 2. In another terminal, check health
curl http://localhost:5103/health
# -> {"ready": true, "model_id": "your-org/your-model"}

# 3. Check info
curl http://localhost:5103/info
# -> {"name": "myvla", "model_id": "...", "action_space": {...}, ...}

# 4. Send a test prediction (with a tiny 2x2 white PNG)
curl -X POST http://localhost:5103/predict \
  -H "Content-Type: application/json" \
  -d '{
    "obs": {
      "image": "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAADklEQVQI12P4z8BQDwAEgAF/QualIQAAAABJRU5ErkJggg==",
      "instruction": "pick up the bowl",
      "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]
    }
  }'
# -> {"actions": [[...]], "chunk_size": 1, "model_id": "..."}
```

### End-to-end with robo-eval

```bash
robo-eval run --vla myvla --suites libero_spatial --episodes 1 --dry-run
```

## Image Transforms and Preprocessing

VLA models trained with image preprocessing (e.g., lerobot's `LiberoProcessorStep` which applies `torch.flip(img, dims=[2,3])`) must declare their expected image transform in the `/info` endpoint so that robo-eval can apply the correct transform at inference time.

### How It Works

1. Your VLA server's `GET /info` response includes an `obs_requirements.image_transform` field.
2. At startup, `SimWrapper` reads this field and stores the transform name.
3. Every observation image (from `/obs`, `/reset`, `/step`) is passed through `_apply_image_transform()` before being sent to the VLA's `/predict` endpoint.

### Declaring the Transform

Add an `obs_requirements` block to your `/info` response:

```json
{
  "name": "myvla",
  "model_id": "your-org/your-model",
  "action_space": {"type": "eef_delta", "dim": 7},
  "state_dim": 8,
  "action_chunk_size": 1,
  "obs_requirements": {
    "cameras": ["primary", "wrist"],
    "state_dim": 8,
    "image_transform": "flip_hw"
  }
}
```

### Known Transforms

| Transform | Operation | Equivalent | Used By |
|-----------|-----------|------------|---------|
| `"flip_hw"` | Flip both height and width (`[::-1, ::-1]`) | `torch.flip(img, dims=[2,3])` — 180° rotation | pi0.5, SmolVLA (lerobot-trained models) |
| `"flip_h"` | Flip height only (`[::-1]`) | Vertical flip | — |
| `"none"` | No transform (pass-through) | Identity | OpenVLA |

### Auto-Detection for lerobot Models

robo-eval auto-detects the `flip_hw` transform for lerobot-based models (pi0.5, SmolVLA) because lerobot's `LiberoProcessorStep` applies `torch.flip(img, dims=[2,3])` during training. The sim worker provides raw (unflipped) images, and the orchestration layer applies the flip before sending to the VLA.

### For Custom VLAs

If you are adding a custom VLA, you **must** examine your model's training preprocessing pipeline to determine the correct image transform:

1. Check if your training code applies any spatial transforms to camera images (flips, rotations, crops).
2. If it does, declare the matching transform in `obs_requirements.image_transform`.
3. If it does not, set `"image_transform": "none"` (or omit the field — defaults to no transform).

> **Warning:** Using the wrong image transform causes **silent failure**. The model will receive images in a different orientation than it was trained on, leading to 0% task success with no error messages. Always verify the transform matches your training preprocessing.

## Common Pitfalls

### Image Format
- Images arrive as **base64-encoded PNG**, 256x256, RGB.
- Decode with `Image.open(BytesIO(base64.b64decode(b64_str))).convert("RGB")`.
- The sim worker already applies the 180-degree flip (`[::-1, ::-1]`) that matches lerobot's `torch.flip(img, dims=[2,3])`. **Do not flip again** in your policy server.

### Action Dimensions
- LIBERO expects **7-dim EEF delta**: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- Gripper convention: **-1 = close, +1 = open** (LIBERO convention).
  - If your model was trained with RLDS convention (1=close, -1=open), you must invert the gripper dim in your server.
- Actions should be **unnormalized** (real-space deltas). If your model outputs normalized actions, unnormalize before returning.

### State Encoding
- LIBERO state is 8-dim: `eef_pos(3) + axisangle(3) + gripper_qpos(2)`.
- Axis-angle is computed from quaternion using lerobot's formula (not scipy `as_rotvec`).
- If your model doesn't use state, set `state_dim: 0` in `/info` and ignore the `state` field.

### Action Chunking
- If `action_chunk_size > 1`, the orchestrator will execute all returned actions before calling `/predict` again.
- Set `action_chunk_size: 1` to replan every step (safer, more reactive).
- Pi0.5 returns 50 actions per call; SmolVLA and OpenVLA return 1.

### Model Loading
- Load the model in the `lifespan` context manager, not at import time. This ensures the FastAPI app starts (and responds to health checks) even if loading takes minutes.
- Set `_ready = True` only after loading succeeds.
- Catch loading errors and store them in `_load_error` so `/health` can report what went wrong.

### GPU Memory
- Only one VLA should be loaded at a time on a single GPU (unless you have >64GB VRAM).
- Kill existing VLA servers before starting a new one: `pkill -f "myvla_policy"`.

## Reference: Existing VLA Servers

| VLA | Port | Model | action_chunk_size | state_dim | Notes |
|-----|------|-------|-------------------|-----------|-------|
| pi05 | 5100 | `lerobot/pi05_libero_finetuned` | 50 | 8 | Largest model (~15GB), uses action chunking |
| openvla | 5101 | `openvla/openvla-7b-finetuned-libero-spatial` | 1 | 0 | No state input, gripper inverted from RLDS |
| smolvla | 5102 | `HuggingFaceVLA/smolvla_libero` | 1 | 8 | Smallest (~3GB), dual camera support |
| cosmos | 5103 | `nvidia/Cosmos-Policy-RoboCasa-Predict2-2B` | 16 | 9 | Diffusion transformer, 3 cameras (224x224), RoboCasa sim |
| internvla | 5104 | `InternRobotics/InternVLA-A1-3B-RoboTwin` | 50 | 32 | Dual-arm joint_pos, lerobot qwena1 type, RoboTwin sim |

## Setup: Cosmos-Policy

Cosmos-Policy uses NVIDIA's [cosmos-policy](https://github.com/NVlabs/cosmos-policy) framework, **not** standard HuggingFace AutoModel loading. The model checkpoint is a `.pt` file with a custom diffusion transformer architecture.

### Prerequisites

```bash
# Clone the cosmos-policy SDK
git clone --depth=1 https://github.com/NVlabs/cosmos-policy /tmp/cosmos-policy

# Install into the cosmos venv
cd /tmp/cosmos-policy
~/.local/bin/uv pip install --python ~/Documents/research/liten-vla/.venvs/cosmos/bin/python -e . --no-deps
# Then install core deps that actually work on your platform:
~/.local/bin/uv pip install --python ~/Documents/research/liten-vla/.venvs/cosmos/bin/python \
    einops hydra-core omegaconf pillow numpy filelock
```

### Model details

- **Checkpoint**: `Cosmos-Policy-RoboCasa-Predict2-2B.pt` (custom `.pt` format)
- **Architecture**: Diffusion transformer (Cosmos-Predict2-2B base, video-to-action)
- **Inputs**: 3 camera images (224x224 RGB) + 9-dim proprioception + text instruction
- **Outputs**: 32 actions of 7-dim (execution horizon = 16), 5 denoising steps
- **Normalization**: Uses `robocasa_dataset_statistics.json` for action unnormalization
- **Text encoding**: Pre-computed T5 embeddings in `robocasa_t5_embeddings.pkl`

### Start

```bash
bash scripts/start_cosmos_policy.sh --port 5103
# Or directly:
MUJOCO_GL=egl .venvs/cosmos/bin/python -m sims.vla_policies.cosmos_policy --port 5103
```

## Setup: InternVLA-A1

InternVLA-A1 uses a custom [lerobot fork](https://github.com/InternRobotics/InternVLA-A1) with a `qwena1` policy type (Qwen3-VL backbone + MoT action expert). It is **not** compatible with standard `AutoModel` loading.

### Prerequisites

```bash
# Clone the InternVLA-A1 lerobot fork
git clone --depth=1 https://github.com/InternRobotics/InternVLA-A1 /tmp/InternVLA-A1

# Install into the internvla venv
cd /tmp/InternVLA-A1
~/.local/bin/uv pip install --python ~/Documents/research/liten-vla/.venvs/internvla/bin/python -e . --no-deps
# Then install core deps:
~/.local/bin/uv pip install --python ~/Documents/research/liten-vla/.venvs/internvla/bin/python \
    einops draccus safetensors datasets qwen-vl-utils
```

### Model details

- **Checkpoint**: `model.safetensors` (lerobot-compatible)
- **Architecture**: Qwen3-VL (2B) + MoT action expert (Mixture-of-Transformers)
- **Policy type**: `qwena1` (registered in InternVLA-A1's lerobot fork)
- **Inputs**: Up to 3 camera images (224x224) + 32-dim state (padded) + text instruction
- **Outputs**: 50 actions of 32-dim (padded; actual RoboTwin dim = 14 for dual-arm)
- **Normalization**: IDENTITY (no normalization applied)
- **Loading**: `QwenA1Policy.from_pretrained("InternRobotics/InternVLA-A1-3B-RoboTwin")`

### Start

```bash
bash scripts/start_internvla_policy.sh --port 5104
# Or directly:
MUJOCO_GL=egl .venvs/internvla/bin/python -m sims.vla_policies.internvla_policy --port 5104
```
