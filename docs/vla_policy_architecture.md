# VLA Policy Architecture — Developer Guide

**Status:** Current (post-refactor, production)
**Branch:** sim-integration
**Last updated:** 2026-03-02

This document is the authoritative reference for adding new simulators and VLA policy servers to
LITEN. Read it before writing any new code. Every major section documents hard-won lessons from
bugs that cost days of debugging.

---

## Table of Contents

1. [Venv Structure](#1-venv-structure)
2. [HTTP Proxy Pattern](#2-http-proxy-pattern)
3. [Policy Server Interface Contract](#3-policy-server-interface-contract)
4. [The make_pre_post_processors Rule](#4-the-make_pre_post_processors-rule)
5. [Sim Worker Patterns](#5-sim-worker-patterns)
6. [Adding a New Simulator](#6-adding-a-new-simulator)
7. [Adding a New VLA Policy Server](#7-adding-a-new-vla-policy-server)
8. [Key Lessons Learned](#8-key-lessons-learned)
9. [Benchmark Reference](#9-benchmark-reference)
10. [Port Reference](#10-port-reference)

---

## 1. Venv Structure

Each component runs in a dedicated virtualenv. **NEVER cross venvs for the wrong purpose.**
Mixing environments causes silent dependency conflicts that are very hard to debug.

| Venv | Python | Purpose | Key packages |
|------|--------|---------|--------------|
| `.venvs/vla/` | 3.11 | pi05 + openvla policy servers | lerobot 0.4.5, torch 2.10+cu130, transformers, safetensors, timm<1.0.0 |
| `.venvs/smolvla/` | 3.11 | smolvla policy server | lerobot 0.4.5, torch 2.10+cu130, libero (editable), robosuite 1.4.0 |
| `.venvs/libero/` | 3.8.20 | LIBERO sim worker | libero (editable, cloned by `scripts/setup_envs.sh`), FastAPI, uvicorn |
| `.venvs/libero_pro/` | 3.8 | LIBERO-Pro sim worker | libero-pro (editable, cloned by `scripts/setup_envs.sh`), FastAPI, uvicorn |
| `.venvs/litellm/` | 3.11 | `run_sim_eval.py` orchestrator | litellm, requests, Pillow — **no lerobot** |
| `.venvs/robocasa/` | 3.11 | RoboCasa sim worker | robocasa, robosuite 1.5.2, FastAPI, uvicorn |
| `.venvs/robotwin/` | 3.10 | RoboTwin sim worker | SAPIEN 3.0.0, torch 2.10+cu130, FastAPI |

**Critical rules:**
- `run_sim_eval.py` MUST run under `.venvs/litellm/bin/python`. It has litellm; the vla venv does not.
- smolvla has a separate venv from vla because `timm>=0.9.10,<1.0.0` (required by openvla) conflicts
  with smolvla's requirements. Each model family gets its own venv when deps conflict.
- The libero sim workers use Python 3.8 because LIBERO's deps (older robosuite, bddl) require it.
- RoboTwin requires `import torch` BEFORE `import sapien` in the same process — enforced by the
  backend's `initialize()` method.

---

## 2. HTTP Proxy Pattern

All components communicate via JSON over HTTP. **There is no shared memory and no direct imports
between components.** This is intentional: it allows each component to run in its own venv with
incompatible dependencies.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  run_sim_eval.py  (.venvs/litellm/, port varies)                        │
│                                                                          │
│   ┌──────────────┐    HTTP     ┌──────────────────────────────────────┐  │
│   │  VLM proxy   │◄──────────►│  VLM (LiteLLM/Gemini)  port 4000    │  │
│   └──────────────┘            └──────────────────────────────────────┘  │
│                                                                          │
│   ┌──────────────┐    HTTP     ┌──────────────────────────────────────┐  │
│   │  Sim worker  │◄──────────►│  sim_worker.py  port 5001+           │  │
│   │  proxy       │            │  (.venvs/libero/ or .venvs/robocasa/) │  │
│   └──────┬───────┘            └─────────────┬────────────────────────┘  │
│          │                                  │                            │
│          │ HTTP (via env_wrapper.py)         │ HTTP                      │
│          ▼                                  ▼                            │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Policy server  port 5100/5101/5102                              │  │
│   │  (.venvs/vla/ or .venvs/smolvla/)                                │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Data flow for each simulation step:**

1. Orchestrator calls sim worker `/reset` → sim worker returns `{image, image2?, state}`
2. Orchestrator builds a subtask command; env_wrapper.py calls policy server `/predict`
3. Policy server returns `{actions: [[...], ...]}` — already in real-space (unnormalized)
4. env_wrapper.py passes actions through (identity, verified at init) and sends to sim worker `/step`
5. Sim worker returns `{image, image2?, state, done, success}`

Each component is a standalone FastAPI server started with uvicorn. The sim worker endpoint
`/init` is called once per task; `/reset` once per episode.

---

## 3. Policy Server Interface Contract

**Every new policy server MUST implement exactly these three endpoints.** The interface is fixed;
sim workers and the orchestrator rely on it without modification.

### GET /health

```json
// Response (200 OK when ready, 503 when not yet loaded)
{
  "ready": true,
  "model_id": "lerobot/pi05_libero_finetuned"
}
// When load failed:
{
  "ready": false,
  "model_id": "HuggingFaceVLA/smolvla_libero",
  "error": "FileNotFoundError: ..."
}
```

Models load at server startup (not on demand). The caller MUST poll `GET /health` and wait for
`ready: true` before sending any `/predict` requests. pi05 takes ~75s to load; smolvla ~40s.

### GET /info

```json
// Response
{
  "name": "SmolVLA",
  "model_id": "HuggingFaceVLA/smolvla_libero",
  "action_space": {
    "type": "eef_delta",
    "dim": 7,
    "description": "End-effector delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]"
  },
  "state_dim": 8,
  "action_chunk_size": 1
}
```

`env_wrapper.py` fetches `/info` at startup to learn the policy's action space type and dim,
then verifies the VLA's output space matches the sim's expected space (strict identity — no
translation, padding, or truncation). Actions pass through unchanged.

**`action_chunk_size`** is the number of actions returned per `/predict` call (i.e. `n_action_steps`
from the model config — NOT `chunk_size`). For smolvla_libero: 1. For pi05: 50. This determines
how often env_wrapper re-queries the policy during a rollout.

**`state_dim: 0`** signals that the model ignores state (e.g. OpenVLA). The sim worker still sends
state; the policy server simply ignores it.

### POST /predict

```json
// Request
{
  "obs": {
    "image": "<base64-encoded PNG string>",
    "image2": "<base64-encoded PNG string or null>",
    "instruction": "pick up the red bowl and place it on the plate",
    "state": [0.45, -0.12, 0.88, 3.14, 0.01, -0.02, 0.035, 0.035]
  }
}

// Response
{
  "actions": [
    [0.002, -0.001, 0.003, 0.0, 0.0, 0.001, 1.0]
  ],
  "chunk_size": 1,
  "model_id": "HuggingFaceVLA/smolvla_libero"
}
```

**`state`**: plain `list[float]` — eef_pos(3) + axisangle(3) + gripper_qpos(2) = 8-dim for LIBERO.
NOT base64-encoded (that was a fixed bug in the original design).

**`image`/`image2`**: base64-encoded PNG strings. The images have already been flipped 180° by
`sim_worker.py`'s `_extract_image()`. Do NOT flip again in the policy server.

**`actions`**: list of lists. Each inner list is one action vector in real space (already
unnormalized). `chunk_size` tells the caller how many to consume before re-querying.

**No `/init` endpoint.** Models are loaded at startup. If a policy needs per-episode reset,
expose a `POST /reset` endpoint (env_wrapper.py calls it best-effort before each episode).

---

## 4. The make_pre_post_processors Rule

This is the single most important rule for adding new lerobot-based policy servers.

### The Rule

**For any new VLA policy server wrapping a lerobot-format checkpoint, ALWAYS delegate ALL
preprocessing and postprocessing to lerobot's `make_pre_post_processors`. NEVER hand-code
image flipping, resizing, normalization, state encoding, or action postprocessing.**

```python
from lerobot.policies.factory import make_pre_post_processors
```

This function loads the exact same pipeline steps that `lerobot-eval` uses, guaranteeing
byte-for-byte identical behavior to the native evaluation tool.

### Why This Matters — Confirmed Bug History

Manual preprocessing diverges from the training distribution in subtle but catastrophic ways.
This was confirmed empirically with smolvla_libero: before fixing, every manual preprocessing
error alone caused **0% task success** on a finetuned model. Bugs included:

- Wrong image normalization (different mean/std than training)
- Missing or incorrect image flip (lerobot uses `torch.flip(img, dims=[2,3])` — a 180° rotation)
- Using `chunk_size=50` instead of `n_action_steps=1` → only ~10 replans per episode instead of 500
- Wrong state normalization (different scale or missing entirely)

### Canonical Pattern (smolvla_policy.py — the gold standard)

```python
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
import torch
from torchvision import transforms

# --- At startup (in _load_model) ---
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.to(torch.device(device))
policy.eval()

# Load preprocessor and postprocessor from saved pipeline configs.
# preprocessor_overrides sets device for image tensors.
# postprocessor_overrides moves output back to CPU for serialization.
preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": device}},
    postprocessor_overrides={"device_processor": {"device": "cpu"}},
)

# Read action_chunk_size from config — use n_action_steps, NOT chunk_size
action_chunk_size = getattr(policy.config, "n_action_steps", 1)

# Read camera keys from model config (do not hardcode "observation.images.image")
camera_key = list(policy.config.image_features.keys())[0]  # primary
camera_key2 = list(policy.config.image_features.keys())[1] if len(...) > 1 else ""


# --- At inference time (in _predict) ---
to_tensor = transforms.ToTensor()
pil_img = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
img_tensor = to_tensor(pil_img)  # (C, H, W) float32 [0, 1]

frame = {
    camera_key: img_tensor,
    "observation.state": torch.tensor(
        state if state else [0.0] * state_dim,
        dtype=torch.float32,
    )[:state_dim],
    "task": instruction,  # raw string — preprocessor handles tokenization
}
if image2_b64 and camera_key2:
    pil_img2 = Image.open(BytesIO(base64.b64decode(image2_b64))).convert("RGB")
    frame[camera_key2] = to_tensor(pil_img2)

# Preprocess: batch, flip, resize, normalize, tokenize, move to device
batch = preprocessor(frame)

# Model forward (select_action manages the internal action queue)
action = policy.select_action(batch)  # (1, action_dim) on device, normalized

# Postprocess: unnormalize, move to CPU
action = postprocessor(action)  # (1, action_dim) real-space, on CPU

# Return as list-of-lists
actions = [action.squeeze(0).tolist()]  # [[float × dim]]
```

**Between episodes**, call `policy.reset()` to clear the internal action queue. Expose this as
`POST /reset` (env_wrapper.py calls it best-effort at the start of each episode).

### Policy-Specific Exceptions

**pi05_policy.py**: Does NOT use `make_pre_post_processors`. Pi05 uses a special prompt format
where the robot state is discretized into 256-bin tokens embedded in the text prompt
(`"Task: ..., State: <bins>;\nAction: "`). This requires direct tokenizer access that the
standard preprocessor pipeline does not expose cleanly. The manual approach in pi05_policy.py
has been validated to match lerobot-eval (confirmed by 93–98% LITEN scores vs 83–93% native).
Do not replace it without verifying parity.

**openvla_policy.py**: OpenVLA is NOT a lerobot model. It uses HuggingFace `AutoProcessor` +
`AutoModelForVision2Seq` from transformers. Do NOT attempt to use `make_pre_post_processors` on
non-lerobot checkpoints — use the model's own native processor.

### Compliance Status

| Policy Server | Model Type | Preprocessing | Status |
|---|---|---|---|
| `pi05_policy.py` | lerobot | Manual (custom tokenization, validated) | Working — do not change without parity test |
| `smolvla_policy.py` | lerobot | `make_pre_post_processors` ✓ | Gold standard template |
| `openvla_policy.py` | non-lerobot (OpenVLA) | Native `AutoProcessor` ✓ | Correct for this model type |

---

## 5. Sim Worker Patterns

### Image Extraction — `_extract_image()`

The flip happens **once and only once**, in `LiberoBackend._extract_image()` in `sim_worker.py`:

```python
def _extract_image(self, obs):
    image = obs.get("agentview_image")
    image2 = obs.get("robot0_eye_in_hand_image")
    primary = image if image is not None else image2
    # Flip both H and W (180° rotation) to match lerobot's LiberoProcessorStep:
    # torch.flip(img, dims=[2,3])
    return (
        np.asarray(primary, dtype=np.uint8)[::-1, ::-1].copy(),
        np.asarray(image2, dtype=np.uint8)[::-1, ::-1].copy() if image2 is not None else None,
    )
```

**Do NOT flip again in the policy server.** Double-flip = 0% success (confirmed).

### State Extraction — `_extract_state()`

State is returned as a plain `list[float]`, not base64-encoded:

```python
def _extract_state(self, obs: dict) -> list:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)      # (3,)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)    # (4,) [x,y,z,w]
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32) # (2,)
    # Convert quaternion to axis-angle using lerobot's _quat2axisangle formula
    x, y, z, w = float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3])
    den = np.sqrt(max(0.0, 1.0 - w * w))
    if den > 1e-10:
        angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        axisangle = np.array([x, y, z], dtype=np.float32) / den * angle
    else:
        axisangle = np.zeros(3, dtype=np.float32)
    state = np.concatenate([eef_pos, axisangle, gripper])  # (8,)
    return state.tolist()
```

**Do NOT use scipy `as_rotvec()`** — it has different sign handling in edge cases.

### Action Spaces — Auto-Discovery & Strict Identity Matching

Action spaces are **auto-discovered** at init via `GET /info` on both the sim worker and VLA
policy server. `SimWrapper._negotiate_spaces()` enforces **strict identity matching**: the VLA's
action type and dim must exactly match the sim's. No padding, truncation, or translation.

| VLA Output | Sim Expected | Result |
|---|---|---|
| `eef_delta` dim=7 | `eef_delta` dim=7 | ✅ identity pass-through |
| `eef_delta` dim=7 | `eef_delta` dim=12 | ❌ **ValueError** — dim mismatch |
| `eef_delta` dim=6 | `eef_delta` dim=7 | ❌ **ValueError** — dim mismatch |
| `eef_delta` | `joint_pos` | ❌ **ValueError** — type mismatch |

The old `SIM_EXPECTED_ACTION_SPACE` dict and `_translate_action()` method have been removed.
Each sim backend's `get_info()` method declares its action space; the VLA's `/info` declares its
output space. Mismatches are hard errors — use a VLA trained for the target sim.

### LIBERO Episode Init — Critical for Correctness

The LIBERO reset sequence must match native lerobot-eval exactly:

```python
def reset(self, episode_index=None):
    self.env.reset()                          # 1. Base physics reset
    init_state = self.init_states[idx % len]  # 2. Select episode-specific init state
    self.env.set_init_state(init_state)       # 3. Apply init state (CRITICAL)
    if self.delta_actions:
        robot.controller.use_delta = True     # 4. Enable delta control
    dummy_action = [0, 0, 0, 0, 0, 0, -1]    # 5. Warmup: gripper closed (not [0]*7!)
    for _ in range(10):
        self.env.step(dummy_action)           # 6. 10 no-op steps to settle physics
```

**Missing `set_init_state()`** = wrong initial object positions = wrong task = 0% success.
**Warmup gripper must be `[0,0,0,0,0,0,-1]`** (closed). Using `[0]*7` (gripper open) diverges
from lerobot's `get_libero_dummy_action()` and causes physics settling differences.

### Starting the Sim Worker

```bash
MUJOCO_GL=egl .venvs/libero/bin/python sims/sim_worker.py --sim libero --port 5001 --headless
```

Flag is `--sim libero` (not `--backend libero`). The `--headless` flag enables EGL offscreen
rendering (required on servers without a display).

The backend is initialized **lazily** on the first `/init` call. At startup, `backend_initialized`
is always `False` in `/health`.

---

## 6. Adding a New Simulator

### Checklist

- [ ] Create a new Backend class in `sims/sim_worker.py`
- [ ] Implement the required methods (see below)
- [ ] Register in the `BACKENDS` dict at the bottom of the backends section
- [ ] Implement `get_info()` returning `action_space`, `obs_space` (auto-discovered by env_wrapper)
- [ ] Add `SIM_MAX_STEPS` entry in `sims/env_wrapper.py`
- [ ] Decide which venv to use (or create a new one if deps conflict)
- [ ] Test with an existing policy server before writing a new one

### Required Backend Methods

```python
class MySimBackend:
    def init(self, task_name: str, camera_resolution: int,
             suite: str = None, headless: bool = True) -> dict:
        """Load task, create env. Return {"task_description": str}."""

    def reset(self, episode_index: int = None):
        """Reset to initial state. Return (img, img2) numpy arrays (uint8)
        or a single img if no wrist camera."""

    def step(self, action: list):
        """Step with action. Return (img, img2, reward, done, info)."""

    def get_obs(self):
        """Return current (img, img2) without stepping."""

    def check_success(self) -> bool:
        """Return True if task is currently succeeded."""

    def close(self):
        """Clean up resources."""

    def _extract_image(self, obs) -> tuple:
        """Extract (primary, wrist) image arrays from raw obs dict.
        Apply any required flips here (once, not in the policy server)."""

    def _extract_state(self, obs) -> list:
        """Extract proprioceptive state as list[float].
        NOT base64 encoded — plain list."""
```

### Registration

```python
BACKENDS = {
    "libero":     LiberoBackend,
    "robocasa":   RoboCasaBackend,
    "robotwin":   RoboTwinBackend,
    "libero_pro": LiberoProBackend,
    "mysim":      MySimBackend,     # add your backend here
}
```

### Key Constraints

- **NEVER modify policy servers** to accommodate a new sim. The HTTP interface is fixed.
- RoboTwin uses a different API (`setup_demo` / `get_obs` / `take_action`) — see `RoboTwinBackend`
  for how to handle non-gym simulators that don't follow the reset/step pattern.
- Import `torch` before `sapien` if your sim uses SAPIEN (RoboTwin pattern).
- Set `MUJOCO_GL=egl` before any MuJoCo import for headless rendering.

---

## 7. Adding a New VLA Policy Server

### Checklist

- [ ] Copy `sims/vla_policies/smolvla_policy.py` as your template
- [ ] For lerobot models: use `make_pre_post_processors()` — no exceptions
- [ ] For non-lerobot models: use the model's own native processor (see openvla_policy.py)
- [ ] Implement `/health`, `/info`, `/predict` endpoints (mandatory)
- [ ] Read `action_chunk_size` from `policy.config.n_action_steps` — NOT `chunk_size`
- [ ] Read camera keys from `policy.config.image_features` — do NOT hardcode key names
- [ ] Pick an unused port (5100=pi05, 5101=openvla, 5102=smolvla, 5103+ for new ones)
- [ ] Create a start script in `scripts/`
- [ ] Test parity with native `lerobot-eval` before declaring the server working
- [ ] Use `.venvs/vla/` or create a new venv if deps conflict

### Template (condensed from smolvla_policy.py)

```python
#!/usr/bin/env python
"""MyModel policy server for LITEN. Port: 5103"""
from contextlib import asynccontextmanager
import argparse, base64, logging, traceback
from io import BytesIO
from typing import Optional
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Module-level state
_policy = _preprocessor = _postprocessor = None
_model_id = _camera_key = _camera_key2 = ""
_action_chunk_size = _action_dim = _state_dim = 0
_ready = False
_load_error: Optional[str] = None
_cli_model_id = "org/my-model"
_cli_device = "cuda"

def _load_model(model_id, device):
    global _policy, _preprocessor, _postprocessor, _model_id, _device
    global _action_chunk_size, _action_dim, _state_dim, _camera_key, _camera_key2, _ready
    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.mymodel.modeling_mymodel import MyModelPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    _policy = MyModelPolicy.from_pretrained(model_id)
    _policy.to(torch.device(device)).eval()

    _preprocessor, _postprocessor = make_pre_post_processors(
        _policy.config, pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    cfg = _policy.config
    _action_chunk_size = getattr(cfg, "n_action_steps", 1)  # NOT chunk_size
    _action_dim = cfg.output_features[ACTION].shape[0]
    _state_dim = cfg.input_features[OBS_STATE].shape[0]

    keys = list(cfg.image_features.keys())
    _camera_key = keys[0]
    _camera_key2 = keys[1] if len(keys) > 1 else ""
    _ready = True

def _predict(image_b64, instruction, state, image2_b64=None):
    import torch
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    frame = {
        _camera_key: to_tensor(Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")),
        "observation.state": torch.tensor(state or [0.0]*_state_dim, dtype=torch.float32)[:_state_dim],
        "task": instruction,
    }
    if image2_b64 and _camera_key2:
        frame[_camera_key2] = to_tensor(Image.open(BytesIO(base64.b64decode(image2_b64))).convert("RGB"))

    batch = _preprocessor(frame)
    action = _policy.select_action(batch)
    action = _postprocessor(action)
    return [action.squeeze(0).tolist()]

@asynccontextmanager
async def lifespan(app):
    global _load_error
    try:
        _load_model(_cli_model_id, _cli_device)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
    yield

app = FastAPI(lifespan=lifespan)

class _ObsRequest(BaseModel):
    image: str; instruction: str
    state: Optional[list[float]] = None
    image2: Optional[str] = None

class PredictRequest(BaseModel):
    obs: _ObsRequest

@app.get("/health")
def health():
    if _load_error:
        return JSONResponse(503, {"ready": False, "model_id": _cli_model_id, "error": _load_error})
    return {"ready": _ready, "model_id": _model_id}

@app.get("/info")
def info():
    return {
        "name": "MyModel",
        "model_id": _model_id or _cli_model_id,
        "action_space": {"type": "eef_delta", "dim": _action_dim},
        "state_dim": _state_dim,
        "action_chunk_size": _action_chunk_size,
    }

@app.post("/reset")
def reset_policy():
    if not _ready:
        return JSONResponse(503, {"error": "Model not ready"})
    _policy.reset()
    return {"success": True}

@app.post("/predict")
def predict(req: PredictRequest):
    if not _ready:
        return JSONResponse(503, {"error": _load_error or "Model not ready"})
    try:
        actions = _predict(req.obs.image, req.obs.instruction, req.obs.state, req.obs.image2)
        return {"actions": actions, "chunk_size": len(actions), "model_id": _model_id}
    except Exception as e:
        return JSONResponse(500, {"error": str(e), "traceback": traceback.format_exc()})
```

### Parity Test Before Declaring Done

Always verify your policy server matches native `lerobot-eval` on at least one task:

```bash
# 1. Run native lerobot-eval (ground truth)
MUJOCO_GL=egl .venvs/vla/bin/lerobot-eval \
  --policy.path=org/my-model --env.type=libero \
  --env.task=libero_spatial --eval.batch_size=1 --eval.n_episodes=5

# 2. Run LITEN pipeline with new policy server
MUJOCO_GL=egl .venvs/litellm/bin/python run_sim_eval.py \
  --no-vlm --suite libero_spatial --episodes 5 \
  --vla-url http://localhost:5103

# Results should be within ~5% of each other.
# If LITEN score is 0% but native is >0%, there is a preprocessing bug.
```

---

## 8. Key Lessons Learned

These are bugs that caused significant wasted effort. Read them before writing new code.

### Image Processing

**Double-flip = 0% success.** `sim_worker._extract_image()` flips images once (180° rotation).
Do NOT flip again in the policy server. The flip in sim_worker matches lerobot's
`LiberoProcessorStep` which does `torch.flip(img, dims=[2,3])`. Adding a second flip sends the
robot an upside-down view of the world and causes 0% success.

### Action Chunk Size

**`n_action_steps` ≠ `chunk_size`.** Always read `n_action_steps` from the model config, not
`chunk_size`. For smolvla_libero: `chunk_size=50` but `n_action_steps=1`. Using `chunk_size=50`
means only ~10 replans per 500-step episode instead of 500 — the robot gets stuck executing
stale actions. This caused 0% success before the fix.

### LIBERO Episode Initialization

**`set_init_state()` is mandatory.** Without it, LIBERO uses random initial object positions that
don't match the task's intended configuration. Also: `episode_index` must be passed correctly —
using index 0 for every episode means the same initial state is reused, which inflates variance
and doesn't match the evaluation protocol.

### OpenVLA Gripper Convention

**RLDS convention is inverted from LIBERO.** OpenVLA was trained with RLDS where `1=close,
-1=open`. LIBERO uses the opposite: `-1=close, 1=open`. The policy server must binarize and
invert: `gripper = -(1.0 if action[-1] > 0.0 else -1.0)`. Without this, the robot's gripper
always does the opposite of what the model intends.

### VLM Context Overflow

**Use `--experience-dir` per task, not shared.** The VLM accumulates experience (subtask history,
scene descriptions) across episodes. If a shared experience directory is used across all tasks in
a suite, the context window fills up and the VLM starts making poor decisions by task 5+. Always
pass a fresh `--experience-dir` for each task.

### VLM Instruction Rewriting Bug

**The VLM rewrites task instructions** (e.g. "black bowl" → "silver bowl"), directing the policy
to the wrong object. This is a known open issue causing degraded col3 (LITEN+VLM) results on
libero_10. The issue is in the VLM planner, not the policy pipeline.

### Port Conflicts and Stale Workers

**Always start fresh on a new port after code changes.** Sim workers stay bound to their ports
even when the process should have exited. After changing `smolvla_policy.py` or `sim_worker.py`,
kill old processes and restart on a new port, then verify the new code is running.

### Model Loading Time

**Poll `/health` before sending requests.** Pi05 takes ~75s to load; smolvla ~40s. Sending a
`/predict` request before the model is ready returns a 503 error. Always poll with exponential
backoff until `ready: true`.

### Native lerobot-eval Episode Count

**`--eval.n_episodes=N` is per task, not total.** With 10 tasks and `--eval.n_episodes=10`,
lerobot-eval runs 100 episodes total. Results should be reported as a percentage of the total.

### GPU Memory

**Kill pi05 server before running native lerobot-eval.** Pi05 and lerobot-eval together exceed
GPU memory on the GB10. Kill the server first: `pkill -f pi05_policy`. Same for smolvla.

### PyTorch 2.10 Compatibility

**Attention mask must be `.bool()`.** PyTorch 2.10 on aarch64/GB10 fails in torch.compile with
`"expected predicate to be bool, got torch.int64"`. Cast the tokenizer's `attention_mask` to
`.bool()` before passing to the model. Also unwrap `_torchdynamo_orig_callable` on compiled
functions (see pi05_policy.py for the pattern).

### Normalization Stats Location

**`meta/stats.json` does NOT always exist.** The actual stats are in
`policy_postprocessor_step_0_unnormalizer_processor.safetensors` in the HF cache snapshot
directory. `make_pre_post_processors()` handles this automatically — yet another reason to use it
instead of manual unnormalization.

---

## 9. Benchmark Reference

### Confirmed Results (10 eps/task, 2026-03-01)

Policy: `lerobot/pi05_libero_finetuned`. LIBERO benchmark.
LITEN runs used `--delta-actions` and `--experience-dir` per task.
Native runs used `lerobot-eval` with `--eval.n_episodes=10`.

| Suite | LITEN no-VLM (col1) | Native lerobot-eval (col2) | LITEN+VLM (col3) |
|-------|--------------------|-----------------------------|------------------|
| libero_spatial | **94%** (94/100) | 85% (170/200)† | 75% (75/100) |
| libero_object  | **96%** (96/100) | 89% (89/100) | 75% (75/100) |
| libero_goal    | **98%** (98/100) | 93% (93/100) | 87% (87/100) |
| libero_10      | **77%** (77/100) | 83% (83/100) | 33% (33/100)‡ |

† libero_spatial native used 20 eps/task (200 total); all others 10 eps/task.
‡ libero_10 col3 degraded by the VLM instruction rewriting bug (open issue).

**LITEN no-VLM exceeds native lerobot-eval on 3 of 4 suites.** The pipeline overhead is
negligible; the advantage comes from deterministic episode initialization and subtask framing.

### Confirmed Results for Other Models (partial)

| Model | Suite | Method | Result |
|-------|-------|--------|--------|
| `openvla/openvla-7b-finetuned-libero-spatial` | libero_spatial | Native (5/10 tasks, 20 eps each) | 76% partial |
| `HuggingFaceVLA/smolvla_libero` | libero_spatial | Native lerobot-eval (1 ep/task) | 90% |
| `lerobot/smolvla_base` | all suites | LITEN col1+col3 | 0% (no gripper output) |

See `docs/benchmark_results.md` for full per-task breakdowns.

---

## 10. Port Reference

| Port | Service | Venv | Start Command |
|------|---------|------|---------------|
| 4000 | LiteLLM VLM proxy (Gemini) | `.venvs/litellm/` | `scripts/start_vlm.sh` |
| 5001 | Sim worker (default) | sim-specific | `MUJOCO_GL=egl .venvs/libero/bin/python sims/sim_worker.py --sim libero --port 5001 --headless` |
| 5100 | pi05 policy server | `.venvs/vla/` | `scripts/start_pi05_policy.sh` |
| 5101 | openvla policy server | `.venvs/vla/` | `scripts/start_openvla_policy.sh` |
| 5102 | smolvla policy server | `.venvs/smolvla/` | `MUJOCO_GL=egl .venvs/smolvla/bin/python -m sims.vla_policies.smolvla_policy --port 5102` |
| 5103+ | future policy servers | new venv | create `scripts/start_<model>_policy.sh` |

**Note:** The sim worker port (default 5001) can be changed with `--port`. Use a different port if
running multiple simulators in parallel (e.g., 5001 for libero, 5002 for robocasa).

Multiple sim workers can run simultaneously on different ports. Each `run_sim_eval.py` invocation
specifies which sim worker and policy server URLs to use via `--sim-url` and `--vla-url`.
