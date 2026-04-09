#!/usr/bin/env python
"""
InternVLA-A1 FastAPI Server — matches the official RoboTwin inference pipeline.

Pipeline (from InternVLA-A1/evaluation/RoboTwin/inference.py):
  1. ResizeImagesWithPadFn(224, 224) – resize each temporal pair to 224×224
  2. Qwen3_VLProcessorTransformFn()  – tokenise instruction + VLM image processor
  3. NormalizeTransformFn            – state normalisation (mean_std)
  4. predict_action_chunk()          – run in float32
  5. UnNormalizeTransformFn          – action unnormalisation
  6. Delta mode: action_pred += current_state (gripper indices zeroed first)
  7. Gripper binarisation: 0 if val < 0.5 else 1  at indices 6 and 13
  8. Return first 30 actions (infer_horizon=30) as absolute qpos list

Usage (standalone):
    python -m sims.vla_policies.internvla_policy --port 5200

Environment variables:
    PORT     – server port (default: 8000)
    VLA_PORT – alternative port setting
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sims.vla_policies.vla_schema import VLAObservation, PredictRequest

# ─── Path setup: make InternVLA-A1 source importable ─────────────────────────
_INTERNVLA_SRC = Path(__file__).parents[3] / "InternVLA-A1" / "src"
if _INTERNVLA_SRC.exists() and str(_INTERNVLA_SRC) not in sys.path:
    sys.path.insert(0, str(_INTERNVLA_SRC))

logger = logging.getLogger(__name__)

# ─── Constants (matching official inference.py) ───────────────────────────────
_MODEL_ID      = "InternRobotics/InternVLA-A1-3B-RoboTwin"
_STATS_KEY     = "aloha"
_RESIZE        = 224           # ResizeImagesWithPadFn target size
_INFER_HORIZON = 30            # first 30 of the n_action_steps chunk
_ACTION_DIM    = 14            # ALOHA: 7 left + 7 right joint positions
_LEFT_GRIP     = 6             # gripper index in left half
_RIGHT_GRIP    = 13            # gripper index in full 14-dim vector
_IMG_HIST_LEN  = 16            # deque capacity: [t-15, …, t] (interval=15)

# ─── Global model state ───────────────────────────────────────────────────────
_policy: Optional[Any] = None
_input_xfm: Optional[Any]  = None   # composed input transform
_unnorm_fn: Optional[Any]  = None   # action un-normaliser
_dtype  = torch.float32              # official default dtype
_device = "cuda" if torch.cuda.is_available() else "cpu"
_ready        = False
_load_error   = ""
_n_action_steps = 50

# ─── Per-episode image history (cleared on /reset) ───────────────────────────
_head_hist:  collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)
_left_hist:  collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)
_right_hist: collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)


# ─── Model / transform loading ────────────────────────────────────────────────

def _resolve_ckpt() -> Path:
    """Return the local HuggingFace snapshot dir for the model."""
    try:
        from huggingface_hub import snapshot_download
        local = Path(snapshot_download(repo_id=_MODEL_ID, local_files_only=True))
        return local
    except Exception:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=_MODEL_ID))


def _load() -> bool:
    """Load the InternVLA-A1-3B-RoboTwin model and build transforms."""
    global _policy, _input_xfm, _unnorm_fn
    global _dtype, _device, _ready, _load_error, _n_action_steps

    logger.info(f"Loading InternVLA-A1-3B-RoboTwin on device={_device}, dtype={_dtype}...")

    try:
        from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import (
            QwenA1Policy,
            QwenA1Config,
        )
        from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import (
            Qwen3_VLProcessorTransformFn,
        )
        from lerobot.transforms.core import (
            NormalizeTransformFn,
            UnNormalizeTransformFn,
            ResizeImagesWithPadFn,
            RemapImageKeyTransformFn,
            compose,
        )
        from lerobot.configs.policies import PreTrainedConfig

        # ── Checkpoint ────────────────────────────────────────────────────
        ckpt_dir = _resolve_ckpt()
        logger.info(f"Checkpoint dir: {ckpt_dir}")

        config = PreTrainedConfig.from_pretrained(ckpt_dir)
        _policy = QwenA1Policy.from_pretrained(
            config=config, pretrained_name_or_path=ckpt_dir
        )
        _policy = _policy.to(_device).to(_dtype).eval()
        _n_action_steps = _policy.config.n_action_steps

        # ── Normalisation statistics ───────────────────────────────────────
        stats_path = Path(ckpt_dir) / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)[_STATS_KEY]
        stat_keys = ["min", "max", "mean", "std"]

        state_stat = {
            "observation.state": {
                k: np.asarray(stats["observation.state"][k]) for k in stat_keys
            }
        }
        action_stat = {
            "action": {
                k: np.asarray(stats["action"][k]) for k in stat_keys
            }
        }

        _unnorm_fn = UnNormalizeTransformFn(
            selected_keys=["action"],
            mode="mean_std",
            norm_stats=action_stat,
        )

        # ── Input transforms (matching official inference.py) ─────────────
        image_keys = [f"observation.images.image{i}" for i in range(3)]
        _input_xfm = compose([
            ResizeImagesWithPadFn(height=_RESIZE, width=_RESIZE),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(
                selected_keys=["observation.state"],
                mode="mean_std",
                norm_stats=state_stat,
            ),
        ])

        _ready = True
        logger.info(
            f"✓ InternVLA loaded: device={_device}, dtype={_dtype}, "
            f"n_action_steps={_n_action_steps}, infer_horizon={_INFER_HORIZON}"
        )
        return True

    except Exception as e:
        _load_error = str(e)
        logger.error(f"✗ Load failed: {e}", exc_info=True)
        return False


# ─── Utility helpers ──────────────────────────────────────────────────────────

def _b64_to_tensor(b64: str) -> torch.Tensor:
    """Decode base64 image → float32 [H, W, C] tensor in [0, 1]."""
    import PIL.Image as PILImage
    img = PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return torch.from_numpy(np.array(img)).float() / 255.0  # [H, W, C]


def _build_temporal_pair(
    hist: collections.deque,
    current: torch.Tensor,
) -> torch.Tensor:
    """Return [past, current] as [2, H, W, C].

    Official code: past_idx = max(len(hist) - image_history_interval - 1, 0)
    → takes the oldest frame in the deque (up to 15 steps back).
    Falls back to zeros if history is empty.
    """
    if len(hist) == 0:
        past = torch.zeros_like(current)
    else:
        past = hist[0]  # oldest frame still in the deque (maxlen=16)
    return torch.stack([past, current], dim=0)  # [2, H, W, C]


# ─── FastAPI application ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting InternVLA-A1 FastAPI server...")
    _load()
    yield
    logger.info("🛑 Shutting down InternVLA-A1 server.")


app = FastAPI(
    title="InternVLA-A1 Server",
    description="Official RoboTwin inference pipeline for InternVLA-A1-3B-RoboTwin",
    lifespan=lifespan,
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — is the model ready?"""
    result: dict = {"ready": _ready, "model_id": _MODEL_ID}
    if not _ready and _load_error:
        result["error"] = _load_error
    return result


@app.get("/info")
async def info():
    """Model metadata for env_wrapper negotiation."""
    return {
        "model":       _MODEL_ID,
        "model_id":    _MODEL_ID,
        "model_type":  "InternVLA-A1-3B",
        "device":      _device,
        "loaded":      _policy is not None,
        "action_space": {
            "type": "joint_pos",
            "dim":  _ACTION_DIM,
            "accepted_dims": [_ACTION_DIM],
        },
        "state_dim":        _ACTION_DIM,
        "action_chunk_size": _INFER_HORIZON,
        "obs_requirements": {
            "cameras":       ["primary"],
            "state_dim":     _ACTION_DIM,
            "state_format":  "flat",
            "image_transform": "none",
        },
    }


@app.post("/reset")
async def reset():
    """Clear per-episode image history — call once at the start of each episode."""
    _head_hist.clear()
    _left_hist.clear()
    _right_hist.clear()
    logger.info("Episode reset: image history cleared.")
    return {"status": "reset"}


@app.post("/predict")
async def predict(req: PredictRequest) -> dict:
    """Predict action chunk from an observation.

    Request images (req.obs.images):
        "primary"   or "head_camera"  – head camera (required)
        "wrist"     or "left_camera"  – left wrist camera (optional)
        "secondary" or "right_camera" – right wrist camera (optional)
        If wrist/right cameras are absent the head image is used as a fallback.

    Request state (req.obs.state):
        {"flat": [float × 14]}  – current joint positions

    Response:
        {"actions": [[float×14] × 30], "chunk_size": 30, "model_id": ...}
        Actions are absolute joint positions (qpos) with binarised grippers.
    """
    if not _ready or _policy is None:
        return JSONResponse(
            status_code=503,
            content={"error": _load_error or "Model not loaded"},
        )

    try:
        imgs = req.obs.images

        # ── 1. Decode images ──────────────────────────────────────────────
        head_b64  = imgs.get("primary") or imgs.get("head_camera")
        left_b64  = imgs.get("wrist")   or imgs.get("left_camera")
        right_b64 = imgs.get("secondary") or imgs.get("right_camera")

        if head_b64 is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No primary/head_camera image in request"},
            )

        head_t  = _b64_to_tensor(head_b64)                                   # [H,W,C]
        left_t  = _b64_to_tensor(left_b64)  if left_b64  else head_t        # fallback
        right_t = _b64_to_tensor(right_b64) if right_b64 else head_t        # fallback

        # ── 2. Build temporal pairs from history ──────────────────────────
        image0 = _build_temporal_pair(_head_hist,  head_t)   # [2, H, W, C]
        image1 = _build_temporal_pair(_left_hist,  left_t)
        image2 = _build_temporal_pair(_right_hist, right_t)

        # Update history with the current frame
        _head_hist.append(head_t)
        _left_hist.append(left_t)
        _right_hist.append(right_t)

        # ── 3. Permute [T,H,W,C] → [T,C,H,W] (before transforms) ────────
        image0 = image0.permute(0, 3, 1, 2)  # [2, C, H, W]
        image1 = image1.permute(0, 3, 1, 2)
        image2 = image2.permute(0, 3, 1, 2)

        # ── 4. State ──────────────────────────────────────────────────────
        state_vals = req.obs.state.get("flat", [0.0] * _ACTION_DIM)
        state_t    = torch.tensor(state_vals, dtype=_dtype)   # [14]

        instruction = req.obs.instruction
        logger.info(
            f"Predict: '{instruction[:60]}' state_dim={len(state_vals)}"
        )

        # ── 5. Build sample dict for input_transforms ─────────────────────
        # Note: RemapImageKeyTransformFn in input_transforms will set all
        # masks to True (it remaps each key to itself and sets mask=True).
        # We expose mask information via the sample dict but it will be
        # overridden by the remap step — this is consistent with official
        # inference.py which post-hoc sets all masks to True in inputs.
        sample = {
            "observation.images.image0": image0,
            "observation.images.image1": image1,
            "observation.images.image2": image2,
            "observation.state": state_t,
            "task": instruction,
        }

        sample = _input_xfm(sample)

        # ── 6. Batch inputs for predict_action_chunk ──────────────────
        inputs: dict = {}
        for key, val in sample.items():
            if key == "task":
                inputs[key] = [val]
                continue
            if not isinstance(val, torch.Tensor):
                continue
            if val.dtype in (torch.int64, torch.bool):
                if val.ndim == 0:
                    inputs[key] = val.unsqueeze(0).to(_device)
                else:
                    inputs[key] = val[None].to(_device)
            else:
                if val.ndim == 0:
                    inputs[key] = val.unsqueeze(0).to(_device, dtype=_dtype)
                else:
                    inputs[key] = val[None].to(_device, dtype=_dtype)

        # Ensure boolean image masks as [B=1] tensors (all True → attend to all)
        for i in range(3):
            inputs[f"observation.images.image{i}_mask"] = torch.tensor([True]).to(_device)

        # ── 7. Inference (float32, no autocast) ───────────────────────
        with torch.no_grad():
            action_pred, _ = _policy.predict_action_chunk(inputs, decode_image=False)

        # action_pred: [1, n_action_steps, padded_action_dim]
        action_pred = action_pred[0, :_INFER_HORIZON, :_ACTION_DIM]   # [30, 14]

        # ── 8. Unnormalise actions ────────────────────────────────────
        action_pred = _unnorm_fn({"action": action_pred})["action"]    # [30, 14]

        # ── 9. Delta → absolute (add current joint positions) ─────────
        init_action = torch.tensor(state_vals, dtype=_dtype, device=action_pred.device)
        # Zero-out gripper dims before adding (official inference.py line 394-395)
        init_action[_LEFT_GRIP]  = 0.0
        init_action[_RIGHT_GRIP] = 0.0
        action_pred = action_pred + init_action[None]   # broadcast [30, 14]

        # ── 10. Binarise gripper dimensions ───────────────────────────
        action_pred[:, _LEFT_GRIP]  = (action_pred[:, _LEFT_GRIP]  >= 0.5).float()
        action_pred[:, _RIGHT_GRIP] = (action_pred[:, _RIGHT_GRIP] >= 0.5).float()

        actions = action_pred.cpu().numpy().tolist()
        logger.info(f"✓ Predicted {len(actions)} actions × {_ACTION_DIM} dims")
        return {
            "actions":    actions,
            "chunk_size": _INFER_HORIZON,
            "model_id":   _MODEL_ID,
        }

    except Exception as e:
        logger.error(f"✗ Predict failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Inference failed: {str(e)}"},
        )


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """Start the InternVLA-A1 FastAPI server."""
    global _MODEL_ID

    parser = argparse.ArgumentParser(description="InternVLA-A1 FastAPI Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000))),
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        default=_MODEL_ID,
        help=f"HuggingFace model ID (default: {_MODEL_ID})",
    )
    args = parser.parse_args()

    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be 1024-65535, got {args.port}")

    _MODEL_ID = args.model

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )

    print(
        f"[internvla_policy] Starting on {args.host}:{args.port} "
        f"model={args.model} device={_device} dtype={_dtype}"
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
