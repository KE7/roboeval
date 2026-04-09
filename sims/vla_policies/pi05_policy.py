#!/usr/bin/env python
"""
Pi 0.5 VLA policy server for robo-eval.

Standalone FastAPI server that loads lerobot/pi05 at startup and serves
action-chunk predictions. Runs in .venvs/vla/ (lerobot 0.4.5, torch 2.10+cu130).

Usage:
    python -m sims.vla_policies.pi05_policy --model-id lerobot/pi05_libero_finetuned --port 5100

Endpoints:
    GET  /health  -> {status, ready, model_id}
    GET  /info    -> {name, model_id, action_space, state_dim, action_chunk_size}
    POST /predict {obs: {image, instruction, state?, image2?}} -> {actions, action_chunk_size, action_space}
"""

from __future__ import annotations

import argparse
import base64
import logging
import traceback
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sims.vla_policies.vla_schema import VLAObservation, PredictRequest

logger = logging.getLogger(__name__)

# Module-level model state (populated at startup)
_policy = None
_preprocessor = None
_postprocessor = None
_model_id: str = ""
_device: str = "cuda"
_action_chunk_size: int = 1
_action_dim: int = 7
_state_dim: int = 8
_camera_key: str = "observation.images.image"
_camera_key2: str = ""
_ready: bool = False
_image_transform: str = "none"  # auto-detected during model loading
_load_error: Optional[str] = None

# CLI args captured in main() before server start
_cli_model_id: str = "lerobot/pi05_libero_finetuned"
_cli_device: str = "cuda"


# ======================================================================
# Image transform auto-detection
# ======================================================================


def _detect_image_transform(model_id: str) -> str:
    """Auto-detect image transform for lerobot models.

    Checks if the model was trained on LIBERO data, which uses 180-degree
    flipped cameras.  When lerobot-eval runs on LIBERO environments it
    injects a ``LiberoProcessorStep`` (``torch.flip(img, dims=[2,3])``)
    into the env processor pipeline.  Since our policy server is decoupled
    from the lerobot env, the *caller* must apply the flip before sending
    images.  We detect this requirement here and expose it via ``/info``.

    Detection logic:
      1. Try to import ``LiberoProcessorStep`` from lerobot — confirms the
         framework knows about the flip.
      2. Check whether the model ID or the config's ``pretrained_path`` /
         ``repo_id`` contains ``"libero"`` — indicates the model was
         trained on LIBERO data with 180-degree-flipped cameras.

    Returns ``"flip_hw"`` if detected, ``"none"`` otherwise.
    """
    short = model_id.split("/")[-1] if "/" in model_id else model_id

    # Step 1: Check that lerobot has LiberoProcessorStep
    try:
        from lerobot.processor.env_processor import LiberoProcessorStep  # noqa: F401
    except ImportError:
        logger.info(
            "[%s] lerobot LiberoProcessorStep not available — image_transform=none",
            short,
        )
        return "none"

    # Step 2: Check model ID and config for LIBERO indicators
    indicators: list[bool] = ["libero" in model_id.lower()]
    if _policy is not None and hasattr(_policy, "config"):
        cfg = _policy.config
        pretrained = getattr(cfg, "pretrained_path", "") or ""
        indicators.append("libero" in pretrained.lower())
        repo = getattr(cfg, "repo_id", "") or ""
        indicators.append("libero" in repo.lower())

    if any(indicators):
        logger.info(
            "[%s] Auto-detected image transform: flip_hw "
            "(from LiberoProcessorStep — model trained on LIBERO data "
            "with 180° flipped cameras)",
            short,
        )
        return "flip_hw"

    logger.info(
        "[%s] No LIBERO training indicators found — image_transform=none",
        short,
    )
    return "none"


# ======================================================================
# Model loading
# ======================================================================


def _load_model(model_id: str, device: str) -> None:
    global _policy, _preprocessor, _postprocessor, _model_id, _device
    global _action_chunk_size, _action_dim, _state_dim, _camera_key, _camera_key2
    global _ready

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION, OBS_STATE

    _model_id = model_id
    _device = device

    logger.info("Loading Pi 0.5 from %s on %s ...", model_id, device)
    _policy = PI05Policy.from_pretrained(model_id)
    _policy.to(torch.device(device))
    _policy.eval()

    # Disable torch.compile on sample_actions.
    # The compiled version fails on PyTorch 2.10 (aarch64/GB10) with:
    #   "expected predicate to be bool, got torch.int64"
    # Unwrap the compiled function if present; otherwise disable dynamo globally.
    inner = _policy.model
    if hasattr(inner, "sample_actions") and hasattr(
        inner.sample_actions, "_torchdynamo_orig_callable"
    ):
        inner.sample_actions = inner.sample_actions._torchdynamo_orig_callable
        logger.info("Disabled torch.compile on sample_actions (PyTorch 2.10 bool mask compat)")
    else:
        try:
            torch._dynamo.reset()
            torch._dynamo.config.disable = True
            logger.info("Disabled torch._dynamo globally")
        except Exception as e:
            logger.warning("Could not disable dynamo: %s", e)

    cfg = _policy.config

    # Load the exact saved lerobot pre/post processor stack for parity with
    # native inference. Override only device placement.
    _preprocessor, _postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    logger.info("Preprocessor and postprocessor loaded from %s", model_id)

    # select_action() dequeues n_action_steps actions from the cached chunk.
    _action_chunk_size = getattr(cfg, "n_action_steps", 1)

    # Determine action_dim from model output features
    if hasattr(cfg, "output_features") and ACTION in cfg.output_features:
        _action_dim = cfg.output_features[ACTION].shape[0]
    else:
        _action_dim = 7

    # Determine actual state_dim from policy config
    if hasattr(cfg, "input_features") and OBS_STATE in cfg.input_features:
        _state_dim = cfg.input_features[OBS_STATE].shape[0]
    else:
        _state_dim = 8

    # Determine camera keys from config (first = primary, second = wrist if present)
    _camera_key = "observation.images.image"
    _camera_key2 = ""
    if hasattr(cfg, "image_features") and cfg.image_features:
        keys = list(cfg.image_features)
        _camera_key = keys[0]
        if len(keys) > 1:
            _camera_key2 = keys[1]

    # Auto-detect image transform (e.g., LIBERO 180° flip)
    global _image_transform
    _image_transform = _detect_image_transform(model_id)

    _ready = True
    logger.info(
        "Pi 0.5 ready: model=%s, action_chunk_size=%d, action_dim=%d, image_transform=%s",
        model_id,
        _action_chunk_size,
        _action_dim,
        _image_transform,
    )

# ======================================================================
# Inference helpers
# ======================================================================


def _decode_image_tensor(image_b64: str):
    import torch
    from PIL import Image

    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")
    img_arr = np.array(pil_img, dtype=np.uint8)
    return torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0


def _build_frame(
    image_b64: str,
    instruction: str,
    state: Optional[list],
    image2_b64: Optional[str],
) -> dict:
    import torch

    frame: dict = {
        _camera_key: _decode_image_tensor(image_b64),
        "observation.state": torch.tensor(
            state if state else [0.0] * _state_dim,
            dtype=torch.float32,
        )[:_state_dim],
        "task": instruction,
    }

    if image2_b64 and _camera_key2:
        frame[_camera_key2] = _decode_image_tensor(image2_b64)

    return frame


def _policy_action_to_list(action) -> list[list[float]]:
    """Convert the HF-postprocessed action to the server's list-of-actions shape.

    The saved lerobot postprocessor already performs the canonical PI0.5
    unnormalization. We intentionally do not apply any extra gripper sign flip
    here so the gripper convention stays identical to HF inference.
    """
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    return [act[:_action_dim].tolist() for act in action]


def _predict(
    image_b64: str,
    instruction: str,
    state: Optional[list],
    image2_b64: Optional[str],
) -> list[list[float]]:
    import torch

    frame = _build_frame(image_b64, instruction, state, image2_b64)
    batch = _preprocessor(frame)

    with torch.no_grad():
        action = _policy.select_action(batch)

    action = _postprocessor(action)
    return _policy_action_to_list(action)


# ======================================================================
# FastAPI application
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _load_error
    try:
        _load_model(_cli_model_id, _cli_device)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        logger.exception("Failed to load Pi 0.5 model: %s", _load_error)
    yield


app = FastAPI(title="Pi 0.5 Policy Server", lifespan=lifespan)






@app.get("/health")
def health():
    if _load_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "ready": False, "model_id": _cli_model_id, "error": _load_error},
        )
    return {"status": "ok", "ready": _ready, "model_id": _model_id}


@app.get("/info")
def info():
    name = _model_id.split("/")[-1] if "/" in _model_id else _model_id
    payload = {
        "name": name,
        "model_id": _model_id,
        "action_space": {
            "type": "eef_delta",
            "dim": _action_dim,
            "description": "End-effector delta: [dx, dy, dz, droll, dpitch, dyaw, gripper_open]",
        },
        "state_dim": 8,
        "action_chunk_size": _action_chunk_size,
        "obs_requirements": {
            "cameras": ["primary"],
            "state_dim": 8,
            "image_resolution": [256, 256],
            "image_transform": _image_transform,
        },
    }
    if _load_error:
        payload["error"] = _load_error
    return payload


@app.post("/reset")
def reset_policy():
    """Reset the policy's internal state between episodes.

    Clears PI0.5's internal action queue so select_action() parity matches
    native lerobot episode boundaries.
    """
    if not _ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready yet"})
    _policy.reset()
    return {"success": True}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not _ready:
            return JSONResponse(
                status_code=503, content={"error": "Model not ready yet"}
            )
        actions = _predict(
            req.obs.images.get("primary"),
            req.obs.instruction,
            req.obs.state.get("flat"),
            req.obs.images.get("wrist"),
        )
        return {
            "actions": actions,
            "action_chunk_size": _action_chunk_size,
            "action_space": {"type": "eef_delta", "dim": _action_dim},
        }
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


# ======================================================================
# CLI entrypoint
# ======================================================================


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Pi 0.5 Policy Server")
    parser.add_argument(
        "--model-id",
        default="lerobot/pi05_libero_finetuned",
        help="HuggingFace model ID (default: lerobot/pi05_libero_finetuned)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5100,
        help="Port to serve on (default: 5100)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device (default: cuda)",
    )
    args = parser.parse_args()

    global _cli_model_id, _cli_device
    _cli_model_id = args.model_id
    _cli_device = args.device

    logging.basicConfig(level=logging.INFO)
    print(f"[pi05_policy] Starting Pi 0.5 server on {args.host}:{args.port}")
    print(f"[pi05_policy] Model: {args.model_id}, Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
