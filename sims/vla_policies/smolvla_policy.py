#!/usr/bin/env python
"""
SmolVLA policy server for robo-eval.

Standalone FastAPI server that loads HuggingFaceVLA/smolvla_libero at startup
and serves action predictions. Uses lerobot's make_pre_post_processors for
preprocessing and postprocessing — byte-for-byte identical to native lerobot-eval.

Usage:
    python -m sims.vla_policies.smolvla_policy --port 5102

Endpoints:
    GET  /health  -> {ready, model_id}
    GET  /info    -> {name, model_id, action_space, state_dim, action_chunk_size}
    POST /predict {obs: {image, image2?, instruction, state?}} -> {actions, chunk_size, model_id}
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

MODEL_ID_DEFAULT = "HuggingFaceVLA/smolvla_libero"

# Module-level model state (populated at startup)
_policy = None
_preprocessor = None
_postprocessor = None
_model_id: str = ""
_device: str = "cuda"
_action_chunk_size: int = 1   # n_action_steps from model config (actions returned per call)
_action_dim: int = 7
_state_dim: int = 8
_camera_key: str = ""
_camera_key2: str = ""
_ready: bool = False
_load_error: Optional[str] = None
_image_transform: str = "none"  # auto-detected during model loading

# CLI args captured in main() before server start
_cli_model_id: str = MODEL_ID_DEFAULT
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
    global _policy, _preprocessor, _postprocessor
    global _model_id, _device
    global _action_chunk_size, _action_dim, _state_dim
    global _camera_key, _camera_key2, _ready

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION

    _model_id = model_id
    _device = device

    logger.info("Loading SmolVLA from %s on %s ...", model_id, device)
    _policy = SmolVLAPolicy.from_pretrained(model_id)
    _policy.to(torch.device(device))
    _policy.eval()

    # Build preprocessor and postprocessor from the saved pipeline configs.
    # This loads the exact same steps as native lerobot-eval, overriding only the device.
    _preprocessor, _postprocessor = make_pre_post_processors(
        _policy.config,
        pretrained_path=model_id,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    logger.info("Preprocessor and postprocessor loaded from %s", model_id)

    # Read config metadata
    cfg = _policy.config

    # n_action_steps: how many actions are dequeued per select_action call.
    # For smolvla_libero: n_action_steps=1, chunk_size=50.
    # → robo-eval replans every step, matching native lerobot-eval behavior.
    _action_chunk_size = getattr(cfg, "n_action_steps", 1)

    # Determine actual action_dim from model output features
    try:
        if hasattr(cfg, "output_features") and ACTION in cfg.output_features:
            _action_dim = cfg.output_features[ACTION].shape[0]
    except Exception:
        pass  # keep default

    # Determine actual state_dim from model input features
    try:
        from lerobot.utils.constants import OBS_STATE
        if hasattr(cfg, "input_features") and OBS_STATE in cfg.input_features:
            _state_dim = cfg.input_features[OBS_STATE].shape[0]
    except Exception:
        pass  # keep default

    # Determine camera keys from image_features (first = primary, second = wrist)
    _camera_key = "observation.images.image"  # fallback
    _camera_key2 = ""
    if hasattr(cfg, "image_features") and cfg.image_features:
        keys = list(cfg.image_features)
        _camera_key = keys[0]
        logger.info("Using camera key from model config: %s", _camera_key)
        if len(keys) > 1:
            _camera_key2 = keys[1]
            logger.info("Using second camera key from model config: %s", _camera_key2)
    else:
        logger.warning("No image_features in model config; using fallback key: %s", _camera_key)

    # Auto-detect image transform (e.g., LIBERO 180° flip)
    global _image_transform
    _image_transform = _detect_image_transform(model_id)

    _ready = True
    logger.info(
        "SmolVLA ready: model=%s, n_action_steps=%d (chunk_size=%d), action_dim=%d, "
        "state_dim=%d, camera=%s, image_transform=%s",
        model_id,
        _action_chunk_size,
        getattr(cfg, "chunk_size", 50),
        _action_dim,
        _state_dim,
        _camera_key,
        _image_transform,
    )


# ======================================================================
# Inference
# ======================================================================


def _predict(
    image_b64: str,
    instruction: str,
    state: Optional[list],
    image2_b64: Optional[str] = None,
) -> list[list[float]]:
    """Run one inference step and return a list of action vectors.

    Uses lerobot's preprocess/postprocess pipelines for correct normalization.
    Returns _action_chunk_size (=1 for smolvla_libero) actions per call.
    """
    import torch
    from PIL import Image
    from torchvision import transforms

    # Decode base64 PNG → PIL Image (RGB) → (C, H, W) float32 [0, 1]
    to_tensor = transforms.ToTensor()

    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")
    img_tensor = to_tensor(pil_img)  # (C, H, W)

    # Build frame dict in lerobot format.
    # batch_to_transition splits:
    #   - "observation.*" keys → observation dict
    #   - "task" key → complementary_data["task"]
    # The preprocessor then handles batching, newline, tokenization,
    # device placement, and state normalization (MEAN_STD).
    # Images use IDENTITY normalization — the model's prepare_images()
    # handles resize-with-pad + [0,1]→[-1,1] scaling internally.
    frame: dict = {
        _camera_key: img_tensor,
        "observation.state": torch.tensor(
            state if state else [0.0] * _state_dim,
            dtype=torch.float32,
        )[:_state_dim],
        "task": instruction,
    }

    if image2_b64 and _camera_key2:
        raw2 = base64.b64decode(image2_b64)
        pil_img2 = Image.open(BytesIO(raw2)).convert("RGB")
        frame[_camera_key2] = to_tensor(pil_img2)

    # Preprocess: batch, newline, tokenize, move to device, normalize state
    batch = _preprocessor(frame)

    # Run model (select_action manages the internal action queue;
    # with n_action_steps=1, the queue is refilled every call)
    action = _policy.select_action(batch)  # (1, action_dim) normalized, on device

    # Postprocess: unnormalize action, move to CPU
    action = _postprocessor(action)  # (1, action_dim) unnormalized, on CPU

    # Convert to Python list-of-lists (one list per returned action).
    # action may be a torch.Tensor or np.ndarray after postprocessing.
    if not isinstance(action, np.ndarray):
        action = np.array(action)
    return [action.squeeze(0).tolist()]


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
        logger.exception("Failed to load SmolVLA model: %s", _load_error)
    yield


app = FastAPI(title="SmolVLA Policy Server", lifespan=lifespan)






@app.get("/health")
def health():
    if _load_error:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "model_id": _cli_model_id, "error": _load_error},
        )
    return {"ready": _ready, "model_id": _model_id}


@app.get("/info")
def info():
    extra = {}
    if _load_error:
        extra["error"] = _load_error
    return {
        "name": "SmolVLA",
        "model_id": _model_id or _cli_model_id,
        "action_space": {
            "type": "eef_delta",
            "dim": _action_dim,
            "description": "End-effector delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]",
        },
        "state_dim": _state_dim,
        "action_chunk_size": _action_chunk_size,
        "obs_requirements": {
            "cameras": ["primary", "wrist"],
            "state_dim": _state_dim,
            "image_resolution": [256, 256],
            "image_transform": _image_transform,
        },
        **extra,
    }


@app.post("/reset")
def reset_policy():
    """Reset the policy's internal state (action queue) between episodes.

    Native lerobot-eval calls policy.reset() at the start of every episode.
    With n_action_steps=1 the queue is always empty between select_action calls,
    making this a no-op for smolvla_libero; it is exposed here for completeness
    and to future-proof against models with n_action_steps > 1.
    """
    if not _ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})
    _policy.reset()
    return {"success": True}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not _ready:
            err = _load_error or "Model not ready yet"
            return JSONResponse(status_code=503, content={"error": err})
        actions = _predict(req.obs.images.get("primary"), req.obs.instruction, req.obs.state.get("flat"), req.obs.images.get("wrist"))
        return {
            "actions": actions,
            "chunk_size": len(actions),
            "model_id": _model_id,
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

    parser = argparse.ArgumentParser(description="SmolVLA Policy Server")
    parser.add_argument(
        "--model-id",
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5102,
        help="Port to serve on (default: 5102)",
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
    print(f"[smolvla_policy] Starting SmolVLA server on {args.host}:{args.port}")
    print(f"[smolvla_policy] Model: {args.model_id}, Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
