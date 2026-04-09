#!/usr/bin/env python
# ==============================================================================
# NOTE: OpenVLA is NOT a lerobot model -- uses its own native AutoProcessor
# ==============================================================================
# OpenVLA (openvla/openvla-7b and finetuned variants) uses HuggingFace's
# AutoProcessor + AutoModelForVision2Seq from the transformers library.
# It does NOT use lerobot's make_pre_post_processors (which is for lerobot models).
#
# All image preprocessing is delegated to the model's own AutoProcessor.
# Unnormalization is handled internally by model.predict_action(unnorm_key=...).
# The only manual postprocessing is gripper convention inversion (RLDS->LIBERO).
#
# For lerobot-based models (pi05, smolvla), see the VLA Integration Contract in
# docs/vla_policy_architecture.md -- those MUST use make_pre_post_processors.
# ==============================================================================
"""
OpenVLA policy server for robo-eval.

Standalone FastAPI server that loads openvla/openvla-7b (or a finetuned variant)
at startup in a background thread and serves single-step action predictions.
Runs in .venvs/vla/.

Usage:
    python -m sims.vla_policies.openvla_policy --port 5101

Environment variables:
    OPENVLA_MODEL_ID  - Override model ID (default: openvla/openvla-7b-finetuned-libero-spatial)

Endpoints:
    GET  /health  -> {ready, model_id[, error]}
    GET  /info    -> {name, model_id, action_space, state_dim, action_chunk_size}
    POST /predict {obs: {image, instruction, state?, image2?}} -> {actions, chunk_size, model_id}
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
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

# OpenVLA outputs 7-dim EEF delta actions (OSC_POSE, use_delta=True)
NATIVE_ACTION_DIM = 7

# Module-level model state (populated in background thread at startup)
_model = None
_processor = None
_model_id: str = ""
_device = None
_unnorm_key: str = "libero_spatial"
_ready: bool = False
_load_error: str = ""

# CLI args captured in main() before server start
_cli_model_id: str = os.environ.get("OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial")
_cli_device: str = "cuda"
_cli_unnorm_key: str = "libero_spatial"


# ======================================================================
# Model loading
# ======================================================================


def _load_model(model_id: str, device: str, unnorm_key: str) -> None:
    global _model, _processor, _model_id, _device, _unnorm_key, _ready, _load_error

    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _model_id = model_id
    _unnorm_key = unnorm_key

    logger.info("Loading OpenVLA processor from %s (local_files_only=True) ...", model_id)
    try:
        _processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:
        _load_error = "model not cached - not downloading"
        logger.error(
            "OpenVLA processor %s not in HF cache. Not downloading. Error: %s",
            model_id,
            e,
        )
        return

    logger.info(
        "Loading OpenVLA model from %s on %s (local_files_only=True) ...", model_id, device
    )
    try:
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",  # avoid _supports_sdpa compat issue w/ transformers>=4.50
        )
    except Exception as e:
        _load_error = "model not cached - not downloading"
        logger.error(
            "OpenVLA model %s not in HF cache. Not downloading. Error: %s",
            model_id,
            e,
        )
        _processor = None
        return

    _device = next(_model.parameters()).device
    _model.eval()
    logger.info(
        "OpenVLA ready on %s (unnorm_key=%s)", _device, _unnorm_key
    )
    _ready = True


def _load_model_bg(model_id: str, device: str, unnorm_key: str) -> None:
    """Background thread wrapper for model loading."""
    try:
        _load_model(model_id, device, unnorm_key)
    except Exception as e:
        global _load_error
        _load_error = str(e)
        logger.error("Model load failed unexpectedly: %s\n%s", e, traceback.format_exc())


# ======================================================================
# Inference helpers
# ======================================================================


def _predict(image_b64: str, instruction: str) -> list[list[float]]:
    """Predict a single action from image and instruction.

    State is ignored (OpenVLA is image+text only).
    Returns a list with one action vector of length NATIVE_ACTION_DIM (7).
    """
    import torch
    from PIL import Image

    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")

    prompt = f"In: What action should the robot take to {instruction}?\nOut:"

    inputs = _processor(prompt, pil_img).to(_device, dtype=torch.bfloat16)

    with torch.no_grad():
        action = _model.predict_action(
            **inputs,
            unnorm_key=_unnorm_key,
            do_sample=False,
        )

    action = np.array(action, dtype=np.float64).flatten()[:NATIVE_ACTION_DIM]

    # Binarize and invert gripper: OpenVLA was trained with RLDS convention
    # where 1=close, -1=open. LIBERO (and other sims) use the opposite:
    # -1=close, 1=open. Binarize to remove ambiguity, then invert.
    gripper = 1.0 if action[-1] > 0.0 else -1.0
    action[-1] = -gripper

    return [action.tolist()]


# ======================================================================
# FastAPI application
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(
        target=_load_model_bg,
        args=(_cli_model_id, _cli_device, _cli_unnorm_key),
        daemon=True,
    )
    t.start()
    yield


app = FastAPI(title="OpenVLA Policy Server", lifespan=lifespan)






@app.get("/health")
def health():
    result: dict = {"ready": _ready, "model_id": _model_id}
    if not _ready and _load_error:
        result["error"] = _load_error
    return result


@app.get("/info")
def info():
    name = _model_id.split("/")[-1] if "/" in _model_id else _model_id
    return {
        "name": name or "openvla",
        "model_id": _model_id,
        "action_space": {
            "type": "eef_delta",
            "dim": NATIVE_ACTION_DIM,
            "description": "EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]",
        },
        "state_dim": 0,         # OpenVLA ignores robot state
        "action_chunk_size": 1, # single-step, not chunk-based
        "obs_requirements": {
            "cameras": ["primary"],
            "state_dim": 0,
            "image_resolution": [256, 256],
            # OpenVLA uses its own AutoProcessor (not lerobot processors), so
            # auto-detection is not possible.  Hardcoded to "flip_hw" because the
            # LIBERO-finetuned variants were trained on 180°-flipped camera data
            # (matching the HuggingFaceVLA/libero convention).
            "image_transform": "flip_hw",
        },
    }


class ReloadRequest(BaseModel):
    model_id: str
    unnorm_key: str = "libero_spatial"


@app.post("/reload")
def reload_model(req: ReloadRequest):
    """Hot-swap the model checkpoint and unnorm_key.

    Used when evaluating multiple LIBERO suites that need different
    fine-tuned checkpoints (e.g., libero_spatial -> libero_object).
    OpenVLA has separate LoRA-finetuned checkpoints per suite.

    If the requested model is already loaded, returns immediately.
    Otherwise, loads the new model synchronously (blocks until ready).
    This endpoint may take several minutes to return while the new
    model loads.
    """
    import torch

    global _ready

    if _model_id == req.model_id and _unnorm_key == req.unnorm_key:
        logger.info("Model %s already loaded with unnorm_key=%s, skipping reload",
                     req.model_id, req.unnorm_key)
        return {"success": True, "reloaded": False, "model_id": _model_id}

    logger.info("Reloading model: %s -> %s (unnorm_key: %s -> %s)",
                _model_id, req.model_id, _unnorm_key, req.unnorm_key)

    _ready = False  # Mark as unavailable during reload

    # Free old model memory before loading new one
    global _model, _processor
    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None
    torch.cuda.empty_cache()

    # Synchronous reload (blocks until the new model is ready)
    _load_model(req.model_id, str(_device or "cuda"), req.unnorm_key)

    if _ready:
        logger.info("Reload complete: %s (unnorm_key=%s)", _model_id, _unnorm_key)
        return {"success": True, "reloaded": True, "model_id": _model_id}
    else:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to reload model: {_load_error}"},
        )


@app.post("/reset")
def reset_policy():
    """Reset the policy's internal state between episodes.

    OpenVLA is stateless (single-step predictions), so this is a no-op.
    Exposed for API consistency with other policy servers.
    """
    if not _ready:
        err = _load_error or "Model not ready yet"
        return JSONResponse(status_code=503, content={"error": err})
    return {"success": True}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not _ready:
            err = _load_error or "Model not ready yet"
            return JSONResponse(status_code=503, content={"error": err})
        actions = _predict(req.obs.images.get("primary"), req.obs.instruction)
        return {
            "actions": actions,
            "chunk_size": 1,
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

    parser = argparse.ArgumentParser(description="OpenVLA Policy Server")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"),
        help="HuggingFace model ID (default: openvla/openvla-7b-finetuned-libero-spatial or $OPENVLA_MODEL_ID)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5101,
        help="Port to serve on (default: 5101)",
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
    parser.add_argument(
        "--unnorm-key",
        default="libero_spatial",
        dest="unnorm_key",
        help=(
            "Dataset key for action unnormalization stats "
            "(default: libero_spatial). Common values: "
            "bridge_orig, libero_spatial, libero_10."
        ),
    )
    args = parser.parse_args()

    global _cli_model_id, _cli_device, _cli_unnorm_key
    _cli_model_id = args.model_id
    _cli_device = args.device
    _cli_unnorm_key = args.unnorm_key

    logging.basicConfig(level=logging.INFO)
    print(f"[openvla_policy] Starting OpenVLA server on {args.host}:{args.port}")
    print(f"[openvla_policy] Model: {args.model_id}, unnorm_key: {args.unnorm_key}")
    print(f"[openvla_policy] NOTE: model loads in background; poll GET /health for ready:true")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
