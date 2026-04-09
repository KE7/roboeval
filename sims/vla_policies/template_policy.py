#!/usr/bin/env python
"""
Template VLA policy server for robo-eval.

Copy this file and fill in the TODOs to integrate a new VLA model.
See docs/adding_a_vla.md for the full walkthrough.

Usage:
    python -m sims.vla_policies.template_policy --port 5103

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

# TODO: Set your default HuggingFace model ID
MODEL_ID_DEFAULT = "your-org/your-model"

# ======================================================================
# Module-level model state (populated at startup in _load_model)
# ======================================================================

_policy = None          # Your model object
_model_id: str = ""     # Actual model ID after loading
_device: str = "cuda"
_ready: bool = False
_load_error: Optional[str] = None

# TODO: Adjust these to match your model's specs
_action_dim: int = 7         # Number of action dimensions (7 for LIBERO EEF delta)
_state_dim: int = 8          # Proprioceptive state dim (8 for LIBERO, 0 if unused)
_action_chunk_size: int = 1  # Actions per /predict call (1 = replan every step)

# CLI args captured in main() before server start
_cli_model_id: str = MODEL_ID_DEFAULT
_cli_device: str = "cuda"


# ======================================================================
# Model loading — called once at startup
# ======================================================================


def _load_model(model_id: str, device: str) -> None:
    """Load model weights, tokenizer, and preprocessor.

    This runs inside the lifespan context manager before the server
    starts accepting requests. Set _ready = True when loading succeeds.

    Args:
        model_id: HuggingFace model ID or local path.
        device: Torch device string ("cuda", "cpu", etc.).
    """
    global _policy, _model_id, _device, _ready
    global _action_dim, _state_dim, _action_chunk_size

    _model_id = model_id
    _device = device

    # TODO: Replace with your model loading code. Example:
    #
    # import torch
    # from transformers import AutoModelForVision2Seq, AutoProcessor
    #
    # _policy = AutoModelForVision2Seq.from_pretrained(
    #     model_id, torch_dtype=torch.bfloat16
    # ).to(device)
    # _policy.eval()
    #
    # # Read action/state dims from model config if available:
    # _action_dim = 7
    # _state_dim = 8
    # _action_chunk_size = 1

    raise NotImplementedError(
        "TODO: Implement _load_model() — load your model weights here. "
        "See sims/vla_policies/smolvla_policy.py for a working example."
    )

    _ready = True
    logger.info(
        "Model ready: %s on %s (action_dim=%d, state_dim=%d, chunk=%d)",
        model_id, device, _action_dim, _state_dim, _action_chunk_size,
    )


# ======================================================================
# Inference — called on every /predict request
# ======================================================================


def _predict(
    image_b64: str,
    instruction: str,
    state: Optional[list],
    image2_b64: Optional[str] = None,
) -> list[list[float]]:
    """Run one inference step and return action vector(s).

    Args:
        image_b64: Base64-encoded PNG of the primary camera (agentview, 256x256 RGB).
                   Already flipped 180 degrees by sim_worker to match lerobot convention.
        instruction: Natural language task instruction (e.g., "pick up the red bowl").
        state: Proprioceptive state as list[float]. For LIBERO:
               [eef_x, eef_y, eef_z, axisangle_x, axisangle_y, axisangle_z, grip_l, grip_r]
               Will be None or empty if the model doesn't use state (state_dim=0).
        image2_b64: Optional base64-encoded PNG of the wrist camera. None if unavailable.

    Returns:
        List of action vectors, length = _action_chunk_size.
        Each action is a list[float] of length _action_dim.
        For LIBERO EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        Gripper: -1 = close, +1 = open (LIBERO convention).
        Actions must be UNNORMALIZED (real-space deltas).
    """
    from PIL import Image

    # Decode base64 PNG -> PIL Image (RGB)
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")

    # Decode optional second camera
    pil_img2 = None
    if image2_b64:
        raw2 = base64.b64decode(image2_b64)
        pil_img2 = Image.open(BytesIO(raw2)).convert("RGB")

    # Parse state (default to zeros if not provided)
    state_vec = state if state else [0.0] * _state_dim

    # TODO: Replace with your model's inference code. Example:
    #
    # import torch
    # inputs = preprocess(pil_img, instruction, state_vec)
    # inputs = {k: v.to(_device) for k, v in inputs.items()}
    # with torch.no_grad():
    #     raw_actions = _policy.predict(inputs)
    # actions = unnormalize(raw_actions)  # if needed
    # return [actions.cpu().numpy().tolist()]

    raise NotImplementedError(
        "TODO: Implement _predict() — run your model's forward pass here. "
        "See sims/vla_policies/smolvla_policy.py for a working example."
    )


# ======================================================================
# FastAPI application
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup; errors are captured, not raised."""
    global _load_error
    try:
        _load_model(_cli_model_id, _cli_device)
    except NotImplementedError:
        _load_error = "Model loading not implemented yet (see TODOs in template)"
        logger.error(_load_error)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        logger.exception("Failed to load model: %s", _load_error)
    yield


app = FastAPI(title="Template VLA Policy Server", lifespan=lifespan)


# -- Request schema --





# -- Endpoints --

@app.get("/health")
def health():
    """Health check. Orchestrator polls this until ready=true."""
    if _load_error:
        return JSONResponse(
            status_code=503,
            content={"ready": False, "model_id": _cli_model_id, "error": _load_error},
        )
    return {"ready": _ready, "model_id": _model_id}


@app.get("/info")
def info():
    """Model metadata. Called once at startup by env_wrapper to configure action translation."""
    extra = {}
    if _load_error:
        extra["error"] = _load_error
    return {
        "name": "template",        # TODO: Change to your model's short name
        "model_id": _model_id or _cli_model_id,
        "action_space": {
            "type": "eef_delta",    # TODO: Change if your model uses "joint_pos"
            "dim": _action_dim,
        },
        "state_dim": _state_dim,
        "action_chunk_size": _action_chunk_size,
        **extra,
    }


@app.post("/reset")
def reset_policy():
    """Reset internal state between episodes (e.g., action queue for chunked models).

    Called at the start of every episode. For models with action_chunk_size=1,
    this is typically a no-op. For chunked models (like Pi0.5 with chunk_size=50),
    this clears the action queue so stale actions from the previous episode
    aren't executed.
    """
    if not _ready:
        return JSONResponse(status_code=503, content={"error": "Model not ready"})
    # TODO: If your model has internal state to reset, do it here.
    # Example: _policy.reset()
    return {"success": True}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run inference on the current observation.

    Returns a list of action vectors. The orchestrator executes them
    sequentially, then calls /predict again.
    """
    try:
        if not _ready:
            err = _load_error or "Model not ready yet"
            return JSONResponse(status_code=503, content={"error": err})
        actions = _predict(
            req.obs.images.get("primary"),
            req.obs.instruction,
            req.obs.state.get("flat"),
            req.obs.images.get("wrist"),
        )
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

    parser = argparse.ArgumentParser(description="Template VLA Policy Server")
    parser.add_argument(
        "--model-id",
        default=MODEL_ID_DEFAULT,
        help=f"HuggingFace model ID (default: {MODEL_ID_DEFAULT})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5103,  # TODO: Pick your default port
        help="Port to serve on (default: 5103)",
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
    print(f"[template_policy] Starting server on {args.host}:{args.port}")
    print(f"[template_policy] Model: {args.model_id}, Device: {args.device}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
