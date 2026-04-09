#!/usr/bin/env python
"""
Cosmos-Policy FastAPI Server (inside Docker or standalone).

This server loads the official NVIDIA Cosmos-Policy model and exposes
a simple HTTP API for inference. All model code, preprocessing, and
inference logic uses NVIDIA's official APIs.

Usage (inside container):
    python -m sims.vla_policies.cosmos_policy --port 5103

Usage (standalone):
    python -m sims.vla_policies.cosmos_policy --port 8000

Endpoints:
    GET  /health  -> {ready, model_id[, error]}
    GET  /info    -> {name, model_id, action_space, state_dim, action_chunk_size}
    POST /reset   -> {success}
    POST /predict {obs: {image, image2?, image3?, instruction, state}} -> {actions, chunk_size, model_id}
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
from types import SimpleNamespace
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sims.vla_policies.vla_schema import VLAObservation, PredictRequest

logger = logging.getLogger(__name__)

# ======================================================================
# Constants
# ======================================================================

MODEL_ID = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
ACTION_DIM = 7       # EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]
PROPRIO_DIM = 9      # gripper_qpos(2) + eef_pos(3) + eef_quat(4)
CHUNK_SIZE = 32      # actions per generation (model's native chunk)
EXEC_HORIZON = 16    # actions we actually execute (open-loop horizon)

# Default paths inside the Docker container (HF cache mount)
_HF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B"
    "/snapshots/4b2a04c80d97202f86127ebec80461e8016ec1dc"
)
_DEFAULTS = dict(
    ckpt=os.environ.get(
        "COSMOS_CKPT_PATH",
        f"{_HF_SNAP}/Cosmos-Policy-RoboCasa-Predict2-2B.pt",
    ),
    config=os.environ.get("COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py"),
    stats=os.environ.get(
        "COSMOS_DATASET_STATS_PATH",
        f"{_HF_SNAP}/robocasa_dataset_statistics.json",
    ),
    t5=os.environ.get(
        "COSMOS_T5_EMBEDDINGS_PATH",
        f"{_HF_SNAP}/robocasa_t5_embeddings.pkl",
    ),
)

# ======================================================================
# Module-level model state (populated at startup in _load)
# ======================================================================

_model = None
_dataset_stats = None
_ready: bool = False
_load_error: str = ""

# Configuration namespace matching NVIDIA's PolicyEvalConfig for RoboCasa.
# get_action() reads these fields to control image preprocessing, normalization,
# and the observation → latent-sequence layout.
_eval_cfg = SimpleNamespace(
    suite="robocasa",
    use_third_person_image=True,
    num_third_person_images=2,    # primary (left) + secondary (right)
    use_wrist_image=True,
    num_wrist_images=1,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=CHUNK_SIZE,
    num_open_loop_steps=EXEC_HORIZON,
    flip_images=True,
    use_jpeg_compression=True,
    trained_with_image_aug=True,
    use_variance_scale=False,
)


# ======================================================================
# Model loading — called once at startup
# ======================================================================


def _load(ckpt: str, config_file: str, stats_path: str, t5_path: str) -> bool:
    """Load the Cosmos-Policy model, dataset stats, and T5 embeddings.

    Runs in a background thread so the FastAPI server is immediately reachable
    (returns 503 on /predict until loading completes).  All required files must
    be pre-cached inside the container — HF_HUB_OFFLINE=1 prevents downloads.

    Returns:
        True if load succeeded, False otherwise.
    """
    global _model, _dataset_stats, _ready, _load_error

    # Block network calls — all weights are pre-cached in the mounted HF volume.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    try:
        # Set constants that NVIDIA's config/model code reads at import time
        import cosmos_policy.constants as cc

        cc.NUM_ACTIONS_CHUNK = CHUNK_SIZE
        cc.ACTION_DIM = ACTION_DIM
        cc.PROPRIO_DIM = PROPRIO_DIM
        cc.ROBOT_PLATFORM = "ROBOCASA"

        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
        from cosmos_policy.experiments.robot.cosmos_utils import (
            DEVICE,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        # Load model checkpoint (same as NVIDIA's get_model(), but we skip HF
        # download logic since weights are already cached in the container)
        logger.info("Loading Cosmos-Policy checkpoint: %s", ckpt)
        model, _ = load_model_from_checkpoint(
            "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
            ckpt,
            config_file,
            load_ema_to_reg=False,
        )
        model.eval()
        _model = model.to(DEVICE)

        # Load dataset statistics (for action unnormalization and proprio rescaling)
        _dataset_stats = load_dataset_stats(stats_path)

        # Pre-load T5 text embeddings cache (avoids cold-start T5 forward pass)
        init_t5_text_embeddings_cache(t5_path)

        _ready = True
        logger.info("Cosmos-Policy ready on %s", DEVICE)
        return True

    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        logger.error("Load failed: %s\n%s", e, traceback.format_exc())
        return False


# ======================================================================
# Inference — called on every /predict request
# ======================================================================


def _predict(
    image_b64: str,
    instruction: str,
    state: list[float],
    secondary_b64: Optional[str] = None,
    wrist_b64: Optional[str] = None,
) -> list[list[float]]:
    """Run one inference step and return a chunk of action vectors.

    Decodes images, applies the RoboCasa vertical flip (np.flipud), then
    delegates entirely to NVIDIA's get_action() which handles all remaining
    preprocessing (JPEG compression, resize, center-crop), T5 embedding
    lookup, diffusion generation, action extraction, and unnormalization.

    Camera mapping (sim_worker → policy server → NVIDIA obs dict):
        image  (req.obs.images.get("primary"))  → primary_image  (robot0_agentview_left)
        image2 (req.obs.images.get("wrist")) → wrist_image    (robot0_eye_in_hand)
        image3 (req.obs.images.get("secondary")) → secondary_image (robot0_agentview_right)

    Args:
        image_b64: Base64-encoded PNG, primary camera (left agentview).
        instruction: Natural language task description.
        state: Proprioceptive state: gripper_qpos(2) + eef_pos(3) + eef_quat(4).
        secondary_b64: Optional base64 PNG, secondary camera (right agentview).
        wrist_b64: Optional base64 PNG, wrist camera (eye-in-hand).

    Returns:
        List of EXEC_HORIZON action vectors, each of length ACTION_DIM.
    """
    import torch
    from PIL import Image
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    def _decode(b64: str) -> np.ndarray:
        """Decode base64 PNG → (H, W, 3) uint8 numpy array."""
        return np.array(
            Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"),
            dtype=np.uint8,
        )

    # Decode and flip images vertically — matches NVIDIA's prepare_observation()
    # in run_robocasa_eval.py which applies np.flipud() per camera for RoboCasa.
    primary = np.flipud(_decode(image_b64))

    if wrist_b64:
        wrist = np.flipud(_decode(wrist_b64))
    else:
        wrist = np.zeros_like(primary)

    if secondary_b64:
        secondary = np.flipud(_decode(secondary_b64))
    else:
        # Cosmos uses primary as fallback when secondary is unavailable
        secondary = primary.copy()

    # Build observation dict matching NVIDIA's prepare_observation() output
    obs = {
        "primary_image": primary,
        "secondary_image": secondary,
        "wrist_image": wrist,
        "proprio": np.array(state, dtype=np.float64),
    }

    # One-line inference: NVIDIA's get_action() does everything
    with torch.no_grad():
        result = get_action(
            cfg=_eval_cfg,
            model=_model,
            dataset_stats=_dataset_stats,
            obs=obs,
            task_label_or_embedding=instruction,
            seed=1,
            randomize_seed=False,
            num_denoising_steps_action=5,
            generate_future_state_and_value_in_parallel=False,
            worker_id=0,
            batch_size=1,
        )

    # Extract actions and free large GPU tensors (generated_latent, data_batch, etc.)
    raw_actions = result["actions"]
    del result
    torch.cuda.empty_cache()

    # Convert to list-of-lists, pad to EXEC_HORIZON if needed
    actions = [
        a.tolist() if isinstance(a, np.ndarray) else list(a)
        for a in raw_actions[:EXEC_HORIZON]
    ]
    while len(actions) < EXEC_HORIZON:
        actions.append([0.0] * ACTION_DIM)

    return actions


# ======================================================================
# FastAPI application
# ======================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model in background thread; errors are captured, not raised."""
    t = threading.Thread(
        target=_load,
        args=(
            _DEFAULTS["ckpt"],
            _DEFAULTS["config"],
            _DEFAULTS["stats"],
            _DEFAULTS["t5"],
        ),
        daemon=True,
    )
    t.start()
    yield


app = FastAPI(title="Cosmos-Policy Server", lifespan=lifespan)


# -- Request schema --






# -- Endpoints --


@app.get("/health")
def health():
    """Health check.  Orchestrator polls this until ready=true."""
    result: dict = {"ready": _ready, "model_id": MODEL_ID}
    if not _ready and _load_error:
        result["error"] = _load_error
    return result


@app.get("/info")
def info():
    """Model metadata.  Called once by the orchestrator to configure action handling."""
    return {
        "name": "Cosmos-Policy-RoboCasa-Predict2-2B",
        "model_id": MODEL_ID,
        "action_space": {
            "type": "eef_delta",
            "dim": ACTION_DIM,
            "description": "EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]",
        },
        "state_dim": PROPRIO_DIM,
        "action_chunk_size": EXEC_HORIZON,
        "obs_requirements": {
            "cameras": ["primary"],
            "optional_cameras": ["secondary", "wrist"],
            "state_dim": PROPRIO_DIM,
            "image_resolution": [224, 224],
            "image_transform": "none",  # flipping handled internally by _predict()
        },
    }


@app.post("/reset")
def reset_policy():
    """Reset internal state between episodes.

    Cosmos-Policy is stateless (chunk-based, no persistent action queue),
    so this is a no-op.  Exposed for API consistency with other policy servers.
    """
    if not _ready:
        return JSONResponse(
            status_code=503, content={"error": _load_error or "not ready"}
        )
    return {"success": True}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run inference on the current observation.

    Camera slot mapping (sim_worker convention):
        image  → primary  (left agentview)
        image2 → wrist    (eye-in-hand)
        image3 → secondary (right agentview, rarely sent)
    """
    try:
        if not _ready:
            return JSONResponse(
                status_code=503,
                content={"error": _load_error or "not ready"},
            )

        actions = _predict(
            req.obs.images.get("primary"),
            req.obs.instruction,
            req.obs.state.get("flat"),
            secondary_b64=req.obs.images.get("secondary"),  # image3 → secondary
            wrist_b64=req.obs.images.get("wrist"),       # image2 → wrist
        )
        return {
            "actions": actions,
            "chunk_size": len(actions),
            "model_id": MODEL_ID,
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

    parser = argparse.ArgumentParser(description="Cosmos-Policy Server (RoboCasa)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000))),
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    # Validate port
    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1024 and 65535, got {args.port}")

    logging.basicConfig(level=logging.INFO)
    print(f"[cosmos_policy] Starting Cosmos-Policy server on {args.host}:{args.port}")
    print(f"[cosmos_policy] Model: {MODEL_ID}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
