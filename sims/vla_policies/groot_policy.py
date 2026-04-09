#!/usr/bin/env python
"""
GR00T Policy Server for robo-eval (runs inside Docker container or standalone).

This is a standalone HTTP server that loads the official Gr00tPolicy and
Gr00tSimPolicyWrapper at startup and serves predictions via FastAPI.
The observation / action semantics match the real GR00T simulation path.

Usage (inside Docker):
    python -m sims.vla_policies.groot_policy --port 5105

Usage (host):
    python -m sims.vla_policies.groot_policy --port 8000

Environment variables:
    PORT              - Server port (default: 8000)
    VLA_PORT          - Alternative port setting (default: not used if PORT is set)
    GROOT_MODEL_ID         - Override model ID (default: nvidia/GR00T-N1.6-3B)
    GROOT_EMBODIMENT_TAG   - Override embodiment (default: robocasa_panda_omron)
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

_policy = None
_modality_config = None
_model_id: str = ""
_embodiment_tag: str = ""
_action_chunk_size: int = 0
_action_keys: list[str] = []
_video_keys: list[str] = []
_state_keys: list[str] = []
_language_key: str = ""
_action_dim: int = 0  # Total action dimensionality (calculated after model load)
_gripper_col_idx: int = -1  # Column index for gripper in the flat action (calculated after model load)
_ready: bool = False
_load_error: str = ""

_cli_model_id: str = os.environ.get("GROOT_MODEL_ID", "nvidia/GR00T-N1.6-3B")
_cli_embodiment_tag: str = os.environ.get("GROOT_EMBODIMENT_TAG", "robocasa_panda_omron")
_cli_device: str = "cuda"


def _decode_image(image_b64: str) -> np.ndarray:
    from PIL import Image

    raw = base64.b64decode(image_b64)
    return np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.uint8)


def _load_model(model_id: str, embodiment_tag: str, device: str) -> None:
    global _policy, _modality_config, _model_id, _embodiment_tag
    global _action_chunk_size, _action_keys, _video_keys, _state_keys, _language_key
    global _action_dim, _gripper_col_idx, _ready, _load_error

    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

        emb_tag = EmbodimentTag(embodiment_tag)
        logger.info(
            "Loading GR00T policy via upstream wrapper: model=%s, embodiment=%s, device=%s",
            model_id, embodiment_tag, device,
        )
        base_policy = Gr00tPolicy(
            embodiment_tag=emb_tag,
            model_path=model_id,
            device=device,
            strict=True,
        )
        _policy = Gr00tSimPolicyWrapper(base_policy, strict=True)
        _modality_config = _policy.get_modality_config()
        _model_id = model_id
        _embodiment_tag = embodiment_tag
        _video_keys = list(_modality_config["video"].modality_keys)
        _state_keys = list(_modality_config["state"].modality_keys)
        _action_keys = list(_modality_config["action"].modality_keys)
        _language_key = _modality_config["language"].modality_keys[0]
        _action_chunk_size = len(_modality_config["action"].delta_indices)

        # Calculate actual action dimensionality using the policy's native method.
        # _policy is Gr00tSimPolicyWrapper; the base Gr00tPolicy (with .processor) is _policy.policy.
        delta_indices = _modality_config["action"].delta_indices
        _action_dim = _policy.policy.processor.state_action_processor.get_action_dim(_embodiment_tag)

        # Compute gripper column index dynamically from norm_params
        _gripper_col_idx = -1
        norm_params = _policy.policy.processor.state_action_processor.norm_params[_embodiment_tag]["action"]
        col = 0
        for key in _action_keys:
            if "gripper" in key.lower():
                _gripper_col_idx = col
                break
            col += int(norm_params[key]["dim"].item())

        _ready = True
        logger.info(
            "GR00T ready: video_keys=%s, state_keys=%s, action_keys=%s, chunk=%d, action_dim=%d",
            _video_keys, _state_keys, _action_keys, _action_chunk_size, _action_dim,
        )
    except Exception as e:
        _load_error = str(e)
        logger.error("GR00T load failed: %s\n%s", e, traceback.format_exc())


def _load_model_bg(model_id: str, embodiment_tag: str, device: str) -> None:
    try:
        _load_model(model_id, embodiment_tag, device)
    except Exception as e:
        global _load_error
        _load_error = str(e)
        logger.error("Unexpected error during GR00T load: %s\n%s", e, traceback.format_exc())


def _require_image(images: dict[str, np.ndarray], source: str, key: str) -> np.ndarray:
    if source not in images:
        raise ValueError(f"GR00T requires image source '{source}' for video.{key}")
    return images[source]


def _build_flat_observation(
    image_b64: str,
    instruction: str,
    state_dict: Optional[dict],
    image2_b64: Optional[str],
    image3_b64: Optional[str],
) -> dict:
    primary = _decode_image(image_b64)
    wrist = _decode_image(image2_b64) if image2_b64 else None
    secondary = _decode_image(image3_b64) if image3_b64 else None

    image_sources = {
        "primary": primary,
        "secondary": secondary,
        "wrist": wrist,
    }

    flat_obs: dict = {}
    for key in _video_keys:
        if "side_0" in key or key in ("image", "image_0", "res256_image_0"):
            img = _require_image(image_sources, "primary", key)
        elif "side_1" in key or "secondary" in key:
            img = _require_image(image_sources, "secondary", key)
        elif "wrist" in key:
            img = _require_image(image_sources, "wrist", key)
        else:
            raise ValueError(f"Unsupported GR00T video key '{key}'")
        flat_obs[f"video.{key}"] = img[None, None]

    if not state_dict:
        raise ValueError(
            "GR00T requires structured proprioceptive state_dict; robo-eval did not provide it."
        )

    for key in _state_keys:
        if key not in state_dict:
            raise ValueError(f"Missing GR00T state key '{key}' in state_dict")
        value = np.asarray(state_dict[key], dtype=np.float32).reshape(1, 1, -1)
        flat_obs[f"state.{key}"] = value

    flat_obs[_language_key] = [instruction]
    return flat_obs


def _flatten_action_chunk(action_dict: dict[str, np.ndarray]) -> list[list[float]]:
    parts = []
    for key in _action_keys:
        action_key = f"action.{key}"
        if action_key not in action_dict:
            raise ValueError(f"Missing GR00T action key '{action_key}' in policy output")
        arr = np.asarray(action_dict[action_key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape for {action_key}: {arr.shape}")
        parts.append(arr[0])
    flat_action = np.concatenate(parts, axis=-1)

    # Apply gripper binarization and convention conversion across all timesteps.
    # GR00T outputs RLDS convention: positive=close, negative=open
    # RoboCasa (robosuite PandaGripper) expects: negative=close, positive=open
    # Therefore we NEGATE the gripper signal.
    # Binarization ensures discrete gripper commands (no continuous values near 0).
    if _gripper_col_idx >= 0:
        flat_action[:, _gripper_col_idx] = -(np.where(flat_action[:, _gripper_col_idx] > 0.0, 1.0, -1.0))

    return flat_action.tolist()


def _predict(
    image_b64: str,
    instruction: str,
    state_dict: Optional[dict],
    image2_b64: Optional[str],
    image3_b64: Optional[str],
) -> list[list[float]]:
    flat_obs = _build_flat_observation(
        image_b64=image_b64,
        instruction=instruction,
        state_dict=state_dict,
        image2_b64=image2_b64,
        image3_b64=image3_b64,
    )
    action_dict, _info = _policy.get_action(flat_obs)
    return _flatten_action_chunk(action_dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(
        target=_load_model_bg,
        args=(_cli_model_id, _cli_embodiment_tag, _cli_device),
        daemon=True,
    )
    t.start()
    yield


app = FastAPI(title="GR00T Policy Server", lifespan=lifespan)






@app.get("/health")
def health():
    result: dict = {"ready": _ready, "model_id": _model_id}
    if not _ready and _load_error:
        result["error"] = _load_error
    return result


@app.get("/info")
def info():
    # Determine action space description based on action keys
    action_desc = "Native GR00T action"
    if _action_keys:
        action_desc = f"GR00T {_embodiment_tag}: {' + '.join(_action_keys)}"

    return {
        "name": (_model_id.split("/")[-1] if "/" in _model_id else _model_id) or "groot",
        "model_id": _model_id,
        "embodiment_tag": _embodiment_tag,
        "action_space": {
            "type": "eef_delta",
            "dim": _action_dim,  # Calculated dynamically from modality config
            "description": action_desc,
        },
        "state_dim": len(_state_keys) if _state_keys else 0,
        "action_chunk_size": _action_chunk_size,
        "obs_requirements": {
            "cameras": ["primary", "secondary", "wrist"],
            "state_dim": len(_state_keys) if _state_keys else 0,
            "state_format": "structured",
            "image_transform": "none",
        },
        "modality_keys": {
            "video": _video_keys,
            "state": _state_keys,
            "action": _action_keys,
            "language": _language_key,
        },
    }


@app.post("/reset")
def reset_policy():
    if not _ready:
        err = _load_error or "Model not ready yet"
        return JSONResponse(status_code=503, content={"error": err})
    try:
        return {"success": True, **(_policy.reset() or {})}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        if not _ready:
            err = _load_error or "Model not ready yet"
            return JSONResponse(status_code=503, content={"error": err})

        actions = _predict(
            req.obs.images.get("primary"),
            req.obs.instruction,
            req.obs.state.get("structured"),
            req.obs.images.get("wrist"),
            req.obs.images.get("secondary"),
        )
        return {
            "actions": actions,
            "chunk_size": len(actions),
            "model_id": _model_id,
        }
    except Exception as e:
        logger.exception("GR00T prediction failed")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="GR00T Policy Server")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GROOT_MODEL_ID", "nvidia/GR00T-N1.6-3B"),
    )
    parser.add_argument(
        "--embodiment-tag",
        default=os.environ.get("GROOT_EMBODIMENT_TAG", "robocasa_panda_omron"),
        dest="embodiment_tag",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000))),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Validate port
    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1024 and 65535, got {args.port}")

    global _cli_model_id, _cli_embodiment_tag, _cli_device
    _cli_model_id = args.model_id
    _cli_embodiment_tag = args.embodiment_tag
    _cli_device = args.device

    logging.basicConfig(level=logging.INFO)
    print(f"[groot_policy] Starting GR00T server on {args.host}:{args.port}")
    print(f"[groot_policy] Model: {args.model_id}, embodiment: {args.embodiment_tag}")
    print("[groot_policy] Using upstream Gr00tSimPolicyWrapper bridge")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
