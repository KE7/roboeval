#!/usr/bin/env python
"""
OpenVLA SGLang Pipelined Inference Server.

Uses SGLang's native runtime for pipelined/batched inference of OpenVLA.
SGLang handles request pipelining, KV cache management, and continuous batching
natively — no custom batching code needed.

Two modes of operation:
  1. Embedded mode (default): Uses sgl.Runtime in-process
  2. HTTP mode: Connects to an external SGLang HTTP server

Usage:
    # Embedded mode (starts SGLang runtime in-process):
    .venvs/sglang/bin/python -m sims.vla_policies.openvla_sglang_server --port 5103

    # HTTP mode (connect to running SGLang server):
    .venvs/sglang/bin/python -m sims.vla_policies.openvla_sglang_server --port 5103 \
        --sglang-url http://localhost:30000

Endpoints (drop-in compatible with openvla_policy.py):
    GET  /health  -> {ready, model_id}
    GET  /info    -> {name, model_id, action_space, state_dim, action_chunk_size}
    POST /predict {obs: {image, instruction}} -> {actions: [[float x 7]], chunk_size, model_id}
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import tempfile
import threading
import time
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

# ===========================================================================
# Constants
# ===========================================================================
NATIVE_ACTION_DIM = 7
VOCAB_SIZE_BASE = 32000
N_ACTION_BINS = 256

# Precompute bin centers for action detokenization
_bins = np.linspace(-1, 1, N_ACTION_BINS)
_bin_centers = (_bins[:-1] + _bins[1:]) / 2.0

# Module-level state
_model_id: str = ""
_unnorm_key: str = "libero_spatial"
_ready: bool = False
_load_error: str = ""
_norm_stats: dict = {}
_runtime = None  # sgl.Runtime instance (embedded mode)
_sglang_url: str = ""  # URL of external SGLang server (HTTP mode)
_image_qa_fn = None  # sgl.function for inference

# CLI defaults
_cli_model_id: str = os.environ.get(
    "OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"
)
_cli_device: str = "cuda"
_cli_unnorm_key: str = "libero_spatial"
_cli_sglang_url: str = ""
_cli_mem_fraction: float = 0.25  # SAFETY: 25% of 128GB = ~32GB max for model+KV cache


# ===========================================================================
# Token-to-action conversion
# ===========================================================================
def _tokens_to_actions(token_ids: np.ndarray) -> np.ndarray:
    """Convert generated token IDs to unnormalized actions."""
    discretized = VOCAB_SIZE_BASE - token_ids
    discretized = np.clip(discretized - 1, a_min=0, a_max=_bin_centers.shape[0] - 1)
    normalized = _bin_centers[discretized]

    # Unnormalize
    stats = _norm_stats[_unnorm_key]["action"]
    mask = stats.get("mask", np.ones_like(stats["q01"], dtype=bool))
    q99, q01 = np.array(stats["q99"]), np.array(stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized + 1) * (q99 - q01) + q01,
        normalized,
    )
    return actions


def _postprocess_action(action: np.ndarray) -> list[float]:
    """Binarize and invert gripper (RLDS -> LIBERO convention), return as list."""
    action = action.flatten()[:NATIVE_ACTION_DIM].astype(np.float64)
    gripper = 1.0 if action[-1] > 0.0 else -1.0
    action[-1] = -gripper
    return action.tolist()


# ===========================================================================
# SGLang inference
# ===========================================================================
def _init_sglang_runtime(model_id: str) -> None:
    """Initialize the SGLang runtime for OpenVLA inference."""
    global _runtime, _ready, _load_error, _norm_stats, _model_id, _image_qa_fn

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import sglang as sgl

    _model_id = model_id

    # Load norm stats from model config
    logger.info("Loading norm stats from %s ...", model_id)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        ).to_dict()
        _norm_stats = config.get("norm_stats", {})
        if _unnorm_key not in _norm_stats:
            logger.warning(
                "unnorm_key '%s' not in norm_stats. Available: %s",
                _unnorm_key, list(_norm_stats.keys())
            )
    except Exception as e:
        _load_error = f"Failed to load norm stats: {e}"
        logger.error(_load_error)
        return

    # Define SGLang function for image-based VLA inference
    @sgl.function
    def image_qa(s, image_path, question):
        s += sgl.image(image_path) + question
        s += sgl.gen("action", max_tokens=NATIVE_ACTION_DIM, temperature=0.0)

    _image_qa_fn = image_qa

    if _cli_sglang_url:
        # HTTP mode: connect to external SGLang server
        logger.info("Connecting to external SGLang server at %s ...", _cli_sglang_url)
        try:
            sgl.set_default_backend(sgl.RuntimeEndpoint(_cli_sglang_url))
            logger.info("Connected to SGLang server at %s", _cli_sglang_url)
            _ready = True
        except Exception as e:
            _load_error = f"Failed to connect to SGLang server: {e}"
            logger.error(_load_error)
    else:
        # Embedded mode: start SGLang runtime in-process
        logger.info("Starting SGLang runtime for %s ...", model_id)
        try:
            _runtime = sgl.Runtime(
                model_path=model_id,
                tokenizer_path=model_id,
                disable_cuda_graph=True,
                disable_radix_cache=True,
                trust_remote_code=True,
                mem_fraction_static=_cli_mem_fraction,
            )
            sgl.set_default_backend(_runtime)
            logger.info("SGLang runtime initialized for %s", model_id)
            _ready = True
        except Exception as e:
            _load_error = f"SGLang runtime init failed: {e}"
            logger.error(_load_error)
            logger.error(traceback.format_exc())


def _init_sglang_bg(model_id: str) -> None:
    """Background thread wrapper."""
    try:
        _init_sglang_runtime(model_id)
    except Exception as e:
        global _load_error
        _load_error = str(e)
        logger.error("Init failed: %s\n%s", e, traceback.format_exc())


def _predict_sglang(image_b64: str, instruction: str) -> list[list[float]]:
    """Run inference via SGLang."""
    from PIL import Image

    # SGLang's image API expects a file path, so save temp file
    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        pil_img.save(f, format="PNG")
        temp_path = f.name

    try:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        state = _image_qa_fn.run(
            image_path=temp_path,
            question=prompt,
            max_new_tokens=NATIVE_ACTION_DIM,
            temperature=0.0,
        )

        # Extract output token IDs from the generation
        meta = state.get_meta_info("action")

        # Try different methods to get token IDs
        if "output_token_logprobs" in meta:
            output_logprobs = meta["output_token_logprobs"]
            token_ids = np.array([logprob[1] for logprob in output_logprobs])
        elif "completion_tokens" in meta:
            token_ids = np.array(meta["completion_tokens"])
        else:
            # Fallback: get the generated text and tokenize
            generated_text = state["action"]
            logger.warning("Could not get token IDs from meta_info, got text: %s", generated_text)
            raise RuntimeError(f"Cannot extract token IDs from SGLang output. Meta keys: {list(meta.keys())}")

        if len(token_ids) < NATIVE_ACTION_DIM:
            logger.warning("Got %d tokens, expected %d", len(token_ids), NATIVE_ACTION_DIM)
            # Pad with zeros
            padded = np.zeros(NATIVE_ACTION_DIM, dtype=token_ids.dtype)
            padded[:len(token_ids)] = token_ids
            token_ids = padded

        token_ids = token_ids[:NATIVE_ACTION_DIM]
        actions = _tokens_to_actions(token_ids)
        return [_postprocess_action(actions)]

    finally:
        os.unlink(temp_path)


# ===========================================================================
# FastAPI application
# ===========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(
        target=_init_sglang_bg,
        args=(_cli_model_id,),
        daemon=True,
    )
    t.start()
    yield
    # Cleanup
    if _runtime is not None:
        try:
            _runtime.shutdown()
        except Exception:
            pass


app = FastAPI(title="OpenVLA SGLang Server", lifespan=lifespan)






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
        "name": name or "openvla-sglang",
        "model_id": _model_id,
        "action_space": {
            "type": "eef_delta",
            "dim": NATIVE_ACTION_DIM,
            "description": "EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]",
        },
        "state_dim": 0,
        "action_chunk_size": 1,
        "backend": "sglang",
    }


@app.post("/reset")
def reset_policy():
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

        actions = _predict_sglang(req.obs.images.get("primary"), req.obs.instruction)

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


# ===========================================================================
# CLI entrypoint
# ===========================================================================
def main():
    global _cli_model_id, _cli_device, _cli_unnorm_key, _cli_sglang_url, _cli_mem_fraction

    import uvicorn

    parser = argparse.ArgumentParser(description="OpenVLA SGLang Server")
    parser.add_argument(
        "--model-id",
        default=os.environ.get(
            "OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"
        ),
    )
    parser.add_argument("--port", type=int, default=5103)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--unnorm-key", default="libero_spatial", dest="unnorm_key",
        help="Dataset key for action unnormalization",
    )
    parser.add_argument(
        "--sglang-url", default="",
        help="URL of external SGLang server (e.g., http://localhost:30000). "
             "If empty, starts embedded SGLang runtime.",
    )
    parser.add_argument(
        "--mem-fraction-static", type=float, default=0.25,
        dest="mem_fraction",
        help="Fraction of GPU memory for static allocation (model weights + KV cache). "
             "SAFETY: On 128GB GB10, 0.25 = ~32GB. Default: 0.25. "
             "NEVER set above 0.50 on shared systems.",
    )
    args = parser.parse_args()

    _cli_model_id = args.model_id
    _cli_device = args.device
    _cli_unnorm_key = args.unnorm_key
    _cli_sglang_url = args.sglang_url
    _cli_mem_fraction = args.mem_fraction

    # SAFETY CHECK: Refuse to start with dangerous memory settings
    if _cli_mem_fraction > 0.50:
        print(f"[openvla_sglang] FATAL: --mem-fraction-static={_cli_mem_fraction} exceeds safety limit of 0.50!")
        print(f"[openvla_sglang] On 128GB GB10 unified memory, this could cause OOM and crash the system.")
        print(f"[openvla_sglang] Use 0.25 (safe default) or at most 0.40.")
        import sys
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    mode = f"HTTP mode ({args.sglang_url})" if args.sglang_url else "Embedded mode"
    print(f"[openvla_sglang] Starting SGLang server on {args.host}:{args.port}")
    print(f"[openvla_sglang] Model: {args.model_id}, unnorm_key: {args.unnorm_key}")
    print(f"[openvla_sglang] Mode: {mode}")
    print(f"[openvla_sglang] Memory fraction: {_cli_mem_fraction:.2f} (~{_cli_mem_fraction * 128:.0f}GB on 128GB system)")
    print(f"[openvla_sglang] NOTE: model loads in background; poll GET /health for ready:true")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
