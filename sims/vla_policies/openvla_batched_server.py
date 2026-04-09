#!/usr/bin/env python
"""
Batched OpenVLA policy server for robo-eval.

Drop-in replacement for openvla_policy.py with dynamic request batching.
Concurrent /predict requests are collected by a batch worker and processed
together. Single requests use model.predict_action() directly (identical
outputs). Batched requests use a minimal monkey-patch that removes the
batch_size=1 check from the model (the underlying ops are batch-compatible).

Usage:
    .venvs/sglang/bin/python -m sims.vla_policies.openvla_batched_server --port 5103

Endpoints (same API as openvla_policy.py):
    GET  /health  -> {ready, model_id, batch_stats}
    GET  /info    -> {name, model_id, action_space, ...}
    POST /predict {obs: {image, instruction}} -> {actions, chunk_size, model_id}
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sims.vla_policies.vla_schema import VLAObservation, PredictRequest

logger = logging.getLogger(__name__)

NATIVE_ACTION_DIM = 7
MAX_BATCH_SIZE = 16
MAX_WAIT_MS = 30

_model = None
_processor = None
_model_id: str = ""
_device = None
_unnorm_key: str = "libero_spatial"
_ready: bool = False
_load_error: str = ""
_converter = None

_cli_model_id: str = os.environ.get(
    "OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"
)
_cli_device: str = "cuda"
_cli_unnorm_key: str = "libero_spatial"

_stats = {
    "total_requests": 0,
    "total_batches": 0,
    "avg_batch_size": 0.0,
    "max_batch_size_seen": 0,
}


# ======================================================================
# Token-to-Action (identical to model.predict_action internals)
# ======================================================================

class TokenToAction:
    def __init__(self, config, unnorm_key: str):
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = config.text_config.vocab_size - config.pad_to_multiple_of
        self.norm_stats = config.norm_stats
        self.unnorm_key = unnorm_key
        assert unnorm_key in self.norm_stats
        self.action_stats = self.norm_stats[unnorm_key]["action"]
        self.action_dim = len(self.action_stats["q01"])

    def convert(self, token_ids: np.ndarray) -> np.ndarray:
        discretized = self.vocab_size - token_ids
        discretized = np.clip(discretized - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized = self.bin_centers[discretized]
        mask = self.action_stats.get("mask", np.ones_like(self.action_stats["q01"], dtype=bool))
        high, low = np.array(self.action_stats["q99"]), np.array(self.action_stats["q01"])
        return np.where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)


# ======================================================================
# Monkey-patch: remove batch_size=1 restrictions
# ======================================================================

def _patch_model_for_batching(model):
    """Minimal patch: remove the two batch_size==1 checks.

    1. prepare_inputs_for_generation: remove ValueError for batch>1
    2. forward (cached generation path): remove assert shape[0]==1

    The underlying model operations (vision backbone, projector, LLaMA)
    are all naturally batch-compatible. Only these artificial checks block it.
    """
    import types

    # --- Patch 1: prepare_inputs_for_generation ---
    # Copy the original but remove the batch check
    def patched_prepare_inputs(
        self, input_ids=None, past_key_values=None, inputs_embeds=None,
        pixel_values=None, attention_mask=None, **kwargs,
    ):
        # Handle past_key_values cache
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        # input_embeds only used in 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update({
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        })
        return model_inputs

    # --- Patch 2: forward ---
    # We save and wrap the original forward, relaxing only the batch assert
    original_forward = model.forward

    def patched_forward(input_ids=None, attention_mask=None, pixel_values=None,
                        labels=None, inputs_embeds=None, past_key_values=None,
                        use_cache=None, output_attentions=None,
                        output_hidden_states=None, output_projector_features=None,
                        return_dict=None, **kwargs):
        """Wrapper that handles batched cached generation (autoregressive steps)."""
        import torch

        # For the cached generation path (input_ids.shape[1]==1, past_key_values!=None),
        # the original code asserts batch_size==1. We bypass this by processing
        # the batch through language_model directly.
        if (input_ids is not None and input_ids.shape[1] == 1 and
                past_key_values is not None and input_ids.shape[0] > 1):
            # Batched cached generation - skip the assert, go directly to LM
            use_cache_val = use_cache if use_cache is not None else True
            if model.training:
                use_cache_val = False
            ret_dict = return_dict if return_dict is not None else model.config.use_return_dict

            language_model_output = model.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache_val,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=ret_dict,
            )

            if not ret_dict:
                return language_model_output

            # Return with the same output type the model expects
            output_cls = type(language_model_output)
            try:
                # Try PrismaticCausalLMOutputWithPast if available
                return model._original_output_class(
                    loss=language_model_output.loss,
                    logits=language_model_output.logits,
                    past_key_values=language_model_output.past_key_values,
                    hidden_states=language_model_output.hidden_states,
                    attentions=language_model_output.attentions,
                    projector_features=None,
                )
            except (AttributeError, TypeError):
                return language_model_output

        # For all other cases (first multimodal forward, unimodal, etc),
        # delegate to the original forward
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, labels=labels, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_projector_features=output_projector_features,
            return_dict=return_dict,
        )

    # Save output class for the wrapper
    try:
        # Import the model's custom output class
        mod = type(model).__module__
        import importlib
        m = importlib.import_module(mod)
        model._original_output_class = getattr(m, 'PrismaticCausalLMOutputWithPast', None)
    except Exception:
        model._original_output_class = None

    model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs, model)
    model.forward = patched_forward  # plain function, not bound method (takes kwargs)
    logger.info("Model patched for batched generation (removed batch=1 checks)")


# ======================================================================
# Batch Queue
# ======================================================================

@dataclass
class PendingRequest:
    image_b64: str
    instruction: str
    future: Future = field(default_factory=Future)
    enqueue_time: float = field(default_factory=time.monotonic)


class BatchQueue:
    def __init__(self):
        self._queue: deque[PendingRequest] = deque()
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, req: PendingRequest):
        with self._lock:
            self._queue.append(req)
        self._event.set()

    def drain(self, max_items: int) -> list[PendingRequest]:
        with self._lock:
            n = min(max_items, len(self._queue))
            batch = [self._queue.popleft() for _ in range(n)]
            if not self._queue:
                self._event.clear()
            return batch

    def wait(self, timeout: float = None) -> bool:
        return self._event.wait(timeout=timeout)

    def __len__(self):
        return len(self._queue)


_batch_queue = BatchQueue()


# ======================================================================
# Model loading
# ======================================================================

def _load_model(model_id: str, device: str, unnorm_key: str) -> None:
    global _model, _processor, _model_id, _device, _unnorm_key
    global _ready, _load_error, _converter

    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _model_id = model_id
    _unnorm_key = unnorm_key

    logger.info("Loading processor from %s ...", model_id)
    try:
        _processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True,
        )
    except Exception as e:
        _load_error = f"Processor load failed: {e}"
        logger.error(_load_error)
        return

    logger.info("Loading model from %s on %s ...", model_id, device)
    try:
        _model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    except Exception as e:
        _load_error = f"Model load failed: {e}"
        logger.error(_load_error)
        _processor = None
        return

    _device = next(_model.parameters()).device
    _model.eval()

    # Monkey-patch for batched generation
    _patch_model_for_batching(_model)

    _converter = TokenToAction(_model.config, unnorm_key)
    logger.info("OpenVLA ready on %s (unnorm_key=%s, action_dim=%d)",
                _device, _unnorm_key, _converter.action_dim)
    _ready = True


def _load_model_bg(model_id: str, device: str, unnorm_key: str) -> None:
    try:
        _load_model(model_id, device, unnorm_key)
    except Exception as e:
        global _load_error
        _load_error = str(e)
        logger.error("Model load failed: %s\n%s", e, traceback.format_exc())


# ======================================================================
# Inference
# ======================================================================

def _predict_single(image_b64: str, instruction: str) -> list[float]:
    """Single prediction using model.predict_action (identical to original)."""
    import torch
    from PIL import Image

    raw = base64.b64decode(image_b64)
    pil_img = Image.open(BytesIO(raw)).convert("RGB")
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = _processor(prompt, pil_img).to(_device, dtype=torch.bfloat16)

    with torch.no_grad():
        action = _model.predict_action(
            **inputs, unnorm_key=_unnorm_key, do_sample=False,
        )

    action = np.array(action, dtype=np.float64).flatten()[:NATIVE_ACTION_DIM]
    gripper = 1.0 if action[-1] > 0.0 else -1.0
    action[-1] = -gripper
    return action.tolist()


def _predict_batch(batch: list[PendingRequest]) -> None:
    """Batched prediction using monkey-patched generate()."""
    import torch
    from PIL import Image

    batch_size = len(batch)

    try:
        pil_images = []
        prompts = []
        for req in batch:
            raw = base64.b64decode(req.image_b64)
            pil_images.append(Image.open(BytesIO(raw)).convert("RGB"))
            prompts.append(f"In: What action should the robot take to {req.instruction}?\nOut:")

        # Process each input and stack
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        for prompt, img in zip(prompts, pil_images):
            inp = _processor(prompt, img)
            all_input_ids.append(inp["input_ids"].squeeze(0))
            all_attention_masks.append(inp["attention_mask"].squeeze(0))
            all_pixel_values.append(inp["pixel_values"].squeeze(0))

        # Left-pad input_ids and attention_mask for generation
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_id = _model.config.pad_token_id

        padded_ids, padded_masks = [], []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([
                    torch.full((pad_len,), pad_id, dtype=ids.dtype), ids]))
                padded_masks.append(torch.cat([
                    torch.zeros(pad_len, dtype=mask.dtype), mask]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        input_ids = torch.stack(padded_ids).to(_device)
        attention_mask = torch.stack(padded_masks).to(_device)
        pixel_values = torch.stack(all_pixel_values).to(_device, dtype=torch.bfloat16)

        # Append special empty token 29871 (matches predict_action)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat([input_ids,
                torch.full((batch_size, 1), 29871, dtype=input_ids.dtype, device=_device)], dim=1)
            attention_mask = torch.cat([attention_mask,
                torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=_device)], dim=1)

        action_dim = _converter.action_dim
        with torch.no_grad():
            generated_ids = _model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=pixel_values,
                max_new_tokens=action_dim, do_sample=False,
            )

        for i, req in enumerate(batch):
            try:
                tids = generated_ids[i, -action_dim:].cpu().numpy()
                actions = _converter.convert(tids)
                actions = np.array(actions, dtype=np.float64).flatten()[:NATIVE_ACTION_DIM]
                gripper = 1.0 if actions[-1] > 0.0 else -1.0
                actions[-1] = -gripper
                req.future.set_result(actions.tolist())
            except Exception as e:
                req.future.set_exception(e)

    except Exception as e:
        logger.exception("Batch processing failed")
        for req in batch:
            if not req.future.done():
                req.future.set_exception(e)


def _process_batch(batch: list[PendingRequest]) -> None:
    global _stats
    bs = len(batch)
    logger.info("Processing batch of %d", bs)

    if bs == 1:
        req = batch[0]
        try:
            req.future.set_result(_predict_single(req.image_b64, req.instruction))
        except Exception as e:
            logger.exception("Single prediction failed")
            req.future.set_exception(e)
    else:
        _predict_batch(batch)

    _stats["total_requests"] += bs
    _stats["total_batches"] += 1
    _stats["avg_batch_size"] = _stats["total_requests"] / _stats["total_batches"]
    _stats["max_batch_size_seen"] = max(_stats["max_batch_size_seen"], bs)


def _batch_worker():
    logger.info("Batch worker started (max_batch=%d, max_wait=%dms)",
                MAX_BATCH_SIZE, MAX_WAIT_MS)
    while True:
        _batch_queue.wait(timeout=1.0)
        if not _ready or len(_batch_queue) == 0:
            continue
        time.sleep(MAX_WAIT_MS / 1000.0)
        batch = _batch_queue.drain(MAX_BATCH_SIZE)
        if batch:
            _process_batch(batch)


# ======================================================================
# FastAPI
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_model_bg,
                     args=(_cli_model_id, _cli_device, _cli_unnorm_key),
                     daemon=True).start()
    threading.Thread(target=_batch_worker, daemon=True).start()
    yield


app = FastAPI(title="OpenVLA Batched Policy Server", lifespan=lifespan)






@app.get("/health")
def health():
    r: dict = {"ready": _ready, "model_id": _model_id,
               "batch_stats": _stats, "queue_depth": len(_batch_queue)}
    if not _ready and _load_error:
        r["error"] = _load_error
    return r


@app.get("/info")
def info():
    name = _model_id.split("/")[-1] if "/" in _model_id else _model_id
    return {
        "name": name or "openvla-batched", "model_id": _model_id,
        "action_space": {"type": "eef_delta", "dim": NATIVE_ACTION_DIM,
                         "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]"},
        "state_dim": 0, "action_chunk_size": 1,
        "batching": {"max_batch_size": MAX_BATCH_SIZE, "max_wait_ms": MAX_WAIT_MS},
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    try:
        if not _ready:
            return JSONResponse(status_code=503,
                                content={"error": _load_error or "Model not ready"})
        pending = PendingRequest(image_b64=req.obs.images.get("primary"), instruction=req.obs.instruction)
        _batch_queue.put(pending)
        import asyncio
        actions = await asyncio.get_event_loop().run_in_executor(
            None, pending.future.result, 60.0)
        return {"actions": [actions], "chunk_size": 1, "model_id": _model_id}
    except Exception as e:
        logger.exception("Prediction failed")
        return JSONResponse(status_code=500,
                            content={"error": str(e), "traceback": traceback.format_exc()})


# ======================================================================
# CLI
# ======================================================================

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="OpenVLA Batched Policy Server")
    parser.add_argument("--model-id", default=os.environ.get(
        "OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"))
    parser.add_argument("--port", type=int, default=5103)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unnorm-key", default="libero_spatial", dest="unnorm_key")
    parser.add_argument("--max-batch-size", type=int, default=16, dest="max_batch_size")
    parser.add_argument("--max-wait-ms", type=int, default=30, dest="max_wait_ms")
    args = parser.parse_args()

    global _cli_model_id, _cli_device, _cli_unnorm_key, MAX_BATCH_SIZE, MAX_WAIT_MS
    _cli_model_id = args.model_id
    _cli_device = args.device
    _cli_unnorm_key = args.unnorm_key
    MAX_BATCH_SIZE = args.max_batch_size
    MAX_WAIT_MS = args.max_wait_ms

    logging.basicConfig(level=logging.INFO)
    print(f"[openvla_batched] Starting on {args.host}:{args.port}")
    print(f"[openvla_batched] Model: {args.model_id}, unnorm_key: {args.unnorm_key}")
    print(f"[openvla_batched] Batching: max_batch={MAX_BATCH_SIZE}, max_wait={MAX_WAIT_MS}ms")
    print(f"[openvla_batched] Poll GET /health for ready:true")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
