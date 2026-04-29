#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "torchvision",
#   "lerobot>=0.4.5",
# ]
# ///
"""SmolVLA policy server.

Usage:
    python -m sims.vla_policies.smolvla_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Uses lerobot's make_pre_post_processors for preprocessing and postprocessing.
"""
from __future__ import annotations

import argparse
import base64
import logging
from io import BytesIO

import numpy as np
from sims.vla_policies.base import VLAPolicyBase, detect_lerobot_image_transform, make_app
from sims.vla_policies.vla_schema import VLAObservation
from robo_eval.specs import (
    ActionObsSpec,
    POSITION_DELTA,
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
)

logger = logging.getLogger(__name__)
_MODEL_ID_DEFAULT = "HuggingFaceVLA/smolvla_libero"


class SmolVLAPolicy(VLAPolicyBase):
    """SmolVLA (HuggingFaceVLA/smolvla_libero and variants).

    n_action_steps=1: replans every step.

    Batching
    --------
    ``predict_batch()`` is overridden to run a single GPU forward pass for all
    observations simultaneously (``supports_batching = True``).  Each sample is
    preprocessed independently (so tokenization is handled correctly per-sample),
    then the resulting tensors are stacked and passed to ``self._policy.model()``
    once, yielding ``[B, chunk_size, action_dim]``.  Action at index 0 is taken
    for each sample (matching ``n_action_steps=1`` behaviour) and postprocessed.

    Recommended server settings: ``max_batch_size=8``, ``max_wait_ms=15.0``.
    """

    supports_batching: bool = True

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _SmolVLAPolicy
        from lerobot.utils.constants import ACTION, OBS_STATE

        self.model_id = model_id
        logger.info("Loading SmolVLA from %s on %s ...", model_id, device)
        self._policy = _SmolVLAPolicy.from_pretrained(model_id)
        self._policy.to(torch.device(device))
        self._policy.eval()

        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        cfg = self._policy.config
        # n_action_steps = how many actions are dequeued per select_action call.
        # For smolvla_libero: n_action_steps=1, chunk_size=50 → replans every step.
        self._action_chunk_size = getattr(cfg, "n_action_steps", 1)
        try:
            self._action_dim = cfg.output_features[ACTION].shape[0]
        except Exception:
            self._action_dim = 7
        try:
            self._state_dim = cfg.input_features[OBS_STATE].shape[0]
        except Exception:
            self._state_dim = 8
        keys = list(cfg.image_features) if getattr(cfg, "image_features", None) else []
        self._camera_key = keys[0] if keys else "observation.images.image"
        self._camera_key2 = keys[1] if len(keys) > 1 else ""
        self._image_transform = detect_lerobot_image_transform(model_id, self._policy)
        self.ready = True
        logger.info(
            "SmolVLA ready: model=%s, n_action_steps=%d, chunk_size=%d, transform=%s",
            model_id, self._action_chunk_size,
            getattr(cfg, "chunk_size", 50), self._image_transform,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from PIL import Image
        from torchvision import transforms

        to_tensor = transforms.ToTensor()

        def decode(b64: str):
            return to_tensor(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        frame = {
            self._camera_key: decode(obs.images["primary"]),
            "observation.state": torch.tensor(state_list, dtype=torch.float32)[: self._state_dim],
            "task": obs.instruction,
        }
        if obs.images.get("wrist") and self._camera_key2:
            frame[self._camera_key2] = decode(obs.images["wrist"])

        batch = self._preprocessor(frame)
        action = self._policy.select_action(batch)
        action = self._postprocessor(action)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return [action.squeeze(0).tolist()]

    def predict_batch(self, obs_list: list[VLAObservation]) -> list[list[list[float]]]:
        """True GPU-batched inference: one forward pass for all observations.

        Implementation notes
        --------------------
        * Each observation is preprocessed independently (so text tokenisation
          is handled correctly per sample by the lerobot preprocessor).
        * Resulting tensors are concatenated along the batch dimension (the
          preprocessor adds a batch dim of 1; ``torch.cat`` merges them).
        * ``self._policy.model(batched)`` is called once → ``[B, chunk, action_dim]``.
        * Action at step 0 is taken for each sample (matches ``n_action_steps=1``).
        * Calls ``self._postprocessor`` per sample, identical to ``predict()``.
        * Does NOT use the per-episode action chunk buffer (``select_action``).
          Each call produces a fresh inference — correct for multi-client serving
          where per-client buffer state would need separate tracking.
        """
        import torch
        from PIL import Image
        from torchvision import transforms

        to_tensor = transforms.ToTensor()

        def decode(b64: str) -> "torch.Tensor":
            return to_tensor(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

        # Step 1: build and preprocess each frame independently
        preprocessed: list[dict] = []
        for obs in obs_list:
            state_list = obs.state.get("flat") or [0.0] * self._state_dim
            frame: dict = {
                self._camera_key: decode(obs.images["primary"]),
                "observation.state": torch.tensor(state_list, dtype=torch.float32)[: self._state_dim],
                "task": obs.instruction,
            }
            if obs.images.get("wrist") and self._camera_key2:
                frame[self._camera_key2] = decode(obs.images["wrist"])
            preprocessed.append(self._preprocessor(frame))

        # Step 2: stack preprocessed tensors along batch dim.
        # The preprocessor adds a batch dim of 1; torch.cat along dim=0 merges them.
        # Variable-length sequences (e.g. tokenised instructions of different lengths)
        # must be padded to the same length before concatenation.  We right-pad with 0
        # (attention_mask: 0 = masked; input_ids: 0 = pad_token_id in most tokenizers).
        def _cat_or_pad(tensors: list) -> "torch.Tensor":
            """Concatenate tensors along dim 0, padding if they differ in other dims."""
            if not tensors:
                return torch.empty(0)
            if all(t.shape == tensors[0].shape for t in tensors):
                return torch.cat(tensors, dim=0)
            # Variable shape: pad each tensor to the max extent on every dim > 0
            max_shape = [max(t.shape[d] for t in tensors) for d in range(tensors[0].dim())]
            padded = []
            for t in tensors:
                pad_spec: list[int] = []
                for d in range(t.dim() - 1, 0, -1):
                    # torch.nn.functional.pad takes (right, left, ...) from last dim
                    pad_spec += [0, max_shape[d] - t.shape[d]]
                if any(p != 0 for p in pad_spec):
                    import torch.nn.functional as F
                    t = F.pad(t, pad_spec, value=0)
                padded.append(t)
            return torch.cat(padded, dim=0)

        batched: dict = {}
        sample0 = preprocessed[0]
        for key, val in sample0.items():
            if isinstance(val, torch.Tensor):
                # cat along existing batch dim (dim=0), padding if variable-length
                batched[key] = _cat_or_pad([p[key] for p in preprocessed])
            elif isinstance(val, str):
                # Text fields: collect as list (model accepts list[str])
                batched[key] = [p[key] for p in preprocessed]
            elif isinstance(val, dict):
                # Nested dict (e.g., tokenised task with input_ids / attention_mask)
                batched[key] = {
                    k: _cat_or_pad([p[key][k] for p in preprocessed])
                    for k in val
                    if isinstance(val[k], torch.Tensor)
                }
            else:
                # Fallback: collect as list
                batched[key] = [p[key] for p in preprocessed]

        # Step 3: single batched forward pass — [B, chunk_size, action_dim]
        # Use _get_action_chunk which correctly prepares images, state, tokens
        # before calling model.sample_actions with the right positional args.
        with torch.no_grad():
            action_chunks = self._policy._get_action_chunk(batched)

        # Step 4: extract action at step 0 for each sample, then postprocess
        results: list[list[list[float]]] = []
        for i in range(len(obs_list)):
            # action_chunks shape: [B, chunk_size, action_dim]
            # Take first action in chunk (n_action_steps=1 convention)
            # Add batch dim [1, action_dim] so postprocessor sees the expected shape
            action = action_chunks[i, 0].unsqueeze(0)  # [1, action_dim]
            action = self._postprocessor(action)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            results.append([action.squeeze(0).tolist()])

        return results

    def get_info(self) -> dict:
        return {
            "name": "SmolVLA",
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",
                "dim": getattr(self, "_action_dim", 7),
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]",
            },
            "state_dim": getattr(self, "_state_dim", 8),
            "action_chunk_size": getattr(self, "_action_chunk_size", 1),
            "obs_requirements": {
                "cameras": ["primary", "wrist"],
                "state_dim": getattr(self, "_state_dim", 8),
                "image_resolution": [256, 256],
                # The camera transform is applied before policy inference; clients
                # should not apply a second flip.
                "image_transform": "applied_in_sim",
            },
        }

    def get_action_spec(self) -> dict:
        """SmolVLA action spec: delta EEF with axis-angle rotation, LIBERO gripper (neg=close)."""
        return {
            "position": POSITION_DELTA,
            "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict:
        """SmolVLA observation spec: primary + wrist RGB, 8-dim eef state, language."""
        state_dim = getattr(self, "_state_dim", 8)
        return {
            "primary": IMAGE_RGB,
            "wrist": IMAGE_RGB,
            "state": ActionObsSpec("state", state_dim, "libero_eef_pos3_aa3_grip2"),
            "instruction": LANGUAGE,
        }

    def reset(self) -> None:
        if getattr(self, "_policy", None) is not None:
            self._policy.reset()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SmolVLA Policy Server")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    parser.add_argument("--port", type=int, default=5102)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-batch-size", type=int, default=8, dest="max_batch_size",
        help="Max requests per GPU batch (default 8; set 1 to disable batching)",
    )
    parser.add_argument(
        "--max-wait-ms", type=float, default=15.0, dest="max_wait_ms",
        help="Max milliseconds to wait before dispatching a partial batch (default 15.0)",
    )
    args = parser.parse_args()

    policy = SmolVLAPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="SmolVLA Policy Server",
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
    )
    logging.basicConfig(level=logging.INFO)
    print(f"[smolvla_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    print(f"[smolvla_policy] Batching: max_batch_size={args.max_batch_size}, max_wait_ms={args.max_wait_ms}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
