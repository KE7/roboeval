#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "huggingface_hub",
#   # "lerobot",  # install with InternVLA-A1 extensions from InternRobotics
# ]
# ///
"""InternVLA-A1 policy server — official RoboTwin inference pipeline.

Pipeline (from InternVLA-A1/evaluation/RoboTwin/inference.py):
  1. ResizeImagesWithPadFn(224, 224)
  2. Qwen3_VLProcessorTransformFn() — tokenise instruction + VLM image processor
  3. NormalizeTransformFn           — state normalisation (mean_std)
  4. predict_action_chunk()         — model runs in float32 by default (see INTERNVLA_DTYPE)
  5. UnNormalizeTransformFn         — action unnormalisation (cast to float32)
  6. Delta mode: action_pred += current_state (gripper indices zeroed first)
  7. Gripper binarisation at indices 6 and 13: 0 if val < 0.5 else 1
  8. Return first 30 actions (infer_horizon=30) as absolute qpos

Environment variables:
    INTERNVLA_DTYPE  – model weight dtype: fp32 (default), bf16, or fp16.
                       fp32 uses ~12 GB weights and is recommended for
                       evaluation. bf16/fp16 reduce memory use but can degrade
                       action quality and should be reserved for local debugging.

Usage:
    python -m sims.vla_policies.internvla_policy [--port PORT] [--model MODEL]
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
from pathlib import Path

import numpy as np
import torch
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

# Make InternVLA-A1 source importable when running as a host process
_INTERNVLA_SRC = Path(__file__).parents[3] / "InternVLA-A1" / "src"
if _INTERNVLA_SRC.exists() and str(_INTERNVLA_SRC) not in sys.path:
    sys.path.insert(0, str(_INTERNVLA_SRC))

try:
    from robo_eval.specs import ActionObsSpec
    # InternVLA-A1-3B-RoboTwin: 14-dim ALOHA absolute joint positions
    # 7 left arm joints (incl. gripper at idx 6) + 7 right arm joints (incl. gripper at idx 13)
    _JOINT_POS_14   = ActionObsSpec("joint_pos", 14, "absolute_joint_positions", None)
    _IMAGE_RGB      = ActionObsSpec("image",      0, "rgb_hwc_uint8")
    _LANGUAGE       = ActionObsSpec("language",   0, "language")
    _STATE_JOINTS14 = ActionObsSpec("state",      14, "joint_positions", None)
    _HAS_SPECS = True
except ImportError:
    _HAS_SPECS = False

logger = logging.getLogger(__name__)

_MODEL_ID      = "InternRobotics/InternVLA-A1-3B-RoboTwin"
_STATS_KEY     = "aloha"
_RESIZE        = 224
_INFER_HORIZON = 30   # first 30 of n_action_steps chunk returned as actions
_ACTION_DIM    = 14   # ALOHA: 7 left + 7 right joint positions
_LEFT_GRIP     = 6    # gripper dim in the left-arm half
_RIGHT_GRIP    = 13   # gripper dim in the full 14-dim vector
_IMG_HIST_LEN  = 16   # temporal deque capacity (interval=15 steps)


class InternVLAPolicy(VLAPolicyBase):
    """InternVLA-A1-3B-RoboTwin — dual-arm ALOHA, absolute joint positions.

    Maintains per-episode image history for temporal pairs (past+current).
    Call POST /reset at the start of every episode to clear the history.
    """

    @staticmethod
    def _resolve_dtype() -> torch.dtype:
        """Return model weight dtype from $INTERNVLA_DTYPE (default: float32).

        float32  → ~12 GB weights; recommended for evaluation.
        bfloat16 → ~6 GB weights; local debugging only.
        float16  → ~6 GB weights; local debugging only.
        """
        _MAP = {
            "fp32": torch.float32,  "float32":  torch.float32,
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16,  "float16":  torch.float16,
        }
        raw = os.environ.get("INTERNVLA_DTYPE", "fp32").lower().strip()
        dtype = _MAP.get(raw)
        if dtype is None:
            logger.warning("Unknown INTERNVLA_DTYPE=%r; using float32", raw)
            return torch.float32
        if dtype is not torch.float32:
            logger.warning(
                "INTERNVLA_DTYPE=%s requested. bf16/fp16 corrupt InternVLA-A1's "
                "flow-matching denoise loop and produce ~0%% task success. "
                "Use only for memory-constrained debugging.", raw,
            )
        return dtype

    def __init__(self) -> None:
        super().__init__()
        self._dtype  = self._resolve_dtype()  # default: float32 (~12 GB)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._head_hist:  collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)
        self._left_hist:  collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)
        self._right_hist: collections.deque = collections.deque(maxlen=_IMG_HIST_LEN)

    def load_model(self, model_id: str, device: str, **_) -> None:
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Policy
        from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
        from lerobot.transforms.core import (
            NormalizeTransformFn, UnNormalizeTransformFn,
            ResizeImagesWithPadFn, RemapImageKeyTransformFn, compose,
        )

        self.model_id = model_id
        self._device  = device
        logger.info("Loading InternVLA-A1-3B on device=%s ...", device)

        try:
            ckpt_dir = Path(snapshot_download(repo_id=model_id, local_files_only=True))
        except Exception:
            ckpt_dir = Path(snapshot_download(repo_id=model_id))
        logger.info("Checkpoint dir: %s", ckpt_dir)

        config = PreTrainedConfig.from_pretrained(ckpt_dir)
        self._policy = QwenA1Policy.from_pretrained(config=config, pretrained_name_or_path=ckpt_dir)
        self._policy = self._policy.to(device).to(self._dtype).eval()
        self._n_action_steps = self._policy.config.n_action_steps

        stats_path = Path(ckpt_dir) / "stats.json"
        with open(stats_path) as f:
            stats = json.load(f)[_STATS_KEY]
        sk = ["min", "max", "mean", "std"]
        action_stat = {"action":              {k: np.asarray(stats["action"][k])              for k in sk}}
        state_stat  = {"observation.state":   {k: np.asarray(stats["observation.state"][k])  for k in sk}}

        self._unnorm_fn = UnNormalizeTransformFn(
            selected_keys=["action"], mode="mean_std", norm_stats=action_stat
        )
        image_keys = [f"observation.images.image{i}" for i in range(3)]
        self._input_xfm = compose([
            ResizeImagesWithPadFn(height=_RESIZE, width=_RESIZE),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(
                selected_keys=["observation.state"], mode="mean_std", norm_stats=state_stat
            ),
        ])
        self.ready = True
        logger.info(
            "InternVLA ready: device=%s, dtype=%s, n_action_steps=%d, infer_horizon=%d",
            device, self._dtype, self._n_action_steps, _INFER_HORIZON,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import PIL.Image as PILImage

        def b64_to_tensor(b64: str) -> torch.Tensor:
            img = PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,C]

        def temporal_pair(hist: collections.deque, cur: torch.Tensor) -> torch.Tensor:
            # Official code: past = oldest frame in deque (up to 15 steps back)
            past = hist[0] if hist else torch.zeros_like(cur)
            return torch.stack([past, cur], dim=0)  # [2, H, W, C]

        imgs = obs.images
        head_b64  = imgs.get("primary")   or imgs.get("head_camera")
        left_b64  = imgs.get("wrist")     or imgs.get("left_camera")
        right_b64 = imgs.get("secondary") or imgs.get("right_camera")
        if head_b64 is None:
            raise ValueError("No primary/head_camera image in request")
        head_t  = b64_to_tensor(head_b64)
        left_t  = b64_to_tensor(left_b64)  if left_b64  else head_t
        right_t = b64_to_tensor(right_b64) if right_b64 else head_t

        # Build [2,C,H,W] temporal pairs, then update history
        image0 = temporal_pair(self._head_hist,  head_t).permute(0, 3, 1, 2)
        image1 = temporal_pair(self._left_hist,  left_t).permute(0, 3, 1, 2)
        image2 = temporal_pair(self._right_hist, right_t).permute(0, 3, 1, 2)
        self._head_hist.append(head_t)
        self._left_hist.append(left_t)
        self._right_hist.append(right_t)

        state_vals = obs.state.get("flat", [0.0] * _ACTION_DIM)
        state_t = torch.tensor(state_vals, dtype=self._dtype)

        sample = {
            "observation.images.image0": image0,
            "observation.images.image1": image1,
            "observation.images.image2": image2,
            "observation.state": state_t,
            "task": obs.instruction,
        }
        sample = self._input_xfm(sample)

        inputs: dict = {}
        for key, val in sample.items():
            if key == "task":
                inputs[key] = [val]
                continue
            if not isinstance(val, torch.Tensor):
                continue
            v = val.unsqueeze(0) if val.ndim == 0 else val[None]
            if val.dtype in (torch.int64, torch.bool):
                inputs[key] = v.to(self._device)
            else:
                inputs[key] = v.to(self._device, dtype=self._dtype)
        # All image masks set to True (attend to all tokens)
        for i in range(3):
            inputs[f"observation.images.image{i}_mask"] = torch.tensor([True]).to(self._device)

        with torch.no_grad():
            action_pred, _ = self._policy.predict_action_chunk(inputs, decode_image=False)

        # [1, n_action_steps, padded_dim] → [30, 14]
        action_pred = action_pred[0, :_INFER_HORIZON, :_ACTION_DIM]

        # Cast to float32 for unnorm / delta / binarise so post-processing
        # precision is independent of model weight dtype.
        action_pred = action_pred.to(torch.float32)
        action_pred = self._unnorm_fn({"action": action_pred})["action"]

        # Delta → absolute: add current joint positions (gripper dims zeroed first)
        init = torch.tensor(state_vals, dtype=torch.float32, device=action_pred.device)
        init[_LEFT_GRIP] = 0.0
        init[_RIGHT_GRIP] = 0.0
        action_pred = action_pred + init[None]

        # Binarise gripper dimensions
        action_pred[:, _LEFT_GRIP]  = (action_pred[:, _LEFT_GRIP]  >= 0.5).float()
        action_pred[:, _RIGHT_GRIP] = (action_pred[:, _RIGHT_GRIP] >= 0.5).float()

        return action_pred.cpu().numpy().tolist()

    def get_info(self) -> dict:
        return {
            "model":    _MODEL_ID,
            "model_id": _MODEL_ID,
            "model_type": "InternVLA-A1-3B",
            "device":   self._device,
            "loaded":   getattr(self, "_policy", None) is not None,
            "action_space": {
                "type": "joint_pos",
                "dim":  _ACTION_DIM,
                "accepted_dims": [_ACTION_DIM],
            },
            "state_dim":         _ACTION_DIM,
            "action_chunk_size": _INFER_HORIZON,
            "obs_requirements": {
                "cameras":          ["primary"],
                "state_dim":        _ACTION_DIM,
                "state_format":     "flat",
                "image_transform":  "none",
            },
        }

    def get_action_spec(self) -> "dict | None":
        """InternVLA-A1-3B-RoboTwin: 14-dim ALOHA absolute joint positions."""
        if not _HAS_SPECS:
            return None
        return {"joint_pos": _JOINT_POS_14}

    def get_observation_spec(self) -> "dict | None":
        """InternVLA-A1-3B-RoboTwin observation spec: primary RGB + 14-dim joint state."""
        if not _HAS_SPECS:
            return None
        return {
            "primary":     _IMAGE_RGB,
            "state":       _STATE_JOINTS14,
            "instruction": _LANGUAGE,
        }

    def reset(self) -> None:
        self._head_hist.clear()
        self._left_hist.clear()
        self._right_hist.clear()
        logger.info("Episode reset: image history cleared.")


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="InternVLA-A1 Policy Server")
    parser.add_argument("--port",  type=int, default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000))))
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--model", default=_MODEL_ID)
    args = parser.parse_args()

    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be 1024–65535, got {args.port}")

    policy = InternVLAPolicy()
    app = make_app(policy, args.model, policy._device, title="InternVLA-A1 Policy Server")
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    print(f"[internvla_policy] Starting on {args.host}:{args.port} model={args.model} device={policy._device}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
