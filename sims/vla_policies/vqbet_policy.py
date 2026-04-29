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
#   "lerobot>=0.4.4",
# ]
# ///
"""VQ-BeT (Vector-Quantized Behavior Transformer) policy server via LeRobot.

Usage:
    python -m sims.vla_policies.vqbet_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Canonical checkpoint: ``lerobot/vqbet_pusht`` — the original BeT paper's
canonical PushT pushing benchmark.  Action space is identical to Diffusion
Policy's PushT checkpoint (2-dim absolute (x,y) end-effector position),
making **VQ-BeT vs Diffusion Policy on gym_pusht** a textbook architectural
head-to-head:

    VQ-BeT          : VQ-VAE-quantized action codes + transformer (autoregressive)
    Diffusion Policy: continuous DDPM denoising over action chunks
    pi05 / openvla  : autoregressive transformer over continuous actions

Three architecturally distinct policy families, identical PushT observations.

Action space (lerobot/vqbet_pusht):
    2-dimensional absolute (x, y) end-effector position in PushT coordinates.

Observation space:
    ``observation.image``: top-down camera (96 x 96).
    ``observation.state``: 2-dim robot keypoint state [x, y].
"""
from __future__ import annotations

import argparse
import base64
import logging
from io import BytesIO

import numpy as np
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation
from robo_eval.specs import ActionObsSpec

logger = logging.getLogger(__name__)

# Canonical checkpoint — PushT task, VQ-VAE + transformer policy.
_MODEL_ID_DEFAULT = "lerobot/vqbet_pusht"

# PushT observation resolution expected by the canonical checkpoint.
_PUSHT_IMAGE_SIZE = 96


class VQBeTPolicyServer(VLAPolicyBase):
    """VQ-BeT via lerobot — wraps ``lerobot.policies.vqbet.VQBeTPolicy``.

    The policy maintains a per-episode action queue internally (via lerobot's
    ``select_action``).  Call ``/reset`` between episodes to clear the queue.

    Architecture
    ------------
    VQ-BeT first trains a VQ-VAE to quantize actions into a discrete codebook,
    then trains a transformer to autoregressively predict action codes from
    observation history.  At inference, the transformer emits codes and the
    VQ-VAE decoder reconstructs continuous actions.

    Checkpoint compatibility
    ------------------------
    * ``lerobot/vqbet_pusht`` — 2-dim (x,y) PushT control (canonical)

    Adapts to action / state dims via ``config.output_features`` and
    ``config.input_features`` exactly like the Diffusion Policy server.
    """

    supports_batching: bool = False  # autoregressive sampling not easily batched

    @staticmethod
    def _load_config_compat(model_id: str):
        """Load VQBeTConfig while stripping fields unknown to the installed lerobot.

        Some checkpoint configs may contain fields absent from the installed
        VQBeTConfig.  draccus.parse() is strict and raises if the JSON contains
        unknown fields, so this loader strips them before parsing.
        """
        import dataclasses
        import json
        import os
        import tempfile

        import draccus
        from huggingface_hub import hf_hub_download

        from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig

        config_path = hf_hub_download(repo_id=model_id, filename="config.json",
                                      local_files_only=True)

        with open(config_path) as f:
            cfg_dict = json.load(f)

        # draccus.parse() rejects `type` (the discriminator field); always pop it.
        cfg_dict.pop("type", None)

        # Fields the installed VQBeTConfig actually knows about.
        known_fields = {f.name for f in dataclasses.fields(VQBeTConfig)}
        unknown = set(cfg_dict.keys()) - known_fields
        if unknown:
            logger.warning(
                "VQBeTConfig compat shim: stripping fields absent from installed "
                "lerobot (%s). Checkpoint was likely saved with an older lerobot. "
                "Stripped: %s",
                model_id,
                sorted(unknown),
            )
            for k in unknown:
                cfg_dict.pop(k)

        # Always use the manual path so we control field stripping.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(cfg_dict, tf)
            clean_path = tf.name
        try:
            with draccus.config_type("json"):
                config = draccus.parse(VQBeTConfig, clean_path, args=[])
        finally:
            os.unlink(clean_path)
        return config

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch
        from safetensors.torch import load_file as _load_safetensors
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy as _VQBeTPolicy
        from huggingface_hub import hf_hub_download as _hf_dl

        self.model_id = model_id
        logger.info("Loading VQBeTPolicy from %s on %s ...", model_id, device)

        # Strip config fields unknown to the installed lerobot version before
        # from_pretrained(config=...) runs its strict parser.
        _compat_config = self._load_config_compat(model_id)
        # Override the checkpoint device so from_pretrained() loads weights to
        # the requested runtime device.
        _compat_config.device = device
        self._policy = _VQBeTPolicy.from_pretrained(model_id, config=_compat_config)
        self._policy.to(torch.device(device))
        self._policy.eval()
        self._device = device

        # Normalization compatibility for checkpoints that store min/max stats
        # in the safetensors weights rather than applying them inside
        # select_action().
        self._norm_stats: dict = {}  # {"norm"/"denorm": {feat_key: {"min": Tensor, ...}}}
        try:
            _st_path = _hf_dl(repo_id=model_id, filename="model.safetensors",
                               local_files_only=True)
            _st = _load_safetensors(_st_path)
            dev_t = torch.device(device)
            # Keys look like:
            #   normalize_inputs.buffer_observation_state.min
            #   normalize_inputs.buffer_observation_state.max
            #   unnormalize_outputs.buffer_action.min
            #   unnormalize_outputs.buffer_action.max
            # Feature name has dots replaced by underscores in the buffer key.
            # We reconstruct original feature names from config.
            all_feat_names = (
                list(_compat_config.input_features.keys())
                + list(_compat_config.output_features.keys())
            )
            for k, v in _st.items():
                for prefix, bucket in [
                    ("normalize_inputs.buffer_", "norm"),
                    ("unnormalize_outputs.buffer_", "denorm"),
                ]:
                    if not k.startswith(prefix):
                        continue
                    # remainder: "<feat_underscored>.<stat>"
                    remainder = k[len(prefix):]
                    stat = remainder.rsplit(".", 1)[-1]        # "min" or "max"
                    buf_feat = remainder[: -(len(stat) + 1)]   # e.g. "observation_state"
                    # Match against known feature names (dots → underscores)
                    feat = next(
                        (f for f in all_feat_names if f.replace(".", "_") == buf_feat),
                        buf_feat.replace("_", "."),  # fallback: just replace underscores
                    )
                    self._norm_stats.setdefault(bucket, {}).setdefault(feat, {})[stat] = v.to(dev_t)
            if self._norm_stats:
                logger.info("Loaded normalization stats from checkpoint: %s",
                            {b: list(v.keys()) for b, v in self._norm_stats.items()})
            else:
                logger.warning(
                    "No normalization buffers found in checkpoint safetensors. "
                    "Model may produce unnormalized actions if it was trained with "
                    "built-in normalization (lerobot < 0.4.x style)."
                )
        except Exception as e:
            logger.warning("Could not load normalization stats from checkpoint: %s", e)

        cfg = self._policy.config

        try:
            self._action_dim = cfg.output_features["action"].shape[0]
        except Exception:
            self._action_dim = 2  # PushT default

        try:
            self._state_dim = cfg.input_features["observation.state"].shape[0]
        except Exception:
            self._state_dim = 2  # PushT default

        # n_action_steps: how many actions to dequeue per select_action call.
        self._n_action_steps = getattr(cfg, "n_action_steps", 5)
        self._chunk_size = getattr(cfg, "action_chunk_size", self._n_action_steps)

        # Camera feature key — typically "observation.image" for PushT.
        image_keys = list(cfg.image_features) if getattr(cfg, "image_features", None) else []
        self._camera_key = image_keys[0] if image_keys else "observation.image"

        # Image size from config if available, else fall back to 96 (PushT default).
        try:
            _ishape = cfg.image_features[self._camera_key].shape  # (C, H, W)
            self._image_h = int(_ishape[1])
            self._image_w = int(_ishape[2])
        except Exception:
            self._image_h = _PUSHT_IMAGE_SIZE
            self._image_w = _PUSHT_IMAGE_SIZE

        self.ready = True
        logger.info(
            "VQBeTPolicy ready: model=%s, action_dim=%d, state_dim=%d, "
            "n_action_steps=%d, image=%dx%d",
            model_id,
            self._action_dim,
            self._state_dim,
            self._n_action_steps,
            self._image_h,
            self._image_w,
        )

    def _min_max_norm(self, x, stats_key: str, feat: str):
        """Apply MIN-MAX normalization to tensor x: output in [-1, 1]."""
        bucket = self._norm_stats.get(stats_key, {})
        feat_stats = bucket.get(feat)
        if feat_stats is None or "min" not in feat_stats or "max" not in feat_stats:
            return x  # no stats available — pass through
        mn = feat_stats["min"]
        mx = feat_stats["max"]
        return 2.0 * (x - mn) / (mx - mn + 1e-8) - 1.0

    def _min_max_denorm(self, x, stats_key: str, feat: str):
        """Undo MIN-MAX normalization: input in [-1, 1], output in original range."""
        bucket = self._norm_stats.get(stats_key, {})
        feat_stats = bucket.get(feat)
        if feat_stats is None or "min" not in feat_stats or "max" not in feat_stats:
            return x  # no stats available — pass through
        mn = feat_stats["min"]
        mx = feat_stats["max"]
        return (x + 1.0) / 2.0 * (mx - mn) + mn

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from PIL import Image
        from torchvision import transforms

        resize = transforms.Compose([
            transforms.Resize((self._image_h, self._image_w)),
            transforms.ToTensor(),
        ])

        # Decode primary camera image.
        img_b64 = obs.images.get("primary") or obs.images.get("image") or ""
        if img_b64:
            pil_img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        else:
            pil_img = Image.new("RGB", (self._image_w, self._image_h), color=(128, 128, 128))
        img_tensor = resize(pil_img)  # (C, H, W)

        # Robot state — 2-dim for PushT.
        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        state_tensor = torch.tensor(
            state_list[: self._state_dim], dtype=torch.float32
        )

        dev = torch.device(self._device)

        # Apply observation normalization (MIN-MAX → [-1, 1]).
        # Apply checkpoint min/max stats when they are available.
        state_norm = self._min_max_norm(
            state_tensor.to(dev), "norm", "observation.state"
        )

        # select_action() expects per-step shapes: image (batch, C, H, W),
        # state (batch, state_dim).  The internal queues handle the n_obs_steps
        # temporal dimension.  Do NOT unsqueeze the time dimension here.
        obs_batch = {
            self._camera_key: img_tensor.unsqueeze(0).to(dev),  # (1, C, H, W)
            "observation.state": state_norm.unsqueeze(0),       # (1, state_dim)
        }

        with torch.no_grad():
            action = self._policy.select_action(obs_batch)

        # Undo action normalization ([-1, 1] → workspace coordinates).
        action = self._min_max_denorm(action, "denorm", "action")

        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        action = np.atleast_1d(action.squeeze())
        return [action.tolist()]

    def get_info(self) -> dict:
        return {
            "name": "VQBeT",
            "model_id": self.model_id,
            "action_space": {
                # "eef_xy" matches GymPushTBackend.get_info() action_space.type;
                # using "absolute_xy" would fail env_wrapper._negotiate_spaces().
                "type": "eef_xy" if self._action_dim == 2 else "joint_pos",
                "dim": getattr(self, "_action_dim", 2),
                "description": (
                    "2-dim absolute (x,y) end-effector in PushT coordinates"
                    if getattr(self, "_action_dim", 2) == 2
                    else f"{getattr(self, '_action_dim', 2)}-dim action vector"
                ),
            },
            "state_dim": getattr(self, "_state_dim", 2),
            "action_chunk_size": getattr(self, "_n_action_steps", 5),
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": getattr(self, "_state_dim", 2),
                "image_resolution": [self._image_h, self._image_w]
                if hasattr(self, "_image_h") else [_PUSHT_IMAGE_SIZE, _PUSHT_IMAGE_SIZE],
                "image_transform": "none",
            },
        }

    def get_action_spec(self) -> dict:
        """VQ-BeT action spec — 2-dim absolute (x,y) for PushT checkpoint.

        Uses ``eef_xy`` key and ``absolute_xy_position`` format to match the
        GymPushTBackend action spec (parity with DiffusionPolicyServer).
        """
        action_dim = getattr(self, "_action_dim", 2)
        if action_dim == 2:
            return {
                "eef_xy": ActionObsSpec(
                    "eef_xy", 2, "absolute_xy_position", (0.0, 512.0),
                    description="2-dim absolute (x,y) end-effector position (PushT coordinates)",
                ),
            }
        else:
            return {
                "joint_pos": ActionObsSpec(
                    "joint_pos", action_dim, "joint_pos_absolute", (-3.15, 3.15),
                    description=f"{action_dim}-dim joint position absolute",
                ),
            }

    def get_observation_spec(self) -> dict:
        """VQ-BeT observation spec: top-down RGB + 2-dim state."""
        state_dim = getattr(self, "_state_dim", 2)
        return {
            "primary": ActionObsSpec("image", 0, "rgb_hwc_uint8"),
            "state": ActionObsSpec(
                "state", state_dim,
                # "agent_xy_position" matches GymPushTBackend observation_spec
                # state.format — parity with DiffusionPolicyServer.
                "agent_xy_position" if state_dim == 2 else "eef_state",
                description=(
                    "2-dim (x,y) agent position for PushT"
                    if state_dim == 2 else
                    f"{state_dim}-dim proprioceptive state"
                ),
            ),
        }

    def reset(self) -> None:
        """Reset the per-episode action queue in the VQ-BeT policy."""
        if getattr(self, "_policy", None) is not None:
            self._policy.reset()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="VQ-BeT Policy Server (lerobot)")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    parser.add_argument("--port", type=int, default=5108)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = VQBeTPolicyServer()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="VQ-BeT Policy Server",
        max_batch_size=1,
        max_wait_ms=0.0,
    )
    logging.basicConfig(level=logging.INFO)
    print(
        f"[vqbet_policy] Starting on {args.host}:{args.port} "
        f"model={args.model_id} device={args.device}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
