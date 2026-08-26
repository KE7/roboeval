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
"""Diffusion Policy server via LeRobot.

Usage:
    python -m sims.vla_policies.diffusion_policy_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Canonical checkpoint: ``lerobot/diffusion_pusht`` — trained on the PushT
2-D pushing task.  This checkpoint targets the PushT gym environment.

The DiffusionPolicy implementation in lerobot uses a Denoising Diffusion
Probabilistic Model (DDPM) for action generation.  It ingests a short
observation history and produces an action chunk via iterative denoising.

Action space (lerobot/diffusion_pusht):
    2-dimensional absolute (x, y) end-effector position in PushT coordinates.

Observation space:
    ``observation.image``: top-down camera (96 × 96).
    ``observation.state``: 2-dim robot state [x, y].
"""

from __future__ import annotations

import argparse
import base64
import logging
from io import BytesIO

import numpy as np

from roboeval.specs import ActionObsSpec
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

# Canonical checkpoint — PushT task, DDPM-based diffusion policy.
# Alternative: "lerobot/diffusion_aloha_sim_insertion_human" (14-dim joint_pos, ALOHA sim).
_MODEL_ID_DEFAULT = "lerobot/diffusion_pusht"

# PushT observation resolution expected by the canonical checkpoint.
_PUSHT_IMAGE_SIZE = 96


class DiffusionPolicyServer(VLAPolicyBase):
    """Diffusion Policy via lerobot — wraps ``lerobot.policies.diffusion.DiffusionPolicy``.

    The policy maintains a per-episode action queue internally (via lerobot's
    ``select_action``).  Call ``/reset`` between episodes to clear the queue.

    Checkpoint compatibility
    ------------------------
    * ``lerobot/diffusion_pusht``              — 2-dim (x,y) PushT control
    * ``lerobot/diffusion_aloha_sim_insertion_human`` — 14-dim ALOHA joint pos

    The policy server adapts to whichever checkpoint is loaded by reading
    ``config.output_features[ACTION].shape[0]`` for the action dim and
    ``config.input_features[OBS_STATE].shape[0]`` for the state dim.
    """

    supports_batching: bool = False  # DDPM denoising loop not easily batched

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch
        from lerobot.policies.diffusion.modeling_diffusion import (
            DiffusionPolicy as _DiffusionPolicy,
        )
        from lerobot.utils.constants import ACTION, OBS_STATE

        self.model_id = model_id
        logger.info("Loading DiffusionPolicy from %s on %s ...", model_id, device)

        self._policy = _DiffusionPolicy.from_pretrained(model_id)
        self._policy.to(torch.device(device))
        self._policy.eval()
        self._device = device

        cfg = self._policy.config

        try:
            self._action_dim = cfg.output_features[ACTION].shape[0]
        except Exception:
            self._action_dim = 2  # PushT default

        try:
            self._state_dim = cfg.input_features[OBS_STATE].shape[0]
        except Exception:
            self._state_dim = 2  # PushT default

        # n_action_steps: how many actions to dequeue per select_action call.
        # DiffusionPolicy default is typically 8 (out of a 16-step chunk).
        self._n_action_steps = getattr(cfg, "n_action_steps", 8)
        self._chunk_size = getattr(cfg, "chunk_size", 16)

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
            "DiffusionPolicy ready: model=%s, action_dim=%d, state_dim=%d, "
            "n_action_steps=%d, chunk_size=%d, image=%dx%d",
            model_id,
            self._action_dim,
            self._state_dim,
            self._n_action_steps,
            self._chunk_size,
            self._image_h,
            self._image_w,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from PIL import Image
        from torchvision import transforms

        # lerobot/diffusion_pusht was trained with:
        #   - Images resized to 96×96 then center-cropped to 84×84 (crop done inside encoder)
        #   - ImageNet mean/std normalization (stored as normalize_inputs.buffer_observation_image.*)
        #   - State min-max normalized to [-1, 1] (buffer_observation_state.max/min)
        #   - Actions min-max normalized to [-1, 1] (buffer_action.max/min)
        # lerobot 0.4.4's from_pretrained does NOT load these normalization buffers
        # ("Unexpected key(s)" warning), so we must apply them manually.
        _IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
        _IMG_STD = [0.229, 0.224, 0.225]  # ImageNet std
        _STATE_MIN = np.array([13.456, 32.938], dtype=np.float32)
        _STATE_MAX = np.array([496.146, 510.958], dtype=np.float32)
        _ACT_MIN = np.array([12.0, 25.0], dtype=np.float32)
        _ACT_MAX = np.array([511.0, 511.0], dtype=np.float32)

        resize = transforms.Compose(
            [
                transforms.Resize((self._image_h, self._image_w)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMG_MEAN, std=_IMG_STD),
            ]
        )

        # Decode primary camera image.
        img_b64 = obs.images.get("primary") or obs.images.get("image") or ""
        if img_b64:
            pil_img = Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB")
        else:
            # Fallback: blank image of correct size.
            pil_img = Image.new("RGB", (self._image_w, self._image_h), color=(128, 128, 128))
        img_tensor = resize(pil_img)  # (C, H, W)

        # Robot state — 2-dim for PushT (agent x,y in pixel space [0,512]).
        # Normalize to [-1, 1] using training dataset statistics.
        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        state_raw = np.array(state_list[: self._state_dim], dtype=np.float32)
        state_norm = 2.0 * (state_raw - _STATE_MIN) / (_STATE_MAX - _STATE_MIN) - 1.0
        state_tensor = torch.tensor(state_norm, dtype=torch.float32)

        # Build observation batch.
        # lerobot 0.4.4 select_action expects:
        #   image:  (B, C, H, W)  — NO explicit temporal dim (queue handled internally)
        #   state:  (B, state_dim)
        dev = torch.device(self._device)
        obs_batch = {
            self._camera_key: img_tensor.unsqueeze(0).to(dev),  # (1, C, H, W)
            "observation.state": state_tensor.unsqueeze(0).to(dev),  # (1, state_dim)
        }

        with torch.no_grad():
            action = self._policy.select_action(obs_batch)

        # select_action returns (action_dim,) or (B, action_dim).
        if not isinstance(action, np.ndarray):
            action = action.cpu().numpy()
        action = np.atleast_1d(action.squeeze())

        # Unnormalize action from [-1, 1] back to pixel space [0, 512].
        action_unnorm = (action + 1.0) / 2.0 * (_ACT_MAX - _ACT_MIN) + _ACT_MIN
        return [action_unnorm.tolist()]

    def get_info(self) -> dict:
        return {
            "name": "DiffusionPolicy",
            "model_id": self.model_id,
            "action_space": {
                # "eef_xy" matches GymPushTBackend.get_info() action_space.type,
                # ensuring _negotiate_spaces() identity pass-through.
                "type": "eef_xy" if self._action_dim == 2 else "joint_pos",
                "dim": getattr(self, "_action_dim", 2),
                "description": (
                    "2-dim absolute (x,y) end-effector in PushT pixel space [0,512]"
                    if getattr(self, "_action_dim", 2) == 2
                    else f"{getattr(self, '_action_dim', 2)}-dim action vector"
                ),
            },
            "state_dim": getattr(self, "_state_dim", 2),
            "action_chunk_size": getattr(self, "_n_action_steps", 8),
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": getattr(self, "_state_dim", 2),
                "image_resolution": [self._image_h, self._image_w]
                if hasattr(self, "_image_h")
                else [_PUSHT_IMAGE_SIZE, _PUSHT_IMAGE_SIZE],
                "image_transform": "none",
            },
        }

    def get_action_spec(self) -> dict:
        """Diffusion Policy action spec — 2-dim absolute (x,y) for PushT checkpoint."""
        action_dim = getattr(self, "_action_dim", 2)
        if action_dim == 2:
            # PushT: 2D absolute (x,y) EEF position in pixel space after unnormalization.
            # Format "absolute_xy_position" matches GymPushTBackend action_spec.accepts.
            return {
                "eef_xy": ActionObsSpec(
                    "eef_xy",
                    2,
                    "absolute_xy_position",
                    (0.0, 512.0),
                    description="2-dim absolute (x,y) end-effector position in PushT pixel space [0,512]",
                ),
            }
        else:
            # ALOHA / other: joint position control.
            return {
                "joint_pos": ActionObsSpec(
                    "joint_pos",
                    action_dim,
                    "joint_pos_absolute",
                    (-3.15, 3.15),
                    description=f"{action_dim}-dim joint position absolute",
                ),
            }

    def get_observation_spec(self) -> dict:
        """Diffusion Policy observation spec: top-down RGB + 2-dim state."""
        state_dim = getattr(self, "_state_dim", 2)
        return {
            "primary": ActionObsSpec("image", 0, "rgb_hwc_uint8"),
            "state": ActionObsSpec(
                "state",
                state_dim,
                # "agent_xy_position" matches GymPushTBackend observation_spec state.format.
                "agent_xy_position" if state_dim == 2 else "eef_state",
                description=(
                    "2-dim (x,y) agent position in PushT pixel space [0,512]"
                    if state_dim == 2
                    else f"{state_dim}-dim proprioceptive state"
                ),
            ),
        }

    def reset(self) -> None:
        """Reset the per-episode action queue in the diffusion policy."""
        if getattr(self, "_policy", None) is not None:
            self._policy.reset()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Diffusion Policy Server (lerobot)")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    parser.add_argument("--port", type=int, default=5103)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = DiffusionPolicyServer()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="Diffusion Policy Server",
        max_batch_size=1,  # DDPM denoising not batched
        max_wait_ms=0.0,
    )
    logging.basicConfig(level=logging.INFO)
    print(
        f"[diffusion_policy] Starting on {args.host}:{args.port} "
        f"model={args.model_id} device={args.device}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
