#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "lerobot>=0.4.5",
# ]
# ///
"""Pi 0.5 VLA policy server.

Usage:
    python -m sims.vla_policies.pi05_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Endpoints (see base.py):
    GET  /health  GET  /info  POST /reset  POST /predict
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


class Pi05Policy(VLAPolicyBase):
    """Lerobot Pi 0.5 policy (lerobot/pi05_libero_finetuned and variants)."""

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.utils.constants import ACTION, OBS_STATE

        self.model_id = model_id
        logger.info("Loading Pi 0.5 from %s on %s ...", model_id, device)
        self._policy = PI05Policy.from_pretrained(model_id)
        self._policy.to(torch.device(device))
        self._policy.eval()

        # Disable torch.compile on sample_actions (PyTorch 2.10 bool mask compat)
        inner = self._policy.model
        if hasattr(inner, "sample_actions") and hasattr(
            inner.sample_actions, "_torchdynamo_orig_callable"
        ):
            inner.sample_actions = inner.sample_actions._torchdynamo_orig_callable
            logger.info("Disabled torch.compile on sample_actions")
        else:
            try:
                torch._dynamo.reset()
                torch._dynamo.config.disable = True
            except Exception:
                pass

        cfg = self._policy.config
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self._action_chunk_size = getattr(cfg, "n_action_steps", 1)
        self._action_dim = (
            cfg.output_features[ACTION].shape[0]
            if hasattr(cfg, "output_features") and ACTION in cfg.output_features
            else 7
        )
        self._state_dim = (
            cfg.input_features[OBS_STATE].shape[0]
            if hasattr(cfg, "input_features") and OBS_STATE in cfg.input_features
            else 8
        )
        keys = list(cfg.image_features) if getattr(cfg, "image_features", None) else []
        self._camera_key = keys[0] if keys else "observation.images.image"
        self._camera_key2 = keys[1] if len(keys) > 1 else ""
        # The camera transform is applied before policy inference, so clients
        # should not apply an additional image flip.
        # Keep this explicit to avoid compatibility issues in transform detection.
        self._image_transform = "flip_hw"  # informational only; get_info() overrides this
        self.ready = True
        logger.info(
            "Pi 0.5 ready: model=%s, chunk=%d, action_dim=%d, transform=applied_in_sim (sim-side)",
            model_id, self._action_chunk_size, self._action_dim,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from PIL import Image

        def decode(b64: str):
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0

        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        frame = {
            self._camera_key: decode(obs.images["primary"]),
            "observation.state": torch.tensor(state_list, dtype=torch.float32)[: self._state_dim],
            "task": obs.instruction,
        }
        if obs.images.get("wrist") and self._camera_key2:
            frame[self._camera_key2] = decode(obs.images["wrist"])

        batch = self._preprocessor(frame)
        with torch.no_grad():
            action = self._policy.select_action(batch)
        action = self._postprocessor(action)
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float64)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        return [row[: self._action_dim].tolist() for row in action]

    def get_info(self) -> dict:
        return {
            "name": self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id,
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",
                "dim": getattr(self, "_action_dim", 7),
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper_open]",
            },
            "state_dim": getattr(self, "_state_dim", 8),
            "action_chunk_size": getattr(self, "_action_chunk_size", 1),
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": getattr(self, "_state_dim", 8),
                "image_resolution": [256, 256],
                # The camera transform is applied before policy inference; clients
                # should not apply a second flip.
                "image_transform": "applied_in_sim",
            },
        }

    def get_action_spec(self) -> dict:
        """Pi 0.5 action spec: delta EEF with axis-angle rotation, LIBERO gripper (neg=close)."""
        return {
            "position": POSITION_DELTA,
            "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict:
        """Pi 0.5 observation spec: primary + wrist RGB, 8-dim eef state, language."""
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

    parser = argparse.ArgumentParser(description="Pi 0.5 Policy Server")
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = Pi05Policy()
    app = make_app(policy, args.model_id, args.device, title="Pi 0.5 Policy Server")
    logging.basicConfig(level=logging.INFO)
    print(f"[pi05_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
