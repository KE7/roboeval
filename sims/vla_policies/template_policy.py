#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   # Add your model-specific deps here, e.g.:
#   # "transformers>=4.40",
#   # "lerobot>=0.4.5",
# ]
# ///
"""Template VLA policy server — copy this file and fill in the TODOs.

See docs/extending.md for the full walkthrough.

Usage:
    python -m sims.vla_policies.your_policy [--model-id MODEL] [--port PORT]
"""

from __future__ import annotations

import argparse
import logging

from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

_MODEL_ID_DEFAULT = "your-org/your-model"


class YourPolicy(VLAPolicyBase):
    """TODO: Replace YourPolicy with your model's name and fill in the three methods."""

    # Set True to load in a daemon thread (server immediately reachable; returns
    # HTTP 503 on /predict until ready).  Leave False for small/fast models.
    load_in_background: bool = False

    def load_model(self, model_id: str, device: str, **kwargs) -> None:
        """Load model weights and set self.ready = True on success.

        kwargs contains any extra flags forwarded from make_app() or the CLI.

        Example:
            import torch
            self.model_id = model_id
            self._model = YourModel.from_pretrained(model_id).to(device).eval()
            self._action_dim  = 7
            self._state_dim   = 8
            self._chunk_size  = 1
            self.ready = True
        """
        raise NotImplementedError("TODO: implement load_model()")

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run inference and return a list of action vectors.

        Input helpers:
            obs.images["primary"]   – base64 PNG, primary camera (required)
            obs.images.get("wrist") – base64 PNG, wrist camera (optional)
            obs.instruction         – natural-language task string
            obs.state.get("flat")   – list[float] proprioceptive state (optional)

        Returns:
            List of action vectors (length = action_chunk_size).
            Each vector is list[float] of length action_dim.
            For LIBERO EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            Actions must be UNNORMALIZED (real-space).

        Example:
            import base64, torch
            from io import BytesIO
            from PIL import Image
            raw   = base64.b64decode(obs.images["primary"])
            img   = Image.open(BytesIO(raw)).convert("RGB")
            state = obs.state.get("flat") or [0.0] * self._state_dim
            # ... run your model ...
            return [[float(a) for a in action_vec]]
        """
        raise NotImplementedError("TODO: implement predict()")

    def get_info(self) -> dict:
        """Return model metadata (returned from GET /info and used by env_wrapper)."""
        return {
            "name": "your-model",  # TODO: short display name
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",  # TODO: or "joint_pos"
                "dim": 7,  # TODO: match your model's output dim
            },
            "state_dim": 8,  # TODO: 0 if model ignores state
            "action_chunk_size": 1,  # TODO: actions returned per /predict call
        }

    def reset(self) -> None:
        """Reset per-episode state (e.g. action queue). Default: no-op."""
        # TODO: if your model has a queue or hidden state, clear it here.
        # Example: self._policy.reset()
        pass


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Template VLA Policy Server")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    parser.add_argument("--port", type=int, default=5103)  # TODO: pick your port
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    # TODO: add model-specific args here, e.g.:
    # parser.add_argument("--unnorm-key", dest="unnorm_key", default="libero_spatial")
    args = parser.parse_args()

    policy = YourPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="Template VLA Policy Server",
        # TODO: forward model-specific args as kwargs, e.g.:
        # unnorm_key=args.unnorm_key,
    )
    logging.basicConfig(level=logging.INFO)
    print(f"[template_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
