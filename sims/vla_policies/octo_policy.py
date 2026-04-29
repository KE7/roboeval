#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "jax>=0.4.25",
#   "jaxlib>=0.4.25",
# ]
# ///
# NOTE: Octo is a JAX-based multi-embodiment VLA (NOT PyTorch / lerobot).
# The `octo` package is NOT on PyPI — install via:
#   pip install git+https://github.com/octo-models/octo
# Checkpoint: rail-berkeley/octo-small-1.5 (HuggingFace, ~3.5 GB).
#
# JAX on aarch64 (NVIDIA GB10):
#   CPU: pip install jax jaxlib   (manylinux_2_27_aarch64 wheels on PyPI)
#   GPU: pip install "jax[cuda12]" (installs jax-cuda12-pjrt + jax-cuda12-plugin
#        from PyPI; confirmed for aarch64 since jax>=0.4.26)
#
# Octo-small-1.5 is trained on the Open-X-Embodiment (OXE) mixture: Bridge V2,
# RT-1, Language-Table, and others.  Use this policy for compatibility checks
# unless the target environment matches one of the training domains.
"""Octo policy server (JAX-based multi-embodiment VLA).

Usage:
    python -m sims.vla_policies.octo_policy [--model-id MODEL] [--port PORT]

Default checkpoint: rail-berkeley/octo-small-1.5
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
from io import BytesIO

import numpy as np
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation
from robo_eval.specs import (
    ActionObsSpec,
    POSITION_DELTA,
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
)

logger = logging.getLogger(__name__)

# Octo-small-1.5 outputs 7-dim actions: [dx, dy, dz, drx, dry, drz, gripper]
# Training datasets include Bridge V2 (absolute EEF) and others (delta EEF).
# The base checkpoint normalizes internally — treated as delta EEF here.
_ACTION_DIM = 7

# History window: Octo models maintain a fixed-length observation history.
# octo-small-1.5 was trained with window_size=2 (current + 1 prior frame).
_WINDOW_SIZE = 2

# Resize target: all Octo checkpoints use 256×256 primary RGB.
_IMG_SIZE = (256, 256)


class OctoPolicy(VLAPolicyBase):
    """Octo (rail-berkeley/octo-small-1.5) — JAX multi-embodiment VLA.

    Architecture: Transformer + diffusion head.
    Training: OXE mixture (Bridge V2, RT-1, Language-Table, …).
    Input: primary RGB image + text instruction.
    Output: 7-dim action chunk (1 step per call).

    Limitation: out-of-distribution environments are useful for compatibility
    checks, but not for model accuracy claims.
    """

    load_in_background = True

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._rng = None
        # Rolling image history buffer: list of np.uint8 (H, W, 3)
        self._history: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(self, model_id: str, device: str = "cuda", **_) -> None:
        """Load Octo checkpoint from HuggingFace Hub (offline-first)."""
        self.model_id = model_id
        logger.info("Loading Octo model: %s", model_id)

        # JAX device selection: GPU if available, else CPU.
        # We do NOT set CUDA_VISIBLE_DEVICES here — the caller is responsible.
        try:
            import jax  # noqa: F401
        except ImportError as exc:
            self.load_error = (
                "jax not installed — run: pip install jax jaxlib  "
                "(CPU) or  pip install 'jax[cuda12]'  (GPU, aarch64+x86)"
            )
            logger.error("Octo load failed: %s", self.load_error)
            return

        try:
            from octo.model.octo_model import OctoModel  # type: ignore[import]
        except ImportError as exc:
            self.load_error = (
                "octo not installed — run: "
                "pip install git+https://github.com/octo-models/octo"
            )
            logger.error("Octo load failed: %s", self.load_error)
            return

        # Use HF-style URI (hf://org/repo) — OctoModel handles the download.
        # Respect TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE like the other policies.
        hf_uri = model_id
        if not hf_uri.startswith("hf://"):
            hf_uri = f"hf://{model_id}"

        try:
            self._model = OctoModel.load_pretrained(hf_uri)
        except Exception as exc:
            self.load_error = f"OctoModel.load_pretrained failed: {exc}"
            logger.error("Octo load failed: %s", exc)
            return

        import jax
        import jax.numpy as jnp  # noqa: F401

        self._rng = jax.random.PRNGKey(0)
        self._history = []

        jax_devices = jax.devices()
        device_types = [str(d) for d in jax_devices]
        logger.info("Octo ready — JAX devices: %s", device_types)
        self.ready = True

    # ------------------------------------------------------------------
    # Per-episode reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear observation history at episode start."""
        self._history = []
        if self._rng is not None:
            import jax
            # Advance RNG each episode so actions are not deterministically identical.
            self._rng, _ = jax.random.split(self._rng)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run one Octo inference step; returns a single 7-dim action."""
        import jax
        import numpy as np
        from PIL import Image

        # ── Decode primary image ──────────────────────────────────────
        raw = base64.b64decode(obs.images["primary"])
        pil_img = Image.open(BytesIO(raw)).convert("RGB").resize(_IMG_SIZE)
        img_np = np.array(pil_img, dtype=np.uint8)  # (256, 256, 3)

        # Maintain rolling window history
        self._history.append(img_np)
        if len(self._history) > _WINDOW_SIZE:
            self._history = self._history[-_WINDOW_SIZE:]

        # Pad short history by repeating first frame
        pad_len = _WINDOW_SIZE - len(self._history)
        padded = [self._history[0]] * pad_len + self._history  # list of (H,W,3)

        # (1, T, H, W, C) — batch=1, time=window
        image_seq = np.stack(padded, axis=0)[None]  # (1, T, H, W, C)
        timestep_pad_mask = np.ones((1, _WINDOW_SIZE), dtype=bool)

        observation = {
            "image_primary": image_seq,
            "timestep_pad_mask": timestep_pad_mask,
        }

        # ── Language task ─────────────────────────────────────────────
        task = self._model.create_tasks(texts=[obs.instruction])

        # ── Advance RNG ───────────────────────────────────────────────
        self._rng, sample_rng = jax.random.split(self._rng)

        # ── Sample action ─────────────────────────────────────────────
        # action shape: (1, pred_horizon, action_dim) — we take the first step.
        actions = self._model.sample_actions(
            observation,
            task,
            unnormalize_actions=True,
            rng=sample_rng,
        )
        action = np.array(actions[0, 0], dtype=np.float64).flatten()[:_ACTION_DIM]

        # Gripper convention normalisation:
        # Octo Bridge-trained gripper: 1.0 = open, 0.0 = close.
        # roboeval convention (GRIPPER_CLOSE_NEG): +1 = open, -1 = close.
        # Map: value > 0.5 → open (+1), else → close (-1).
        gripper_raw = float(action[-1])
        action[-1] = 1.0 if gripper_raw > 0.5 else -1.0

        return [action.tolist()]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        name = (self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id) or "octo"
        return {
            "name": name,
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",
                "dim": _ACTION_DIM,
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]",
            },
            "state_dim": 0,          # Octo does not consume robot state
            "action_chunk_size": 1,  # single step per call (pred_horizon[0])
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": 0,
                "image_resolution": list(_IMG_SIZE),
                "image_transform": "none",
            },
            "notes": (
                "Octo-small-1.5 is trained on OXE (Bridge V2, RT-1, …). "
                "No in-distribution sim in v0.1 — infra smoke only. "
                "See configs/bridge_octo_smoke.yaml."
            ),
        }

    def get_action_spec(self) -> dict:
        """Octo action spec: delta EEF, binary gripper (neg=close convention)."""
        return {
            "position": POSITION_DELTA,
            "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict:
        """Octo observation spec: primary RGB only, language instruction."""
        return {
            "primary": IMAGE_RGB,
            "instruction": LANGUAGE,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Octo Policy Server (JAX)")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("OCTO_MODEL_ID", "rail-berkeley/octo-small-1.5"),
        help="HuggingFace model ID (e.g. rail-berkeley/octo-small-1.5)",
    )
    parser.add_argument("--port", type=int, default=5106)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Hint only — JAX selects GPU automatically when available.",
    )
    args = parser.parse_args()

    policy = OctoPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="Octo Policy Server",
    )

    logging.basicConfig(level=logging.INFO)
    print(f"[octo_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    print("[octo_policy] JAX model loads in background — poll GET /health for ready:true")
    print("[octo_policy] GPU: JAX auto-selects accelerator. CPU fallback is available.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
