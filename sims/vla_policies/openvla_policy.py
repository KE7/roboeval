#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "transformers>=4.40",
#   "timm>=0.9.10,<1.0.0",
# ]
# ///
# NOTE: OpenVLA uses HuggingFace AutoProcessor + AutoModelForVision2Seq, NOT lerobot.
# See docs/vla_policy_architecture.md for the full integration contract.
"""OpenVLA policy server.

Usage:
    python -m sims.vla_policies.openvla_policy [--model-id MODEL] [--port PORT] [--unnorm-key KEY]

Extra endpoint beyond the standard four:
    POST /reload {model_id, unnorm_key} → hot-swap checkpoint (blocks until ready)
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
from io import BytesIO

import numpy as np
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from roboeval.specs import (
    GRIPPER_CLOSE_NEG,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    ActionObsSpec,
)
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)
_ACTION_DIM = 7


class OpenVLAPolicy(VLAPolicyBase):
    """OpenVLA (openvla/openvla-7b or finetuned variants).

    Loads in a background thread so the server is immediately reachable.
    Image+text only — robot state is ignored.
    Gripper convention: RLDS (1=close) → LIBERO (-1=close) via binarise+invert.
    """

    load_in_background = True

    def load_model(
        self, model_id: str, device: str, unnorm_key: str = "libero_spatial", **_
    ) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.model_id = model_id
        self._unnorm_key = unnorm_key

        logger.info("Loading OpenVLA processor from %s (offline) ...", model_id)
        try:
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            self.load_error = "model not cached — not downloading"
            logger.error("OpenVLA processor not cached: %s", exc)
            return

        logger.info("Loading OpenVLA model from %s on %s ...", model_id, device)
        try:
            # Keep the default attention implementation for the supported
            # transformers version; forcing eager attention can trigger a
            # causal-mask shape mismatch in this model's custom code.
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:
            self.load_error = "model not cached — not downloading"
            logger.error("OpenVLA model not cached: %s", exc)
            self._processor = None
            return

        self._device = next(self._model.parameters()).device
        self._model.eval()
        logger.info("OpenVLA ready on %s (unnorm_key=%s)", self._device, unnorm_key)
        self.ready = True

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from PIL import Image

        raw = base64.b64decode(obs.images["primary"])
        pil_img = Image.open(BytesIO(raw)).convert("RGB")
        prompt = f"In: What action should the robot take to {obs.instruction}?\nOut:"
        inputs = self._processor(prompt, pil_img).to(self._device, dtype=torch.bfloat16)

        with torch.no_grad():
            action = self._model.predict_action(
                **inputs, unnorm_key=self._unnorm_key, do_sample=False
            )

        action = np.array(action, dtype=np.float64).flatten()[:_ACTION_DIM]
        # Binarise + invert gripper: RLDS (1=close) → LIBERO (-1=close)
        action[-1] = -(1.0 if action[-1] > 0.0 else -1.0)
        return [action.tolist()]

    def get_info(self) -> dict:
        name = (
            self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id
        ) or "openvla"
        return {
            "name": name,
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",
                "dim": _ACTION_DIM,
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]",
            },
            "state_dim": 0,  # OpenVLA ignores robot state
            "action_chunk_size": 1,  # single-step, not chunk-based
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": 0,
                "image_resolution": [256, 256],
                # The camera transform is applied before policy inference; clients
                # should not apply a second flip.
                "image_transform": "applied_in_sim",
            },
        }

    def get_action_spec(self) -> dict:
        """OpenVLA action spec: delta EEF with axis-angle rotation, LIBERO gripper (neg=close).

        Note: openvla converts from RLDS (1=close) → LIBERO (-1=close) internally,
        so the emitted gripper value follows the binary_close_negative convention.
        """
        return {
            "position": POSITION_DELTA,
            "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
            "gripper": GRIPPER_CLOSE_NEG,
        }

    def get_observation_spec(self) -> dict:
        """OpenVLA observation spec: primary RGB only (ignores state and wrist), language."""
        return {
            "primary": IMAGE_RGB,
            "instruction": LANGUAGE,
        }


class _ReloadRequest(BaseModel):
    model_id: str
    unnorm_key: str = "libero_spatial"


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="OpenVLA Policy Server")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("OPENVLA_MODEL_ID", "openvla/openvla-7b-finetuned-libero-spatial"),
    )
    parser.add_argument("--port", type=int, default=5101)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--unnorm-key",
        dest="unnorm_key",
        default="libero_spatial",
        help="Action unnorm key (libero_spatial | libero_10 | bridge_orig …)",
    )
    args = parser.parse_args()

    policy = OpenVLAPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="OpenVLA Policy Server",
        unnorm_key=args.unnorm_key,
    )

    # ------------------------------------------------------------------
    # OpenVLA-specific: hot-swap checkpoint between LIBERO suites
    # ------------------------------------------------------------------
    @app.post("/reload")
    def reload(req: _ReloadRequest):
        import torch

        if (
            policy.model_id == req.model_id
            and getattr(policy, "_unnorm_key", None) == req.unnorm_key
        ):
            return {"success": True, "reloaded": False, "model_id": policy.model_id}

        logger.info(
            "Reloading: %s → %s  (unnorm_key: %s → %s)",
            policy.model_id,
            req.model_id,
            getattr(policy, "_unnorm_key", ""),
            req.unnorm_key,
        )
        policy.ready = False
        for attr in ("_model", "_processor"):
            setattr(policy, attr, None)
        torch.cuda.empty_cache()

        policy.load_model(
            req.model_id,
            str(getattr(policy, "_device", "cuda")),
            unnorm_key=req.unnorm_key,
        )
        if policy.ready:
            return {"success": True, "reloaded": True, "model_id": policy.model_id}
        return JSONResponse(
            status_code=500,
            content={"error": f"Reload failed: {policy.load_error}"},
        )

    logging.basicConfig(level=logging.INFO)
    print(f"[openvla_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    print("[openvla_policy] Model loads in background — poll GET /health for ready:true")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
