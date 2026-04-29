#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   # TODO [STEP 1]: add your model-specific dependencies here, e.g.:
#   # "transformers>=4.40",
#   # "lerobot>=0.4.5",
# ]
# ///
"""Template VLA policy server — copy this file and fill in the four TODO sections.

Quick-start (3 steps for a working server):

    1. Rename ``MyModelPolicy`` to something meaningful.
    2. Fill in ``load_model()``, ``predict()``, and ``get_info()`` (TODO markers below).
    3. Launch:
           python -m sims.vla_policies.my_model_policy --model-id my-org/my-model

The server exposes four HTTP endpoints automatically via ``make_app()``:

    GET  /health  → {"status": "ok", "ready": true/false, "model_id": "..."}
    GET  /info    → result of get_info() (+ action_spec / observation_spec if declared)
    POST /reset   → {"success": true}           # calls reset() before each episode
    POST /predict → {"actions": [[...]], ...}   # calls predict() for inference

See docs/extending.md for the full walkthrough and pitfall list.
"""

from __future__ import annotations

import argparse
import logging

# ── ActionObsSpec (optional but recommended) ───────────────────────────────────────
# Declaring typed specs lets the orchestrator validate VLA × sim compatibility
# before running evaluations with mismatched conventions.
# Pre-built constants cover the most common formats; compose custom ActionObsSpecs
# for anything else.  Import only what you need.
from robo_eval.specs import (
    ActionObsSpec,
)

# ── Framework imports ─────────────────────────────────────────────────────────
# VLAPolicyBase: abstract class that wires your three methods to HTTP endpoints.
# make_app():    FastAPI factory — wraps your policy in the standard 4-endpoint app.
from sims.vla_policies.base import VLAPolicyBase, make_app

# VLAObservation: the observation payload your predict() receives.
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

# TODO [STEP 2a]: set a sensible default model ID for your model
_MODEL_ID_DEFAULT = "your-org/your-model"


# =============================================================================
# REQUIRED HOOK 1 — Subclass VLAPolicyBase
# =============================================================================
class MyModelPolicy(VLAPolicyBase):
    """TODO [STEP 2b]: Replace MyModelPolicy with your model's name and fill the
    three abstract methods below.  Everything else (HTTP wiring, batching,
    lifespan, error handling) is handled by the base class + make_app().
    """

    # ── Class-level flags (change if needed) ─────────────────────────────────
    # Set True to load in a background daemon thread. The server will be
    # immediately reachable on /health and /info but return HTTP 503 on /predict
    # until self.ready = True.  Good for large models (>10 GB) where startup
    # time would otherwise delay the health-check poll.
    load_in_background: bool = False

    # Set True when you override predict_batch() with a real GPU-batched
    # implementation (single forward pass for N observations).  Leave False
    # to use the serial fallback in VLAPolicyBase.
    supports_batching: bool = False

    # =========================================================================
    # REQUIRED HOOK 2 — load_model
    # =========================================================================
    def load_model(self, model_id: str, device: str, **kwargs) -> None:
        """Load model weights.  Called once at server startup inside the ASGI
        lifespan (or in a daemon thread when load_in_background=True).

        Contract:
          - Store the model on ``self``.
          - Set ``self.model_id = model_id`` so /health can report it.
          - Set ``self.ready = True`` at the end on success.
          - Raise (or set ``self.load_error``) on failure — the base class
            catches exceptions and reports them via /health.

        ``kwargs`` contains any extra flags forwarded from the ``make_app()``
        call or the CLI (e.g. ``unnorm_key``).

        Example (HuggingFace model):
        ─────────────────────────────
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            self.model_id = model_id
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = (
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
                .to(device)
                .eval()
            )
            self._action_dim  = 7    # number of action dimensions your model outputs
            self._state_dim   = 8    # proprioceptive state dimensions expected (0 = ignore)
            self._chunk_size  = 1    # actions returned per predict() call
            self.ready = True
        """
        # TODO [STEP 3]: replace the NotImplementedError with real loading code
        raise NotImplementedError("implement load_model()")

    # =========================================================================
    # REQUIRED HOOK 3 — predict
    # =========================================================================
    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run inference for a single observation.

        Input:
            obs.images["primary"]    – base64 PNG/JPEG, primary camera (always present)
            obs.images.get("wrist")  – base64 PNG/JPEG, wrist camera (may be absent)
            obs.images.get("secondary") – base64 PNG/JPEG, side camera (may be absent)
            obs.instruction          – natural-language task string
            obs.state.get("flat")    – list[float] of proprioceptive state
                                       (may be an empty dict if the sim doesn't send state)

        Output:
            A list of action vectors.
            - Length = action_chunk_size (1 for single-step models).
            - Each inner list has length = action_dim.
            - Values must be UNNORMALIZED (real-world units / radians / meters).
            - For standard LIBERO EEF delta: [dx, dy, dz, droll, dpitch, dyaw, gripper]
              where gripper = -1.0 (close) or +1.0 (open).

        ── Image transform contract ───────────────────────────────────────────
        LIBERO cameras are physically mounted upside-down and require a 180°
        flip.  The flip has ALREADY been applied by the simulator backend
        before the image reaches this function — do NOT apply it again here.
        Your model should consume the images as-is.

        If your model was trained on a dataset that stores images WITHOUT the
        flip (e.g. raw RoboCasa, RoboTwin), do not apply any transform.
        Document the expected orientation in ``get_info()`` under obs_requirements.

        Example (HuggingFace VLM):
        ──────────────────────────
            from PIL import Image

            raw   = base64.b64decode(obs.images["primary"])
            img   = Image.open(BytesIO(raw)).convert("RGB")
            state = obs.state.get("flat") or [0.0] * self._state_dim
            prompt = f"Task: {obs.instruction}\\nAction:"

            with torch.no_grad():
                output = self._model.predict_action(img, prompt, state)

            action = np.array(output, dtype=np.float64).flatten()[:self._action_dim]
            return [action.tolist()]
        """
        # TODO [STEP 4]: replace the NotImplementedError with real inference code
        raise NotImplementedError("implement predict()")

    # =========================================================================
    # REQUIRED HOOK 4 — get_info
    # =========================================================================
    def get_info(self) -> dict:
        """Return model metadata exposed via GET /info.

        This dict is used by the orchestrator and env_wrapper to:
          - Confirm the action space matches the simulator.
          - Decide the image resolution to request from the sim.
          - Apply (or skip) image transforms.
          - Log the model identity for result files.

        Required keys:
            name              – short display name (no slashes)
            model_id          – full model identifier (HF repo or local path)
            action_space.type – "eef_delta" | "joint_pos" | "eef_absolute" | …
            action_space.dim  – number of action dimensions
            action_chunk_size – number of action steps returned per /predict call
            state_dim         – proprioceptive state dimensions consumed (0 = ignore state)

        Optional but recommended:
            obs_requirements.image_transform – "applied_in_sim" | "none" | "flip_hw"
                "applied_in_sim": the sim already flipped the image; skip client-side flip.
                "none":          no flip needed for this model.
                "flip_hw":       env_wrapper should flip before sending to predict().

        TODO [STEP 5]: fill in the correct values for your model.
        """
        return {
            "name": "my-model",  # TODO: short display name
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",  # TODO: or "joint_pos", "eef_absolute", …
                "dim": 7,  # TODO: your model's output dimensionality
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]",  # TODO
            },
            "state_dim": 8,  # TODO: 0 if your model ignores proprioceptive state
            "action_chunk_size": 1,  # TODO: actions per /predict call (1 for single-step)
            "obs_requirements": {
                "cameras": ["primary"],  # TODO: add "wrist" or "secondary" if needed
                "state_dim": 8,  # TODO: match state_dim above
                "image_resolution": [256, 256],  # TODO: what the sim should render
                # CRITICAL: image_transform declaration.
                # "applied_in_sim": the 180° LIBERO flip is done by the sim backend,
                #                   env_wrapper must NOT apply a second flip.
                # "none":           no flip needed (e.g. RoboCasa, RoboTwin models).
                # "flip_hw":        env_wrapper should flip before calling predict().
                "image_transform": "applied_in_sim",  # TODO: match your training data
            },
        }

    # =========================================================================
    # OPTIONAL HOOK — reset (override if your model keeps per-episode state)
    # =========================================================================
    def reset(self) -> None:
        """Reset per-episode hidden state (action queues, attention caches, etc.).

        Called before every new episode via POST /reset.  The default in
        VLAPolicyBase is a no-op — only override if your model is stateful.

        Example:
            if self._policy is not None:
                self._policy.reset()
        """
        pass  # TODO: clear any per-episode buffers here

    # =========================================================================
    # OPTIONAL HOOK — get_action_spec (strongly recommended for new models)
    # =========================================================================
    def get_action_spec(self) -> dict[str, ActionObsSpec] | None:
        """Return the typed action spec (what this model PRODUCES).

        The orchestrator calls check_specs() with this dict and the benchmark's
        action_spec to catch convention mismatches (wrong rotation format,
        inverted gripper sign, delta-vs-absolute) before evaluation starts.

        Return None to opt out of spec checking (legacy / unknown format).

        Example — standard LIBERO EEF delta model:
            return {
                "position": POSITION_DELTA,   # 3-dim delta_xyz, range [-1, 1]
                "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
                "gripper":  GRIPPER_CLOSE_NEG,  # -1=close, +1=open
            }
        """
        # TODO: declare your action spec or return None to skip validation
        return None

    # =========================================================================
    # OPTIONAL HOOK — get_observation_spec (strongly recommended for new models)
    # =========================================================================
    def get_observation_spec(self) -> dict[str, ActionObsSpec] | None:
        """Return the typed observation spec (what this model CONSUMES).

        The orchestrator uses this to confirm the sim provides the required
        cameras, state format, and language instruction.

        Return None to opt out of spec checking.

        Example — model using primary + wrist cameras and 8-dim EEF state:
            return {
                "primary":     IMAGE_RGB,
                "wrist":       IMAGE_RGB,
                "state":       ActionObsSpec("state", 8, "libero_eef_pos3_aa3_grip2"),
                "instruction": LANGUAGE,
            }
        """
        # TODO: declare your observation spec or return None to skip validation
        return None


# =============================================================================
# Entry point
# =============================================================================
def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="My Model VLA Policy Server")
    parser.add_argument(
        "--model-id", default=_MODEL_ID_DEFAULT, help="HuggingFace repo or local path to the model."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5106,  # TODO: pick a unique port
        help="Port to serve on.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--device", default="cuda", help="Torch device string (cuda, cpu, cuda:1, …)."
    )
    # TODO: add model-specific CLI args here, e.g.:
    # parser.add_argument("--unnorm-key", dest="unnorm_key", default="libero_spatial")
    args = parser.parse_args()

    policy = MyModelPolicy()

    # make_app() wires the four standard endpoints to policy and returns a
    # FastAPI app.  Model loading happens inside the ASGI lifespan.
    # Extra kwargs are forwarded verbatim to load_model(**kwargs).
    app = make_app(
        policy,
        model_id=args.model_id,
        device=args.device,
        title="My Model Policy Server",
        # TODO: forward model-specific args as kwargs, e.g.:
        # unnorm_key=args.unnorm_key,
    )

    logging.basicConfig(level=logging.INFO)
    print(f"[my_model_policy] Starting on {args.host}:{args.port}  model={args.model_id}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
