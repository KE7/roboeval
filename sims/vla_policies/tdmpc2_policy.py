#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "tensordict>=0.6",
#   # Optional upstream tdmpc2 package.
#   # "tdmpc2>=1.0.0.dev0",
# ]
# ///
"""TDMPC2 (Temporal Difference Model-Predictive Control v2) policy server.

Usage:
    python -m sims.vla_policies.tdmpc2_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Background
----------
TDMPC2 (Hansen et al. 2024, https://www.tdmpc2.com) is a model-based
reinforcement-learning algorithm.  The
agent learns a latent world model and plans short-horizon trajectories with
MPPI at every step, falling back to a learned Q-function for longer-horizon
value estimation.

Compatibility
-------------
1. **Action-dim match.**  The MT80 metaworld checkpoint uses Sawyer's native
   4-dim end-effector control ([dx, dy, dz, gripper]).  The metaworld sim
   backend declares the same 4-dim eef_delta contract — the
   ``ActionObsSpec`` gate accepts the pairing exactly, so metaworld is
   a direct action-space match.
2. **Paradigm diversity.**  TDMPC2 provides a model-based RL policy alongside
   imitation-learning policy servers.

Checkpoints
-----------
The default checkpoint targets metaworld button-press-v2 (single-task):
    nicklashansen/tdmpc2  (HF model card, MIT licence)
    file: metaworld/mw-button-press-1.pt

Override via env var TDMPC2_CHECKPOINT (e.g. "metaworld/mw-reach-1.pt")
or via model_id colon notation: "nicklashansen/tdmpc2:metaworld/mw-reach-1.pt".

Loading strategy
----------------
1. Try ``lerobot.policies.tdmpc2.modeling_tdmpc2.TDMPC2Policy``.
2. Fall back to ``lerobot.common.policies.tdmpc2.modeling_tdmpc2.TDMPC2Policy``
   (older lerobot path).
3. Fall back to an installed ``tdmpc2`` package compatible with the upstream
   nicklashansen implementation.

Action space
------------
* Format: ``eef_delta_xyz_gripper`` (4-dim continuous, range ~[-1, 1]).
* Compatible sim: metaworld (Sawyer eef-delta).

Observation space
-----------------
* 39-dim metaworld_obs proprioceptive state vector (upstream checkpoint).
* Language instruction is accepted but ignored (TDMPC2 is task-conditioned
  via task ID embedding, not free text).

Note: The upstream single-task metaworld checkpoints are state-based (no
image encoder), so the RGB camera observation is accepted but ignored.
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np

from roboeval.specs import (
    IMAGE_RGB,
    LANGUAGE,
    ActionObsSpec,
)
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

_MODEL_ID_DEFAULT = "nicklashansen/tdmpc2"

# TDMPC2 metaworld defaults (Sawyer eef-delta).
_ACTION_DIM = 4
_STATE_DIM = 39
_CHUNK_SIZE = 1  # MPC re-plans every step (planning horizon is internal)

# Default single-task checkpoint (metaworld button-press-v2 seed-1, 5M params).
# Override with TDMPC2_CHECKPOINT env var or by passing model_id with colon notation.
_DEFAULT_CHECKPOINT = "metaworld/mw-button-press-1.pt"


def _make_upstream_cfg(action_dim: int = 4, state_dim: int = 39, episode_length: int = 500):
    """Build a SimpleNamespace config for the upstream TDMPC2 single-task model.

    Architecture matches MODEL_SIZE[5] (5M params), used by the single-task
    metaworld checkpoints in nicklashansen/tdmpc2.
    """
    import types

    cfg = types.SimpleNamespace(
        obs="state",
        obs_shape={"state": [state_dim]},
        action_dim=action_dim,
        episode_length=episode_length,
        multitask=False,
        task_dim=0,
        tasks=["mw-button-press"],
        obs_shapes={"state": [state_dim]},
        action_dims=[action_dim],
        episode_lengths=[episode_length],
        # Architecture (MODEL_SIZE[5])
        enc_dim=256,
        mlp_dim=512,
        latent_dim=512,
        num_enc_layers=2,
        num_q=5,
        # Discrete regression bins
        num_bins=101,
        vmin=-10,
        vmax=10,
        bin_size=20.0 / 100.0,
        # Regularization
        dropout=0.01,
        simnorm_dim=8,
        log_std_min=-10,
        log_std_max=2,
        # MPPI planning
        mpc=True,
        iterations=6,
        num_samples=512,
        num_elites=64,
        num_pi_trajs=24,
        horizon=3,
        min_std=0.05,
        max_std=2.0,
        temperature=0.5,
        # Discount schedule
        discount_denom=5,
        discount_min=0.95,
        discount_max=0.995,
        # Training (not used for inference but needed by __init__)
        lr=3e-4,
        enc_lr_scale=0.3,
        episodic=False,
        compile=False,
    )
    cfg.get = lambda key, default=None: getattr(cfg, key, default)
    return cfg


class TDMPC2Policy(VLAPolicyBase):
    """TDMPC2 model-based RL agent paired with the metaworld sim backend.

    Loading strategy
    ----------------
    1. Try ``lerobot.policies.tdmpc2.modeling_tdmpc2.TDMPC2Policy``.
    2. Fall back to ``lerobot.common.policies.tdmpc2.modeling_tdmpc2.TDMPC2Policy``
       (older lerobot path).
    3. Fall back to an installed upstream-compatible ``tdmpc2`` package.

    Inference (backend = "tdmpc2_upstream")
    ----------------------------------------
    TDMPC2 plans with MPPI internally each call; from the server's point of
    view ``predict()`` returns a single 4-dim action vector.  ``reset()``
    flushes the planner's per-episode trajectory cache (``_prev_mean``).

    The upstream single-task metaworld checkpoints are state-based — they
    use only the 39-dim proprioceptive observation (no image).
    """

    supports_batching: bool = False

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch

        self.model_id = model_id
        logger.info("Loading TDMPC2 from %s on %s ...", model_id, device)

        loaded = False
        load_err = None
        self._t0 = True  # Episode-start flag for MPPI warm-start

        # Path 1: primary lerobot policy wrapper.
        try:
            from lerobot.policies.tdmpc2.modeling_tdmpc2 import (  # type: ignore[import-not-found]
                TDMPC2Policy as _TDMPC2,
            )

            self._policy = _TDMPC2.from_pretrained(model_id)
            self._backend = "lerobot.policies.tdmpc2"
            loaded = True
        except Exception as e:
            load_err = e

        # Path 2: lerobot common.* legacy path
        if not loaded:
            try:
                from lerobot.common.policies.tdmpc2.modeling_tdmpc2 import (  # type: ignore[import-not-found]
                    TDMPC2Policy as _TDMPC2,
                )

                self._policy = _TDMPC2.from_pretrained(model_id)
                self._backend = "lerobot.common.policies.tdmpc2"
                loaded = True
            except Exception as e:
                load_err = e

        # Path 3: upstream-compatible nicklashansen/tdmpc2 package.
        if not loaded:
            try:
                import tdmpc2 as _tdmpc2_pkg  # type: ignore[import-not-found]
                from huggingface_hub import hf_hub_download

                # Resolve repo and checkpoint path.
                # Supports:
                #   "nicklashansen/tdmpc2"               → use TDMPC2_CHECKPOINT env or default
                #   "nicklashansen/tdmpc2:metaworld/..."  → explicit sub-path
                if ":" in model_id:
                    repo_id, ckpt_path = model_id.split(":", 1)
                else:
                    repo_id = model_id
                    ckpt_path = os.environ.get("TDMPC2_CHECKPOINT", _DEFAULT_CHECKPOINT)

                logger.info("TDMPC2 upstream: repo=%s  checkpoint=%s", repo_id, ckpt_path)

                # Instantiate model with single-task metaworld config (5M params).
                cfg = _make_upstream_cfg(
                    action_dim=_ACTION_DIM,
                    state_dim=_STATE_DIM,
                    episode_length=500,
                )
                agent = _tdmpc2_pkg.TDMPC2(cfg)

                # Download checkpoint from HuggingFace Hub.
                local_path = hf_hub_download(repo_id, ckpt_path)
                agent.load(local_path)
                logger.info("TDMPC2 checkpoint loaded from %s", local_path)

                self._policy = agent
                self._backend = "tdmpc2_upstream"
                loaded = True
            except Exception as e:
                load_err = e

        if not loaded:
            raise RuntimeError(
                f"Could not load TDMPC2 from {model_id}: tried lerobot.policies.tdmpc2, "
                f"lerobot.common.policies.tdmpc2, and upstream tdmpc2 package. "
                f"Last error: {load_err!r}"
            )

        # Backend-specific setup.
        if self._backend == "tdmpc2_upstream":
            # Upstream TDMPC2 hardcodes cuda:0 internally; do not try to move it.
            self._device = device
            self._action_dim = _ACTION_DIM
            self._state_dim = _STATE_DIM
        else:
            self._policy.to(torch.device(device))
            self._policy.eval()
            self._device = device

            cfg = getattr(self._policy, "config", None)
            try:
                from lerobot.utils.constants import ACTION, OBS_STATE

                self._action_dim = cfg.output_features[ACTION].shape[0]
                self._state_dim = cfg.input_features[OBS_STATE].shape[0]
            except Exception:
                self._action_dim = _ACTION_DIM
                self._state_dim = _STATE_DIM

            try:
                image_keys = (
                    list(cfg.image_features)
                    if (cfg is not None and getattr(cfg, "image_features", None))
                    else []
                )
            except Exception:
                image_keys = []
            self._camera_key = image_keys[0] if image_keys else "observation.images.corner"

        self._chunk_size = _CHUNK_SIZE
        self.ready = True
        logger.info(
            "TDMPC2 ready: model=%s, backend=%s, action_dim=%d, state_dim=%d",
            model_id,
            self._backend,
            self._action_dim,
            self._state_dim,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run a single TDMPC2 MPC planning step and return a 4-dim action."""
        import torch

        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        state_tensor = torch.tensor(state_list, dtype=torch.float32)[: self._state_dim]

        if self._backend == "tdmpc2_upstream":
            # Upstream API: act(obs, t0, eval_mode) — state tensor on cuda:0.
            state_tensor = state_tensor.to("cuda:0")
            action = self._policy.act(state_tensor, t0=self._t0, eval_mode=True)
            self._t0 = False  # Only True for the first step of each episode
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
        else:
            # lerobot API: select_action(batch_dict)
            import base64
            from io import BytesIO

            from PIL import Image
            from torchvision import transforms

            to_tensor = transforms.ToTensor()

            def decode(b64: str) -> torch.Tensor:
                return to_tensor(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

            batch: dict = {
                self._camera_key: decode(obs.images["primary"]).unsqueeze(0),
                "observation.state": state_tensor.unsqueeze(0),
            }
            batch = {
                k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.no_grad():
                action = self._policy.select_action(batch)
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = action.flatten()[: self._action_dim]
        return [action.tolist()]

    def get_info(self) -> dict:
        return {
            "name": "TDMPC2",
            "model_id": self.model_id,
            "action_space": {
                "type": "eef_delta",
                "dim": getattr(self, "_action_dim", _ACTION_DIM),
                "description": ("Sawyer end-effector delta: [dx, dy, dz, gripper]"),
            },
            "state_dim": getattr(self, "_state_dim", _STATE_DIM),
            "action_chunk_size": getattr(self, "_chunk_size", _CHUNK_SIZE),
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": getattr(self, "_state_dim", _STATE_DIM),
                "image_resolution": [224, 224],
                "image_transform": "applied_in_sim",
            },
            "paradigm": "model_based_rl",
        }

    def get_action_spec(self) -> dict:
        """TDMPC2 action spec: 4-dim Sawyer eef-delta — exact match for metaworld."""
        action_dim = getattr(self, "_action_dim", _ACTION_DIM)
        return {
            "eef_delta": ActionObsSpec(
                "eef_delta",
                action_dim,
                "eef_delta_xyz_gripper",
                (-1.0, 1.0),
                description=(
                    "Sawyer 4-dim eef-delta: [dx, dy, dz, gripper] — matches "
                    "metaworld backend exactly"
                ),
            ),
        }

    def get_observation_spec(self) -> dict:
        """TDMPC2 obs spec: primary RGB + 39-dim metaworld state; language accepted-but-ignored."""
        state_dim = getattr(self, "_state_dim", _STATE_DIM)
        return {
            "primary": IMAGE_RGB,
            "state": ActionObsSpec(
                "state",
                state_dim,
                "metaworld_obs",
                description="metaworld 39-dim observation vector (eef + object features)",
            ),
            "instruction": LANGUAGE,
        }

    def reset(self) -> None:
        """Flush per-episode planner state (MPPI warm-start cache)."""
        self._t0 = True  # Mark next predict() call as episode start
        policy = getattr(self, "_policy", None)
        if policy is None:
            return
        backend = getattr(self, "_backend", "")
        if backend == "tdmpc2_upstream":
            # Zero out the MPPI previous-mean buffer to avoid warm-starting
            # from a stale trajectory.
            prev_mean = getattr(policy, "_prev_mean", None)
            if prev_mean is not None:
                prev_mean.zero_()
        elif hasattr(policy, "reset"):
            policy.reset()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="TDMPC2 Policy Server")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    # Port 5108 is reserved for vqbet; tdmpc2 uses 5109.
    parser.add_argument("--port", type=int, default=5109)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = TDMPC2Policy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="TDMPC2 Policy Server",
    )
    logging.basicConfig(level=logging.INFO)
    print(f"[tdmpc2_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
