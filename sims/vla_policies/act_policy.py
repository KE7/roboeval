#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   "lerobot>=0.4.4",
# ]
# ///
"""ACT (Action Chunking Transformer) policy server via lerobot.

Usage:
    python -m sims.vla_policies.act_policy [--model-id MODEL] [--port PORT] [--device DEVICE]

Canonical checkpoints (gym-aloha bimanual tasks):
    lerobot/act_aloha_sim_transfer_cube_human  ← default; best smoke coverage on AlohaTransferCube-v0
    lerobot/act_aloha_sim_insertion_human      ← AlohaInsertion-v0

Action space:  14-dim absolute joint positions (7 joints × 2 arms).
Observation:   single top-view RGB + 14-dim joint state; NO language conditioning.
Chunk size:    100 actions generated per forward pass; chunk replayed step-by-step.

Design notes
------------
* ACT uses lerobot's ``select_action`` interface, which dequeues one action at a
  time from an internal chunk buffer (``temporal_ensemble_coeff`` controls
  ensembling).  ``reset()`` flushes the buffer between episodes.
* No language conditioning: ACT is a behaviour-cloning model trained on a single
  task; the task is encoded implicitly in the checkpoint.  The ``instruction``
  field is accepted but ignored.
* Image transform: aloha sim cameras do NOT need the 180° LIBERO flip.
  ``image_transform="none"`` is reported in ``/info``.
* GPU coordination: this server can run alongside simulator processes on the
  same GPU when memory allows.
"""

from __future__ import annotations

import argparse
import base64
import logging
from io import BytesIO

import numpy as np

from roboeval.specs import (
    IMAGE_RGB,
    LANGUAGE,
    ActionObsSpec,
)
from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

_MODEL_ID_DEFAULT = "lerobot/act_aloha_sim_transfer_cube_human"

# ACT aloha: 14-dim joint positions (7 per arm, absolute)
_ACTION_DIM = 14
_STATE_DIM = 14
_CHUNK_SIZE = 100  # n_action_steps for aloha ACT checkpoints


class ACTPolicy(VLAPolicyBase):
    """ACT (Action Chunking Transformer) via lerobot, targeting gym-aloha bimanual sim.

    The lerobot ``ACTPolicy.select_action()`` dequeues one action per call from a
    temporal-ensemble buffer of length ``chunk_size``.  The policy replans every
    ``n_action_steps`` executed steps (or immediately if the queue is empty).

    Supported checkpoints
    ---------------------
    * ``lerobot/act_aloha_sim_transfer_cube_human``  (AlohaTransferCube-v0)
    * ``lerobot/act_aloha_sim_insertion_human``      (AlohaInsertion-v0)
    """

    supports_batching: bool = False

    def load_model(self, model_id: str, device: str, **_) -> None:
        import torch

        self.model_id = model_id
        logger.info("Loading ACT from %s on %s ...", model_id, device)

        # Support both old (lerobot.common.policies.*) and new (lerobot.policies.*) paths.
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy as _ACT
        except ImportError:
            from lerobot.common.policies.act.modeling_act import (
                ACTPolicy as _ACT,  # type: ignore[no-redef]
            )

        self._policy = _ACT.from_pretrained(model_id)
        self._policy.to(torch.device(device))
        self._policy.eval()
        self._device = device

        cfg = self._policy.config
        self._chunk_size = getattr(cfg, "chunk_size", _CHUNK_SIZE)
        self._n_action_steps = getattr(cfg, "n_action_steps", self._chunk_size)

        # Infer action and state dims from config or fall back to aloha defaults.
        try:
            from lerobot.utils.constants import ACTION, OBS_STATE

            self._action_dim = cfg.output_features[ACTION].shape[0]
            self._state_dim = cfg.input_features[OBS_STATE].shape[0]
        except Exception:
            self._action_dim = _ACTION_DIM
            self._state_dim = _STATE_DIM

        # Determine image key (ACT aloha checkpoints use "observation.images.top").
        try:
            image_keys = list(cfg.image_features) if getattr(cfg, "image_features", None) else []
        except Exception:
            image_keys = []
        self._camera_key = image_keys[0] if image_keys else "observation.images.top"

        # ── Normalization stats (for backward-compat with old lerobot checkpoints) ──
        # Newer lerobot versions (≥0.4.4) embed normalization inside the model and
        # apply it automatically.  Older checkpoints (like the HuggingFace ALOHA ones)
        # stored stats in separate "normalize_inputs" / "unnormalize_outputs" buffers
        # that are NOT loaded by newer lerobot (they appear as "unexpected keys").
        # We load them manually from the safetensors file so that predict() can apply
        # the correct input / output transforms regardless of lerobot version.
        self._img_mean: torch.Tensor | None = None
        self._img_std: torch.Tensor | None = None
        self._state_mean: torch.Tensor | None = None
        self._state_std: torch.Tensor | None = None
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._needs_manual_norm = False  # set True if old-style buffers found

        try:
            from pathlib import Path as _Path

            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file as _load_sf

            try:
                _ckpt = _Path(snapshot_download(model_id, local_files_only=True))
            except Exception:
                _ckpt = _Path(snapshot_download(model_id))

            _sf = _ckpt / "model.safetensors"
            if _sf.exists():
                _w = _load_sf(str(_sf))
                _img_m = _w.get("normalize_inputs.buffer_observation_images_top.mean")
                _img_s = _w.get("normalize_inputs.buffer_observation_images_top.std")
                _st_m = _w.get("normalize_inputs.buffer_observation_state.mean")
                _st_s = _w.get("normalize_inputs.buffer_observation_state.std")
                _act_m = _w.get("unnormalize_outputs.buffer_action.mean")
                _act_s = _w.get("unnormalize_outputs.buffer_action.std")
                if _act_m is not None and _act_s is not None:
                    self._img_mean = _img_m.float() if _img_m is not None else None
                    self._img_std = _img_s.float() if _img_s is not None else None
                    self._state_mean = _st_m.float() if _st_m is not None else None
                    self._state_std = _st_s.float() if _st_s is not None else None
                    self._action_mean = _act_m.float()
                    self._action_std = _act_s.float()
                    self._needs_manual_norm = True
                    logger.info(
                        "ACT: loaded old-style normalization buffers from checkpoint; "
                        "will apply manual norm/unnorm in predict() "
                        "(action mean[0..4]=%s)",
                        _act_m[:5].tolist(),
                    )
        except Exception as _e:
            logger.warning(
                "ACT: could not load normalization stats (%s); using raw model output", _e
            )

        self.ready = True
        logger.info(
            "ACT ready: model=%s, chunk_size=%d, n_action_steps=%d, "
            "action_dim=%d, state_dim=%d, camera_key=%s, manual_norm=%s",
            model_id,
            self._chunk_size,
            self._n_action_steps,
            self._action_dim,
            self._state_dim,
            self._camera_key,
            self._needs_manual_norm,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        """Run ACT inference.

        Returns a single action vector (dequeued from the chunk buffer).
        The buffer is refilled by a new forward pass when empty.

        Normalization: old-style ALOHA checkpoints (lerobot/act_aloha_sim_*) store
        mean/std buffers that newer lerobot ignores.  When ``_needs_manual_norm`` is
        True, we apply them here:
          - Images:  (img_[0,1] − img_mean) / img_std  before the model
          - State:   (state − state_mean) / state_std   before the model
          - Action:  action * action_std + action_mean  after the model
        """
        import torch
        from PIL import Image
        from torchvision import transforms

        to_tensor = transforms.ToTensor()

        def decode(b64: str) -> torch.Tensor:
            return to_tensor(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

        state_list = obs.state.get("flat") or [0.0] * self._state_dim
        state_tensor = torch.tensor(state_list, dtype=torch.float32)[: self._state_dim]
        img_tensor = decode(obs.images["primary"])  # [C, H, W] in [0, 1]

        if self._needs_manual_norm:
            # Apply input normalisations from old-style checkpoint buffers.
            if self._img_mean is not None and self._img_std is not None:
                img_tensor = (img_tensor - self._img_mean) / (self._img_std + 1e-8)
            if self._state_mean is not None and self._state_std is not None:
                state_tensor = (state_tensor - self._state_mean) / (self._state_std + 1e-8)

        # Build lerobot-compatible observation batch (batch dim = 1).
        batch: dict = {
            self._camera_key: img_tensor.unsqueeze(0),
            "observation.state": state_tensor.unsqueeze(0),
        }

        # Move tensors to device.
        batch = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        with torch.no_grad():
            action = self._policy.select_action(batch)

        if isinstance(action, torch.Tensor):
            action = action.cpu()

        if (
            self._needs_manual_norm
            and self._action_mean is not None
            and self._action_std is not None
        ):
            # Unnormalise: action_real = action_normalised * std + mean
            action = action * self._action_std + self._action_mean

        if isinstance(action, torch.Tensor):
            action = action.numpy()
        if not isinstance(action, np.ndarray):
            action = np.array(action)

        # select_action returns shape [action_dim]; wrap in list[list[float]].
        action = action.flatten()
        return [action.tolist()]

    def get_info(self) -> dict:
        return {
            "name": "ACT",
            "model_id": self.model_id,
            "action_space": {
                "type": "joint_pos",
                "dim": getattr(self, "_action_dim", _ACTION_DIM),
                "description": (
                    "Absolute joint positions: [q0..q6 (left arm), q0..q6 (right arm)]"
                ),
            },
            "state_dim": getattr(self, "_state_dim", _STATE_DIM),
            "action_chunk_size": getattr(self, "_n_action_steps", _CHUNK_SIZE),
            "obs_requirements": {
                "cameras": ["primary"],
                "state_dim": getattr(self, "_state_dim", _STATE_DIM),
                "image_resolution": [480, 640],
                # Aloha cameras are already upright — no flip needed.
                "image_transform": "none",
            },
        }

    def get_action_spec(self) -> dict:
        """ACT action spec: absolute joint positions, 14-dim for aloha.

        Key and format align with AlohaGymBackend / RoboTwinBackend:
          - key = "joint_pos"
          - format = "absolute_joint_positions"
        """
        action_dim = getattr(self, "_action_dim", _ACTION_DIM)
        return {
            "joint_pos": ActionObsSpec(
                "joint_pos",
                action_dim,
                "absolute_joint_positions",
                (-3.15, 3.15),
                description="Absolute joint positions (radians), 14-dim aloha bimanual",
            ),
        }

    def get_observation_spec(self) -> dict:
        """ACT observation spec: single top-view RGB + 14-dim joint state.

        State format aligns with AlohaGymBackend:
          - format = "joint_positions"
        """
        state_dim = getattr(self, "_state_dim", _STATE_DIM)
        return {
            "primary": IMAGE_RGB,
            "state": ActionObsSpec(
                "state",
                state_dim,
                "joint_positions",
                description="Absolute joint positions (radians), 14-dim aloha bimanual",
            ),
            "instruction": LANGUAGE,
        }

    def reset(self) -> None:
        """Flush the ACT action-chunk buffer between episodes."""
        if getattr(self, "_policy", None) is not None:
            self._policy.reset()


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="ACT Policy Server")
    parser.add_argument("--model-id", default=_MODEL_ID_DEFAULT)
    parser.add_argument("--port", type=int, default=5107)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = ACTPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="ACT Policy Server",
    )
    logging.basicConfig(level=logging.INFO)
    print(f"[act_policy] Starting on {args.host}:{args.port} model={args.model_id}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
