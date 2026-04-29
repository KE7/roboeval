#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   # "gr00t",  # install from https://github.com/NVIDIA/Isaac-GR00T (N1.6, e29d8fc)
# ]
# ///
"""GR00T-N1.6 policy server (NVIDIA Isaac-GR00T).

Pinned to upstream N1.6 (public main, commit ``e29d8fc``, 2026-02-03).
Loads the model directly via :class:`gr00t.policy.gr00t_policy.Gr00tPolicy`
(no ``Gr00tSimPolicyWrapper``) and feeds nested ``video``/``state``/``language``
dict observations.

Usage::

    python -m sims.vla_policies.groot_policy [--model-id MODEL] [--embodiment-tag TAG] [--port PORT]

Environment variables:
    GROOT_MODEL_ID          – HF repo id (default: ``0xAnkitSingh/GR00T-N1.6-LIBERO``)
    GROOT_EMBODIMENT_TAG    – embodiment tag (default: ``LIBERO_PANDA``)
    GROOT_MODEL_SUBFOLDER   – optional HF repo subfolder

Canonical pairings
------------------
  0xAnkitSingh/GR00T-N1.6-LIBERO  LIBERO_PANDA           (libero,   7-dim EEF delta)
  nvidia/GR00T-N1.6-3B            ROBOCASA_PANDA_OMRON   (robocasa, 12-dim mobile-base EEF delta)

Both checkpoints use the publicly accessible ``nvidia/Eagle-Block2A-2B-v2``
backbone shipped inside the Isaac-GR00T clone — no license-gated downloads.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np

from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

try:
    from roboeval.specs import (
        GRIPPER_CLOSE_POS,
        IMAGE_RGB,
        LANGUAGE,
        POSITION_DELTA,
        ActionObsSpec,
    )

    _SPECS_AVAILABLE = True
except ImportError:
    _SPECS_AVAILABLE = False
    ActionObsSpec = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


def _ensure_writable_hf_modules_cache() -> None:
    """Default GR00T dynamic modules to a writable cache before Transformers imports."""
    if os.environ.get("HF_MODULES_CACHE"):
        return

    cache_root = Path(tempfile.gettempdir()) / "roboeval-hf-modules" / str(os.getuid()) / "groot"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(cache_root)


_ensure_writable_hf_modules_cache()


def _resolve_embodiment_tag(tag_str: str):
    """Return an :class:`EmbodimentTag` for the given name-or-value string.

    Accepts canonical enum names (``LIBERO_PANDA``) and value strings
    (``libero_panda``), case-insensitively.  Falls back to the raw string if
    the enum has neither.
    """
    from gr00t.data.embodiment_tags import EmbodimentTag

    # Try value-based constructor first (e.g. "libero_panda" → LIBERO_PANDA).
    try:
        return EmbodimentTag(tag_str)
    except ValueError:
        pass
    # Try name-based attribute access, both as-given and uppercased.
    for candidate in (tag_str, tag_str.upper()):
        if hasattr(EmbodimentTag, candidate):
            return getattr(EmbodimentTag, candidate)
    return EmbodimentTag(tag_str)  # raises ValueError with a helpful message


class GR00TPolicy(VLAPolicyBase):
    """NVIDIA GR00T-N1.6 via :class:`gr00t.policy.gr00t_policy.Gr00tPolicy`.

    Loads in a background thread — server is immediately reachable while
    the model warms up (returns 503 until ready).

    Observation format: nested ``video``/``state``/``language`` dicts (the
    canonical Gr00tPolicy interface).  No flat-key wrapper is used.
    Gripper convention: GR00T outputs in [0, 1]; we binarise at 0.5 and emit
    ±1 (robosuite convention) per checkpoint:
      ``gripper_close``: 1 = close → close when g > 0.5 (RoboCasa).
      ``gripper``       : 0 = close → close when g < 0.5 (LIBERO RLDS).
    """

    load_in_background = True

    def load_model(
        self,
        model_id: str,
        device: str,
        embodiment_tag: str = "LIBERO_PANDA",
        model_subfolder: str = "",
        action_key_filter: str = "",
        **_,
    ) -> None:
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.model_id = model_id
        emb_tag = _resolve_embodiment_tag(embodiment_tag)

        # Optional HF subfolder support for checkpoints stored below repo root.
        model_path: str = model_id
        if model_subfolder:
            from huggingface_hub import snapshot_download

            local_dir = snapshot_download(
                model_id,
                allow_patterns=[f"{model_subfolder}/**"],
                ignore_patterns=["**/optim_states*", "**/mp_rank_00_model_states.pt"],
            )
            model_path = str(os.path.join(local_dir, model_subfolder))
            logger.info(
                "GR00T subfolder resolved: repo=%s, subfolder=%s → %s",
                model_id,
                model_subfolder,
                model_path,
            )

        logger.info(
            "Loading GR00T-N1.6: model=%s, embodiment=%s (%s), device=%s",
            model_path,
            embodiment_tag,
            getattr(emb_tag, "value", emb_tag),
            device,
        )

        self._policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=emb_tag,
            device=device,
            strict=False,
        )

        cfg = self._policy.get_modality_config()
        self._video_keys = list(cfg["video"].modality_keys)
        self._state_keys = list(cfg["state"].modality_keys)
        self._action_keys = list(cfg["action"].modality_keys)
        self._language_key = self._policy.language_key
        self._embodiment_tag = embodiment_tag
        self._embodiment_value = getattr(emb_tag, "value", str(emb_tag))
        self._action_chunk_size = len(cfg["action"].delta_indices)

        # Resolve action-dim and gripper column from the processor's
        # state-action processor norm_params (N1.6 path).
        sap = self._policy.processor.state_action_processor
        self._action_dim = int(sap.get_action_dim(self._embodiment_value))
        norm_params = sap.norm_params[self._embodiment_value]["action"]

        # Optional action-key filter: select a subset of action keys (e.g.
        # "left_arm,right_arm" for a GR1 checkpoint driving a bimanual robot).
        # Must be applied BEFORE recomputing action_dim and gripper_col.
        if action_key_filter:
            filter_keys = [k.strip() for k in action_key_filter.split(",") if k.strip()]
            self._action_keys = [k for k in self._action_keys if k in filter_keys]
            self._action_dim = sum(
                int(norm_params[k]["dim"].item()) for k in self._action_keys if k in norm_params
            )

        # Store per-key state dims for flat-state → structured decomposition.
        # Needed when the sim returns a flat proprioceptive vector (e.g.
        # RoboTwin) instead of a keyed state_dict (e.g. LIBERO / RoboCasa).
        state_norm = sap.norm_params[self._embodiment_value].get("state", {})
        self._state_key_dims: dict = {
            k: int(state_norm[k]["dim"].item()) for k in self._state_keys if k in state_norm
        }

        col = 0
        self._gripper_col_idx = -1
        self._gripper_key_name = ""
        for key in self._action_keys:
            if "gripper" in key.lower():
                self._gripper_col_idx = col
                self._gripper_key_name = key
                break
            col += int(norm_params[key]["dim"].item())

        self.ready = True
        logger.info(
            "GR00T ready: chunk=%d, action_dim=%d, video_keys=%s, state_keys=%s, action_keys=%s",
            self._action_chunk_size,
            self._action_dim,
            self._video_keys,
            self._state_keys,
            self._action_keys,
        )

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        def decode(b64: str) -> np.ndarray:
            from PIL import Image

            return np.asarray(
                Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8
            )

        sources = {
            "primary": decode(obs.images["primary"]),
            "secondary": decode(obs.images["secondary"]) if obs.images.get("secondary") else None,
            "wrist": decode(obs.images["wrist"]) if obs.images.get("wrist") else None,
        }

        # Build nested observation dict (Gr00tPolicy native interface).
        video_obs: dict = {}
        for key in self._video_keys:
            if "side_0" in key or "head" in key or key in ("image", "image_0", "res256_image_0"):
                img = sources["primary"]
            elif "side_1" in key or "secondary" in key:
                img = sources["secondary"]
                if img is None:
                    raise ValueError(f"GR00T requires 'secondary' image for video key '{key}'")
            elif "wrist" in key:
                img = sources["wrist"]
                if img is None:
                    raise ValueError(f"GR00T requires 'wrist' image for video key '{key}'")
            else:
                # Fallback: map unknown keys to primary camera with a warning.
                logger.warning("Unknown GR00T video key '%s'; falling back to primary camera.", key)
                img = sources["primary"]
            # Gr00tPolicy expects (B=1, T=1, H, W, C) uint8.
            video_obs[key] = img[None, None]

        state_dict_in = obs.state.get("structured") or {}
        flat_state = obs.state.get("flat")
        state_obs: dict = {}
        if state_dict_in:
            # Preferred path: structured state_dict provided by the sim (LIBERO, RoboCasa).
            for key in self._state_keys:
                if key not in state_dict_in:
                    raise ValueError(f"Missing GR00T state key '{key}' in state_dict")
                state_obs[key] = np.asarray(state_dict_in[key], dtype=np.float32).reshape(1, 1, -1)
        elif flat_state is not None:
            # Fallback: sim returns a flat proprioceptive vector (e.g. RoboTwin joint_pos).
            # Decompose into per-key arrays using dims from statistics.json.
            # Keys beyond the flat vector's length are zeroed (e.g. hand/waist for GR1
            # when the robot only exposes 14 arm DOFs).
            flat_arr = np.asarray(flat_state, dtype=np.float32).flatten()
            offset = 0
            key_dims = getattr(self, "_state_key_dims", {})
            for key in self._state_keys:
                dim = key_dims.get(key, 1)
                if offset + dim <= len(flat_arr):
                    val = flat_arr[offset : offset + dim]
                else:
                    val = np.zeros(dim, dtype=np.float32)
                state_obs[key] = val.reshape(1, 1, -1)
                offset += dim
        else:
            raise ValueError(
                "GR00T requires either structured or flat proprioceptive state; "
                "roboeval did not provide either."
            )

        # Gr00tPolicy language schema: dict[str, list[list[str]]] with shape (B=1, T=1).
        language_obs = {self._language_key: [[obs.instruction]]}

        observation = {"video": video_obs, "state": state_obs, "language": language_obs}
        action_dict, _ = self._policy.get_action(observation)

        parts = []
        for key in self._action_keys:
            # Gr00tPolicy returns plain keys; some checkpoints include an
            # "action." prefix.
            arr = action_dict.get(key)
            if arr is None:
                arr = action_dict[f"action.{key}"]
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 3:
                raise ValueError(f"Unexpected shape for action key '{key}': {arr.shape}")
            parts.append(arr[0])  # drop batch → (T, D)
        flat_action = np.concatenate(parts, axis=-1)

        # Binarise gripper at 0.5 threshold; both RoboCasa and LIBERO sims use
        # robosuite convention (+1.0 = close, -1.0 = open).
        if self._gripper_col_idx >= 0:
            g = flat_action[:, self._gripper_col_idx]
            logger.info(
                "RAW gripper '%s' pre-binarize: min=%.4f max=%.4f mean=%.4f vals=%s",
                self._gripper_key_name,
                float(g.min()),
                float(g.max()),
                float(g.mean()),
                [round(float(x), 4) for x in g[:8]],
            )
            if self._gripper_key_name == "gripper_close":
                # RoboCasa: 1=close → close when g > 0.5
                flat_action[:, self._gripper_col_idx] = np.where(g > 0.5, 1.0, -1.0)
            else:
                # LIBERO RLDS and others: 0=close → close when g < 0.5
                flat_action[:, self._gripper_col_idx] = np.where(g < 0.5, 1.0, -1.0)
            logger.info(
                "POST-binarize gripper: close=%d/%d",
                int((flat_action[:, self._gripper_col_idx] > 0).sum()),
                len(g),
            )

        return flat_action.tolist()

    def get_info(self) -> dict:
        vk = getattr(self, "_video_keys", [])
        sk = getattr(self, "_state_keys", [])
        ak = getattr(self, "_action_keys", [])
        et = getattr(self, "_embodiment_tag", "")

        # Derive required camera roles from the model's actual video_keys
        # (different N1.6 checkpoints use different camera sets).
        _cameras: list[str] = []
        for k in vk:
            if "wrist" in k:
                _cameras.append("wrist")
            elif "side_1" in k or "secondary" in k:
                _cameras.append("secondary")
            else:
                _cameras.append("primary")
        cameras = list(dict.fromkeys(_cameras))  # deduplicate, preserve order

        return {
            "name": (self.model_id.split("/")[-1] if "/" in self.model_id else self.model_id)
            or "groot",
            "model_id": self.model_id,
            "embodiment_tag": et,
            "action_space": {
                "type": "joint_pos" if self._is_joint_pos_mode() else "eef_delta",
                "dim": getattr(self, "_action_dim", 0),
                "description": f"GR00T {et}: {' + '.join(ak)}" if ak else "GR00T native action",
            },
            "state_dim": len(sk),
            "action_chunk_size": getattr(self, "_action_chunk_size", 0),
            "obs_requirements": {
                "cameras": cameras,
                "state_dim": len(sk),
                "state_format": "structured",
                "image_transform": "none",
            },
            "modality_keys": {
                "video": vk,
                "state": sk,
                "action": ak,
                "language": getattr(self, "_language_key", ""),
            },
        }

    def _is_joint_pos_mode(self) -> bool:
        """Return True when the loaded checkpoint outputs absolute joint positions.

        Recognised case: all action keys contain "joint" (e.g. a future
        ALOHA or RoboTwin-specific checkpoint with explicit joint keys).
        LIBERO_PANDA / ROBOCASA_PANDA_OMRON use EEF-delta keys and are not affected.
        """
        ak = getattr(self, "_action_keys", [])
        if not ak:
            return False
        # Explicit joint-keyed checkpoints
        if all("joint" in k.lower() for k in ak):
            return True
        return False

    def get_action_spec(self) -> dict[str, ActionObsSpec] | None:
        """GR00T action spec: derived from the loaded checkpoint's action keys.

        Joint-position mode (future ALOHA / RoboTwin checkpoint):
            ``{"joint_pos": ActionObsSpec("joint_pos", N, "absolute_joint_positions")}``

        Delta-EEF mode (current LIBERO_PANDA / ROBOCASA_PANDA_OMRON checkpoints):
            ``{"position": ..., "rotation": ..., "gripper": ...}``
        """
        if not _SPECS_AVAILABLE:
            return None
        if self._is_joint_pos_mode():
            action_dim = getattr(self, "_action_dim", 14)
            return {
                "joint_pos": ActionObsSpec(
                    "joint_pos",
                    action_dim,
                    "absolute_joint_positions",
                    None,
                    description=f"GR00T absolute joint positions ({getattr(self, '_embodiment_tag', '')})",
                ),
            }
        # Default: delta EEF (LIBERO_PANDA, ROBOCASA_PANDA_OMRON).
        return {
            "position": POSITION_DELTA,
            "rotation": ActionObsSpec("rotation", 3, "delta_axisangle", (-3.15, 3.15)),
            "gripper": GRIPPER_CLOSE_POS,
        }

    def get_observation_spec(self) -> dict[str, ActionObsSpec] | None:
        """GR00T observation spec: cameras derived from video_keys + structured state + language."""
        if not _SPECS_AVAILABLE:
            return None
        spec: dict[str, ActionObsSpec] = {}
        for role in ("primary", "wrist", "secondary"):
            vk = getattr(self, "_video_keys", [])
            needs = any(
                (
                    role == "primary"
                    and "wrist" not in k
                    and "secondary" not in k
                    and "side_1" not in k
                )
                or (role == "wrist" and "wrist" in k)
                or (role == "secondary" and ("side_1" in k or "secondary" in k))
                for k in vk
            )
            if needs:
                spec[role] = IMAGE_RGB
        # State format depends on the checkpoint's action mode.
        # Joint-position mode (ALOHA / RoboTwin checkpoint): declare "joint_positions"
        #   so the sim's joint_positions state passes spec validation.
        # EEF-delta mode (LIBERO_PANDA / ROBOCASA checkpoints): embodiment-specific format.
        if self._is_joint_pos_mode():
            action_dim = getattr(self, "_action_dim", 14)
            spec["state"] = ActionObsSpec("state", action_dim, "joint_positions")
        else:
            # N1.6-3B (robocasa_panda_omron)               → 9-dim: grip2 + eef_pos3 + quat4
            # 0xAnkitSingh/GR00T-N1.6-LIBERO (libero_panda) → 8-dim: eef_pos3 + aa3 + grip2
            et = getattr(self, "_embodiment_tag", "")
            if "robocasa" in et.lower():
                state_fmt = "robocasa_grip2_eef_pos3_quat4"
            else:
                state_fmt = "libero_eef_pos3_aa3_grip2"
            spec["state"] = ActionObsSpec("state", 0, state_fmt)
        spec["instruction"] = LANGUAGE
        return spec

    def reset(self) -> None:
        if getattr(self, "_policy", None) is not None:
            try:
                self._policy.reset()
            except Exception:
                pass


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="GR00T-N1.6 Policy Server")
    parser.add_argument(
        "--model-id", default=os.environ.get("GROOT_MODEL_ID", "0xAnkitSingh/GR00T-N1.6-LIBERO")
    )
    parser.add_argument(
        "--embodiment-tag",
        dest="embodiment_tag",
        default=os.environ.get("GROOT_EMBODIMENT_TAG", "LIBERO_PANDA"),
    )
    parser.add_argument(
        "--model-subfolder",
        dest="model_subfolder",
        default=os.environ.get("GROOT_MODEL_SUBFOLDER", ""),
        help="Optional HuggingFace repo subfolder (legacy; not needed "
        "for the default 0xAnkitSingh/GR00T-N1.6-LIBERO repo). "
        "Set via GROOT_MODEL_SUBFOLDER env var.",
    )
    parser.add_argument(
        "--action-keys",
        dest="action_keys",
        default=os.environ.get("GROOT_ACTION_KEYS", ""),
        help="Comma-separated subset of action keys to concatenate "
        "(e.g. 'left_arm,right_arm' for GR1→RoboTwin 14-dim). "
        "Set via GROOT_ACTION_KEYS env var.",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000)))
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be 1024–65535, got {args.port}")

    policy = GR00TPolicy()
    app = make_app(
        policy,
        args.model_id,
        args.device,
        title="GR00T-N1.6 Policy Server",
        embodiment_tag=args.embodiment_tag,
        model_subfolder=args.model_subfolder,
        action_key_filter=args.action_keys,
    )
    logging.basicConfig(level=logging.INFO)
    print(
        f"[groot_policy] Starting on {args.host}:{args.port} model={args.model_id} subfolder={args.model_subfolder!r} embodiment={args.embodiment_tag} action_keys={args.action_keys!r}"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
