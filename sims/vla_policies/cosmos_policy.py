#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "fastapi>=0.100",
#   "uvicorn[standard]",
#   "numpy",
#   "pillow",
#   "torch>=2.0",
#   # "cosmos_policy",
# ]
# ///
"""Cosmos-Policy server for RoboCasa.

Usage:
    python -m sims.vla_policies.cosmos_policy [--port PORT]

Model paths can be overridden with COSMOS_CKPT_PATH, COSMOS_CONFIG_FILE,
COSMOS_DATASET_STATS_PATH, and COSMOS_T5_EMBEDDINGS_PATH.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
from io import BytesIO
from types import SimpleNamespace

import numpy as np

from sims.vla_policies.base import VLAPolicyBase, make_app
from sims.vla_policies.vla_schema import VLAObservation

logger = logging.getLogger(__name__)

_MODEL_ID = "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
_ACTION_DIM = 7  # EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]
_PROPRIO_DIM = 9  # gripper_qpos(2) + eef_pos(3) + eef_quat(4)
_CHUNK_SIZE = 32  # actions per diffusion generation
_EXEC_HORIZON = 16  # actions actually executed (open-loop horizon)

_HF_SNAP = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-RoboCasa-Predict2-2B"
    "/snapshots/4b2a04c80d97202f86127ebec80461e8016ec1dc"
)

# Runtime config consumed by the policy action helper.
_EVAL_CFG = SimpleNamespace(
    suite="robocasa",
    use_third_person_image=True,
    num_third_person_images=2,  # primary (left) + secondary (right)
    use_wrist_image=True,
    num_wrist_images=1,
    use_proprio=True,
    normalize_proprio=True,
    unnormalize_actions=True,
    chunk_size=_CHUNK_SIZE,
    num_open_loop_steps=_EXEC_HORIZON,
    flip_images=True,
    use_jpeg_compression=True,
    trained_with_image_aug=True,
    use_variance_scale=False,
)


class CosmosPolicy(VLAPolicyBase):
    """Cosmos-Policy-RoboCasa-Predict2-2B wrapper.

    Loads in a background thread because the checkpoint is large and health
    checks should remain responsive while initialization is in progress.
    """

    load_in_background = True

    def load_model(
        self,
        model_id: str,
        device: str,
        ckpt: str = "",
        config_file: str = "",
        stats_path: str = "",
        t5_path: str = "",
        **_,
    ) -> None:
        import cosmos_policy.constants as cc
        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
        from cosmos_policy.experiments.robot.cosmos_utils import (
            DEVICE,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.model_id = model_id

        cc.NUM_ACTIONS_CHUNK = _CHUNK_SIZE
        cc.ACTION_DIM = _ACTION_DIM
        cc.PROPRIO_DIM = _PROPRIO_DIM
        cc.ROBOT_PLATFORM = "ROBOCASA"

        logger.info("Loading Cosmos-Policy checkpoint: %s", ckpt)
        self._model, _ = load_model_from_checkpoint(
            "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
            ckpt,
            config_file,
            load_ema_to_reg=False,
        )
        self._model.eval()
        self._model = self._model.to(DEVICE)
        self._dataset_stats = load_dataset_stats(stats_path)
        init_t5_text_embeddings_cache(t5_path)
        self.ready = True
        logger.info("Cosmos-Policy ready on %s", DEVICE)

    def predict(self, obs: VLAObservation) -> list[list[float]]:
        import torch
        from cosmos_policy.experiments.robot.cosmos_utils import get_action
        from PIL import Image

        def decode(b64: str) -> np.ndarray:
            return np.array(
                Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"), dtype=np.uint8
            )

        # RoboCasa camera frames are vertically flipped for this model.
        primary = np.flipud(decode(obs.images["primary"]))
        wrist_b64 = obs.images.get("wrist")
        secondary_b64 = obs.images.get("secondary")
        wrist = np.flipud(decode(wrist_b64)) if wrist_b64 else np.zeros_like(primary)
        secondary = np.flipud(decode(secondary_b64)) if secondary_b64 else primary.copy()

        obs_dict = {
            "primary_image": primary,
            "secondary_image": secondary,
            "wrist_image": wrist,
            "proprio": np.array(obs.state.get("flat", []), dtype=np.float64),
        }
        with torch.no_grad():
            result = get_action(
                cfg=_EVAL_CFG,
                model=self._model,
                dataset_stats=self._dataset_stats,
                obs=obs_dict,
                task_label_or_embedding=obs.instruction,
                seed=1,
                randomize_seed=False,
                num_denoising_steps_action=5,
                generate_future_state_and_value_in_parallel=False,
                worker_id=0,
                batch_size=1,
            )
        raw = result["actions"]
        del result
        torch.cuda.empty_cache()

        actions = [
            a.tolist() if isinstance(a, np.ndarray) else list(a) for a in raw[:_EXEC_HORIZON]
        ]
        while len(actions) < _EXEC_HORIZON:
            actions.append([0.0] * _ACTION_DIM)
        return actions

    def get_info(self) -> dict:
        return {
            "name": "Cosmos-Policy-RoboCasa-Predict2-2B",
            "model_id": _MODEL_ID,
            "action_space": {
                "type": "eef_delta",
                "dim": _ACTION_DIM,
                "description": "EEF delta: [dx,dy,dz,droll,dpitch,dyaw,gripper]",
            },
            "state_dim": _PROPRIO_DIM,
            "action_chunk_size": _EXEC_HORIZON,
            "obs_requirements": {
                "cameras": ["primary"],
                "optional_cameras": ["secondary", "wrist"],
                "state_dim": _PROPRIO_DIM,
                "image_resolution": [224, 224],
                "image_transform": "none",  # applied in predict()
            },
        }


def main():
    import uvicorn

    _ckpt_def = os.environ.get(
        "COSMOS_CKPT_PATH", f"{_HF_SNAP}/Cosmos-Policy-RoboCasa-Predict2-2B.pt"
    )
    _cfg_def = os.environ.get("COSMOS_CONFIG_FILE", "cosmos_policy/config/config.py")
    _stats_def = os.environ.get(
        "COSMOS_DATASET_STATS_PATH", f"{_HF_SNAP}/robocasa_dataset_statistics.json"
    )
    _t5_def = os.environ.get("COSMOS_T5_EMBEDDINGS_PATH", f"{_HF_SNAP}/robocasa_t5_embeddings.pkl")

    parser = argparse.ArgumentParser(description="Cosmos-Policy Server (RoboCasa)")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", os.environ.get("VLA_PORT", 8000)))
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--ckpt", default=_ckpt_def)
    parser.add_argument("--config-file", dest="config_file", default=_cfg_def)
    parser.add_argument("--stats", default=_stats_def)
    parser.add_argument("--t5", default=_t5_def)
    args = parser.parse_args()

    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be 1024–65535, got {args.port}")

    policy = CosmosPolicy()
    app = make_app(
        policy,
        _MODEL_ID,
        "cuda",
        title="Cosmos-Policy Server",
        ckpt=args.ckpt,
        config_file=args.config_file,
        stats_path=args.stats,
        t5_path=args.t5,
    )
    logging.basicConfig(level=logging.INFO)
    print(f"[cosmos_policy] Starting on {args.host}:{args.port} model={_MODEL_ID}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
