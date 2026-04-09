#!/usr/bin/env python
"""
Simulator HTTP server for robo-eval.

This script runs inside a simulator-specific virtualenv and exposes the
simulator as a FastAPI HTTP service. Each simulator has different Python
version requirements and dependencies, so this isolation is necessary.

Usage:
    /path/to/venv/bin/python sims/sim_worker.py --sim libero --port 5001

Endpoints:
    POST /init   — Initialize simulator  {sim, task, suite?, camera_resolution?}
    POST /reset  — Reset environment      -> {image, success}
    POST /step   — Step with action       {action: [float,...]} -> {image, reward, done, success}
    GET  /obs    — Current observation    -> {image}
    GET  /info   — Sim capabilities       -> {sim, action_space, obs_space, ...}
    GET  /success — Check task success    -> {success}
    POST /close  — Shutdown server
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import pathlib
import signal
import traceback
from io import BytesIO
from typing import Optional

import logging

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Module-level headless flag — set in main() before uvicorn starts so that
# MUJOCO_GL is in the environment before any MuJoCo imports occur.
# Default is False (windowed); --headless CLI flag sets it to True.
_headless: bool = False


def encode_image_b64(img_array: np.ndarray) -> str:
    """Encode a numpy RGB image array as a base64 PNG string."""
    from PIL import Image

    img = Image.fromarray(img_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_images_response(
    img: np.ndarray,
    img2: Optional[np.ndarray] = None,
    img3: Optional[np.ndarray] = None,
) -> dict:
    """Build the image portion of an HTTP response with role-keyed ``images`` dict.

    Returns a dict containing:
      - ``"image"``: base64 primary image (backward compat for older readers)
      - ``"image2"``: base64 wrist image (backward compat, only if *img2* given)
      - ``"image3"``: base64 secondary image (backward compat, only if *img3* given)
      - ``"images"``: ``{"primary": b64, "wrist": b64, "secondary": b64}``
        Role-keyed dict — the canonical representation going forward.
    """
    primary_b64 = encode_image_b64(img)
    images: dict = {"primary": primary_b64}
    resp: dict = {"image": primary_b64}  # backward compat

    if img2 is not None:
        wrist_b64 = encode_image_b64(img2)
        images["wrist"] = wrist_b64
        resp["image2"] = wrist_b64  # backward compat

    if img3 is not None:
        secondary_b64 = encode_image_b64(img3)
        images["secondary"] = secondary_b64
        resp["image3"] = secondary_b64  # backward compat

    resp["images"] = images
    return resp


# ======================================================================
# Simulator-specific backends
#
# To add a new backend:
# 1. Create a class with methods: init(), reset(), step(), get_obs(),
#    check_success(), close(). See docs/adding_a_benchmark.md for details.
# 2. Return conventions: reset() -> (image, image2), step() -> (image,
#    image2, reward, done, info), get_obs() -> (image, image2).
#    image2 is None if no wrist camera.
# 3. Optional richer observation helpers can be exposed via backend methods
#    such as get_state_dict() or _extract_secondary_image(); the HTTP layer
#    forwards these fields when present without changing the base contract.
# 3. Register your class in the BACKENDS dict at the end of this section.
# ======================================================================


class LiberoBackend:
    """Backend for LIBERO benchmark environments."""

    def __init__(self):
        self.env = None
        self.benchmark = None
        self.task = None
        self._ep_idx = 0       # tracks which init_state to use per episode
        self._last_obs: dict = {}  # cache for state extraction (Bug 2)
        self._last_img2 = None   # cache for wrist-camera image (image2)

    def _find_task_idx(self, task_name, task_names, suite):
        """Find task index by name or numeric index."""
        if task_name.isdigit():
            return int(task_name)
        matching = [
            i for i, t in enumerate(task_names)
            if task_name.lower() in t.lower()
        ]
        if not matching:
            raise ValueError(
                f"Task '{task_name}' not found in {suite}. "
                f"Available: {task_names}"
            )
        return matching[0]

    def _get_init_states(self, task_idx):
        """Load episode init states for a task. Subclasses may override."""
        return self.benchmark.get_task_init_states(task_idx)

    def init(self, task_name: str, camera_resolution: int, suite: str = None, headless: bool = True, sim_config: dict = None):
        """Initialize a LIBERO environment for a specific task.

        Sets MUJOCO_GL, loads the benchmark suite, creates the
        OffScreenRenderEnv, and fetches init states for episode resets.
        """
        # MUJOCO_GL must be set before MuJoCo is imported. Since these are lazy
        # imports inside init(), setting it here is safe for the first call.
        os.environ["MUJOCO_GL"] = "egl" if headless else "glfw"
        self.headless = headless

        from libero.libero import benchmark as bm_module
        from libero.libero.envs import OffScreenRenderEnv

        suite = suite or "libero_spatial"
        self.benchmark = bm_module.get_benchmark(suite)()
        task_names = self.benchmark.get_task_names()
        task_idx = self._find_task_idx(task_name, task_names, suite)

        self.task = self.benchmark.get_task(task_idx)
        task_description = self.task.language
        self.init_states = self._get_init_states(task_idx)

        task_bddl = self.benchmark.get_task_bddl_file_path(task_idx)
        env_args = {
            "bddl_file_name": task_bddl,
            "camera_heights": camera_resolution,
            "camera_widths": camera_resolution,
            # has_renderer=True opens a live GLFW window; False = no GUI.
            "has_renderer": not headless,
            # has_offscreen_renderer is always needed for camera image capture.
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
        }
        self.env = OffScreenRenderEnv(**env_args)
        self.env.reset()
        return {"task_description": task_description}

    def reset(self, episode_index: int = None):
        """Reset the environment to a specific episode's initial state.

        Performs: base reset → set_init_state → 10 physics warmup steps (no-op)
        → use_delta (if delta_actions). Returns ``(image, image2)`` tuple.

        Args:
            episode_index: Which init_state to load. If None, uses an
                auto-incrementing counter.
        """
        self.env.reset()
        if hasattr(self, "init_states") and len(self.init_states) > 0:
            idx = episode_index if episode_index is not None else self._ep_idx
            init_state = self.init_states[idx % len(self.init_states)]
            self._ep_idx = idx + 1
            self.env.set_init_state(init_state)

        # Warmup: 10 no-op steps to let physics settle after set_init_state
        # (matches lerobot's LiberoEnv.reset() which uses num_steps_wait=10).
        _dummy_action = [0, 0, 0, 0, 0, 0, -1]  # no-op, gripper close
        for _ in range(10):
            self.env.step(_dummy_action)

        # Match lerobot's LIBERO reset ordering: let the env settle first,
        # then switch the controller into delta-action mode for rollout steps.
        if getattr(self, "delta_actions", False):
            for robot in self.env.env.robots:
                robot.controller.use_delta = True

        obs = self.env.env._get_observations()
        self._last_obs = obs
        img, img2 = self._extract_image(obs)
        self._last_img2 = img2
        return img, img2

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # In windowed mode, push the new frame to the live GLFW viewer.
        # Guard against headless servers where GLFW cannot create a window and
        # viewer is None — calling render() on None raises AttributeError.
        if not getattr(self, "headless", True) and getattr(self.env.env, "viewer", None) is not None:
            self.env.env.render()
        self._last_obs = obs
        img, img2 = self._extract_image(obs)
        self._last_img2 = img2
        return img, img2, reward, done, info

    def get_obs(self):
        obs = self.env.env._get_observations()
        self._last_obs = obs
        img, img2 = self._extract_image(obs)
        self._last_img2 = img2
        return img, img2

    def check_success(self):
        return self.env.env._check_success()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_info(self) -> dict:
        """Return action/obs space metadata for auto-discovery."""
        cam_res = getattr(self, "_cam_res", 256)
        return {
            "action_space": {"type": "eef_delta", "dim": 7, "accepted_dims": [7]},
            "obs_space": {
                "cameras": [
                    {"key": "agentview_image", "resolution": [cam_res, cam_res], "role": "primary"},
                    {"key": "robot0_eye_in_hand_image", "resolution": [cam_res, cam_res], "role": "wrist"},
                ],
                "state": {"dim": 8, "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"},
            },
            "max_steps": 280,
            "delta_actions": getattr(self, "delta_actions", False),
        }

    def _extract_image(self, obs):
        """Return (image, image2) where image=agentview and image2=wrist camera.

        Returns raw images without any transforms.  Image flipping (e.g. the
        180° flip required by lerobot-trained VLAs) is now applied by the
        orchestration layer (env_wrapper) based on the VLA's declared
        ``obs_requirements.image_transform``.
        """
        image = obs.get("agentview_image")
        image2 = obs.get("robot0_eye_in_hand_image")
        if image is None and image2 is None:
            raise KeyError(
                f"No camera image found in obs keys: {list(obs.keys())}"
            )
        # Use agentview as primary if available, else wrist camera
        primary = image if image is not None else image2
        return np.asarray(primary, dtype=np.uint8).copy(), (
            np.asarray(image2, dtype=np.uint8).copy() if image2 is not None else None
        )

    def _extract_state(self, obs: dict) -> list:
        """Extract proprioceptive state: eef_pos(3) + axisangle(3) + gripper(2) = 8-dim.

        Returns a plain list[float] for direct use by the policy server.
        """
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)       # (3,)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)     # (4,) [x,y,z,w]
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)  # (2,)

        # Convert quaternion [x,y,z,w] to axis-angle using lerobot's _quat2axisangle
        # formula (matches training data). Scipy as_rotvec() differs in edge case handling.
        x, y, z, w = float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3])
        den = np.sqrt(max(0.0, 1.0 - w * w))
        if den > 1e-10:
            angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
            axisangle = np.array([x, y, z], dtype=np.float32) / den * angle
        else:
            axisangle = np.zeros(3, dtype=np.float32)

        state = np.concatenate([eef_pos, axisangle, gripper])  # (8,)
        return state.tolist()


class LiberoProBackend(LiberoBackend):
    """Backend for LIBERO-Pro (same API as LIBERO, different repo).

    Overrides only init() to change the default suite name and gracefully
    handle missing init-state files (common in LIBERO-Pro OOD variants).
    All other methods (reset, step, get_obs, etc.) are inherited.
    """

    # Default suite for LIBERO-Pro (overrides LiberoBackend's "libero_spatial")
    DEFAULT_SUITE = "libero_pro"

    def _get_init_states(self, task_idx):
        """Load init states, gracefully handling missing files."""
        try:
            return self.benchmark.get_task_init_states(task_idx)
        except FileNotFoundError as e:
            logger.warning("Init states not found (%s); "
                          "episodes will start from default MuJoCo state.", e)
            return []

    def init(self, task_name: str, camera_resolution: int, suite: str = None, headless: bool = True, sim_config: dict = None):
        """Initialize a LIBERO-Pro environment.

        Delegates to LiberoBackend.init() with a different default suite.
        Gracefully handles missing init-state files by returning an empty list.
        """
        return super().init(
            task_name=task_name,
            camera_resolution=camera_resolution,
            suite=suite or self.DEFAULT_SUITE,
            headless=headless,
            sim_config=sim_config,
        )


class RoboCasaBackend:
    """Backend for RoboCasa kitchen task environments.

    Defaults to PandaOmron (12-dim: 7 arm + 5 base) — required by RoboCasa
    kitchen environments which need a mobile base.  The robot type is
    configurable via ``sim_config={"robot": "PandaOmron"}``.  Panda (fixed
    base) is NOT supported by RoboCasa kitchen scenes.

    Environment creation follows the Cosmos-Policy evaluation pattern:
    controller configs loaded from a fixed pickle, proper camera setup,
    obj_instance_split="B" (held-out test objects), and deterministic
    layout/style IDs for reproducibility.

    When a 7-dim policy (e.g. Cosmos) sends actions, :meth:`step` pads
    them to 12-dim with an idle mobile base.

    Returns ``(image, image2)`` where image2 is the wrist camera.
    """

    # Task name aliases: Cosmos (short) → RoboCasa (full) naming convention.
    # Callers may pass either form; both are accepted.
    _TASK_ALIASES = {
        "PnPCounterToCab": "PickPlaceCounterToCabinet",
        "PnPCabToCounter": "PickPlaceCabinetToCounter",
        "PnPCounterToSink": "PickPlaceCounterToSink",
        "PnPSinkToCounter": "PickPlaceSinkToCounter",
        "PnPCounterToMicrowave": "PickPlaceCounterToMicrowave",
        "PnPMicrowaveToCounter": "PickPlaceMicrowaveToCounter",
        "PnPCounterToStove": "PickPlaceCounterToStove",
        "PnPStoveToCounter": "PickPlaceStoveToCounter",
        "OpenSingleDoor": "OpenCabinet",
        "CloseSingleDoor": "CloseCabinet",
        "CoffeePressButton": "StartCoffeeMachine",
    }

    # Ordered task list for numeric index mapping (runner passes --task 0, 1, ...).
    # First 10 match the Cosmos-Policy "RoboCasa-10" benchmark.
    _TASK_INDEX = [
        "PickPlaceCounterToCabinet", "PickPlaceCabinetToCounter",
        "PickPlaceCounterToSink", "PickPlaceSinkToCounter",
        "PickPlaceCounterToMicrowave", "PickPlaceMicrowaveToCounter",
        "PickPlaceCounterToStove", "PickPlaceStoveToCounter",
        "OpenCabinet", "CloseCabinet",
        "OpenDrawer", "CloseDrawer",
        "TurnOnStove", "TurnOffStove",
        "TurnOnSinkFaucet", "TurnOffSinkFaucet",
        "TurnSinkSpout",
        "CoffeeSetupMug", "CoffeeServeMug", "StartCoffeeMachine",
        "TurnOnMicrowave", "TurnOffMicrowave",
    ]

    # Max steps per task (from Cosmos evaluation benchmarks)
    _TASK_MAX_STEPS = {
        "PickPlaceCounterToCabinet": 500, "PickPlaceCabinetToCounter": 500,
        "PickPlaceCounterToSink": 700, "PickPlaceSinkToCounter": 500,
        "PickPlaceCounterToMicrowave": 600, "PickPlaceMicrowaveToCounter": 500,
        "PickPlaceCounterToStove": 500, "PickPlaceStoveToCounter": 500,
        "OpenCabinet": 500, "CloseCabinet": 500,
        "OpenDrawer": 500, "CloseDrawer": 500,
        "TurnOnStove": 500, "TurnOffStove": 500,
        "TurnOnSinkFaucet": 500, "TurnOffSinkFaucet": 500,
        "TurnSinkSpout": 500,
        "CoffeeSetupMug": 600, "CoffeeServeMug": 600, "StartCoffeeMachine": 300,
        "TurnOnMicrowave": 500, "TurnOffMicrowave": 500,
    }

    # Default layout and style IDs — 5 test scenes from Cosmos eval
    _DEFAULT_LAYOUT_STYLE_IDS = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))

    # Camera setup matching Cosmos evaluation
    _CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

    # Mobile-base padding: 5 zeros + brake flag (used when receiving 7D
    # actions but running PandaOmron/PandaMobile with 12D action space)
    _MOBILE_BASE_PAD = np.array([0.0, 0.0, 0.0, 0.0, -1.0])

    # Path to controller configs pickle (copied from Cosmos-Policy repo)
    _CONTROLLER_CONFIGS_PATH = os.path.join(
        os.path.dirname(__file__), "robocasa_controller_configs.pkl"
    )

    def __init__(self):
        self.env = None
        self._last_obs: dict = {}
        self._robot_type: str = "PandaOmron"
        self._cam_res: int = 256
        self._task_name: str = ""
        self._resolved_task: str = ""  # after alias resolution

    def _resolve_task_name(self, task_name: str) -> str:
        """Resolve task identifiers to full RoboCasa env names.

        Accepts numeric indices (e.g. "0" -> first task), Cosmos short
        names (e.g. "PnPCounterToCab"), or full RoboCasa names.
        """
        # Handle numeric task indices (runner passes --task 0, 1, ...)
        if task_name.isdigit():
            idx = int(task_name)
            if idx < len(self._TASK_INDEX):
                return self._TASK_INDEX[idx]
            raise ValueError(
                f"Task index {idx} out of range. "
                f"Valid range: 0-{len(self._TASK_INDEX) - 1}"
            )
        return self._TASK_ALIASES.get(task_name, task_name)

    def _load_controller_configs(self):
        """Load OSC_POSE controller configs from the Cosmos pickle."""
        import pickle
        with open(self._CONTROLLER_CONFIGS_PATH, "rb") as f:
            return pickle.load(f)

    def init(self, task_name: str, camera_resolution: int, suite: str = None, headless: bool = True, sim_config: dict = None):
        """Initialize a RoboCasa environment by task name.

        Args:
            task_name: Task name — accepts both Cosmos short names
                (e.g. ``PnPCounterToCab``) and full RoboCasa names
                (e.g. ``PickPlaceCounterToCabinet``).  Defaults to
                ``PnPCounterToCab`` when called without a task.
            camera_resolution: Image resolution for all cameras.
            suite: Ignored (RoboCasa has a single suite).
            headless: If True, use offscreen rendering (default).
            sim_config: Optional dict with overrides:
                - ``robot``: ``"PandaOmron"`` (12D, default) or
                  ``"PandaMobile"`` (12D alias).  Panda (fixed base)
                  is NOT supported by RoboCasa kitchen scenes.
                - ``obj_instance_split``: Object split (default ``"target"``,
                  equivalent to Cosmos "B" = held-out test objects).
                - ``layout_and_style_ids``: Tuple of (layout, style) pairs.
                - ``seed``: Random seed for environment.
        """
        import robocasa  # noqa: F401 — registers kitchen envs with robosuite
        import robosuite

        cfg = sim_config or {}
        self._robot_type = cfg.get("robot", "PandaOmron")
        self._cam_res = camera_resolution
        self._task_name = task_name or "PnPCounterToCab"
        self._resolved_task = self._resolve_task_name(self._task_name)

        # Load fixed OSC_POSE controller configs (matches Cosmos eval)
        controller_configs = self._load_controller_configs()

        # Parse layout and style IDs
        layout_and_style_ids = cfg.get(
            "layout_and_style_ids", self._DEFAULT_LAYOUT_STYLE_IDS
        )

        # Create environment using robosuite.make() directly (matches Cosmos
        # evaluation pattern) with full kwargs for reproducible evals.
        env_kwargs = dict(
            env_name=self._resolved_task,
            robots=self._robot_type,
            controller_configs=controller_configs,
            camera_names=self._CAMERA_NAMES,
            camera_widths=camera_resolution,
            camera_heights=camera_resolution,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=cfg.get("seed", None),
            # Cosmos uses "B" (held-out test objects); our robocasa version
            # uses "target" for the equivalent split.
            obj_instance_split=cfg.get("obj_instance_split", "target"),
            generative_textures=None,
            randomize_cameras=cfg.get("randomize_cameras", False),
            layout_and_style_ids=layout_and_style_ids,
            translucent_robot=False,
        )
        self.env = robosuite.make(**env_kwargs)

        # Auto-reset so that /obs works immediately after /init, and run
        # stabilization steps (matching Cosmos eval warmup pattern).
        obs = self.env.reset()
        _NUM_WARMUP = 10
        dummy = np.zeros(self.env.action_spec[0].shape)
        for _ in range(_NUM_WARMUP):
            obs, _, _, _ = self.env.step(dummy)
        self._last_obs = obs

        # Derive a natural-language task description from the env name
        # e.g. "PickPlaceCounterToCabinet" → "pick place counter to cabinet"
        import re
        task_desc = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", self._resolved_task).lower()
        return {"task_description": task_desc}

    def reset(self, episode_index: int = None):
        """Reset environment and run warmup stabilization steps.

        ``episode_index`` is accepted but ignored (RoboCasa has no init states).
        """
        obs = self.env.reset()
        # Warmup: let objects settle (matches Cosmos eval pattern)
        _NUM_WARMUP = 10
        dummy = np.zeros(self.env.action_spec[0].shape)
        for _ in range(_NUM_WARMUP):
            obs, _, _, _ = self.env.step(dummy)
        self._last_obs = obs
        img = self._extract_primary_image(obs)
        img2 = self._extract_wrist_image(obs)
        return img, img2

    def step(self, action):
        """Step the environment.

        Accepts 7-dim arm actions and pads to 12-dim for PandaOmron.
        Accepts 16-dim GR00T Omron actions and trims the last 4 dims.
        Also accepts native-dim actions matching the robot's action space.
        Returns ``(image, image2, reward, done, info)``.
        """
        action = np.array(action, dtype=np.float64)
        # Pad 7-dim policy output → 12-dim env action for PandaOmron
        if action.shape[-1] == 7 and self._robot_type in ("PandaOmron", "PandaMobile"):
            action = np.concatenate([action, self._MOBILE_BASE_PAD])
        # Trim 16-dim GR00T Omron action → 12-dim env action
        elif action.shape[-1] == 16 and self._robot_type in ("PandaOmron", "PandaMobile"):
            # GR00T Omron action mapping:
            # 0-2: eef pos (3)
            # 3-5: eef rot (3)
            # 6: gripper (1) - but robosuite uses 1-dim gripper at the very end
            # 7-11: base motion (5)
            # 12-15: control modes (4)
            # RoboCasa Omron expects: 12-dim (7 arm + 5 base) -> [0-5 arm_pos_rot, 6-10 base, 11 gripper]
            trimmed = np.zeros(12, dtype=np.float64)
            trimmed[0:6] = action[0:6]   # arm pos + rot
            trimmed[6:11] = action[7:12] # base motion
            trimmed[11] = action[6]      # gripper
            action = trimmed
            
        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        img = self._extract_primary_image(obs)
        img2 = self._extract_wrist_image(obs)
        return img, img2, reward, done, info

    def get_obs(self):
        """Get current observation. Returns ``(image, image2)``."""
        if self._last_obs:
            return self._extract_primary_image(self._last_obs), self._extract_wrist_image(self._last_obs)
        # Fallback: query the env directly
        inner = getattr(self.env, 'env', self.env)
        obs = inner._get_observations()
        self._last_obs = obs
        return self._extract_primary_image(obs), self._extract_wrist_image(obs)

    def check_success(self):
        """Check task success via the environment's internal condition."""
        inner = getattr(self.env, 'env', self.env)
        return inner._check_success()

    def close(self):
        """Close the environment and free resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_state(self) -> list:
        """Extract 9-dim proprioceptive state: gripper_qpos(2) + eef_pos(3) + eef_quat(4)."""
        obs = self._last_obs
        parts = []
        for key in ("robot0_gripper_qpos", "robot0_eef_pos", "robot0_eef_quat"):
            if key in obs:
                parts.append(np.asarray(obs[key]).flatten())
        if parts:
            return np.concatenate(parts).tolist()
        return []

    def get_state_dict(self) -> dict:
        """Return the structured GR00T RoboCasa state using real sim observations."""
        obs = self._last_obs
        if not obs:
            return {}
        state = {}
        for src_key, dst_key in (
            ("robot0_base_to_eef_pos", "end_effector_position_relative"),
            ("robot0_base_to_eef_quat", "end_effector_rotation_relative"),
            ("robot0_gripper_qpos", "gripper_qpos"),
            ("robot0_base_pos", "base_position"),
            ("robot0_base_quat", "base_rotation"),
        ):
            if src_key in obs:
                state[dst_key] = np.asarray(obs[src_key], dtype=np.float32).flatten().tolist()
        return state

    @property
    def _action_dim(self) -> int:
        """Native action dimensionality based on robot type."""
        if self._robot_type in ("PandaOmron", "PandaMobile"):
            return 12
        return 7  # Panda

    def get_info(self) -> dict:
        """Return action/obs space metadata for auto-discovery.

        Reports the correct action dim based on robot type:
        - Panda: 7-dim (6 EEF delta + 1 gripper)
        - PandaOmron/PandaMobile: 12-dim (7 arm + 5 base)

        When using PandaOmron with a 7-dim policy, :meth:`step` handles
        padding to 12-dim internally.

        ``accepted_dims`` tells the orchestrator which VLA action dims this
        backend can consume.  PandaOmron/PandaMobile accept both 7 (arm-only,
        auto-padded to 12) and 12 (full mobile-base).  Panda only accepts 7.
        """
        cam_res = self._cam_res
        dim = self._action_dim
        max_steps = self._TASK_MAX_STEPS.get(self._resolved_task, 500)
        # PandaOmron/PandaMobile can accept 7-dim (arm-only, padded) or 12-dim (full)
        if self._robot_type in ("PandaOmron", "PandaMobile"):
            accepted_dims = [7, 12]
        else:
            accepted_dims = [dim]
        return {
            "action_space": {"type": "eef_delta", "dim": dim, "accepted_dims": accepted_dims},
            "obs_space": {
                "cameras": [
                    {"key": "robot0_agentview_left_image", "resolution": [cam_res, cam_res], "role": "primary"},
                    {"key": "robot0_agentview_right_image", "resolution": [cam_res, cam_res], "role": "secondary"},
                    {"key": "robot0_eye_in_hand_image", "resolution": [cam_res, cam_res], "role": "wrist"},
                ],
                "state": {"dim": 9, "format": "gripper_qpos(2)+eef_pos(3)+eef_quat(4)"},
            },
            "max_steps": max_steps,
            "delta_actions": True,
        }

    def _extract_primary_image(self, obs):
        """Extract the primary camera image (left agentview)."""
        for key in [
            "robot0_agentview_left_image",
            "robot0_agentview_image",
            "agentview_image",
            "robot0_robotview_image",
        ]:
            if key in obs:
                img = obs[key]
                return np.asarray(img, dtype=np.uint8).copy()
        raise KeyError(
            f"No primary camera image found in obs keys: {list(obs.keys())}"
        )

    def _extract_wrist_image(self, obs):
        """Extract the wrist camera image, or None if unavailable."""
        key = "robot0_eye_in_hand_image"
        if key in obs:
            return np.asarray(obs[key], dtype=np.uint8).copy()
        return None

    def _extract_secondary_image(self, obs):
        """Extract the secondary side camera image, or None if unavailable."""
        key = "robot0_agentview_right_image"
        if key in obs:
            return np.asarray(obs[key], dtype=np.uint8).copy()
        return None


class RoboTwinBackend:
    """Backend for RoboTwin 2.0 (SAPIEN-based bimanual, 14-dim actions).

    Uses setup_demo() / get_obs() / take_action() — NOT reset() / step().
    CRITICAL: torch must be imported before sapien in the same process.
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.env = None

    def initialize(self, seed: int = 42) -> dict:
        import sys
        import importlib
        import yaml

        # CRITICAL: import torch before any sapien import to avoid CUDA segfault.
        import torch  # noqa: F401

        _vendors_base = pathlib.Path(os.environ.get(
            "ROBO_EVAL_VENDORS_DIR",
            str(pathlib.Path.home() / ".local" / "share" / "robo-eval" / "vendors"),
        )).resolve()
        robotwin_dir = str(_vendors_base / "RoboTwin")
        os.chdir(robotwin_dir)
        if robotwin_dir not in sys.path:
            sys.path.insert(0, robotwin_dir)

        # S-0: Resolve integer task index → actual task module name.
        # The runner passes task_name as a decimal string ("0", "1", ...).
        # RoboTwin task modules live in envs/<task_name>.py; they are NOT numbered.
        if self.task_name.isdigit():
            envs_dir = pathlib.Path(robotwin_dir) / "envs"
            task_list = sorted(
                p.stem for p in envs_dir.glob("*.py")
                if not p.stem.startswith("_")
            )
            idx = int(self.task_name)
            if idx >= len(task_list):
                raise ValueError(
                    f"RoboTwin task index {idx} out of range (0-{len(task_list)-1}). "
                    f"Available tasks: {task_list}"
                )
            resolved = task_list[idx]
            import logging as _logging
            _logging.getLogger(__name__).info(
                "RoboTwin: resolved task index %s -> '%s'", idx, resolved
            )
            self.task_name = resolved

        from envs import CONFIGS_PATH

        with open("./task_config/demo_clean.yml", "r", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
            emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        robot_file = emb_types["aloha-agilex"]["file_path"]
        with open(os.path.join(robot_file, "config.yml"), "r", encoding="utf-8") as f:
            emb_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), "r", encoding="utf-8") as f:
            cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_cam_type = args["camera"]["head_camera_type"]
        args.update({
            "task_name": self.task_name,
            "left_robot_file": robot_file,
            "right_robot_file": robot_file,
            "dual_arm_embodied": True,
            "left_embodiment_config": emb_cfg,
            "right_embodiment_config": emb_cfg,
            "head_camera_h": cam_cfg[head_cam_type]["h"],
            "head_camera_w": cam_cfg[head_cam_type]["w"],
            "eval_mode": True,
            "eval_video_save_dir": None,
            "save_freq": None,
        })

        module = importlib.import_module(f"envs.{self.task_name}")
        env_class = getattr(module, self.task_name)
        self.env = env_class()
        self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        self.env.set_instruction(self.task_name.replace("_", " "))

        return {"status": "ok", "action_dim": 14}

    def reset(self, seed: int = 42) -> dict:
        if self.env is not None:
            self.env.close_env()
            self.env = None
        return self.initialize(seed)

    def get_obs(self) -> dict:
        obs = self.env.get_obs()
        head_b64 = encode_image_b64(obs["observation"]["head_camera"]["rgb"])
        # Role-keyed images dict (canonical) + camera-name keys (backward compat)
        images = {"head_camera": head_b64, "primary": head_b64}
        # S-1: joint_action["vector"] is the current joint positions (confirmed from
        # official InternVLA inference.py which uses this key for state).
        joint_pos = obs["joint_action"]["vector"]
        result: dict = {
            "images": images,
            "image": head_b64,  # backward compat for env_wrapper
            "state": joint_pos.tolist() if hasattr(joint_pos, "tolist") else list(joint_pos),
        }
        # Add wrist cameras with both camera-name and role keys.
        obs_cameras = obs.get("observation", {})
        left_b64 = None
        for cam in ("left_camera", "left_wrist", "cam_left_wrist"):
            if cam in obs_cameras:
                left_b64 = encode_image_b64(obs_cameras[cam]["rgb"])
                images[cam] = left_b64
                break
        right_b64 = None
        for cam in ("right_camera", "right_wrist", "cam_right_wrist"):
            if cam in obs_cameras:
                right_b64 = encode_image_b64(obs_cameras[cam]["rgb"])
                images[cam] = right_b64
                break
        if left_b64 is not None:
            images["wrist"] = left_b64
            result["image2"] = left_b64   # backward compat
        if right_b64 is not None:
            images["secondary"] = right_b64
            result["image3"] = right_b64  # backward compat
        return result

    def step(self, action: list) -> dict:
        self.env.take_action(np.array(action))
        # Flatten obs into top-level response so env_wrapper can access
        # resp["image"], resp["state"], resp["done"], resp["success"] directly.
        obs_dict = self.get_obs()
        return {
            **obs_dict,
            "success": bool(self.env.eval_success),
            "done": bool(
                self.env.eval_success or self.env.take_action_cnt >= self.env.step_lim
            ),
        }

    def get_info(self) -> dict:
        """Return action/obs space metadata for auto-discovery."""
        step_lim = getattr(self.env, "step_lim", 600) if self.env is not None else 600
        return {
            "action_space": {"type": "joint_pos", "dim": 14, "accepted_dims": [14, 32]},
            "obs_space": {
                "cameras": [
                    {"key": "head_camera", "resolution": [480, 640], "role": "primary"},
                ],
                "state": {"dim": 14, "format": "joint_positions(14)"},
            },
            "max_steps": step_lim,
            "delta_actions": False,
        }

    def check_success(self) -> bool:
        """Return True if the current episode was successful."""
        if self.env is None:
            return False
        return bool(getattr(self.env, "eval_success", False))

    def close(self):
        if self.env is not None:
            self.env.close_env()
            self.env = None


class LiberoInfinityBackend:
    """Backend for LIBERO-Infinity: Scenic-based perturbation testing.

    Uses the lower-level LIBEROSimulator + LIBEROSimulation API so that
    the BDDL context manager (temp file) stays alive for the full episode.

    Lifecycle per episode
    ─────────────────────
    init()  → resolve BDDL, generate Scenic file, compile scenario once
    reset() → sample scene, enter BDDL context, createSimulation + setup
    step()  → drive policy action through sim
    close() → destroy sim, exit BDDL context, unlink Scenic file
    """

    def __init__(self):
        self._scenario = None         # compiled Scenic scenario (cached across resets)
        self._sim = None              # current LIBEROSimulation
        self._bddl_ctx = None         # active bddl_for_scene() context manager
        self._bddl_path = None        # path to original BDDL file
        self._orig_obj_classes = None # {instance_name: class_name} from BDDL
        self._perturbation = "position"
        self._max_steps = 300
        self._env_kwargs: dict = {}
        self._scenic_path = None      # generated .scenic file (cleaned up on close)
        self._camera_resolution = 256
        self._run_seed = 42           # base seed for deterministic episode seeding
        self._ep_idx = 0              # auto-incrementing episode counter
        self._last_obs: dict = {}     # cache last obs dict for state extraction
        self._max_reset_attempts = 5
        self._post_reset_settle_steps = 80
        self._post_reset_stable_steps = 10
        self._post_reset_pos_tol = 0.002
        self._post_reset_vel_tol = 0.03
        self._post_reset_rot_tol = 0.03
        self._post_reset_ang_vel_tol = 0.2
        self._post_reset_target_rot_tol = 0.35

    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str = None,
        headless: bool = True,
        sim_config: dict = None,
    ):
        """Resolve BDDL, compile Scenic, and prime the first observation.

        Args:
            task_name: Integer index or name substring of the BDDL task.
            camera_resolution: Image resolution for both cameras.
            suite: Suite name (e.g. "libero_infinity_spatial"). Required.
            headless: If True, use EGL offscreen rendering.
            sim_config: Optional dict with keys:
                perturbation  (str)  — default "position"
                max_steps     (int)  — default 300
                seed          (int)  — base RNG seed, default 42
                max_distractors (int)
                min_distractors (int)
        """
        os.environ["MUJOCO_GL"] = "egl" if headless else "glfw"

        sim_config = sim_config or {}
        self._camera_resolution = camera_resolution
        self._perturbation = sim_config.get("perturbation", "position")
        self._max_steps = sim_config.get("max_steps", 300)
        self._run_seed = sim_config.get("seed", 42)
        self._max_reset_attempts = max(1, int(sim_config.get("max_reset_attempts", 5)))
        self._post_reset_settle_steps = max(
            0, int(sim_config.get("post_reset_settle_steps", 80))
        )
        self._post_reset_stable_steps = max(
            1, int(sim_config.get("post_reset_stable_steps", 10))
        )
        self._post_reset_pos_tol = float(sim_config.get("post_reset_pos_tol", 0.002))
        self._post_reset_vel_tol = float(sim_config.get("post_reset_vel_tol", 0.03))
        self._post_reset_rot_tol = float(sim_config.get("post_reset_rot_tol", 0.03))
        self._post_reset_ang_vel_tol = float(
            sim_config.get("post_reset_ang_vel_tol", 0.2)
        )
        self._post_reset_target_rot_tol = float(
            sim_config.get("post_reset_target_rot_tol", 0.35)
        )

        # Build distractor kwargs to forward to generate_scenic_file
        distractor_kwargs = {}
        if "max_distractors" in sim_config:
            distractor_kwargs["max_distractors"] = int(sim_config["max_distractors"])
        if "min_distractors" in sim_config:
            distractor_kwargs["min_distractors"] = int(sim_config["min_distractors"])

        # env_kwargs forwarded to OffScreenRenderEnv inside LIBEROSimulation.setup()
        self._env_kwargs = {
            "camera_heights": camera_resolution,
            "camera_widths": camera_resolution,
            "has_renderer": not headless,
            "has_offscreen_renderer": True,
            "use_camera_obs": True,
        }

        # Resolve the BDDL file path for this (suite, task_name) pair
        self._bddl_path = self._resolve_bddl_path(suite, task_name)

        # Parse the BDDL to get task metadata and generate the Scenic program
        from libero_infinity.task_config import TaskConfig
        from libero_infinity.scenic_generator import generate_scenic_file

        cfg = TaskConfig.from_bddl(self._bddl_path)

        # Clean up any previous generated Scenic file
        if self._scenic_path is not None:
            pathlib.Path(self._scenic_path).unlink(missing_ok=True)
            self._scenic_path = None

        self._scenic_path = generate_scenic_file(
            cfg,
            perturbation=self._perturbation,
            **distractor_kwargs,
        )

        # Compile the Scenic scenario once (expensive — involves Scenic
        # parsing + Python code generation). Reused across all resets.
        import scenic

        self._scenario = scenic.scenarioFromFile(
            self._scenic_path,
            params={"bddl_path": self._bddl_path},
        )

        # Cache original object classes for BDDL asset substitution
        from libero_infinity.bddl_preprocessor import parse_object_classes

        self._orig_obj_classes = parse_object_classes(
            pathlib.Path(self._bddl_path).read_text()
        )

        # Match the sim-worker contract expected by SimWrapper: /init must leave
        # the backend ready for an immediate /obs call. LIBERO-Infinity only
        # compiles the Scenic scenario here, so materialize the first sampled
        # scene now and then restore the episode counter for the real eval loop.
        self.reset(episode_index=0)
        self._ep_idx = 0

        return {"task_description": cfg.language}

    def reset(self, episode_index: int = None):
        """Sample a new perturbed scene and set up the LIBERO simulation.

        Seeds the RNG deterministically as sha256(run_seed:episode_index)
        so each (run, episode) pair produces a unique but reproducible scene.

        Returns:
            (img, img2): agentview + wrist-camera images after setup.
        """
        import random
        from libero_infinity.bddl_preprocessor import bddl_for_scene
        from libero_infinity.simulator import LIBEROSimulator

        # Destroy previous simulation
        if self._sim is not None:
            try:
                self._sim.destroy()
            except Exception:
                pass
            self._sim = None

        # Exit previous BDDL context (releases temp file if any)
        if self._bddl_ctx is not None:
            try:
                self._bddl_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._bddl_ctx = None

        # Compute a deterministic seed for this episode
        ep_idx = episode_index if episode_index is not None else self._ep_idx
        self._ep_idx = ep_idx + 1
        last_exc = None

        for attempt in range(self._max_reset_attempts):
            seed_bytes = f"{self._run_seed}:{ep_idx}:{attempt}".encode()
            seed_i = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2 ** 31)

            # Seed Python and NumPy RNGs so Scenic sampling is reproducible
            random.seed(seed_i)
            np.random.seed(seed_i)

            try:
                # Sample a scene from the compiled Scenic scenario
                scene, _ = self._scenario.generate(maxIterations=1000, verbosity=0)

                # Open the BDDL context manager manually so the temp file stays alive
                # for the entire episode (not just until the end of this method).
                self._bddl_ctx = bddl_for_scene(scene, self._bddl_path, self._orig_obj_classes)
                effective_bddl = self._bddl_ctx.__enter__()

                # Create and set up the LIBERO simulation
                simulator = LIBEROSimulator(bddl_path=effective_bddl, env_kwargs=self._env_kwargs)
                self._sim = simulator.createSimulation(
                    scene, maxSteps=self._max_steps, verbosity=0
                )
                self._sim.setup()
                self._post_reset_settle()

                obs = self._sim.last_obs or {}
                self._last_obs = obs
                return self._extract_images(obs)
            except RuntimeError as exc:
                if "Invalid Scenic sample after MuJoCo settling" not in str(exc):
                    raise
                last_exc = exc
                if self._sim is not None:
                    try:
                        self._sim.destroy()
                    except Exception:
                        pass
                    self._sim = None
                if self._bddl_ctx is not None:
                    try:
                        self._bddl_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                    self._bddl_ctx = None

        if last_exc is not None:
            raise RuntimeError(
                f"{last_exc} (after {self._max_reset_attempts} Scenic resample attempts)"
            ) from last_exc
        raise RuntimeError("failed to reset LIBERO-Infinity scene")

    def step(self, action):
        """Execute one action step.

        Returns:
            (img, img2, reward, done, info)
        """
        obs, reward, done, info = self._sim.step_with_action(np.asarray(action))
        self._last_obs = obs if obs else {}
        img, img2 = self._extract_images(obs)
        return img, img2, reward, bool(done), info

    def get_obs(self):
        """Return current observation without stepping.

        NOTE: ``self._sim.last_obs`` is cached from the last step() call and
        is therefore stale — it reflects the state *after* the most recent
        action, not a freshly queried frame.  LIBEROSimulation does not
        expose a method to query a new observation without stepping, so this
        is the best we can do.  Callers that need a guaranteed-fresh frame
        should call reset() or step() first.

        Returns:
            (img, img2)
        """
        obs = self._sim.last_obs or {}
        self._last_obs = obs
        return self._extract_images(obs)

    def check_success(self):
        """Return True if the task success condition is met."""
        return bool(self._sim.check_success()) if self._sim is not None else False

    def close(self):
        """Release all resources: sim, BDDL context, generated Scenic file."""
        if self._sim is not None:
            try:
                self._sim.destroy()
            except Exception:
                pass
            self._sim = None

        if self._bddl_ctx is not None:
            try:
                self._bddl_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._bddl_ctx = None

        if self._scenic_path is not None:
            pathlib.Path(self._scenic_path).unlink(missing_ok=True)
            self._scenic_path = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_state(self, obs: dict) -> list:
        """Extract proprioceptive state: eef_pos(3) + axisangle(3) + gripper(2) = 8-dim.

        Mirrors LiberoBackend._extract_state() so that VLAs requiring
        proprioceptive state work identically with both backends.

        Returns a plain list[float] for direct use by the policy server.
        """
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)       # (3,)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)     # (4,) [x,y,z,w]
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)  # (2,)

        # Convert quaternion [x,y,z,w] to axis-angle using lerobot's _quat2axisangle
        # formula (matches training data). Scipy as_rotvec() differs in edge-case handling.
        x, y, z, w = float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3])
        den = np.sqrt(max(0.0, 1.0 - w * w))
        if den > 1e-10:
            angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
            axisangle = np.array([x, y, z], dtype=np.float32) / den * angle
        else:
            axisangle = np.zeros(3, dtype=np.float32)

        state = np.concatenate([eef_pos, axisangle, gripper])  # (8,)
        return state.tolist()

    def _post_reset_settle(self) -> None:
        """Advance zero-action control steps until the sampled scene is stable."""
        if self._sim is None or self._post_reset_settle_steps <= 0:
            return

        env_wrapper = self._sim.libero_env
        env = env_wrapper.env
        sim = env.sim
        zero_action = np.zeros(env.action_spec[0].shape[0], dtype=float)
        saved_timestep = getattr(env, "timestep", 0)
        saved_cur_time = getattr(env, "cur_time", 0.0)
        saved_done = getattr(env, "done", False)

        body_ids = []
        body_names = []
        for obj in self._sim.scene.objects:
            libero_name = getattr(obj, "libero_name", None)
            if not libero_name or libero_name.startswith("distractor_"):
                continue
            for candidate in (libero_name, libero_name + "_main"):
                try:
                    body_ids.append(sim.model.body_name2id(candidate))
                    body_names.append(libero_name)
                    break
                except Exception:
                    continue

        if not body_ids:
            obs = env._get_observations(force_update=True)
            self._sim._last_obs = obs
            self._last_obs = obs
            return

        prev_pos = np.stack(
            [np.array(sim.data.body_xpos[bid][:3], dtype=float) for bid in body_ids]
        )
        prev_rot = np.stack(
            [np.array(sim.data.body_xmat[bid], dtype=float).reshape(3, 3) for bid in body_ids]
        )
        target_rot = prev_rot.copy()
        stable_steps = 0
        max_pos_delta = float("inf")
        max_rot_delta = float("inf")
        max_speed = float("inf")
        max_ang_speed = float("inf")
        obs = None

        for _ in range(self._post_reset_settle_steps):
            obs, _reward, _done, _info = env_wrapper.step(zero_action)

            curr_pos = np.stack(
                [np.array(sim.data.body_xpos[bid][:3], dtype=float) for bid in body_ids]
            )
            curr_rot = np.stack(
                [np.array(sim.data.body_xmat[bid], dtype=float).reshape(3, 3) for bid in body_ids]
            )
            linear_speeds = np.array(
                [np.linalg.norm(sim.data.cvel[bid][3:]) for bid in body_ids],
                dtype=float,
            )
            angular_speeds = np.array(
                [np.linalg.norm(sim.data.cvel[bid][:3]) for bid in body_ids],
                dtype=float,
            )
            pos_deltas = np.linalg.norm(curr_pos - prev_pos, axis=1)
            rel_rot = np.einsum("nij,njk->nik", np.transpose(prev_rot, (0, 2, 1)), curr_rot)
            rot_deltas = np.arccos(
                np.clip((np.trace(rel_rot, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
            )
            max_pos_delta = float(np.max(pos_deltas))
            max_rot_delta = float(np.max(rot_deltas))
            max_speed = float(np.max(linear_speeds))
            max_ang_speed = float(np.max(angular_speeds))

            if (
                max_pos_delta <= self._post_reset_pos_tol
                and max_rot_delta <= self._post_reset_rot_tol
                and max_speed <= self._post_reset_vel_tol
                and max_ang_speed <= self._post_reset_ang_vel_tol
            ):
                stable_steps += 1
                if stable_steps >= self._post_reset_stable_steps:
                    break
            else:
                stable_steps = 0

            prev_pos = curr_pos
            prev_rot = curr_rot

        env.timestep = saved_timestep
        env.cur_time = saved_cur_time
        env.done = saved_done
        self._sim._done = False
        if obs is None:
            obs = env._get_observations(force_update=True)
        self._sim._last_obs = obs
        self._last_obs = obs

        final_rot = np.stack(
            [np.array(sim.data.body_xmat[bid], dtype=float).reshape(3, 3) for bid in body_ids]
        )
        rel_target_rot = np.einsum(
            "nij,njk->nik", np.transpose(target_rot, (0, 2, 1)), final_rot
        )
        target_rot_error = np.arccos(
            np.clip(
                (np.trace(rel_target_rot, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
        )
        toppled = [
            f"{name} rotated {err:.2f} rad from its intended pose"
            for name, err in zip(body_names, target_rot_error)
            if err > self._post_reset_target_rot_tol
        ]

        if toppled:
            raise RuntimeError(
                "Invalid Scenic sample after MuJoCo settling: " + "; ".join(toppled)
            )

        if stable_steps < self._post_reset_stable_steps:
            raise RuntimeError(
                "Invalid Scenic sample after MuJoCo settling: "
                f"scene remained in motion (max_pos_delta={max_pos_delta:.4f}, "
                f"max_rot_delta={max_rot_delta:.4f}, max_speed={max_speed:.4f}, "
                f"max_ang_speed={max_ang_speed:.4f})"
            )

    def get_info(self) -> dict:
        """Return action/obs space metadata for auto-discovery."""
        cam_res = self._camera_resolution
        return {
            "action_space": {"type": "eef_delta", "dim": 7, "accepted_dims": [7]},
            "obs_space": {
                "cameras": [
                    {"key": "agentview_image", "resolution": [cam_res, cam_res], "role": "primary"},
                    {"key": "robot0_eye_in_hand_image", "resolution": [cam_res, cam_res], "role": "wrist"},
                ],
                "state": {"dim": 8, "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"},
            },
            "max_steps": self._max_steps,
            "delta_actions": getattr(self, "delta_actions", False),
        }

    def _extract_images(self, obs: dict):
        """Extract (agentview, wrist-camera) images from an obs dict.

        Returns raw images without any transforms.  Image flipping is now
        applied by the orchestration layer (env_wrapper) based on the VLA's
        declared ``obs_requirements.image_transform``.

        Returns:
            (img, img2) where img2 is None if no wrist camera in obs.
        """
        image = obs.get("agentview_image")
        image2 = obs.get("robot0_eye_in_hand_image")
        if image is None and image2 is None:
            raise KeyError(
                f"No camera image found in obs keys: {list(obs.keys())}"
            )
        primary = image if image is not None else image2
        img = np.asarray(primary, dtype=np.uint8).copy()
        img2 = (
            np.asarray(image2, dtype=np.uint8).copy()
            if image2 is not None
            else None
        )
        return img, img2

    def _resolve_bddl_path(self, suite: str, task_name: str) -> str:
        """Resolve a BDDL file path from (suite, task_name).

        Supports three task_name formats:
          - Integer string ("0", "10"): index into sorted file list
          - Exact stem match: returns unique match or raises
          - Substring match: returns unique match or raises (ambiguous / missing)

        Args:
            suite: Suite name, e.g. "libero_infinity_spatial".
            task_name: Integer index or task name (exact or substring).

        Returns:
            Absolute path to the BDDL file as a string.

        Raises:
            ValueError: If suite is None, directory not found, no files found,
                        task not found, or task name is ambiguous.
        """
        import libero_infinity as _li_pkg

        if suite is None:
            raise ValueError(
                "suite is required for LiberoInfinityBackend "
                "(e.g. 'libero_infinity_spatial')"
            )

        pkg_root = pathlib.Path(_li_pkg.__file__).parent

        # "libero_infinity_spatial" → strip prefix → "spatial" → prepend "libero_" → "libero_spatial"
        raw_suite = "libero_" + suite.removeprefix("libero_infinity_")
        bddl_dir = pkg_root / "data" / "libero_runtime" / "bddl_files" / raw_suite

        if not bddl_dir.exists():
            raise ValueError(
                f"BDDL directory not found: {bddl_dir}. "
                f"Available suites: {[d.name for d in bddl_dir.parent.iterdir() if d.is_dir()]}"
            )

        bddl_files = sorted(f for f in bddl_dir.iterdir() if f.suffix == ".bddl")
        if not bddl_files:
            raise ValueError(f"No BDDL files found in {bddl_dir}")

        # Integer index
        if task_name.isdigit():
            idx = int(task_name)
            if idx >= len(bddl_files):
                raise ValueError(
                    f"Task index {idx} out of range "
                    f"(suite '{raw_suite}' has {len(bddl_files)} tasks)"
                )
            return str(bddl_files[idx])

        # Exact stem match (highest priority — avoids substring ambiguity)
        exact = [f for f in bddl_files if f.stem == task_name]
        if len(exact) == 1:
            return str(exact[0])
        if len(exact) > 1:
            raise ValueError(
                f"Ambiguous exact match for task '{task_name}': "
                f"{[f.name for f in exact]}"
            )

        # Substring match
        matches = [f for f in bddl_files if task_name.lower() in f.stem.lower()]
        if len(matches) == 1:
            return str(matches[0])
        if not matches:
            raise ValueError(
                f"Task '{task_name}' not found in {bddl_dir}. "
                f"Available: {[f.stem for f in bddl_files]}"
            )
        raise ValueError(
            f"Ambiguous task name '{task_name}' matches multiple files in "
            f"'{raw_suite}': {[f.name for f in matches]}"
        )


# ======================================================================
# Backend registry — add new backends here (key must match SimConfig.name)
# ======================================================================

BACKENDS = {
    "libero": LiberoBackend,
    "robocasa": RoboCasaBackend,
    "robotwin": RoboTwinBackend,
    "libero_pro": LiberoProBackend,
    "libero_infinity": LiberoInfinityBackend,
}


# ======================================================================
# Pydantic request models
# ======================================================================


class InitRequest(BaseModel):
    sim: str
    task: str
    suite: Optional[str] = None
    camera_resolution: int = 256
    # headless=True → EGL offscreen (CI/server default).
    # headless=False → GLFW window with live viewer.
    # If None, falls back to the server-level --headless CLI flag.
    headless: Optional[bool] = None
    # delta_actions=True → set robot.controller.use_delta=True after each reset.
    # Required for Pi0.5 which outputs relative/delta actions.
    delta_actions: bool = False
    # sim_config: opaque dict forwarded to the backend's init(). Used by
    # LiberoInfinityBackend (perturbation, max_steps, seed, max_distractors…).
    # All other backends accept and ignore it.
    sim_config: Optional[dict] = None


class StepRequest(BaseModel):
    action: list


class ResetRequest(BaseModel):
    seed: int = 42
    episode_index: Optional[int] = None


# ======================================================================
# FastAPI application
# ======================================================================

app = FastAPI(title="robo-eval Sim Worker")
backend = None
_current_sim_name: str = ""
_current_suite: str = ""


@app.get("/health")
def health():
    """Health check: returns whether a backend has been initialized."""
    return {"status": "ok", "backend_initialized": backend is not None}


@app.get("/info")
def sim_info():
    """Return the sim's self-reported action/obs space capabilities.

    Only available after /init has been called (backend must be initialized).
    """
    if backend is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Backend not initialized. Call /init first."},
        )
    try:
        info = backend.get_info()
        info["sim"] = _current_sim_name
        info["suite"] = _current_suite
        return info
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/init")
def init_env(req: InitRequest):
    """Initialize (or re-initialize) the simulator backend for a task."""
    global backend, _current_sim_name, _current_suite
    try:
        _current_sim_name = req.sim
        _current_suite = req.suite or ""
        if req.sim not in BACKENDS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unknown sim '{req.sim}'. Must be one of: {list(BACKENDS.keys())}"},
            )
        # req.headless=None means "use server default" set by --headless CLI flag.
        use_headless = req.headless if req.headless is not None else _headless
        # Close the old backend before creating a new one to free memory.
        # Without this, each /init call leaks an OffScreenRenderEnv (MuJoCo
        # scene + GPU resources), eventually causing OOM after ~4 tasks.
        old_backend = backend
        if old_backend is not None:
            try:
                old_backend.close()
            except Exception:
                pass  # best-effort cleanup

        if req.sim == "robotwin":
            # RoboTwin takes task_name at construction time and uses initialize().
            # Initialize before assigning to global to avoid race: a concurrent
            # /step would see backend.env=None if we assigned before init().
            new_backend = RoboTwinBackend(task_name=req.task)
            result = new_backend.initialize(seed=42)
            backend = new_backend
        else:
            backend_cls = BACKENDS[req.sim]
            # Initialize before assigning to global to avoid race condition:
            # if we did `backend = backend_cls()` first, a concurrent /step
            # would see the new backend with env=None before init() sets it.
            new_backend = backend_cls()
            result = new_backend.init(
                task_name=req.task,
                camera_resolution=req.camera_resolution,
                suite=req.suite,
                headless=use_headless,
                sim_config=req.sim_config,
            )
            # Store delta_actions flag so reset() can apply use_delta.
            new_backend.delta_actions = req.delta_actions
            backend = new_backend
        return {"success": True, **(result or {})}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = None):
    """Reset the environment to a specific episode init state."""
    try:
        global backend
        
        # If the environment crashed in a previous episode and set self.env = None,
        # we need to re-initialize it before resetting.
        if backend is not None and hasattr(backend, "env") and backend.env is None:
            # Re-call the init method with the saved config
            import logging
            logging.warning("backend.env is None during /reset, attempting to re-initialize.")
            if isinstance(backend, RoboTwinBackend):
                backend.initialize()
            else:
                # We need to recreate the offscreen render env.
                # The easiest way is to let the client handle it by returning a 503,
                # but we can try to re-init here if we have the args.
                pass # The client expects a reset, but if env is None, the backend methods will fail.
                # Actually, raising an explicit error telling the client to re-init is safer.
                return JSONResponse(
                    status_code=503,
                    content={"error": "backend.env is None. Environment crashed. Please call /init again.", "traceback": ""},
                )

        if isinstance(backend, RoboTwinBackend):
            seed = req.seed if req is not None else 42
            result = backend.reset(seed=seed)
            # After reset, fetch an initial observation so env_wrapper gets
            # image / state / image2 / image3 in the reset response.
            try:
                obs_result = backend.get_obs()
                result.update(obs_result)
            except Exception:
                pass  # If get_obs fails after reset, fall back to metadata-only response
            return {"success": True, **result}

        # LiberoBackend, LiberoProBackend, and RoboCasaBackend all return
        # (image, image2) tuples.  episode_index is forwarded where supported.
        episode_index = req.episode_index if req is not None else None
        img, img2 = backend.reset(episode_index=episode_index)
        # Extract optional secondary (third) camera image
        img3 = None
        if hasattr(backend, "_extract_secondary_image") and getattr(backend, "_last_obs", None):
            img3 = backend._extract_secondary_image(backend._last_obs)
        resp = _build_images_response(img, img2, img3)
        resp["success"] = True
        if hasattr(backend, '_extract_state') and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, 'get_state'):
            resp["state"] = backend.get_state()
        if hasattr(backend, "get_state_dict"):
            state_dict = backend.get_state_dict()
            if state_dict:
                resp["state_dict"] = state_dict
        return resp
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/step")
def step_env(req: StepRequest):
    """Execute one action step and return the new observation."""
    try:
        if isinstance(backend, RoboTwinBackend):
            # RoboTwin has its own response format.
            # Truncate padded actions (e.g. 32-dim from InternVLA) to native 14-dim.
            action = req.action[:14] if len(req.action) > 14 else req.action
            return backend.step(action)

        # LiberoBackend, LiberoProBackend, RoboCasaBackend all return
        # (image, image2, reward, done, info).
        img, img2, reward, done, info = backend.step(req.action)
        # Extract optional secondary (third) camera image
        img3 = None
        if hasattr(backend, "_extract_secondary_image") and getattr(backend, "_last_obs", None):
            img3 = backend._extract_secondary_image(backend._last_obs)
        resp = _build_images_response(img, img2, img3)
        resp["reward"] = float(reward) if reward is not None else 0.0
        resp["done"] = bool(done)
        resp["success"] = bool(info.get("success", False)) if isinstance(info, dict) else False
        # Include proprioceptive state for LIBERO backends
        if hasattr(backend, '_extract_state') and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, 'get_state'):
            resp["state"] = backend.get_state()
        if hasattr(backend, "get_state_dict"):
            state_dict = backend.get_state_dict()
            if state_dict:
                resp["state_dict"] = state_dict
        return resp
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/obs")
def get_obs():
    """Fetch the current observation (image + state) without stepping."""
    try:
        if isinstance(backend, RoboTwinBackend):
            return backend.get_obs()

        # LiberoBackend, LiberoProBackend, RoboCasaBackend all return (image, image2).
        img, img2 = backend.get_obs()
        # Extract optional secondary (third) camera image
        img3 = None
        if hasattr(backend, "_extract_secondary_image") and getattr(backend, "_last_obs", None):
            img3 = backend._extract_secondary_image(backend._last_obs)
        resp = _build_images_response(img, img2, img3)
        # Include proprioceptive state for LIBERO backends
        if hasattr(backend, '_extract_state') and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, 'get_state'):
            resp["state"] = backend.get_state()
        if hasattr(backend, "get_state_dict"):
            state_dict = backend.get_state_dict()
            if state_dict:
                resp["state_dict"] = state_dict
        return resp
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/success")
def get_success():
    """Check whether the current task's success condition is met."""
    try:
        success = backend.check_success()
        return {"success": bool(success)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/close")
def close_env():
    """Close the backend and shut down the server process."""
    try:
        if backend is not None:
            backend.close()
        # Schedule shutdown after response is sent
        os.kill(os.getpid(), signal.SIGTERM)
        return {"success": True}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


# ======================================================================
# CLI entrypoint
# ======================================================================


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="robo-eval Sim Worker HTTP Server")
    parser.add_argument(
        "--sim",
        required=True,
        choices=list(BACKENDS.keys()),
        help="Which simulator backend to use.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to serve on (default: 5001).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help=(
            "Enable headless EGL offscreen rendering (no display required). "
            "Use this for CI/servers. Default: windowed GLFW rendering."
        ),
    )
    args = parser.parse_args()

    # Set the module-level default and MUJOCO_GL *before* uvicorn starts so
    # that the env var is in place before any MuJoCo import happens anywhere
    # in the process. Individual backends also re-assert it in their init().
    global _headless
    _headless = args.headless
    os.environ["MUJOCO_GL"] = "egl" if args.headless else "glfw"

    logger.info(
        "Starting %s server on %s:%d (%s)",
        args.sim, args.host, args.port,
        "headless/EGL" if args.headless else "windowed/GLFW",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
