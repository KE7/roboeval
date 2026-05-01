#!/usr/bin/env python
"""
Simulator HTTP server for roboeval.

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
import logging
import os
import pathlib
import signal
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager as _contextmanager
from io import BytesIO

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
    img2: np.ndarray | None = None,
    img3: np.ndarray | None = None,
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
# Abstract base class for all simulator backends
# ======================================================================


class SimBackendBase(ABC):
    """Abstract base class for all roboeval simulator backends.

    Mirrors :class:`sims.vla_policies.base.VLAPolicyBase` on the VLA side.
    Every new sim backend **must** subclass this class and implement all
    abstract methods.  The ``BACKENDS`` dict at module level serves as the
    name→class registry; ``SimBackendBase`` provides the type contract.

    Lifecycle per episode::

        backend = BACKENDS["libero"]()
        backend.init(task_name, camera_resolution, suite, headless)
        image, image2 = backend.reset(episode_index=0)
        while not done:
            image, image2, reward, done, info = backend.step(action)
        backend.close()

    Required abstract methods
    -------------------------
    ``init``           — load the simulator, resolve task, fetch init states.
    ``reset``          — reset to a known init state; return initial image pair.
    ``step``           — apply one action; return (image, image2, reward, done, info).
    ``get_obs``        — return current (image, image2) without stepping.
    ``check_success``  — return whether the current task has been solved.
    ``close``          — teardown (free env, GPU memory, etc.).
    ``get_info``       — return the ``/info`` metadata dict including typed
                         ``action_spec`` / ``observation_spec`` blocks.
    """

    @abstractmethod
    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        """Initialize the simulator for a given task.

        Must set up the environment so that :meth:`reset` can be called
        immediately afterwards.

        Returns
        -------
        dict
            At minimum ``{"task_description": str}``.
        """

    @abstractmethod
    def reset(self, episode_index: int | None = None):
        """Reset to a known init state.

        Parameters
        ----------
        episode_index:
            Which init state to use.  ``None`` uses an auto-incrementing counter.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            ``(primary_image, wrist_image)`` — wrist image may be ``None``.
        """

    @abstractmethod
    def step(self, action):
        """Apply one action and advance the simulation by one step.

        Returns
        -------
        tuple
            ``(primary_image, wrist_image, reward, done, info)``
        """

    @abstractmethod
    def get_obs(self):
        """Return the current observation without stepping.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            ``(primary_image, wrist_image)``
        """

    @abstractmethod
    def check_success(self) -> bool:
        """Return whether the current task has been solved."""

    @abstractmethod
    def close(self) -> None:
        """Teardown — free the environment, GPU memory, open files, etc."""

    @abstractmethod
    def get_info(self) -> dict:
        """Return the ``/info`` metadata dict.

        Must include at minimum:
        - ``"action_space"`` — legacy action space descriptor.
        - ``"action_spec"`` — typed :class:`roboeval.specs.ActionObsSpec`
          dicts (serialized via ``ActionObsSpec.to_dict()``).
        - ``"observation_spec"`` — same format for observations.
        """


# ======================================================================
# Simulator-specific backends
#
# To add a new backend:
# 1. Subclass SimBackendBase and implement all abstract methods.
#    See docs/extending.md for details.
# 2. Return conventions: reset() -> (image, image2), step() -> (image,
#    image2, reward, done, info), get_obs() -> (image, image2).
#    image2 is None if no wrist camera.
# 3. Declare typed ActionObsSpec contracts in get_info() under
#    "action_spec" and "observation_spec" keys.
# 4. Register your class in the BACKENDS dict at the end of this section.
# ======================================================================


class LiberoBackend(SimBackendBase):
    """Backend for LIBERO benchmark environments."""

    def __init__(self):
        self.env = None
        self.benchmark = None
        self.task = None
        self._ep_idx = 0  # tracks which init_state to use per episode
        self._last_obs: dict = {}  # cache for state extraction
        self._last_img2 = None  # cache for wrist-camera image (image2)

    def _find_task_idx(self, task_name, task_names, suite):
        """Find task index by name or numeric index."""
        if task_name.isdigit():
            return int(task_name)
        matching = [i for i, t in enumerate(task_names) if task_name.lower() in t.lower()]
        if not matching:
            raise ValueError(f"Task '{task_name}' not found in {suite}. Available: {task_names}")
        return matching[0]

    def _get_init_states(self, task_idx):
        """Load episode init states for a task. Subclasses may override."""
        import os

        import torch
        from libero.libero import get_libero_path

        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.benchmark.tasks[task_idx].problem_folder,
            self.benchmark.tasks[task_idx].init_states_file,
        )
        # weights_only=False required for legacy LIBERO .pruned_init files which
        # contain numpy arrays (torch.load default changed to True in PyTorch 2.6+).
        return torch.load(init_states_path, weights_only=False)

    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str = None,
        headless: bool = True,
        sim_config: dict = None,
    ):
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
        if (
            not getattr(self, "headless", True)
            and getattr(self.env.env, "viewer", None) is not None
        ):
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
                    {
                        "key": "robot0_eye_in_hand_image",
                        "resolution": [cam_res, cam_res],
                        "role": "wrist",
                    },
                ],
                "state": {"dim": 8, "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"},
                "image_transform": "applied_in_sim",
            },
            "max_steps": 280,
            "delta_actions": getattr(self, "delta_actions", False),
            # Typed ActionObsSpec contracts — what this sim CONSUMES (action) and PROVIDES (obs).
            # Serialized as plain dicts (ActionObsSpec.to_dict() format) so no roboeval import
            # is required in the sim venv (Python 3.8).
            "action_spec": {
                "position": {
                    "name": "position",
                    "dims": 3,
                    "format": "delta_xyz",
                    "range": [-1, 1],
                    "accepts": ["delta_xyz", "delta_axisangle"],
                },
                "rotation": {
                    "name": "rotation",
                    "dims": 3,
                    "format": "delta_axisangle",
                    "range": [-3.15, 3.15],
                    "accepts": ["delta_axisangle", "axis_angle"],
                },
                "gripper": {
                    "name": "gripper",
                    "dims": 1,
                    "format": "binary_close_negative",
                    "range": [-1, 1],
                    "accepts": ["binary_close_positive", "binary_close_negative"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 8, "format": "libero_eef_pos3_aa3_grip2"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
        }

    def _extract_image(self, obs):
        """Return (image, image2) where image=agentview and image2=wrist camera.

        Both images are flipped 180° (``[::-1, ::-1]``) to match the orientation
        expected by lerobot-trained VLAs (pi05, smolvla, openvla).  The flip is
        applied here — in the sim layer — which is the canonical single location
        for this transform.

        Consumers (env_wrapper, policy servers) must NOT apply an additional flip.
        The obs_space metadata from ``get_info()`` advertises
        ``image_transform="applied_in_sim"`` to signal this.
        """
        image = obs.get("agentview_image")
        image2 = obs.get("robot0_eye_in_hand_image")
        if image is None and image2 is None:
            raise KeyError(f"No camera image found in obs keys: {list(obs.keys())}")
        # Use agentview as primary if available, else wrist camera
        primary = image if image is not None else image2
        return np.asarray(primary, dtype=np.uint8).copy()[::-1, ::-1], (
            np.asarray(image2, dtype=np.uint8).copy()[::-1, ::-1] if image2 is not None else None
        )

    def _extract_state(self, obs: dict) -> list:
        """Extract proprioceptive state: eef_pos(3) + axisangle(3) + gripper(2) = 8-dim.

        Returns a plain list[float] for direct use by the policy server.
        """
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)  # (3,)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)  # (4,) [x,y,z,w]
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

    def get_state_dict(self) -> dict:
        """Return structured state dict for GR00T libero_sim embodiment tag.

        Maps the LIBERO proprioceptive state to the per-key format expected by
        the GR00T ``libero_sim`` modality config (each key is a 1-element list):

            x, y, z       — EEF position (eef_pos)
            roll, pitch, yaw — EEF orientation as axis-angle
            gripper       — mean of gripper_qpos (scalar, range ~[0, 1])
        """
        obs = self._last_obs
        if not obs:
            return {}
        eef_pos = np.asarray(obs.get("robot0_eef_pos", [0.0, 0.0, 0.0]), dtype=np.float32)
        eef_quat = np.asarray(obs.get("robot0_eef_quat", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        gripper = np.asarray(obs.get("robot0_gripper_qpos", [0.0, 0.0]), dtype=np.float32)

        # Quaternion [x,y,z,w] → axis-angle (matches _extract_state and training data)
        x, y, z, w = float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3])
        den = np.sqrt(max(0.0, 1.0 - w * w))
        if den > 1e-10:
            angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
            axisangle = np.array([x, y, z], dtype=np.float32) / den * angle
        else:
            axisangle = np.zeros(3, dtype=np.float32)

        return {
            "x": [float(eef_pos[0])],
            "y": [float(eef_pos[1])],
            "z": [float(eef_pos[2])],
            "roll": [float(axisangle[0])],
            "pitch": [float(axisangle[1])],
            "yaw": [float(axisangle[2])],
            "gripper": [float(gripper[0]), float(gripper[1])],
        }


class LiberoProBackend(LiberoBackend):
    """Backend for LIBERO-Pro (same API as LIBERO, different repo).

    Overrides only init() to change the default suite name and gracefully
    handle missing init-state files (common in LIBERO-Pro OOD variants).
    All other methods (reset, step, get_obs, etc.) are inherited.
    """

    # Default LIBERO benchmark suite used when no suite is specified.
    # "libero_pro" is the sim backend name, not a LIBERO benchmark key.
    DEFAULT_SUITE = "libero_spatial"

    # Map caller-facing suite names → registered LIBERO benchmark keys.
    # "libero_pro" and "libero_pro_*" are roboeval identifiers, not keys
    # in libero.libero.benchmark.BENCHMARK_MAPPING.
    _SUITE_ALIASES: dict = {
        "libero_pro": "libero_spatial",
        "libero_pro_spatial_object": "libero_spatial",
        "libero_pro_spatial_with_mug": "libero_spatial",
        "libero_pro_goal_swap": "libero_goal",
    }

    def _get_init_states(self, task_idx):
        """Load init states directly with weights_only=False, gracefully handling missing files.

        Uses torch.load() directly (like LiberoBackend) rather than calling
        benchmark.get_task_init_states() which omits weights_only=False and
        would fail on PyTorch >= 2.6 with legacy .pruned_init files containing
        numpy arrays.  Returns an empty list when the file is absent (common
        for LIBERO-Pro OOD variants that don't ship init states).
        """
        import torch
        from libero.libero import get_libero_path

        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.benchmark.tasks[task_idx].problem_folder,
            self.benchmark.tasks[task_idx].init_states_file,
        )
        try:
            # weights_only=False is required for legacy LIBERO .pruned_init files
            # which contain numpy arrays (torch.load default changed in PyTorch 2.6+).
            return torch.load(init_states_path, weights_only=False)
        except FileNotFoundError as e:
            logger.warning(
                "Init states not found (%s); episodes will start from default MuJoCo state.", e
            )
            return []

    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str = None,
        headless: bool = True,
        sim_config: dict = None,
    ):
        """Initialize a LIBERO-Pro environment.

        Resolves the suite name (mapping roboeval aliases like "libero_pro"
        to the registered LIBERO benchmark key), then delegates to
        LiberoBackend.init().  Gracefully handles missing init-state files.
        """
        resolved = suite or self.DEFAULT_SUITE
        # Normalise roboeval suite identifiers to LIBERO benchmark keys.
        # E.g. "libero_pro" (the sim-backend name) → "libero_spatial".
        resolved = self._SUITE_ALIASES.get(resolved, resolved)
        return super().init(
            task_name=task_name,
            camera_resolution=camera_resolution,
            suite=resolved,
            headless=headless,
            sim_config=sim_config,
        )


class RoboCasaBackend(SimBackendBase):
    """Backend for RoboCasa kitchen task environments.

    Defaults to PandaOmron (12-dim: 7 arm + 5 base) — required by RoboCasa
    kitchen environments which need a mobile base.  The robot type is
    configurable via ``sim_config={"robot": "PandaOmron"}``.  Panda (fixed
    base) is NOT supported by RoboCasa kitchen scenes.

    Environment creation uses fixed controller configs, explicit camera setup,
    held-out test objects, and deterministic layout/style IDs for
    reproducibility.

    When a 7-dim policy sends actions, :meth:`step` pads
    them to 12-dim with an idle mobile base.

    Returns ``(image, image2)`` where image2 is the wrist camera.
    """

    # Task name aliases: short names → full RoboCasa environment names.
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
    # First 10 cover the common RoboCasa-10 task set.
    _TASK_INDEX = [
        "PickPlaceCounterToCabinet",
        "PickPlaceCabinetToCounter",
        "PickPlaceCounterToSink",
        "PickPlaceSinkToCounter",
        "PickPlaceCounterToMicrowave",
        "PickPlaceMicrowaveToCounter",
        "PickPlaceCounterToStove",
        "PickPlaceStoveToCounter",
        "OpenCabinet",
        "CloseCabinet",
        "OpenDrawer",
        "CloseDrawer",
        "TurnOnStove",
        "TurnOffStove",
        "TurnOnSinkFaucet",
        "TurnOffSinkFaucet",
        "TurnSinkSpout",
        "CoffeeSetupMug",
        "CoffeeServeMug",
        "StartCoffeeMachine",
        "TurnOnMicrowave",
        "TurnOffMicrowave",
    ]

    # Max steps per task.
    _TASK_MAX_STEPS = {
        "PickPlaceCounterToCabinet": 500,
        "PickPlaceCabinetToCounter": 500,
        "PickPlaceCounterToSink": 700,
        "PickPlaceSinkToCounter": 500,
        "PickPlaceCounterToMicrowave": 600,
        "PickPlaceMicrowaveToCounter": 500,
        "PickPlaceCounterToStove": 500,
        "PickPlaceStoveToCounter": 500,
        "OpenCabinet": 500,
        "CloseCabinet": 500,
        "OpenDrawer": 500,
        "CloseDrawer": 500,
        "TurnOnStove": 500,
        "TurnOffStove": 500,
        "TurnOnSinkFaucet": 500,
        "TurnOffSinkFaucet": 500,
        "TurnSinkSpout": 500,
        "CoffeeSetupMug": 600,
        "CoffeeServeMug": 600,
        "StartCoffeeMachine": 300,
        "TurnOnMicrowave": 500,
        "TurnOffMicrowave": 500,
    }

    # Default layout and style IDs for reproducible test scenes.
    _DEFAULT_LAYOUT_STYLE_IDS = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))

    # Camera setup used for RoboCasa evaluation.
    _CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

    # Mobile-base padding: 5 zeros + brake flag (used when receiving 7D
    # actions but running PandaOmron/PandaMobile with 12D action space)
    _MOBILE_BASE_PAD = np.array([0.0, 0.0, 0.0, 0.0, -1.0])

    # Path to controller configs pickle.
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

        Accepts numeric indices (e.g. "0" -> first task), short aliases
        (e.g. "PnPCounterToCab"), or full RoboCasa names.
        """
        # Handle numeric task indices (runner passes --task 0, 1, ...)
        if task_name.isdigit():
            idx = int(task_name)
            if idx < len(self._TASK_INDEX):
                return self._TASK_INDEX[idx]
            raise ValueError(
                f"Task index {idx} out of range. Valid range: 0-{len(self._TASK_INDEX) - 1}"
            )
        return self._TASK_ALIASES.get(task_name, task_name)

    def _load_controller_configs(self):
        """Load OSC_POSE controller configs."""
        import pickle

        with open(self._CONTROLLER_CONFIGS_PATH, "rb") as f:
            return pickle.load(f)

    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str = None,
        headless: bool = True,
        sim_config: dict = None,
    ):
        """Initialize a RoboCasa environment by task name.

        Args:
            task_name: Task name — accepts both short aliases
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
                - ``obj_instance_split``: Object split (default ``"target"``).
                - ``layout_and_style_ids``: Tuple of (layout, style) pairs.
                - ``seed``: Random seed for environment.
        """
        import robocasa  # registers kitchen envs with robosuite
        import robosuite

        cfg = sim_config or {}
        self._robot_type = cfg.get("robot", "PandaOmron")
        self._cam_res = camera_resolution
        self._task_name = task_name or "PnPCounterToCab"
        self._resolved_task = self._resolve_task_name(self._task_name)

        # Load fixed OSC_POSE controller configs.
        controller_configs = self._load_controller_configs()

        # Parse layout and style IDs
        layout_and_style_ids = cfg.get("layout_and_style_ids", self._DEFAULT_LAYOUT_STYLE_IDS)

        # Create environment using robosuite.make() directly with full kwargs
        # for reproducible evaluation.
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
            # Use the held-out object split by default.
            obj_instance_split=cfg.get("obj_instance_split", "target"),
            # Limit to lightwheel objects only; the lightweight asset set
            # suffices for evaluation.  Passing objaverse would cause a
            # NaN-probability crash when the objaverse asset dir is missing.
            obj_registries=tuple(cfg.get("obj_registries", ["lightwheel"])),
            generative_textures=None,
            randomize_cameras=cfg.get("randomize_cameras", False),
            layout_and_style_ids=layout_and_style_ids,
            translucent_robot=False,
        )
        self.env = robosuite.make(**env_kwargs)

        # Auto-reset so that /obs works immediately after /init, and run
        # stabilization steps.
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
        # Warmup: let objects settle.
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
            trimmed[0:6] = action[0:6]  # arm pos + rot
            trimmed[6:11] = action[7:12]  # base motion
            trimmed[11] = action[6]  # gripper
            action = trimmed

        obs, reward, done, info = self.env.step(action)
        self._last_obs = obs
        img = self._extract_primary_image(obs)
        img2 = self._extract_wrist_image(obs)
        return img, img2, reward, done, info

    def get_obs(self):
        """Get current observation. Returns ``(image, image2)``."""
        if self._last_obs:
            return self._extract_primary_image(self._last_obs), self._extract_wrist_image(
                self._last_obs
            )
        # Fallback: query the env directly
        inner = getattr(self.env, "env", self.env)
        obs = inner._get_observations()
        self._last_obs = obs
        return self._extract_primary_image(obs), self._extract_wrist_image(obs)

    def check_success(self):
        """Check task success via the environment's internal condition."""
        inner = getattr(self.env, "env", self.env)
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
                    {
                        "key": "robot0_agentview_left_image",
                        "resolution": [cam_res, cam_res],
                        "role": "primary",
                    },
                    {
                        "key": "robot0_agentview_right_image",
                        "resolution": [cam_res, cam_res],
                        "role": "secondary",
                    },
                    {
                        "key": "robot0_eye_in_hand_image",
                        "resolution": [cam_res, cam_res],
                        "role": "wrist",
                    },
                ],
                "state": {"dim": 9, "format": "gripper_qpos(2)+eef_pos(3)+eef_quat(4)"},
                # RoboCasa images are passed through unchanged.
                "image_transform": "none",
            },
            "max_steps": max_steps,
            "delta_actions": True,
            # ── Typed ActionObsSpec contracts ───────────────────────────────────────
            # IMPORTANT: RoboCasa state is a *9-dim quaternion* representation
            # (gripper_qpos(2) + eef_pos(3) + eef_quat(4)).  This is structurally
            # incompatible with the 8-dim axis-angle state that LIBERO-trained
            # VLAs (pi05, smolvla, openvla) expect.  By declaring the spec
            # honestly here, the orchestrator's ActionObsSpec gate refuses to start an
            # episode for incompatible (model, sim) pairs instead of silently
            # serving a wrong-shaped state vector.
            "action_spec": {
                "position": {
                    "name": "position",
                    "dims": 3,
                    "format": "delta_xyz",
                    "range": [-1, 1],
                    "accepts": ["delta_xyz"],
                },
                "rotation": {
                    "name": "rotation",
                    "dims": 3,
                    "format": "delta_axisangle",
                    "range": [-3.15, 3.15],
                    "accepts": ["delta_axisangle", "axis_angle"],
                },
                "gripper": {
                    "name": "gripper",
                    "dims": 1,
                    "format": "binary_close_negative",
                    "range": [-1, 1],
                    "accepts": ["binary_close_positive", "binary_close_negative"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "secondary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                # 9-dim quaternion state — declared explicitly so ActionObsSpec
                # validation flags any 8-dim axis-angle consumer at episode start.
                "state": {"name": "state", "dims": 9, "format": "robocasa_grip2_eef_pos3_quat4"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
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
        raise KeyError(f"No primary camera image found in obs keys: {list(obs.keys())}")

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


# ---------------------------------------------------------------------------
# RoboTwin fast-init helpers
# ---------------------------------------------------------------------------


class _EvalGripperPlanner:
    """Minimal gripper planner shim for eval-only RoboTwin startup.

    RoboTwin's qpos evaluation path calls ``plan_grippers()`` during env
    setup, but never uses CuRobo path planning afterwards.  This shim
    satisfies the gripper interpolation contract while skipping the expensive
    CuRobo warmup triggered by ``Robot.set_planner()``.
    """

    def plan_grippers(self, now_val: float, target_val: float):
        num_step = 200
        per_step = (target_val - now_val) / num_step
        vals = np.linspace(now_val, target_val, num_step)
        return {"num_step": num_step, "per_step": per_step, "result": vals}

    def update_point_cloud(self, world_pcd, resolution: float = 0.02):
        return None

    def plan_path(self, *args, **kwargs):
        raise RuntimeError(
            "RoboTwin eval fast-path: CuRobo path planning is disabled. "
            "This error means a task script tried to use the full motion planner "
            "during an eval episode — unexpected for qpos evaluation."
        )

    def plan_batch(self, *args, **kwargs):
        raise RuntimeError("RoboTwin eval fast-path: CuRobo batch planning is disabled.")


def _make_fast_set_planner(robot_mod):
    """Return a ``Robot.set_planner`` replacement that skips CuRobo init.

    When ``robot.need_topp`` is True, instantiates the real
    ``robot_mod.MplibPlanner`` (mplib==0.2.1, with the ``or collide`` patch
    applied during environment setup) so that ``TOPP()`` performs genuine time-optimal
    path parameterisation (velocity/accel limits).

    CuRobo warmup in ``Robot.set_planner()`` is bypassed here
    by not calling ``super().set_planner()``; gripper planning is handled by
    ``_EvalGripperPlanner``.
    """

    def _set_planner_fast(self, scene=None):
        self.communication_flag = False
        self.left_planner = _EvalGripperPlanner()
        self.right_planner = _EvalGripperPlanner()

        if getattr(self, "need_topp", False):
            # Instantiate the real mplib.MplibPlanner so that TOPP() runs
            # genuine time-optimal path parameterisation (velocity/accel limits).
            # Requires mplib==0.2.1 with the collision-check patch applied.
            self.left_mplib_planner = robot_mod.MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = robot_mod.MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    return _set_planner_fast


@_contextmanager
def _patched_robot_set_planner():
    """Temporarily replace ``Robot.set_planner`` to skip CuRobo warmup.

    RoboTwin calls ``Robot.set_planner()`` during ``setup_demo()``.  That
    method imports and initialises CuRobo, which is not installed in this
    venv.  The missing import causes a ``ModuleNotFoundError`` that races
    with SAPIEN's background XCB thread and produces an unrecoverable
    assertion failure.

    This context manager monkey-patches ``Robot.set_planner`` with
    ``_make_fast_set_planner()`` for the duration of ``setup_demo()``, then
    restores the prior implementation.
    """
    import envs.robot.robot as _robot_mod

    original = _robot_mod.Robot.set_planner
    _robot_mod.Robot.set_planner = _make_fast_set_planner(_robot_mod)
    try:
        yield
    finally:
        _robot_mod.Robot.set_planner = original


# ---------------------------------------------------------------------------


class RoboTwinBackend(SimBackendBase):
    """Backend for RoboTwin 2.0 (SAPIEN-based bimanual, 14-dim actions).

    Uses setup_demo() / get_obs() / take_action() — NOT reset() / step().
    CRITICAL: torch must be imported before sapien in the same process.

    Lifecycle divergence (vs. LiberoBackend / RoboCasaBackend)
    ----------------------------------------------------------
    RoboTwin's underlying env_class takes the task name at *construction* time,
    so this backend exposes ``initialize(seed)`` (instead of a no-arg
    ``__init__`` followed by ``init(task_name, ...)``).  The HTTP layer in
    ``/init`` special-cases this branch.  ``init(...)`` is provided as a thin
    compatibility wrapper so callers that follow the LIBERO lifecycle still
    work; the sim_worker route still runs the canonical RoboTwin path.
    """

    def __init__(self, task_name: str = ""):
        # ``task_name`` is accepted at construction time (legacy RoboTwin path)
        # but defaults to empty so a future caller can use the standard
        # ``backend = RoboTwinBackend(); backend.init(task_name=...)`` lifecycle.
        self.task_name = task_name
        self.env = None

    def init(
        self,
        task_name: str,
        camera_resolution: int = 480,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        """Compatibility shim matching the LIBERO/RoboCasa lifecycle.

        Allows external callers to follow the canonical ``init(task_name, ...)``
        contract.  Internally still delegates to ``initialize()``.
        """
        self.task_name = task_name
        seed = (sim_config or {}).get("seed", 42)
        return self.initialize(seed=seed)

    def initialize(self, seed: int = 42) -> dict:
        import importlib
        import sys

        # CRITICAL: import torch before any sapien import to avoid CUDA segfault.
        import torch
        import yaml

        _vendors_base = pathlib.Path(
            os.environ.get(
                "ROBOEVAL_VENDORS_DIR",
                str(pathlib.Path.home() / ".local" / "share" / "roboeval" / "vendors"),
            )
        ).resolve()
        robotwin_path = _vendors_base / "RoboTwin"
        if not robotwin_path.exists():
            for entry in sys.path:
                candidate = pathlib.Path(entry)
                if (candidate / "envs").is_dir() and (
                    candidate / "task_config" / "demo_clean.yml"
                ).is_file():
                    robotwin_path = candidate.resolve()
                    break

        robotwin_dir = str(robotwin_path)
        os.chdir(robotwin_dir)
        if robotwin_dir not in sys.path:
            sys.path.insert(0, robotwin_dir)

        # Resolve integer task index → actual task module name.
        # The runner passes task_name as a decimal string ("0", "1", ...).
        # RoboTwin task modules live in envs/<task_name>.py; they are NOT numbered.
        if self.task_name.isdigit():
            envs_dir = pathlib.Path(robotwin_dir) / "envs"
            task_list = sorted(p.stem for p in envs_dir.glob("*.py") if not p.stem.startswith("_"))
            idx = int(self.task_name)
            if idx >= len(task_list):
                raise ValueError(
                    f"RoboTwin task index {idx} out of range (0-{len(task_list) - 1}). "
                    f"Available tasks: {task_list}"
                )
            resolved = task_list[idx]
            import logging as _logging

            _logging.getLogger(__name__).info(
                "RoboTwin: resolved task index %s -> '%s'", idx, resolved
            )
            self.task_name = resolved

        from envs import CONFIGS_PATH

        with open("./task_config/demo_clean.yml", encoding="utf-8") as f:
            args = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
            emb_types = yaml.load(f.read(), Loader=yaml.FullLoader)

        robot_file = emb_types["aloha-agilex"]["file_path"]
        with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
            emb_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        with open(os.path.join(CONFIGS_PATH, "_camera_config.yml"), encoding="utf-8") as f:
            cam_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        head_cam_type = args["camera"]["head_camera_type"]
        args.update(
            {
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
            }
        )

        module = importlib.import_module(f"envs.{self.task_name}")
        env_class = getattr(module, self.task_name)
        self.env = env_class()
        # Wrap setup_demo() in _patched_robot_set_planner so that
        # Robot.set_planner() uses _EvalGripperPlanner instead of trying to
        # import and warm-up CuRobo.  Without this, the missing curobo import
        # causes a ModuleNotFoundError that races with SAPIEN's background XCB
        # thread and produces an unrecoverable assertion failure:
        #   "Assertion '!xcb_xlib_unknown_seq_number' failed."
        with _patched_robot_set_planner():
            self.env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        self.env.set_instruction(self.task_name.replace("_", " "))

        # Return task_description so callers can use it as the instruction
        # (without this, task_name falls back to the raw task index "0" in the
        # policy request, which can break task-conditioned models).
        return {
            "status": "ok",
            "action_dim": 14,
            "task_description": self.task_name.replace("_", " "),
        }

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
        # joint_action["vector"] contains the current joint positions.
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
            result["image2"] = left_b64  # backward compat
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
            "done": bool(self.env.eval_success or self.env.take_action_cnt >= self.env.step_lim),
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
                "image_transform": "none",
            },
            "max_steps": step_lim,
            "delta_actions": False,
            # Typed ActionObsSpec contracts.  RoboTwin uses absolute joint targets
            # (NOT delta EEF), so VLAs trained for eef_delta will fail the gate.
            "action_spec": {
                "joint_pos": {
                    "name": "joint_pos",
                    "dims": 14,
                    "format": "absolute_joint_positions",
                    "accepts": ["absolute_joint_positions"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 14, "format": "joint_positions"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
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


class LiberoInfinityBackend(LiberoBackend):
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
        self._scenario = None  # compiled Scenic scenario (cached across resets)
        self._sim = None  # current LIBEROSimulation
        self._bddl_ctx = None  # active bddl_for_scene() context manager
        self._bddl_path = None  # path to original BDDL file
        self._orig_obj_classes = None  # {instance_name: class_name} from BDDL
        self._perturbation = "position"
        self._max_steps = 300
        self._env_kwargs: dict = {}
        self._scenic_path = None  # generated .scenic file (cleaned up on close)
        self._camera_resolution = 256
        self._run_seed = 42  # base seed for deterministic episode seeding
        self._ep_idx = 0  # auto-incrementing episode counter
        self._last_obs: dict = {}  # cache last obs dict for state extraction
        self._max_reset_attempts = 5
        self._post_reset_settle_steps = 80
        self._post_reset_stable_steps = 10
        self._post_reset_pos_tol = 0.002
        self._post_reset_vel_tol = 0.03
        self._post_reset_rot_tol = 0.03
        self._post_reset_ang_vel_tol = 0.2
        self._post_reset_target_rot_tol = 0.35

    _PERTURBATION_AXIS_ORDER = (
        "position",
        "object",
        "robot",
        "camera",
        "lighting",
        "texture",
        "distractor",
        "background",
        "articulation",
    )
    _PERTURBATION_PRESETS = {
        "combined": (
            "position",
            "object",
            "robot",
            "camera",
            "lighting",
            "distractor",
            "background",
        ),
        "full": _PERTURBATION_AXIS_ORDER,
        "all": _PERTURBATION_AXIS_ORDER,
        "all_axes": _PERTURBATION_AXIS_ORDER,
        "all-axes": _PERTURBATION_AXIS_ORDER,
    }

    @classmethod
    def _normalize_perturbation(cls, value) -> str:
        """Normalize LIBERO-Infinity perturbation config for upstream compiler APIs."""
        if value is None:
            return "position"

        if isinstance(value, str):
            raw_parts = [part.strip().lower() for part in value.split(",")]
        elif isinstance(value, (list, tuple, set, frozenset)):
            raw_parts = []
            for item in value:
                if not isinstance(item, str):
                    raise ValueError(
                        "LIBERO-Infinity sim_config.perturbation list values must be strings; "
                        f"got {type(item).__name__}: {item!r}"
                    )
                raw_parts.extend(part.strip().lower() for part in item.split(","))
        else:
            raise ValueError(
                "LIBERO-Infinity sim_config.perturbation must be a string or list of strings; "
                f"got {type(value).__name__}: {value!r}"
            )

        requested = [part for part in raw_parts if part]
        if not requested:
            return "position"

        presets = dict(cls._PERTURBATION_PRESETS)
        try:
            from libero_infinity.planner.composition import AXIS_PRESETS

            presets.update({name: tuple(axes) for name, axes in AXIS_PRESETS.items()})
            if "full" in presets:
                presets.setdefault("all", presets["full"])
        except Exception:
            pass

        known_axes = set(cls._PERTURBATION_AXIS_ORDER)
        for axes in presets.values():
            known_axes.update(axes)

        expanded: set[str] = set()
        unknown: list[str] = []
        for part in requested:
            if part in presets:
                expanded.update(presets[part])
            elif part in known_axes:
                expanded.add(part)
            else:
                unknown.append(part)

        if unknown:
            valid_axes = ", ".join(cls._PERTURBATION_AXIS_ORDER)
            valid_presets = ", ".join(sorted(presets))
            raise ValueError(
                "Unknown LIBERO-Infinity perturbation axis/preset "
                f"{', '.join(repr(axis) for axis in unknown)}. "
                f"Known axes: {valid_axes}. Presets: {valid_presets}. "
                "Use a scalar string, a comma-separated string like "
                "'position,camera,distractor', or a YAML list like "
                "['position', 'camera']."
            )

        ordered = [axis for axis in cls._PERTURBATION_AXIS_ORDER if axis in expanded]
        ordered.extend(sorted(expanded.difference(ordered)))
        return ",".join(ordered)

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
        self._perturbation = self._normalize_perturbation(sim_config.get("perturbation"))
        self._max_steps = sim_config.get("max_steps", 300)
        self._run_seed = sim_config.get("seed", 42)
        self._max_reset_attempts = max(1, int(sim_config.get("max_reset_attempts", 5)))
        self._post_reset_settle_steps = max(0, int(sim_config.get("post_reset_settle_steps", 80)))
        self._post_reset_stable_steps = max(1, int(sim_config.get("post_reset_stable_steps", 10)))
        self._post_reset_pos_tol = float(sim_config.get("post_reset_pos_tol", 0.002))
        self._post_reset_vel_tol = float(sim_config.get("post_reset_vel_tol", 0.03))
        self._post_reset_rot_tol = float(sim_config.get("post_reset_rot_tol", 0.03))
        self._post_reset_ang_vel_tol = float(sim_config.get("post_reset_ang_vel_tol", 0.2))
        self._post_reset_target_rot_tol = float(sim_config.get("post_reset_target_rot_tol", 0.35))

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
        try:
            from libero_infinity.compiler import generate_scenic_file
        except ImportError:
            from libero_infinity import scenic_generator as _scenic_generator

            generate_scenic_file = _scenic_generator.generate_scenic_file
            if not getattr(_scenic_generator, "PERTURBATION_AXES", None):
                logger.warning(
                    "Installed libero_infinity.scenic_generator does not advertise "
                    "perturbation axes; sim_config.perturbation=%r will be forwarded "
                    "but may be ignored by that installed generator. Upgrade "
                    "libero-infinity or install an editable compiler-enabled version "
                    "for full-axis perturbations.",
                    self._perturbation,
                )
        from libero_infinity.task_config import TaskConfig

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

        self._orig_obj_classes = parse_object_classes(pathlib.Path(self._bddl_path).read_text())

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
            seed_i = int(hashlib.sha256(seed_bytes).hexdigest(), 16) % (2**31)

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
                self._sim = simulator.createSimulation(scene, maxSteps=self._max_steps, verbosity=0)
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
        rel_target_rot = np.einsum("nij,njk->nik", np.transpose(target_rot, (0, 2, 1)), final_rot)
        target_rot_error = np.arccos(
            np.clip(
                (np.trace(rel_target_rot, axis1=1, axis2=2) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
        )
        toppled = [
            f"{name} rotated {err:.2f} rad from its intended pose"
            for name, err in zip(body_names, target_rot_error, strict=False)
            if err > self._post_reset_target_rot_tol
        ]

        if toppled:
            raise RuntimeError("Invalid Scenic sample after MuJoCo settling: " + "; ".join(toppled))

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
                    {
                        "key": "robot0_eye_in_hand_image",
                        "resolution": [cam_res, cam_res],
                        "role": "wrist",
                    },
                ],
                "state": {"dim": 8, "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)"},
                # The 180° flip is applied here in the sim layer (see _extract_images
                # below), matching LiberoBackend.  This is the canonical convention
                # consumers must not apply a second flip.
                "image_transform": "applied_in_sim",
            },
            "max_steps": self._max_steps,
            "delta_actions": getattr(self, "delta_actions", False),
            # Typed ActionObsSpec contracts (mirror LiberoBackend exactly so the same
            # LIBERO-trained VLAs work identically against both backends).
            "action_spec": {
                "position": {
                    "name": "position",
                    "dims": 3,
                    "format": "delta_xyz",
                    "range": [-1, 1],
                    "accepts": ["delta_xyz", "delta_axisangle"],
                },
                "rotation": {
                    "name": "rotation",
                    "dims": 3,
                    "format": "delta_axisangle",
                    "range": [-3.15, 3.15],
                    "accepts": ["delta_axisangle", "axis_angle"],
                },
                "gripper": {
                    "name": "gripper",
                    "dims": 1,
                    "format": "binary_close_negative",
                    "range": [-1, 1],
                    "accepts": ["binary_close_positive", "binary_close_negative"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 8, "format": "libero_eef_pos3_aa3_grip2"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
        }

    def _extract_images(self, obs: dict):
        """Extract (agentview, wrist-camera) images from an obs dict.

        Both images are flipped 180° (``[::-1, ::-1]``) here in the sim layer
        to match the orientation expected by lerobot-trained VLAs.  This is
        identical to LiberoBackend._extract_image and is the single location
        for the flip.  ``obs_space.image_transform`` is advertised as
        ``"applied_in_sim"`` so callers do not apply a second flip.

        Returns:
            (img, img2) where img2 is None if no wrist camera in obs.
        """
        image = obs.get("agentview_image")
        image2 = obs.get("robot0_eye_in_hand_image")
        if image is None and image2 is None:
            raise KeyError(f"No camera image found in obs keys: {list(obs.keys())}")
        primary = image if image is not None else image2
        img = np.asarray(primary, dtype=np.uint8).copy()[::-1, ::-1]
        img2 = np.asarray(image2, dtype=np.uint8).copy()[::-1, ::-1] if image2 is not None else None
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
                "suite is required for LiberoInfinityBackend (e.g. 'libero_infinity_spatial')"
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
                f"Ambiguous exact match for task '{task_name}': {[f.name for f in exact]}"
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


class AlohaGymBackend(SimBackendBase):
    """Backend for ``gym-aloha`` (HuggingFace's pure-Python ALOHA bimanual sim).

    gym-aloha is a uv-clean, aarch64-friendly bimanual ALOHA simulator built on
    MuJoCo + dm_control + gymnasium.  It complements :class:`RoboTwinBackend`:
    same 14-dim joint-position action contract (so InternVLA drives both
    natively), but no SAPIEN, no curobo, no mplib, no conda — pure PyPI wheels.

    Tasks (gymnasium ids; ``init`` accepts the trailing name or a numeric index):

    * ``AlohaTransferCube-v0`` — right arm picks the cube, hands it to left arm.
    * ``AlohaInsertion-v0``    — both arms pick socket+peg, insert mid-air.

    Lifecycle mirrors :class:`LiberoBackend` (no construction-time task name,
    standard ``init → reset → step`` flow).  ``reset`` returns ``(image, image2)``
    where ``image`` is the ``top`` agent-view camera and ``image2`` is the
    ``angle`` (side) view used as a stand-in for a wrist camera.

    **Image orientation**: gym-aloha's dm_control cameras render in the correct
    (upright) orientation — **no 180° flip is applied**, unlike LIBERO backends.
    ACT and similar checkpoints were trained on lerobot's native gym-aloha data
    which also has no flip.  The ``obs_space.image_transform`` is ``"none"``.
    """

    # Tasks are gymnasium ids registered by `gym_aloha`.  Keep this list aligned
    # with upstream — if HF adds tasks (e.g. ALOHA-Sim2Real), append them here.
    TASKS: list[str] = [
        "AlohaTransferCube-v0",
        "AlohaInsertion-v0",
    ]

    def __init__(self) -> None:
        self.env = None
        self._task_id: str = ""
        self._cam_res: int = 480
        self._last_obs = (
            None  # None until first reset(); empty dict was falsy, breaking pre-reset /obs
        )
        self._last_img2 = None
        self._last_reward: float = 0.0
        # gym-aloha rewards saturate at 4.0 on full task completion (both
        # TransferCube and Insertion).  ``check_success`` compares against this.
        self._success_threshold: float = 4.0
        self._step_count: int = 0
        self._max_steps: int = 400

    # ------------------------------------------------------------------ helpers
    def _resolve_task(self, task_name: str) -> str:
        """Map a user-supplied task name (or numeric index) to a gym id."""
        # Numeric index → list lookup (matches RoboTwinBackend).
        if task_name.isdigit():
            idx = int(task_name)
            if not 0 <= idx < len(self.TASKS):
                raise ValueError(
                    f"AlohaGym task index {idx} out of range "
                    f"(0-{len(self.TASKS) - 1}). Available: {self.TASKS}"
                )
            return self.TASKS[idx]
        # Allow short forms: "transfer_cube", "insertion", "AlohaTransferCube-v0", etc.
        norm = task_name.lower().replace("_", "").replace("-", "")
        for tid in self.TASKS:
            if norm in tid.lower().replace("-", ""):
                return tid
        raise ValueError(f"AlohaGym task '{task_name}' not recognised. Available: {self.TASKS}")

    # ------------------------------------------------------------------- init
    def init(
        self,
        task_name: str,
        camera_resolution: int = 480,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        # MuJoCo backend selection — must precede any mujoco import.
        os.environ["MUJOCO_GL"] = "egl" if headless else "glfw"
        self.headless = headless
        self._cam_res = int(camera_resolution) if camera_resolution else 480

        import gym_aloha  # registers the gym ids
        import gymnasium as gym

        self._task_id = self._resolve_task(task_name)
        # ``obs_type="pixels_agent_pos"`` returns dict of camera frames + 14-dim
        # qpos.  ``render_mode="rgb_array"`` enables headless image capture.
        self.env = gym.make(
            f"gym_aloha/{self._task_id}",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        # Discover step limit from gymnasium's TimeLimit wrapper if present.
        spec_max = getattr(self.env.spec, "max_episode_steps", None)
        if spec_max:
            self._max_steps = int(spec_max)

        # Friendly per-task instruction string (no built-in language label).
        instruction = {
            "AlohaTransferCube-v0": "Right arm picks up the red cube and transfers it to the left gripper.",
            "AlohaInsertion-v0": "Pick up the socket and peg with the two arms and insert them mid-air.",
        }.get(self._task_id, self._task_id.replace("-", " "))

        return {"task_description": instruction, "task_id": self._task_id}

    # ------------------------------------------------------------------ reset
    def reset(self, episode_index: int | None = None):
        seed = 42 if episode_index is None else 1000 + int(episode_index)
        obs, _info = self.env.reset(seed=seed)
        self._last_obs = obs
        self._last_reward = 0.0
        self._step_count = 0
        return self._extract_image(obs)

    # ------------------------------------------------------------------- step
    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        # Tolerate VLA chunks longer than 14 (e.g. InternVLA emits 32-dim).
        if action_arr.shape[-1] > 14:
            action_arr = action_arr[..., :14]
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        self._last_obs = obs
        self._last_reward = float(reward)
        self._step_count += 1
        success = self._last_reward >= self._success_threshold
        done = bool(terminated or truncated or success)
        info = dict(info) if isinstance(info, dict) else {}
        info["success"] = bool(success)
        img, img2 = self._extract_image(obs)
        return img, img2, self._last_reward, done, info

    # ---------------------------------------------------------------- get_obs
    def get_obs(self):
        return self._extract_image(self._last_obs)

    def check_success(self) -> bool:
        return bool(self._last_reward >= self._success_threshold)

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None

    # --------------------------------------------------------------- get_info
    def get_info(self) -> dict:
        cam_res = self._cam_res
        return {
            "action_space": {"type": "joint_pos", "dim": 14, "accepted_dims": [14, 32]},
            "obs_space": {
                "cameras": [
                    {"key": "top", "resolution": [cam_res, cam_res], "role": "primary"},
                    {"key": "angle", "resolution": [cam_res, cam_res], "role": "wrist"},
                ],
                "state": {"dim": 14, "format": "joint_positions(14)"},
                # gym-aloha dm_control cameras render images in the correct (upright)
                # orientation — no in-sim flip is applied.  VLA consumers must NOT
                # apply an additional flip either (ACT was trained on un-flipped images).
                "image_transform": "none",
            },
            "max_steps": self._max_steps,
            "delta_actions": False,
            # gym-aloha consumes ABSOLUTE 14-dim joint targets — the same
            # contract as RoboTwin, so InternVLA's joint_pos head drives it.
            "action_spec": {
                "joint_pos": {
                    "name": "joint_pos",
                    "dims": 14,
                    "format": "absolute_joint_positions",
                    "accepts": ["absolute_joint_positions"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 14, "format": "joint_positions"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
        }

    # --------------------------------------------------------- image helpers
    def _extract_image(self, obs):
        """Return ``(primary, wrist)`` — in natural (un-flipped) camera orientation.

        gym-aloha with ``obs_type="pixels_agent_pos"`` exposes a ``pixels`` dict
        with a ``top`` key (overhead camera).  An ``angle`` (side) key may also
        be present in some gym-aloha builds; if absent, ``wrist`` is ``None``.

        **No 180° flip is applied here.**  gym-aloha's dm_control cameras render
        images in the correct (upright) orientation — the same orientation used
        when lerobot collected ACT training data.  Applying a flip (as LIBERO
        backends do via LiberoProcessorStep) would present upside-down images to
        ACT, causing it to produce near-random actions.

        Returns a black placeholder pair when called before the first reset
        (``obs is None``) so that ``GET /obs`` immediately after ``POST /init``
        does not crash ``encode_image_b64``.
        """
        if obs is None:
            # Pre-reset state: return black frames so /obs before /reset is safe.
            blank = np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)
            return blank, None
        pixels = obs.get("pixels") if isinstance(obs, dict) else None
        if pixels is None:
            blank = np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)
            return blank, None
        top = pixels.get("top")
        angle = pixels.get("angle")
        if top is None and angle is None:
            raise KeyError(
                f"No camera image found in gym-aloha obs pixels keys: {list(pixels.keys())}"
            )
        primary = top if top is not None else angle
        primary = np.asarray(primary, dtype=np.uint8).copy()
        wrist = np.asarray(angle, dtype=np.uint8).copy() if angle is not None else None
        self._last_img2 = wrist
        return primary, wrist

    def _extract_state(self, obs: dict) -> list:
        """Return the 14-dim joint position vector ALOHA-style."""
        if obs is None:
            return [0.0] * 14
        agent_pos = obs.get("agent_pos") if isinstance(obs, dict) else None
        if agent_pos is None:
            return [0.0] * 14
        arr = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
        return arr.tolist()


# === START gym_pusht ===
# ======================================================================
# GymPushTBackend — HuggingFace gym-pusht (PushT pushing benchmark)
#
# gym-pusht is the canonical companion simulator for Diffusion Policy.
# A T-shaped block must be pushed into a target zone using a 2-dim
# (x, y) end-effector action.  Pure Python / PyPI — no MuJoCo, no
# conda, no GPU required.  aarch64-clean.
#
# Package : gym-pusht  (PyPI)
# Python  : 3.11+
# Action  : 2-dim continuous (x, y) normalised to [-1, 1]
# Task    : push_t  (single task)
# ======================================================================


class GymPushTBackend(SimBackendBase):
    """Backend for ``gym-pusht`` (HuggingFace's PushT pushing benchmark).

    gym-pusht is a lightweight pure-Python 2-D pushing simulator built on
    `pymunk` (2-D physics) + `pygame` (rendering).  It is the **canonical**
    companion to Diffusion Policy; the reference DP model was trained and
    evaluated exclusively on this environment.

    **Task**: push a T-shaped block into a target zone drawn on the
    workspace floor using a disk-shaped end-effector.

    **Action space**: 2-dim ``(x, y)`` continuous, normalised to [-1, 1],
    representing the desired end-effector position on the 2-D workspace.

    **Success criterion**: T-block coverage of the target zone ≥ 90 %
    (``env.reward() >= 0.9``).

    **Install**: ``pip install gym-pusht`` (PyPI, no extras needed).
    Python 3.11+, aarch64-clean.

    Lifecycle mirrors :class:`AlohaGymBackend` (``init → reset → step``).
    ``reset`` returns a single ``(primary, None)`` image tuple.
    """

    # Single task id registered by gym_pusht
    TASKS: list[str] = ["gym_pusht/PushT-v0"]

    def __init__(self) -> None:
        self.env = None
        self._task_id: str = ""
        self._cam_res: int = 96
        self._last_obs = None
        self._last_reward: float = 0.0
        # Coverage ≥ 0.9 → success (matches DP paper threshold)
        self._success_threshold: float = 0.9
        self._step_count: int = 0
        self._max_steps: int = 300

    # ---------------------------------------------------------------- helpers
    def _resolve_task(self, task_name: str) -> str:
        """Map a user-supplied task name or index to the gymnasium id."""
        if task_name.isdigit():
            idx = int(task_name)
            if idx != 0:
                raise ValueError(f"GymPushT only has task index 0 (push_t). Got {idx}.")
            return self.TASKS[0]
        norm = task_name.lower().replace("_", "").replace("-", "").replace("/", "")
        for tid in self.TASKS:
            if norm in tid.lower().replace("_", "").replace("-", "").replace("/", ""):
                return tid
        # Accept bare "push_t" or "pusht" as aliases
        if norm in ("pusht", "pusht", "pushtv0"):
            return self.TASKS[0]
        raise ValueError(
            f"GymPushT task '{task_name}' not recognised. Available: {self.TASKS} or index 0."
        )

    # ------------------------------------------------------------------- init
    def init(
        self,
        task_name: str,
        camera_resolution: int = 96,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        self._cam_res = int(camera_resolution) if camera_resolution else 96
        self._task_id = self._resolve_task(task_name)

        import gym_pusht  # registers gym_pusht/PushT-v0
        import gymnasium as gym

        # render_mode="rgb_array" for headless pixel observations.
        # obs_type="pixels_agent_pos" returns dict with "pixels" and "agent_pos".
        self.env = gym.make(
            self._task_id,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
        )
        spec_max = getattr(self.env.spec, "max_episode_steps", None)
        if spec_max:
            self._max_steps = int(spec_max)

        return {
            "task_description": (
                "Push the T-shaped block into the target zone using the disk end-effector."
            ),
            "task_id": self._task_id,
        }

    # ------------------------------------------------------------------ reset
    def reset(self, episode_index: int | None = None):
        seed = 42 if episode_index is None else 1000 + int(episode_index)
        obs, _info = self.env.reset(seed=seed)
        self._last_obs = obs
        self._last_reward = 0.0
        self._step_count = 0
        return self._extract_image(obs), None

    # ------------------------------------------------------------------- step
    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        # Accept 2-dim actions; truncate longer vectors (e.g. from chunked VLAs)
        action_arr = action_arr.reshape(-1)[:2]
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        self._last_obs = obs
        self._last_reward = float(reward)
        self._step_count += 1
        success = self._last_reward >= self._success_threshold
        done = bool(terminated or truncated or success)
        info = dict(info) if isinstance(info, dict) else {}
        info["success"] = bool(success)
        img = self._extract_image(obs)
        return img, None, self._last_reward, done, info

    # ---------------------------------------------------------------- get_obs
    def get_obs(self):
        return self._extract_image(self._last_obs), None

    def check_success(self) -> bool:
        return bool(self._last_reward >= self._success_threshold)

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None

    # --------------------------------------------------------------- get_info
    def get_info(self) -> dict:
        cam_res = self._cam_res
        return {
            "action_space": {
                "type": "eef_xy",
                "dim": 2,
                "accepted_dims": [2],
            },
            "obs_space": {
                "cameras": [
                    {"key": "pixels", "resolution": [cam_res, cam_res], "role": "primary"},
                ],
                "state": {"dim": 2, "format": "agent_xy_position"},
                "image_transform": "none",
            },
            "max_steps": self._max_steps,
            "delta_actions": False,
            "action_spec": {
                "eef_xy": {
                    "name": "eef_xy",
                    "dims": 2,
                    "format": "absolute_xy_position",
                    "accepts": ["absolute_xy_position"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 2, "format": "agent_xy_position"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
        }

    # ------------------------------------------------------------- image helper
    def _extract_image(self, obs) -> np.ndarray:
        """Return the RGB pixel observation from gym-pusht."""
        if obs is None:
            return np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)
        if isinstance(obs, dict):
            pixels = obs.get("pixels")
            if pixels is not None:
                return np.asarray(pixels, dtype=np.uint8)
        # Fallback: render directly
        rendered = self.env.render()
        if rendered is not None:
            return np.asarray(rendered, dtype=np.uint8)
        return np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)

    def _extract_state(self, obs) -> list:
        """Return the 2-dim agent (x, y) position."""
        if obs is None:
            return [0.0, 0.0]
        if isinstance(obs, dict):
            agent_pos = obs.get("agent_pos")
            if agent_pos is not None:
                return np.asarray(agent_pos, dtype=np.float32).reshape(-1)[:2].tolist()
        return [0.0, 0.0]


# === END gym_pusht ===


# === START maniskill2 ===
# ======================================================================
# ManiSkill2Backend — Hao Su lab manipulation benchmark (SAPIEN-based)
#
# aarch64 availability note
# ──────────────────────────────────────────────────────────────────────
# mani_skill2==0.5.3 requires sapien==2.2.2, which ships ONLY
# manylinux2014_x86_64 wheels on PyPI.  sapien 3.x (used by mani-skill3)
# also lacks aarch64 wheels.  The SAPIEN nightly index (used by RoboTwin)
# does not publish sapien 2.2.2 for aarch64 either.
#
# On aarch64 this backend remains discoverable:
#   - registered in BACKENDS for introspection,
#   - get_info() returns a valid spec for tooling introspection,
#   - init() raises RuntimeError with remediation guidance.
#
# x86_64 users can install the full ManiSkill2 stack.
#
# Package   : mani_skill2  (PyPI)
# Python    : 3.9+
# Renderer  : SAPIEN (requires sapien==2.2.2, x86_64 only on PyPI)
# Action    : pd_ee_delta_pose  7-dim (6 EEF delta + 1 gripper)
# Tasks     : PickCube-v0, StackCube-v0, PegInsertionSide-v0
# ======================================================================


class ManiSkill2Backend(SimBackendBase):
    """Backend for ManiSkill2 (Hao Su lab, SAPIEN-based manipulation benchmark).

    .. warning:: **aarch64 availability**

        ``mani_skill2==0.5.3`` requires ``sapien==2.2.2``, which ships only
        ``manylinux2014_x86_64`` wheels on PyPI.  ``sapien 3.x`` (used by
        ``mani-skill3``) also lacks aarch64 wheels.  The SAPIEN nightly index
        (used by RoboTwin) does not publish ``sapien==2.2.2`` for aarch64.

        On aarch64 this backend remains introspectable: ``get_info()`` returns
        a valid typed spec so tooling can introspect it; ``init()`` raises
        ``RuntimeError`` with remediation guidance.

        x86_64 environments can install the full ManiSkill2 stack.

    Tasks (ManiSkill2 gymnasium env ids):

    * ``PickCube-v0``         — pick up a cube and lift it to a target height.
    * ``StackCube-v0``        — stack the green cube on top of the red cube.
    * ``PegInsertionSide-v0`` — insert a peg into a side-hole box.

    Action space: ``pd_ee_delta_pose`` controller — 7-dim
    ``(Δx, Δy, Δz, Δax, Δay, Δaz, gripper)`` — compatible with pi05,
    smolvla, and openvla single-arm checkpoints.

    Camera: ``base_camera`` 256×256 RGB (primary); no wrist camera by default.

    Lifecycle mirrors :class:`LiberoBackend` (``init → reset → step``).
    """

    # Canonical ManiSkill2 task ids shipped with the package.  All three share
    # the same 7-dim pd_ee_delta_pose action space so they work with pi05 /
    # smolvla / openvla without action-space remapping.
    TASKS: list[str] = [
        "PickCube-v0",
        "StackCube-v0",
        "PegInsertionSide-v0",
    ]

    # Human-readable task descriptions (ManiSkill2 has no built-in language labels).
    _TASK_DESCRIPTIONS: dict[str, str] = {
        "PickCube-v0": "Pick up the red cube and lift it above the table.",
        "StackCube-v0": "Stack the green cube on top of the red cube.",
        "PegInsertionSide-v0": "Insert the peg into the side-hole box.",
    }

    # Max episode steps for each task (ManiSkill2 defaults).
    _TASK_MAX_STEPS: dict[str, int] = {
        "PickCube-v0": 200,
        "StackCube-v0": 200,
        "PegInsertionSide-v0": 500,
    }

    def __init__(self) -> None:
        self.env = None
        self._task_id: str = ""
        self._cam_res: int = 256
        self._last_obs = None
        self._last_reward: float = 0.0
        self._step_count: int = 0
        self._max_steps: int = 200

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _aarch64_blocker_message() -> str:
        """Return a descriptive RuntimeError message for aarch64 installs."""
        return (
            "ManiSkill2 is not available on aarch64: mani_skill2==0.5.3 requires "
            "sapien==2.2.2, which ships only manylinux2014_x86_64 wheels on PyPI.  "
            "sapien 3.x (mani-skill3) also lacks aarch64 wheels as of 2026-04-26.  "
            "Options:\n"
            "  1. Run on x86_64 and use: ./scripts/setup.sh maniskill2\n"
            "  2. Watch https://github.com/haosulab/SAPIEN/releases for an "
            "aarch64 wheel and re-run setup once published.\n"
            "  3. Use gym-aloha (aloha_gym) or RoboTwin for aarch64-compatible "
            "benchmarks in this harness."
        )

    def _resolve_task(self, task_name: str) -> str:
        """Map a user-supplied task name or numeric index to a ManiSkill2 env id."""
        if task_name.isdigit():
            idx = int(task_name)
            if not 0 <= idx < len(self.TASKS):
                raise ValueError(
                    f"ManiSkill2 task index {idx} out of range "
                    f"(0-{len(self.TASKS) - 1}). Available: {self.TASKS}"
                )
            return self.TASKS[idx]
        # Exact match (case-insensitive, normalised)
        norm = task_name.lower().replace("-", "").replace("_", "")
        for tid in self.TASKS:
            if norm == tid.lower().replace("-", "").replace("_", ""):
                return tid
        # Substring match
        matches = [t for t in self.TASKS if norm in t.lower().replace("-", "").replace("_", "")]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"ManiSkill2 task '{task_name}' not recognised. "
            f"Available: {self.TASKS} (or numeric indices 0-{len(self.TASKS) - 1})"
        )

    # ------------------------------------------------------------------- init

    def init(
        self,
        task_name: str,
        camera_resolution: int = 256,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        """Initialise ManiSkill2 for *task_name*.

        .. warning::
            Raises ``RuntimeError`` on **aarch64** — ``sapien==2.2.2`` has no
            aarch64 wheel.  On x86_64 this performs a full ``mani_skill2``
            import and ``gymnasium.make()`` call.
        """
        import platform

        if platform.machine() == "aarch64":
            raise RuntimeError(self._aarch64_blocker_message())

        # x86_64 path — attempt real import
        try:
            import gymnasium as gym
            import mani_skill2.envs  # registers all task envs
        except ImportError as exc:
            raise RuntimeError(
                f"mani_skill2 import failed: {exc}.  Run: ./scripts/setup.sh maniskill2"
            ) from exc

        self._task_id = self._resolve_task(task_name)
        self._cam_res = int(camera_resolution) if camera_resolution else 256
        sim_config = sim_config or {}
        self._max_steps = int(
            sim_config.get("max_steps", self._TASK_MAX_STEPS.get(self._task_id, 200))
        )

        self.env = gym.make(
            self._task_id,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            render_mode="cameras",
            camera_cfgs={"width": self._cam_res, "height": self._cam_res},
        )
        self._step_count = 0
        self._last_reward = 0.0

        description = self._TASK_DESCRIPTIONS.get(self._task_id, self._task_id)
        return {"task_description": description, "task_id": self._task_id}

    # ------------------------------------------------------------------ reset

    def reset(self, episode_index: int | None = None) -> tuple:
        seed = 42 if episode_index is None else 1000 + int(episode_index)
        obs, _info = self.env.reset(seed=seed)
        self._last_obs = obs
        self._last_reward = 0.0
        self._step_count = 0
        return self._extract_image(obs)

    # ------------------------------------------------------------------- step

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim > 1:
            action_arr = action_arr.flatten()
        # pd_ee_delta_pose expects exactly 7 dims.
        if action_arr.shape[0] > 7:
            action_arr = action_arr[:7]
        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        self._last_obs = obs
        self._last_reward = float(reward)
        self._step_count += 1
        success = bool(info.get("success", False))
        done = bool(terminated or truncated or success)
        info = dict(info) if isinstance(info, dict) else {}
        info["success"] = success
        img, img2 = self._extract_image(obs)
        return img, img2, self._last_reward, done, info

    # ---------------------------------------------------------------- get_obs

    def get_obs(self) -> tuple:
        return self._extract_image(self._last_obs)

    # ---------------------------------------------------------- check_success

    def check_success(self) -> bool:
        if self._last_obs is None:
            return False
        # ManiSkill2 reports success in info; mirror via env if available.
        if hasattr(self.env, "get_attr"):
            try:
                return bool(self.env.get_attr("is_success")[0])
            except Exception:
                pass
        return False

    # ----------------------------------------------------------------- close

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            finally:
                self.env = None

    # ---------------------------------------------------------------- get_info

    def get_info(self) -> dict:
        """Return action/observation spec.

        Returns a valid spec even on aarch64 (where ``init()`` is blocked) so
        tooling can introspect the expected contract without installing
        ManiSkill2.
        """
        cam_res = self._cam_res
        task = self._task_id or "PickCube-v0"
        max_steps = self._TASK_MAX_STEPS.get(task, 200)
        return {
            "sim": "maniskill2",
            "action_space": {
                "type": "eef_delta",
                "dim": 7,
                "accepted_dims": [7],
                "controller": "pd_ee_delta_pose",
            },
            "obs_space": {
                "cameras": [
                    {"key": "base_camera", "resolution": [cam_res, cam_res], "role": "primary"},
                ],
                "state": {"dim": 9, "format": "qpos(7)+qvel(2)"},
                "image_transform": "none",
            },
            "max_steps": max_steps,
            "delta_actions": True,
            # Typed ActionObsSpec — matches the 7-dim eef_delta contract used by
            # pi05, smolvla, and openvla so those VLAs can drive ManiSkill2
            # without remapping (same format as LiberoBackend).
            "action_spec": {
                "position": {
                    "name": "position",
                    "dims": 3,
                    "format": "delta_xyz",
                    "range": [-1, 1],
                    "accepts": ["delta_xyz"],
                },
                "rotation": {
                    "name": "rotation",
                    "dims": 3,
                    "format": "delta_axisangle",
                    "range": [-3.15, 3.15],
                    "accepts": ["delta_axisangle", "axis_angle"],
                },
                "gripper": {
                    "name": "gripper",
                    "dims": 1,
                    "format": "binary_close_negative",
                    "range": [-1, 1],
                    "accepts": ["binary_close_negative"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 9, "format": "qpos7_qvel2"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
            # Advertise the aarch64 blocker so callers can surface it cleanly.
            "aarch64_blocked": True,
            "aarch64_note": (
                "sapien==2.2.2 (required by mani_skill2==0.5.3) has no aarch64 "
                "wheel on PyPI.  init() raises RuntimeError on aarch64."
            ),
        }

    # ------------------------------------------------------- image helpers

    def _extract_image(self, obs) -> tuple:
        """Extract ``(primary, None)`` from a ManiSkill2 obs dict.

        ManiSkill2 obs_mode='rgbd' nests camera images under
        ``obs["image"]["base_camera"]["rgb"]``.  Returns ``(primary, None)``
        since there is no wrist camera in the default single-arm config.
        """
        if obs is None:
            blank = np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)
            return blank, None
        try:
            rgb = obs["image"]["base_camera"]["rgb"]
            img = np.asarray(rgb, dtype=np.uint8).copy()
            return img, None
        except (KeyError, TypeError):
            # Fallback — try flat "rgb" key
            if isinstance(obs, dict) and "rgb" in obs:
                img = np.asarray(obs["rgb"], dtype=np.uint8).copy()
                return img, None
            blank = np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)
            logger.warning(
                "ManiSkill2Backend: could not extract image from obs; "
                "returning blank frame.  obs keys: %s",
                list(obs.keys()) if isinstance(obs, dict) else type(obs),
            )
            return blank, None


# === END maniskill2 ===


# === START metaworld ===

# Per-task natural-language descriptions used by ``MetaWorldBackend.init()``.
_MW_TASK_DESCRIPTIONS: dict[str, str] = {
    # Keys use V3 names (installed metaworld package is V3).
    # V2 names (e.g. "button-press-v2") are accepted by _resolve_task() and
    # automatically mapped to the corresponding V3 name.
    "assembly-v3": "Insert the peg into the ring.",
    "basketball-v3": "Dunk the basketball into the basket.",
    "bin-picking-v3": "Pick up the object and place it in the bin.",
    "box-close-v3": "Close the box by pushing the lid down.",
    "button-press-topdown-v3": "Press the red button from above.",
    "button-press-topdown-wall-v3": "Press the button from above, reaching over the wall.",
    "button-press-v3": "Press the red button.",
    "button-press-wall-v3": "Press the button on the other side of the wall.",
    "coffee-button-v3": "Push the button on the coffee machine.",
    "coffee-pull-v3": "Pull the coffee mug out of the machine.",
    "coffee-push-v3": "Push the coffee mug into the machine.",
    "dial-turn-v3": "Rotate the dial to the target position.",
    "disassemble-v3": "Remove the peg from the ring.",
    "door-close-v3": "Close the door.",
    "door-lock-v3": "Lock the door by turning the lock clockwise.",
    "door-open-v3": "Open the door.",
    "door-unlock-v3": "Unlock the door by turning the lock counter-clockwise.",
    "hand-insert-v3": "Insert the hand into the hole.",
    "drawer-close-v3": "Close the drawer.",
    "drawer-open-v3": "Open the drawer.",
    "faucet-open-v3": "Turn the faucet handle to open.",
    "faucet-close-v3": "Turn the faucet handle to close.",
    "hammer-v3": "Hammer the nail into the board.",
    "handle-press-side-v3": "Press the handle down from the side.",
    "handle-press-v3": "Press the handle down.",
    "handle-pull-side-v3": "Pull the handle up from the side.",
    "handle-pull-v3": "Pull the handle up.",
    "lever-pull-v3": "Pull the lever towards you.",
    "peg-insert-side-v3": "Insert the peg into the hole from the side.",
    "pick-place-wall-v3": "Pick up the puck and place it at the goal past the wall.",
    "pick-out-of-hole-v3": "Pick the peg out of the hole.",
    "reach-v3": "Move the robot hand to the red goal sphere.",
    "push-back-v3": "Push the puck to the goal behind the robot.",
    "push-v3": "Push the puck to the goal.",
    "pick-place-v3": "Pick up the puck and place it at the goal.",
    "plate-slide-v3": "Slide the plate forward to the goal.",
    "plate-slide-side-v3": "Slide the plate sideways to the goal.",
    "plate-slide-back-v3": "Slide the plate backwards to the goal.",
    "plate-slide-back-side-v3": "Slide the plate diagonally back to the goal.",
    "peg-unplug-side-v3": "Remove the peg from the socket on the side.",
    "soccer-v3": "Kick the ball into the goal.",
    "stick-push-v3": "Use the stick to push the puck to the goal.",
    "stick-pull-v3": "Use the stick to pull the puck to the goal.",
    "push-wall-v3": "Push the puck past the wall to the goal.",
    "reach-wall-v3": "Reach to the goal past the wall.",
    "shelf-place-v3": "Lift the puck and place it on the shelf.",
    "sweep-into-v3": "Sweep the puck into the hole.",
    "sweep-v3": "Sweep the puck to the goal.",
    "window-close-v3": "Close the window.",
    "window-open-v3": "Open the window.",
}


class MetaWorldBackend(SimBackendBase):
    """Backend for the Meta-World benchmark (~50 single-arm manipulation tasks).

    Meta-World provides a well-known suite of tabletop manipulation tasks
    (push-button, pick-place, door-open, drawer-close, etc.) built on top of
    MuJoCo and the Sawyer robot.  It is a pure PyPI / uv-clean package with
    official aarch64-compatible wheels (MuJoCo provides aarch64 wheels; no
    SAPIEN, no curobo, no conda).

    **Action space** (4-dim end-effector delta, range [−1, 1])::

        dim 0: Δx  (EEF position delta, forward/back)
        dim 1: Δy  (EEF position delta, left/right)
        dim 2: Δz  (EEF position delta, up/down)
        dim 3: gripper open/close  (−1 = closed, +1 = open)

    **Action-dim mismatch with 7-dim VLAs (Pi0.5, SmolVLA, OpenVLA):**
    These VLAs emit 7-dim actions (6-DoF EEF delta + gripper).  The
    ``action_spec`` declared by this backend has ``dims=4`` and
    ``accepted_dims=[4]``, so the orchestrator's ``ActionObsSpec`` gate will
    BLOCK those pairings at episode start — producing a "spec-gate-blocked"
    verdict.  This is expected behaviour for v0.1; it proves the gate works.
    A 4-dim adapter or a 4-dim-native VLA is required for those pairings.

    The backend still implements a defensive 7→4 truncation inside ``step()``
    (keeps dims 0-2 for xyz and dim 6 for gripper) so isolated unit tests can
    drive the env with a 7-dim action without crashing.

    **Cameras:** ``corner`` (primary agent view) and ``behindGripper`` (wrist
    stand-in).  Both are flipped 180° (``[::-1, ::-1]``) inside
    ``_get_images()`` to match the in-sim flip convention used by every other
    backend.

    **Python:** 3.11.

    Lifecycle mirrors :class:`LiberoBackend` and :class:`AlohaGymBackend`:
    no construction-time task name; standard ``init → reset → step`` flow.
    """

    #: Canonical MT50 task names aligned with ``metaworld.MT50.train_classes``.
    TASKS: list[str] = sorted(_MW_TASK_DESCRIPTIONS.keys())

    def __init__(self) -> None:
        self.env = None
        self._task_name: str = ""
        self._cam_res: int = 256
        self._last_obs = None
        self._last_reward: float = 0.0
        self._last_success: bool = False
        self._step_count: int = 0
        self._max_steps: int = 500
        # Persistent MuJoCo renderers reused across calls to avoid the
        # overhead of re-creating them on every step() / get_obs().
        self._mj_renderer = None
        self._mj_renderer_wrist = None
        # Task-variation objects (list[metaworld.Task]); populated in init().
        self._tasks: list = []

    # ----------------------------------------------------------------- helpers

    def _resolve_task(self, task_name: str) -> str:
        """Map a user-supplied task name or integer index to a canonical MT50 name.

        Accepts V3 names (canonical), V2 names (automatically promoted to V3),
        integer indices, or unambiguous substrings.
        """
        if task_name.isdigit():
            idx = int(task_name)
            if not 0 <= idx < len(self.TASKS):
                raise ValueError(
                    f"MetaWorld task index {idx} out of range "
                    f"(0-{len(self.TASKS) - 1}).  Available: {self.TASKS}"
                )
            return self.TASKS[idx]
        if task_name in self.TASKS:
            return task_name
        # V2 → V3 automatic promotion: "button-press-v2" → "button-press-v3".
        if task_name.endswith("-v2"):
            v3_name = task_name[:-3] + "-v3"
            if v3_name in self.TASKS:
                return v3_name
        # Fuzzy: normalise underscores to dashes and try substring match.
        norm = task_name.lower().replace("_", "-")
        matches = [t for t in self.TASKS if norm in t.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"MetaWorld task '{task_name}' is ambiguous — matches {matches}.")
        raise ValueError(
            f"MetaWorld task '{task_name}' not recognised.  "
            f"Available ({len(self.TASKS)}): {self.TASKS}"
        )

    def _init_renderers(self) -> None:
        """(Re-)create mujoco.Renderer instances for corner and behindGripper cameras."""
        self._close_renderers()
        try:
            import mujoco

            self._mj_renderer = mujoco.Renderer(
                self.env.model, height=self._cam_res, width=self._cam_res
            )
            self._mj_renderer_wrist = mujoco.Renderer(
                self.env.model, height=self._cam_res, width=self._cam_res
            )
        except Exception:
            # Graceful degradation: _render_camera returns zeros when renderer is None.
            self._mj_renderer = None
            self._mj_renderer_wrist = None

    def _close_renderers(self) -> None:
        for attr in ("_mj_renderer", "_mj_renderer_wrist"):
            r = getattr(self, attr, None)
            if r is not None:
                try:
                    r.close()
                except Exception:
                    pass
            setattr(self, attr, None)

    def _render_camera(self, camera_name: str, renderer) -> np.ndarray:
        """Render one MuJoCo camera view; return 180°-flipped RGB uint8 array."""
        if renderer is not None:
            try:
                renderer.update_scene(self.env.data, camera=camera_name)
                img = np.asarray(renderer.render(), dtype=np.uint8).copy()
                # Apply the canonical in-sim 180° flip (matches all other backends).
                return img[::-1, ::-1]
            except Exception:
                pass
        # Fallback: black frame when rendering is unavailable.
        return np.zeros((self._cam_res, self._cam_res, 3), dtype=np.uint8)

    def _get_images(self):
        """Return ``(primary, wrist)`` — corner camera + behindGripper camera."""
        primary = self._render_camera("corner", self._mj_renderer)
        wrist = self._render_camera("behindGripper", self._mj_renderer_wrist)
        return primary, wrist

    # ------------------------------------------------------------------- init

    def init(
        self,
        task_name: str,
        camera_resolution: int = 256,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        # MuJoCo backend selection must precede any mujoco import.
        os.environ["MUJOCO_GL"] = "egl" if headless else "glfw"
        self.headless = headless
        self._cam_res = int(camera_resolution) if camera_resolution else 256

        import metaworld

        task_name_resolved = self._resolve_task(task_name)
        self._task_name = task_name_resolved

        # MT1 provides 50 procedurally generated variations of one task.
        mt1 = metaworld.MT1(task_name_resolved, seed=42)
        env_cls = mt1.train_classes[task_name_resolved]
        self.env = env_cls()
        self._tasks = list(mt1.train_tasks)

        # Deterministic: always start with variation 0.
        self.env.set_task(self._tasks[0])
        obs, _ = self.env.reset()
        self._last_obs = obs
        self._last_reward = 0.0
        self._last_success = False
        self._step_count = 0

        # Build MuJoCo renderers now that env.model/env.data are populated.
        self._init_renderers()

        desc = _MW_TASK_DESCRIPTIONS.get(task_name_resolved, task_name_resolved)
        return {
            "task_description": f"Meta-World {task_name_resolved}: {desc}",
            "task_id": task_name_resolved,
        }

    # ------------------------------------------------------------------ reset

    def reset(self, episode_index: int | None = None):
        # Rotate task variation by episode_index for diverse initial states.
        if self._tasks:
            task_idx = 0 if episode_index is None else int(episode_index) % len(self._tasks)
            self.env.set_task(self._tasks[task_idx])
        obs, _ = self.env.reset()
        self._last_obs = obs
        self._last_reward = 0.0
        self._last_success = False
        self._step_count = 0
        return self._get_images()

    # ------------------------------------------------------------------- step

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        # Adapt incoming action to the native 4-dim MetaWorld action space.
        # 7-dim VLA: [dx, dy, dz, drx, dry, drz, gripper] → keep [dx, dy, dz, gripper].
        # The ActionObsSpec gate should block 7-dim VLAs before reaching here;
        # this fallback supports isolated unit tests and ROBOEVAL_STRICT_SPECS=0.
        n = action_arr.shape[0]
        if n >= 7:
            action_4d = np.concatenate([action_arr[:3], action_arr[6:7]])
        elif n == 4:
            action_4d = action_arr
        elif n < 4:
            action_4d = np.pad(action_arr, (0, 4 - n))
        else:
            # 5 or 6 dims: keep first 3 + last.
            action_4d = np.concatenate([action_arr[:3], action_arr[-1:]])

        obs, reward, terminated, truncated, info = self.env.step(action_4d)
        self._last_obs = obs
        self._last_reward = float(reward)
        self._last_success = bool(info.get("success", False))
        self._step_count += 1
        done = bool(terminated or truncated or self._last_success)
        info_out = dict(info) if isinstance(info, dict) else {}
        info_out["success"] = self._last_success
        img, img2 = self._get_images()
        return img, img2, self._last_reward, done, info_out

    # ---------------------------------------------------------------- get_obs

    def get_obs(self):
        return self._get_images()

    def check_success(self) -> bool:
        return self._last_success

    def get_state(self) -> list:
        """Return the 39-dim metaworld proprioceptive state as a flat float list.

        The upstream TDMPC2 single-task checkpoints are state-based; they
        require this vector rather than camera images.
        """
        obs = self._last_obs
        if obs is None:
            return [0.0] * 39
        return np.asarray(obs, dtype=np.float32).flatten().tolist()

    def close(self) -> None:
        self._close_renderers()
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            finally:
                self.env = None

    # --------------------------------------------------------------- get_info

    def get_info(self) -> dict:
        cam_res = self._cam_res
        return {
            # 4-dim eef-delta: [dx, dy, dz, gripper].
            # ``accepted_dims=[4]`` intentionally excludes 7 so the spec gate
            # catches mismatches with 7-dim VLAs (Pi0.5, SmolVLA, OpenVLA).
            "action_space": {
                "type": "eef_delta",
                "dim": 4,
                "accepted_dims": [4],
            },
            "obs_space": {
                "cameras": [
                    {"key": "corner", "resolution": [cam_res, cam_res], "role": "primary"},
                    {"key": "behindGripper", "resolution": [cam_res, cam_res], "role": "wrist"},
                ],
                # MetaWorld observation vector is typically 39-dim for MT-Bench tasks:
                # [EEF pos (3), EEF vel (3), gripper (1), object features (~32)].
                "state": {"dim": 39, "format": "metaworld_obs(39)"},
                "image_transform": "applied_in_sim",
            },
            "max_steps": self._max_steps,
            "delta_actions": True,
            # ActionObsSpec contract.
            # INTENTIONAL MISMATCH vs 7-dim VLAs — see class docstring.
            "action_spec": {
                "eef_delta": {
                    "name": "eef_delta",
                    "dims": 4,
                    "format": "eef_delta_xyz_gripper",
                    "accepts": ["eef_delta_xyz_gripper"],
                },
            },
            "observation_spec": {
                "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
                "state": {"name": "state", "dims": 39, "format": "metaworld_obs"},
                "instruction": {"name": "language", "dims": 0, "format": "language"},
            },
        }


# === END metaworld ===


# ======================================================================
# Backend registry — add new backends here (key must match SimConfig.name)
# ======================================================================

BACKENDS: dict[str, type[SimBackendBase]] = {
    "libero": LiberoBackend,
    "robocasa": RoboCasaBackend,
    "robotwin": RoboTwinBackend,
    "libero_pro": LiberoProBackend,
    "libero_infinity": LiberoInfinityBackend,
    "aloha_gym": AlohaGymBackend,
    "gym_pusht": GymPushTBackend,
    "maniskill2": ManiSkill2Backend,
    "metaworld": MetaWorldBackend,
}


# ======================================================================
# Pydantic request models
# ======================================================================


class InitRequest(BaseModel):
    # sim is optional: if omitted, the server uses the --sim CLI startup arg.
    sim: str | None = None
    task: str
    suite: str | None = None
    camera_resolution: int = 256
    # headless=True → EGL offscreen (CI/server default).
    # headless=False → GLFW window with live viewer.
    # If None, falls back to the server-level --headless CLI flag.
    headless: bool | None = None
    # delta_actions=True → set robot.controller.use_delta=True after each reset.
    # Required for Pi0.5 which outputs relative/delta actions.
    delta_actions: bool = False
    # sim_config: opaque dict forwarded to the backend's init(). Used by
    # LiberoInfinityBackend (perturbation, max_steps, seed, max_distractors…).
    # All other backends accept and ignore it.
    sim_config: dict | None = None


class StepRequest(BaseModel):
    action: list


class ResetRequest(BaseModel):
    seed: int = 42
    episode_index: int | None = None


# ======================================================================
# FastAPI application
# ======================================================================

app = FastAPI(title="roboeval Sim Worker")
backend = None
_current_sim_name: str = ""
_current_suite: str = ""
# Set in main() from --sim CLI arg; used as default when InitRequest.sim is None.
_server_sim: str = ""


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
        # Use the request's sim field; fall back to the server-level --sim arg
        # so that clients can omit "sim" when the server was started with --sim.
        sim_name = req.sim or _server_sim
        _current_sim_name = sim_name
        _current_suite = req.suite or ""
        if sim_name not in BACKENDS:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Unknown sim '{sim_name}'. Must be one of: {list(BACKENDS.keys())}"
                },
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

        if sim_name == "robotwin":
            # RoboTwin takes task_name at construction time and uses initialize().
            # Initialize before assigning to global to avoid race: a concurrent
            # /step would see backend.env=None if we assigned before init().
            new_backend = RoboTwinBackend(task_name=req.task)
            result = new_backend.initialize(seed=42)
            backend = new_backend
        else:
            backend_cls = BACKENDS[sim_name]
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
def reset_env(req: ResetRequest | None = None):
    """Reset the environment to a specific episode init state."""
    try:
        global backend

        # If the environment crashed in a previous episode and set self.env = None,
        # we need to re-initialize it before resetting.
        if backend is not None and hasattr(backend, "env") and backend.env is None:
            # Attempt backend-local recovery when supported.
            import logging

            logging.warning("backend.env is None during /reset, attempting to re-initialize.")
            if isinstance(backend, RoboTwinBackend):
                backend.initialize()
            else:
                # The offscreen render environment must be recreated by /init.
                pass
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "backend.env is None. Environment crashed. Please call /init again.",
                        "traceback": "",
                    },
                )

        if isinstance(backend, RoboTwinBackend):
            # Seed formula matching RoboTwin's stable evaluation range:
            #   st_seed = 100_000 * (1 + episode_index)
            # Seeds 0–99 produce many UnStableError failures; the 100k-scale
            # range is where RoboTwin's physics stability checks pass reliably.
            # If UnStableError is raised, increment seed by 1 and retry (up to
            # 20 attempts).
            if req is not None and req.episode_index is not None:
                base_seed = 100_000 * (1 + int(req.episode_index))
            elif req is not None:
                base_seed = req.seed
            else:
                base_seed = 100_000

            import logging as _log

            _logger = _log.getLogger(__name__)
            result = None
            last_exc = None
            for _attempt in range(20):
                seed = base_seed + _attempt
                try:
                    result = backend.reset(seed=seed)
                    break
                except Exception as _e:
                    _logger.warning(
                        "RoboTwin reset seed=%d attempt=%d failed: %s",
                        seed,
                        _attempt,
                        _e,
                    )
                    last_exc = _e
            if result is None:
                raise RuntimeError(
                    f"RoboTwin reset failed after 20 seed attempts (base={base_seed}): {last_exc}"
                ) from last_exc
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
        if hasattr(backend, "_extract_state") and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, "get_state"):
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
        if hasattr(backend, "_extract_state") and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, "get_state"):
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
        if hasattr(backend, "_extract_state") and backend._last_obs:
            resp["state"] = backend._extract_state(backend._last_obs)
        elif hasattr(backend, "get_state"):
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

    parser = argparse.ArgumentParser(description="roboeval Sim Worker HTTP Server")
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
    global _headless, _server_sim
    _headless = args.headless
    _server_sim = args.sim
    os.environ["MUJOCO_GL"] = "egl" if args.headless else "glfw"

    logger.info(
        "Starting %s server on %s:%d (%s)",
        args.sim,
        args.host,
        args.port,
        "headless/EGL" if args.headless else "windowed/GLFW",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
