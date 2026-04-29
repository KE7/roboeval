"""Template simulator backend — copy this file and fill in the six TODO sections.

Quick-start (4 steps for a working sim backend):

    1. Rename ``MySimBackend`` to something meaningful (e.g. ``ManiskillBackend``).
    2. Fill in ``init()``, ``reset()``, ``step()``, and ``get_obs()``.
    3. Implement ``check_success()`` and ``get_info()``.
    4. Register the class in ``BACKENDS`` in ``sims/sim_worker.py``.

The sim backend is wrapped by a FastAPI server (``sims/sim_worker.py``) that
exposes these HTTP endpoints:

    POST /init     → {"success": true, "task_description": "..."}
    POST /reset    → {"image": b64, "images": {...}, "state": [...], "success": true}
    POST /step     → {"image": b64, "images": {...}, "reward": 0.0, "done": false, ...}
    GET  /obs      → {"image": b64, "images": {...}, "state": [...]}
    GET  /info     → result of get_info()
    GET  /success  → {"success": true/false}

Your backend does NOT implement HTTP — that is handled entirely by sim_worker.py.
You only implement the six Python methods below.

Return conventions:
    reset()   → (img: np.ndarray, img2: np.ndarray | None)
    step()    → (img, img2, reward: float, done: bool, info: dict)
    get_obs() → (img, img2)

    img   = H×W×3 uint8 NumPy array, primary camera (required)
    img2  = H×W×3 uint8 NumPy array, wrist camera  (None if unavailable)

Image flip contract:
    If your simulator's camera is mounted upside-down relative to the training
    data (as LIBERO cameras are), apply the 180° flip here in the backend:

        img = np.asarray(raw, dtype=np.uint8).copy()[::-1, ::-1]

    If your camera is already right-side-up, do NOT flip.  Whichever choice you
    make, set obs_requirements.image_transform in get_info() to communicate it:
        "applied_in_sim"  — flip was done here; env_wrapper must skip client flip
        "none"            — no flip needed

    The VLA policy will also declare its expected transform.  The orchestrator
    compares them and raises an error on mismatch.

See docs/extending.md for the full walkthrough, pitfall list, and checklist.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# TODO [STEP 1]: Add any imports your simulator needs here.
# =============================================================================
# import your_sim_package


class MySimBackend:
    """Template simulator backend.

    TODO [STEP 2]: Replace MySimBackend with your simulator's name and implement
    the six required methods.
    """

    def __init__(self) -> None:
        # TODO: declare instance variables used across methods.
        self.env = None  # the actual simulator environment object
        self._last_obs: dict = {}  # cache last raw observation dict
        self._cam_res: int = 256  # camera resolution (set in init())
        self._task_name: str = ""
        self._delta_actions: bool = False

    # =========================================================================
    # REQUIRED METHOD 1 — init
    # =========================================================================
    def init(
        self,
        task_name: str,
        camera_resolution: int,
        suite: str | None = None,
        headless: bool = True,
        sim_config: dict | None = None,
    ) -> dict:
        """Initialize the simulator environment for a specific task.

        Called once at the start of an evaluation run (or when /init is called).
        After this method returns, the backend must be ready for reset() calls.

        Args:
            task_name:         Task identifier — may be a name string, numeric index
                               string ("0", "1"), or task-suite-specific format.
            camera_resolution: Pixel size for rendered images (square: res×res).
            suite:             Optional suite/dataset qualifier (e.g. "libero_spatial").
                               Some simulators have a single suite and can ignore this.
            headless:          True = EGL offscreen rendering (CI/server default).
                               False = GLFW windowed rendering (local development).
            sim_config:        Optional dict of backend-specific overrides.  The keys
                               are not standardized — document yours in get_info().

        Returns:
            Dict that will be merged into the /init HTTP response.
            Must include at minimum: {"task_description": "<natural language task>"}

        TODO [STEP 3]: implement this method.
        """
        raise NotImplementedError("implement init()")

    # =========================================================================
    # REQUIRED METHOD 2 — reset
    # =========================================================================
    def reset(self, episode_index: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Reset to the start of a new episode.

        Args:
            episode_index: Which pre-recorded initial state to use, or None for
                           auto-increment.  Simulators without fixed init states
                           (e.g. RoboCasa) can ignore this.

        Returns:
            (img, img2):
              img   — H×W×3 uint8 NumPy array, primary camera view.
              img2  — H×W×3 uint8 NumPy array, wrist camera view, or None.

        ── Image flip ────────────────────────────────────────────────────────
        Apply the 180° flip here if your camera is physically upside-down:

            img = np.asarray(raw_img).copy()[::-1, ::-1]

        Do NOT flip if the image is already right-side-up.
        Mirror your choice in get_info() → obs_requirements.image_transform.

        TODO [STEP 4]: implement this method.
        """
        raise NotImplementedError("implement reset()")

    # =========================================================================
    # REQUIRED METHOD 3 — step
    # =========================================================================
    def step(self, action: list) -> tuple[np.ndarray, np.ndarray | None, float, bool, dict]:
        """Execute one action step and return the result.

        Args:
            action: Flat list of floats from the VLA policy.
                    Length and format depend on what your sim declared in
                    get_info() → action_space.dim / action_space.type.

                    For EEF delta:   [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    For joint_pos:   [q0, q1, …, q_{n-1}]

                    Gripper sign depends on convention — see get_info() and docs.

        Returns:
            (img, img2, reward, done, info):
              img    — H×W×3 uint8 NumPy array, primary camera.
              img2   — H×W×3 uint8 NumPy array, wrist camera, or None.
              reward — scalar float (0.0 for sparse reward, 1.0 on success typical).
              done   — True when the episode should terminate (success OR timeout).
              info   — dict; must include {"success": bool}.

        TODO [STEP 5]: implement this method.
        """
        raise NotImplementedError("implement step()")

    # =========================================================================
    # REQUIRED METHOD 4 — get_obs
    # =========================================================================
    def get_obs(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Return the current observation without stepping.

        Called by GET /obs.  Typically returns the cached observation from the
        last reset() or step() call rather than re-querying the simulator.

        Returns:
            (img, img2) — same format as reset().

        TODO [STEP 6]: implement this method.
        """
        raise NotImplementedError("implement get_obs()")

    # =========================================================================
    # REQUIRED METHOD 5 — check_success
    # =========================================================================
    def check_success(self) -> bool:
        """Return True if the current episode's success condition is met.

        Called by GET /success.  This signal is used for final result tallying.
        It should be consistent with the ``"success"`` key in step()'s info dict.

        If your simulator provides success via the step() info dict only (not a
        separate query), cache it in an instance variable and return it here.

        TODO [STEP 7]: implement this method.
        """
        raise NotImplementedError("implement check_success()")

    # =========================================================================
    # REQUIRED METHOD 6 — close
    # =========================================================================
    def close(self) -> None:
        """Release simulator resources (rendering contexts, GPU memory, etc.).

        Called on /close or when the backend is replaced via a new /init call.
        Must be safe to call multiple times and on a partially-initialized backend.
        """
        if self.env is not None:
            # TODO: call the appropriate cleanup method for your sim, e.g.:
            # self.env.close()
            self.env = None

    # =========================================================================
    # REQUIRED METHOD 7 — get_info
    # =========================================================================
    def get_info(self) -> dict:
        """Return action/observation space metadata for spec checking and logging.

        Called by GET /info after /init.  This dict drives:
          - Orchestrator spec compatibility checks (action dims, gripper sign, …).
          - env_wrapper image transform decisions.
          - Result file metadata.

        Required keys:
            action_space.type        – "eef_delta" | "joint_pos" | "eef_absolute" | …
            action_space.dim         – native action dimensionality
            action_space.accepted_dims – list of dims this backend can accept (after
                                         any internal padding/trimming); e.g. [7, 12]
            obs_space.cameras        – list of dicts with {"key", "resolution", "role"}
                                       roles: "primary" (required), "wrist", "secondary"
            obs_space.state          – {"dim": N, "format": "<name>"}
            max_steps                – hard episode time limit (steps, not seconds)
            delta_actions            – True if actions are relative EEF deltas;
                                       False if they are absolute joint/EEF targets

        Typed spec keys (optional but strongly recommended):
            action_spec      – dict of component_name → ActionObsSpec.to_dict()
            observation_spec – dict of role_name → ActionObsSpec.to_dict()

        TODO [STEP 8]: fill in the correct values for your simulator.

        ── Gripper sign convention ────────────────────────────────────────────
        Two conventions exist in the ecosystem — declare yours explicitly:
            LIBERO:  -1 = close, +1 = open   (binary_close_negative)
            RLDS:    +1 = close, -1 = open   (binary_close_positive)
        Mismatching this between the VLA and the sim causes the gripper to work
        in reverse. Always check this when integrating a new VLA × sim pair.

        ── State format ──────────────────────────────────────────────────────
        Most LIBERO-style sims encode state as:
            eef_pos(3) + axis_angle(3) + gripper_qpos(2)  → 8-dim
        Do NOT use quaternion for the rotation component unless your VLA was
        trained with quaternion state. LIBERO-style VLAs typically expect
        axis-angle state.
        If your sim natively provides quaternion, convert it using the
        formula in LiberoBackend._extract_state() in sims/sim_worker.py.
        """
        cam_res = self._cam_res
        return {
            # ── Action space ──────────────────────────────────────────────
            "action_space": {
                "type": "eef_delta",  # TODO: your action type
                "dim": 7,  # TODO: your native action dim
                "accepted_dims": [7],  # TODO: dims you can handle after padding/trimming
            },
            # ── Observation space ─────────────────────────────────────────
            "obs_space": {
                "cameras": [
                    {
                        "key": "agentview_image",  # TODO: obs dict key in your sim
                        "resolution": [cam_res, cam_res],
                        "role": "primary",  # always "primary" for the main camera
                    },
                    # Optional wrist camera:
                    # {
                    #     "key": "wrist_image",
                    #     "resolution": [cam_res, cam_res],
                    #     "role": "wrist",
                    # },
                ],
                "state": {
                    "dim": 8,  # TODO: state vector length (0 if none)
                    "format": "eef_pos(3)+axisangle(3)+gripper_qpos(2)",  # TODO: your format
                },
                # Communicate the image flip decision you made in reset()/step():
                #   "applied_in_sim" — flip done in this backend; env_wrapper must NOT flip again.
                #   "none"           — no flip needed for this camera orientation.
                "image_transform": "applied_in_sim",  # TODO: match your reset() / step()
            },
            # ── Episode limits ────────────────────────────────────────────
            "max_steps": 280,  # TODO: hard time limit for your tasks
            "delta_actions": False,  # TODO: True for EEF delta, False for absolute targets
            # ── Typed ActionObsSpec contracts (optional but recommended) ────────
            # These are serialized via ActionObsSpec.to_dict() — see robo_eval/specs.py.
            # Uncomment and fill in to enable spec checking:
            #
            # "action_spec": {
            #     "position": {"name": "position", "dims": 3, "format": "delta_xyz",
            #                  "range": [-1, 1], "accepts": ["delta_xyz"]},
            #     "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle",
            #                  "range": [-3.15, 3.15], "accepts": ["delta_axisangle", "axis_angle"]},
            #     "gripper":  {"name": "gripper",  "dims": 1, "format": "binary_close_negative",
            #                  "range": [-1, 1],  "accepts": ["binary_close_negative"]},
            # },
            # "observation_spec": {
            #     "primary":     {"name": "image",    "dims": 0, "format": "rgb_hwc_uint8"},
            #     "state":       {"name": "state",    "dims": 8, "format": "libero_eef_pos3_aa3_grip2"},
            #     "instruction": {"name": "language", "dims": 0, "format": "language"},
            # },
        }

    # =========================================================================
    # OPTIONAL — _extract_state
    # =========================================================================
    def _extract_state(self, obs: dict) -> list:
        """Extract proprioceptive state from a raw simulator observation dict.

        sim_worker.py calls this automatically if it exists and self._last_obs
        is set.  The result is included in the HTTP response under "state".

        Standard LIBERO format (8-dim):
            eef_pos(3) + axisangle(3) + gripper_qpos(2)

        The axis-angle MUST match the format expected by the VLA.  For
        LIBERO-style state, use the same quaternion-to-axis-angle conversion
        used during VLA training.  See LiberoBackend._extract_state() in
        sims/sim_worker.py for the reference implementation.

        TODO: implement if your sim provides proprioceptive state.
        """
        # Example skeleton for an 8-dim EEF state:
        # eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)   # (3,)
        # eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)  # (4,) [x,y,z,w]
        # gripper  = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)  # (2,)
        #
        # # Quaternion [x,y,z,w] → axis-angle (lerobot formula)
        # x, y, z, w = float(eef_quat[0]), float(eef_quat[1]), float(eef_quat[2]), float(eef_quat[3])
        # den = np.sqrt(max(0.0, 1.0 - w * w))
        # if den > 1e-10:
        #     angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
        #     axisangle = np.array([x, y, z], dtype=np.float32) / den * angle
        # else:
        #     axisangle = np.zeros(3, dtype=np.float32)
        #
        # return np.concatenate([eef_pos, axisangle, gripper]).tolist()
        raise NotImplementedError("implement _extract_state() or remove if not needed")


# =============================================================================
# Registration — add to BACKENDS dict in sims/sim_worker.py
# =============================================================================
#
# After implementing the class above, register it:
#
#   # In sims/sim_worker.py, find the BACKENDS dict and add:
#   from sims.my_sim_backend import MySimBackend   # or keep inline
#   BACKENDS = {
#       "libero":   LiberoBackend,
#       "robocasa": RoboCasaBackend,
#       "robotwin": RoboTwinBackend,
#       # ── NEW ──────────────────────────────────────
#       "my_sim":   MySimBackend,          # key used in --sim and config YAML
#   }
#
# Then register the default port/venv in robo_eval/server_runner.py:
#
#   _SIM_DEFAULT_PORTS["my_sim"]  = 5304   # pick an unused port
#   _SIM_DEFAULT_VENVS["my_sim"]  = ".venvs/my_sim"
#
# =============================================================================
