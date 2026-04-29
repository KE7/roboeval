"""
Simulator Environment Wrapper for LITEN.

Wraps robotic simulators (LIBERO, RoboCasa, RoboTwin, LIBERO-Pro) as a
BaseWorldStub by communicating with a sim_worker HTTP server.
"""

from __future__ import annotations

import base64
import logging
import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from roboeval.world_stubs import BaseWorldStub

logger = logging.getLogger(__name__)

try:
    from roboeval.specs import ActionObsSpec, check_specs

    _SPECS_AVAILABLE = True
except ImportError:
    _SPECS_AVAILABLE = False


class SpecMismatchError(RuntimeError):
    """Raised when VLA and sim spec contracts are incompatible (HARD severity)."""


class ActionChunkBuffer:
    """
    Buffer for action chunks from a VLA policy with configurable trimming and blending.

    When the policy returns a chunk of N actions this buffer:

    1. Trims the incoming chunk to ``chunk_size`` (prevents executing stale tail
       of a large chunk — e.g. pi0.5 returns 50 but effective chunk_size may be 10).
    2. Optionally blends overlapping positions with any *remaining* buffered actions
       from the previous inference call (reduces jerkiness on re-plan).
    3. Pops one action per env step; inference is only called again when the
       buffer drains.

    Blending strategies
    -------------------
    ``"newest"`` (default)
        Incoming chunk overwrites the remaining buffer entirely.
    ``"average"``
        Element-wise mean at overlapping positions; tail of new chunk appended.
    ``"ema"``
        ``alpha * new + (1 - alpha) * old`` at overlapping positions;
        tail of new chunk appended.  ``alpha`` is controlled by ``ema_alpha``.
    """

    def __init__(
        self,
        chunk_size: int = None,
        action_ensemble: str = "newest",
        ema_alpha: float = 0.5,
    ):
        """
        Args:
            chunk_size: Max actions to keep from each incoming chunk.
                ``None`` means keep the full chunk as returned by the policy.
            action_ensemble: Blending strategy (``"newest"``, ``"average"``,
                or ``"ema"``).
            ema_alpha: Blend coefficient for ``"ema"`` strategy.
                ``alpha * new + (1 - alpha) * old``.  Must be in ``[0, 1]``.
        """
        if action_ensemble not in ("newest", "average", "ema"):
            raise ValueError(
                f"action_ensemble must be one of 'newest', 'average', 'ema'; "
                f"got {action_ensemble!r}"
            )
        self._chunk_size = chunk_size
        self._action_ensemble = action_ensemble
        self._ema_alpha = ema_alpha
        self._buffer: list = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def empty(self) -> bool:
        """True when no actions remain in the buffer."""
        return len(self._buffer) == 0

    def push(self, new_chunk: list) -> None:
        """Push a new chunk into the buffer, trimming and blending as configured.

        Args:
            new_chunk: List of action vectors returned by the VLA policy.
                Each element is a ``list[float]`` (one action).
        """
        # Trim to configured chunk_size (None = keep all)
        if self._chunk_size is not None:
            new_chunk = list(new_chunk[: self._chunk_size])
        else:
            new_chunk = list(new_chunk)

        if not new_chunk:
            return

        if self._action_ensemble == "newest" or not self._buffer:
            # Replace remaining buffer entirely with the new chunk.
            self._buffer = new_chunk
            return

        overlap = min(len(self._buffer), len(new_chunk))

        if self._action_ensemble == "average":
            blended = []
            for i in range(overlap):
                old_a = np.array(self._buffer[i], dtype=float)
                new_a = np.array(new_chunk[i], dtype=float)
                blended.append(((old_a + new_a) / 2.0).tolist())
        elif self._action_ensemble == "ema":
            alpha = self._ema_alpha
            blended = []
            for i in range(overlap):
                old_a = np.array(self._buffer[i], dtype=float)
                new_a = np.array(new_chunk[i], dtype=float)
                blended.append((alpha * new_a + (1.0 - alpha) * old_a).tolist())
        else:
            # Should never reach here given __init__ validation, but be safe.
            blended = list(new_chunk[:overlap])

        # Append the non-overlapping tail of the new chunk.
        blended.extend(new_chunk[overlap:])
        self._buffer = blended

    def pop(self):
        """Pop and return the next action vector, or ``None`` if empty."""
        if not self._buffer:
            return None
        return self._buffer.pop(0)

    def clear(self) -> None:
        """Discard all buffered actions (call on episode reset or new subtask)."""
        self._buffer = []


# Expected action space per simulator (type + dim) for action translation.
# Also used for random-action fallback sizing (via the "dim" field).
SIM_EXPECTED_ACTION_SPACE = {
    "libero": {"type": "eef_delta", "dim": 7},
    "libero_pro": {"type": "eef_delta", "dim": 7},
    "libero_infinity": {"type": "eef_delta", "dim": 7},
    "robocasa": {"type": "eef_delta", "dim": 12},
    "robotwin": {"type": "joint_pos", "dim": 14},
    # gym-aloha bimanual sim: 14-dim absolute joint positions (7 per arm)
    "aloha_gym": {"type": "joint_pos", "dim": 14},
    # gym-pusht: 2-dim (x, y) end-effector position in pixel space [0,512]
    "gym_pusht": {"type": "eef_xy", "dim": 2},
    # Meta-World (Sawyer robot): 4-dim Sawyer eef-delta [dx, dy, dz, gripper].
    "metaworld": {"type": "eef_delta", "dim": 4},
}

# Derived: default action dim per simulator (for backward compat and convenience)
SIM_ACTION_DIM = {k: v["dim"] for k, v in SIM_EXPECTED_ACTION_SPACE.items()}

# Default rollout horizon per simulator (fallback if no suite-specific value)
SIM_MAX_STEPS = {
    "libero": 500,
    "libero_infinity": 300,
    "robocasa": 500,
    "robotwin": 300,
    "libero_pro": 500,
    # gym-aloha: 400 steps is the registered spec max_episode_steps
    "aloha_gym": 400,
    # gym-pusht: 300 steps (standard PushT horizon)
    "gym_pusht": 300,
    # Meta-World: 500 steps per episode (standard metaworld horizon)
    "metaworld": 500,
}

# Suite-specific rollout horizons.
# CANONICAL definition lives in roboeval/config.py:SUITE_MAX_STEPS.
# env_wrapper runs in the sim venv (Python 3.8) which cannot import roboeval,
# so we keep a local copy here. If you update one, update the other.
SUITE_MAX_STEPS = {
    # Standard LIBERO suites (matching lerobot TASK_SUITE_MAX_STEPS values)
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
    # LIBERO-INF suites
    "libero_infinity_spatial": 300,
    "libero_infinity_object": 300,
    "libero_infinity_goal": 300,
    "libero_infinity_10": 520,
    # LIBERO-PRO suites: 300 steps (conservative default)
    "libero_spatial_object": 300,
    "libero_goal_swap": 300,
    "libero_spatial_with_mug": 300,
    # RoboCasa and RoboTwin benchmark-scoped suite names
    "robocasa_kitchen": 500,
    "robotwin_aloha_agilex": 300,
    # gym-aloha task suites (gymnasium IDs used as suite names)
    "AlohaTransferCube-v0": 400,
    "AlohaInsertion-v0": 400,
}

VALID_SIMS = list(SIM_ACTION_DIM.keys())

# Default policy action space (used if /info call fails)
_DEFAULT_POLICY_ACTION_SPACE = {"type": "eef_delta", "dim": 7}


def _decode_b64_image(b64_str: str) -> Image.Image:
    """Decode a base64 string into a PIL Image."""
    raw = base64.b64decode(b64_str)
    return Image.open(BytesIO(raw))


def _image_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy RGB array."""
    return np.array(img.convert("RGB"))


def _apply_image_transform(img: Image.Image, transform: str) -> Image.Image:
    if not transform or transform == "none":
        return img
    # "applied_in_sim" means the sim backend already applied the flip;
    # env_wrapper must not apply a second flip.
    if transform == "applied_in_sim":
        return img
    arr = np.array(img)
    if transform == "flip_hw":
        arr = arr[::-1, ::-1]
    elif transform == "flip_h":
        arr = arr[::-1]
    else:
        raise ValueError(f"Unknown image_transform: {transform}")
    return Image.fromarray(arr)


class SimWrapper(BaseWorldStub):
    """
    World stub that wraps a robotic simulator environment via HTTP.

    The simulator runs as a separate FastAPI server (sim_worker.py),
    potentially on another machine or inside a different virtualenv.
    Communication happens through HTTP+JSON.
    """

    def __init__(
        self,
        sim_server_url: str,
        sim_name: str,
        task_name: str,
        camera_resolution: int = 256,
        suite: str = None,
        max_steps: int = None,
        vla_server_url: str = os.environ.get("VLA_URL", "http://localhost:5100"),
        headless: bool = False,
        delta_actions: bool = False,
        no_vlm: bool = False,
        sim_config: dict | None = None,
        chunk_size: int = None,
        action_ensemble: str = "newest",
        ema_alpha: float = 0.5,
    ):
        self.sim_server_url = sim_server_url.rstrip("/")
        self._sim_config = sim_config or {}
        self.vla_server_url = vla_server_url.rstrip("/")
        self._no_vlm = no_vlm
        self.sim_name = sim_name
        self.task_name = task_name
        self.camera_resolution = camera_resolution
        self.suite = suite
        self._headless = headless
        # Priority: explicit max_steps arg > suite-specific > sim-type default
        if max_steps:
            self.max_steps = max_steps
        elif suite and suite in SUITE_MAX_STEPS:
            self.max_steps = SUITE_MAX_STEPS[suite]
        else:
            self.max_steps = SIM_MAX_STEPS.get(sim_name, 300)
        self.action_dim = SIM_ACTION_DIM.get(sim_name, 7)

        # Attributes expected by BaseWorldStub
        self.finetuning_tasks = None
        self.execution_trace = None
        self._current_state: list = []  # latest robot proprioceptive state (list[float])
        # Role-keyed dict of camera images (e.g. "primary", "wrist", "secondary")
        self._current_images: dict = {}  # role -> PIL.Image.Image

        # Policy info from /info endpoint; sim expected action space from lookup table
        self._policy_info: dict = {}
        self._policy_action_space: dict = _DEFAULT_POLICY_ACTION_SPACE.copy()
        self._sim_expected_action_space: dict = SIM_EXPECTED_ACTION_SPACE.get(
            sim_name, _DEFAULT_POLICY_ACTION_SPACE
        )
        self._delta_actions = delta_actions
        self._image_transform = "none"
        self._current_state_dict = None

        # Action chunk buffer configuration.
        # _chunk_size_override=None means "use the model's action_chunk_size from /info".
        # _effective_chunk_size is resolved in _fetch_policy_info() once /info is available.
        self._chunk_size_override = chunk_size
        self._action_ensemble = action_ensemble
        self._ema_alpha = ema_alpha
        self._effective_chunk_size: int = chunk_size  # preliminary; updated after /info
        # The buffer itself is created after _effective_chunk_size is known (see
        # _fetch_policy_info).  We pre-create it here as a safety net so that
        # physical_reset() can always call _action_buffer.clear() unconditionally.
        self._action_buffer = ActionChunkBuffer(
            chunk_size=self._effective_chunk_size,
            action_ensemble=self._action_ensemble,
            ema_alpha=self._ema_alpha,
        )
        self._fetch_policy_info()

        # Initialize the environment via HTTP, passing the headless flag so the
        # sim worker can configure MUJOCO_GL and has_renderer accordingly.
        resp = self._post(
            "/init",
            {
                "sim": sim_name,
                "task": task_name,
                "camera_resolution": camera_resolution,
                "suite": suite,
                "headless": headless,
                "delta_actions": delta_actions,
                "sim_config": self._sim_config,
            },
        )
        if not resp.get("success"):
            raise RuntimeError(f"Failed to init sim env: {resp.get('error', 'unknown')}")

        # Fetch sim info and negotiate
        self._fetch_sim_info()
        self._negotiate_spaces()
        self._negotiate_obs()
        self._validate_specs()

        # Use the task description from the sim if available, else the raw name
        task_desc = resp.get("task_description", task_name)

        # Get initial observation
        obs_img = self._get_obs_image()

        # Initialize the base class with the first image.
        # When no_vlm=True, pass None to skip the VLM object-identification call
        # in BaseWorldStub.__init__ → refresh_objects().
        super().__init__(
            initial_image=None if no_vlm else obs_img,
            task_instruction=task_desc,
        )
        # Always store the actual image for the eval loop regardless of VLM mode
        self.current_image = obs_img

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, json_data: dict = None) -> dict:
        """Send a POST request to the sim server and return the JSON response."""
        url = f"{self.sim_server_url}{path}"
        r = requests.post(url, json=json_data or {}, timeout=120)
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"[SimWrapper] POST {path} returned HTTP {r.status_code}: {err}")
        return r.json()

    def _get(self, path: str) -> dict:
        """Send a GET request to the sim server and return the JSON response."""
        url = f"{self.sim_server_url}{path}"
        r = requests.get(url, timeout=120)
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"[SimWrapper] GET {path} returned HTTP {r.status_code}: {err}")
        return r.json()

    def _parse_images_from_resp(self, resp: dict) -> None:
        """Populate ``self._current_images`` from an HTTP response dict.

        Supports two response formats:

        * **New (role-keyed):** ``resp["images"]`` is a dict mapping role
          names (``"primary"``, ``"wrist"``, ``"secondary"``) to base64
          strings.  This is the canonical format produced by
          ``_build_images_response()`` in sim_worker.py.
        * **Legacy (flat keys):** ``resp["image"]``, ``resp["image2"]``,
          ``resp["image3"]`` — for backward compatibility with older sim
          workers.

        After this call, ``self._current_images`` contains PIL Images
        keyed by role with the configured ``_image_transform`` already
        applied.
        """
        if "images" in resp and isinstance(resp["images"], dict):
            self._current_images = {
                role: _apply_image_transform(_decode_b64_image(b64), self._image_transform)
                for role, b64 in resp["images"].items()
            }
        else:
            # Legacy flat-key fallback
            self._current_images = {}
            if "image" in resp:
                self._current_images["primary"] = _apply_image_transform(
                    _decode_b64_image(resp["image"]), self._image_transform
                )
            if "image2" in resp:
                self._current_images["wrist"] = _apply_image_transform(
                    _decode_b64_image(resp["image2"]), self._image_transform
                )
            if "image3" in resp:
                self._current_images["secondary"] = _apply_image_transform(
                    _decode_b64_image(resp["image3"]), self._image_transform
                )

    def _fetch_sim_info(self) -> None:
        try:
            r = requests.get(f"{self.sim_server_url}/info", timeout=5)
            self._sim_info = r.json()
            self._sim_action_space = self._sim_info.get("action_space", {})
        except Exception as e:
            logger.warning("Could not fetch /info from sim server: %s", e)
            self._sim_info = {}
            self._sim_action_space = self._sim_expected_action_space

    def _negotiate_spaces(self):
        p_type = self._policy_action_space.get("type")
        p_dim = self._policy_action_space.get("dim")

        s_type = self._sim_action_space.get("type")
        s_dim = self._sim_action_space.get("dim")
        s_accepted = self._sim_action_space.get("accepted_dims", [])

        if p_type != s_type:
            raise ValueError(f"Action space type mismatch: policy='{p_type}', sim='{s_type}'")

        if s_accepted:
            if p_dim not in s_accepted:
                raise ValueError(
                    f"Action space dim mismatch: policy={p_dim} not in sim accepted={s_accepted}"
                )
        else:
            if p_dim != s_dim:
                raise ValueError(f"Action space dim mismatch: policy={p_dim}, sim={s_dim}")

        self.action_dim = p_dim

    def _negotiate_obs(self):
        req = self._policy_info.get("obs_requirements")
        if not req:
            self._image_transform = "none"
            return

        req_cams = req.get("cameras", [])
        sim_cams = [c.get("role") for c in self._sim_info.get("obs_space", {}).get("cameras", [])]
        for c in req_cams:
            if c not in sim_cams:
                raise ValueError(
                    f"Camera mismatch: VLA requires '{c}' but sim only provides {sim_cams}"
                )

        req_state_format = req.get("state_format", "flat")
        req_state_dim = req.get("state_dim", 0)
        sim_state_dim = self._sim_info.get("obs_space", {}).get("state", {}).get("dim", 0)
        if req_state_dim > 0:
            if sim_state_dim == 0:
                raise ValueError("State mismatch: VLA requires state but sim provides none")
            if req_state_format != "structured" and req_state_dim != sim_state_dim:
                # Only compare scalars-to-scalars for flat state formats.
                # Structured state: GR00T has 5 "keys" != RoboCasa 9 "floats" — incomparable units.
                raise ValueError(f"State dim mismatch: VLA={req_state_dim}, sim={sim_state_dim}")

        self._image_transform = req.get("image_transform", "none")

    def _validate_specs(self) -> None:
        """Cross-validate VLA and sim ActionObsSpec contracts after both /info calls complete.

        Severity rules applied here:
        - HARD → raise SpecMismatchError (unless ROBOEVAL_STRICT_SPECS=0)
        - WARN → log warning
        - IGNORE / legacy (no specs) → log once at INFO, continue

        image_transform HARD conditions (checked first, independently of ActionObsSpec):
        - sim says "applied_in_sim" AND vla says "flip_hw" or "flip_h" → double flip
        - sim says "none" or absent AND vla says "flip_hw" or "flip_h" → missing backend transform

        Gate: set ROBOEVAL_STRICT_SPECS=0 to demote all HARD failures to WARN
        (escape hatch for legacy servers in CI).
        """
        strict = os.environ.get("ROBOEVAL_STRICT_SPECS", "1") not in ("0", "false", "False")

        # ── image_transform HARD check ────────────────────────────────────────
        sim_img_xfm = self._sim_info.get("obs_space", {}).get("image_transform", "none")
        vla_img_xfm = self._image_transform  # already stored by _negotiate_obs()

        hard_msgs: list[str] = []

        if sim_img_xfm == "applied_in_sim" and vla_img_xfm in ("flip_hw", "flip_h"):
            hard_msgs.append(
                f"image_transform conflict: sim='applied_in_sim' but VLA='{vla_img_xfm}' "
                f"— would cause a double flip; ensure the VLA advertises 'applied_in_sim' or 'none'"
            )
        elif sim_img_xfm not in ("applied_in_sim",) and vla_img_xfm in ("flip_hw", "flip_h"):
            hard_msgs.append(
                f"image_transform conflict: sim='{sim_img_xfm}' (not applying flip) but "
                f"VLA='{vla_img_xfm}' (expects flip); "
                f"ensure the sim applies the flip and advertises 'applied_in_sim'"
            )

        for msg in hard_msgs:
            if strict:
                raise SpecMismatchError(f"HARD spec mismatch: {msg}")
            else:
                logger.warning("HARD spec mismatch (strict=off, continuing): %s", msg)

        # ── ActionObsSpec check via check_specs() ──────────────────────────────────
        if not _SPECS_AVAILABLE:
            logger.debug("roboeval.specs not available; skipping ActionObsSpec validation")
            return

        # Deserialize spec dicts from both sides
        def _load_spec(raw: dict) -> dict[str, ActionObsSpec]:
            if not raw:
                return {}
            result = {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    try:
                        result[k] = ActionObsSpec.from_dict(v)
                    except Exception as exc:
                        logger.warning("Could not deserialize ActionObsSpec for '%s': %s", k, exc)
            return result

        vla_action_spec = _load_spec(self._policy_info.get("action_spec", {}))
        vla_obs_spec = _load_spec(self._policy_info.get("observation_spec", {}))
        sim_action_spec = _load_spec(self._sim_info.get("action_spec", {}))
        sim_obs_spec = _load_spec(self._sim_info.get("observation_spec", {}))

        # ── Legacy / one-sided spec declaration ─────────────────────────────
        # In strict mode, one-sided declarations are treated as HARD failures:
        # if the model declares a contract, the sim must declare a contract too
        # (and vice versa), or compatibility cannot be verified.
        sim_has_specs = bool(sim_action_spec or sim_obs_spec)
        vla_has_specs = bool(vla_action_spec or vla_obs_spec)
        if not sim_has_specs or not vla_has_specs:
            if not sim_has_specs and not vla_has_specs:
                # Both sides legacy — emit a WARN and skip (no contract to check).
                logger.warning(
                    "Spec WARN: no action_spec or observation_spec declared (legacy server); "
                    "skipping ActionObsSpec validation"
                )
                logger.warning("spec validation passed (legacy mode)")
                return
            # One-sided declarations are HARD failures under strict mode.
            if not sim_has_specs:
                msg = (
                    "sim declares no ActionObsSpec contracts but VLA does; cannot verify "
                    "state/action compatibility (e.g. RoboCasa 9-dim quat vs LIBERO 8-dim "
                    "axis-angle).  Add get_action_spec()/get_observation_spec() to the sim backend."
                )
            else:
                msg = (
                    "VLA declares no ActionObsSpec contracts but sim does; cannot verify "
                    "state/action compatibility.  Add get_action_spec()/get_observation_spec() "
                    "to the VLA policy server."
                )
            if strict:
                raise SpecMismatchError(f"HARD spec mismatch: {msg}")
            logger.warning("HARD spec mismatch (strict=off, continuing): %s", msg)
            return

        issues = check_specs(
            server_action=vla_action_spec,
            bench_action=sim_action_spec,
            server_obs=vla_obs_spec,
            bench_obs=sim_obs_spec,
        )

        has_hard = False
        hard_details: list[str] = []
        for severity, msg in issues:
            if severity == "HARD":
                has_hard = True
                hard_details.append(msg)
                if not strict:
                    logger.warning("HARD spec mismatch (strict=off, continuing): %s", msg)
                else:
                    logger.error("HARD spec mismatch: %s", msg)
            elif severity == "WARN":
                logger.warning("Spec WARN: %s", msg)

        if has_hard and strict:
            raise SpecMismatchError(
                "Spec validation failed (HARD):\n" + "\n".join(f"  • {m}" for m in hard_details)
            )

        if issues:
            logger.warning(
                "spec validation passed (with %d warning(s))",
                sum(1 for s, _ in issues if s == "WARN"),
            )
        else:
            logger.warning("spec validation passed")

    def _fetch_policy_info(self) -> None:
        """Fetch /info from the policy server and store action space metadata.

        Called at init. If the policy server is not yet reachable, falls back to
        the default eef_delta dim=7 space and tries again lazily on first act().

        Also resolves ``_effective_chunk_size``:
          * If the user provided ``chunk_size`` (``_chunk_size_override``), use it.
          * Otherwise fall back to the model's ``action_chunk_size`` from ``/info``
            (default 1 if the field is absent).

        After resolving the effective chunk size, re-creates ``_action_buffer``
        so it uses the correct trimming configuration.
        """
        try:
            r = requests.get(f"{self.vla_server_url}/info", timeout=5)
            self._policy_info = r.json()
            self._policy_action_space = self._policy_info.get(
                "action_space", _DEFAULT_POLICY_ACTION_SPACE
            )
            logger.info(
                "Policy info: model=%s, action_space=%s",
                self._policy_info.get("model_id", "?"),
                self._policy_action_space,
            )
        except Exception:
            logger.warning(
                "Could not reach policy server at %s/info. Using default action space %s.",
                self.vla_server_url,
                self._policy_action_space,
            )

        # Resolve effective chunk size: user override takes priority over model default.
        model_chunk_size = self._policy_info.get("action_chunk_size", 1)
        self._effective_chunk_size = self._chunk_size_override or model_chunk_size
        logger.info(
            "Action chunk buffer: effective_chunk_size=%d "
            "(override=%s, model_default=%d), ensemble=%s, ema_alpha=%.2f",
            self._effective_chunk_size,
            self._chunk_size_override,
            model_chunk_size,
            self._action_ensemble,
            self._ema_alpha,
        )
        # Rebuild the buffer now that we know the effective chunk size.
        self._action_buffer = ActionChunkBuffer(
            chunk_size=self._effective_chunk_size,
            action_ensemble=self._action_ensemble,
            ema_alpha=self._ema_alpha,
        )

    def _validate_action_chunk(self, chunk: list, expected_dim: int | None = None) -> None:
        """Reject NaN / inf / wrong-shape actions before they hit the sim.

        Without this check, NaN actions slip into MuJoCo and surface as a
        SIGABRT inside sim_worker — the orchestrator just sees ``nonzero_exit_-6``
        with no indication that the *VLA* was at fault.
        """
        if not isinstance(chunk, list) or not chunk:
            raise ValueError(
                f"VLA returned an empty or non-list action chunk; got: {type(chunk).__name__}"
            )
        for i, a in enumerate(chunk):
            arr = np.asarray(a, dtype=float)
            if expected_dim is not None and arr.size != expected_dim:
                raise ValueError(
                    f"VLA action chunk[{i}] has dim={arr.size} but the "
                    f"negotiated action_dim is {expected_dim}. Check the "
                    f"policy's get_action_spec() and that obs preprocessing "
                    f"matches its training data."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"VLA action chunk[{i}] contains NaN or inf: {arr.tolist()!r}. "
                    f"This usually means the policy is uninitialized, the input "
                    f"normalisation is wrong, or training diverged."
                )

    def _translate_action(self, action: list, from_space: dict, to_space: dict) -> list:
        """Translate an action from policy output space to simulator input space.

        Supported translations:
          - Same type + dim: identity pass-through
          - eef_delta dim=7 -> eef_delta dim=12: zero-pad dims 7-11 (base motion = 0)
          - eef_delta -> joint_pos: NOT supported (raises ValueError)

        Gripper dimension is clipped to [-1, 1] for eef_delta actions only
        (it's a normalized open/close in all supported sims). Real-space EEF
        deltas and joint positions are NOT clipped.
        """
        f_type, f_dim = from_space.get("type"), from_space.get("dim", len(action))
        t_type, t_dim = to_space.get("type"), to_space.get("dim", len(action))

        if f_type != t_type:
            raise ValueError(
                f"Action space type mismatch: policy outputs '{f_type}' but "
                f"sim expects '{t_type}'. No translation supported."
            )

        act = list(action)

        if f_dim == t_dim:
            # Identity
            pass
        elif f_type == "eef_delta" and f_dim == 7 and t_dim == 12:
            # Zero-pad dims 7-11 (RoboCasa mobile base motion = 0)
            act = act + [0.0] * (t_dim - f_dim)
        elif f_type == "eef_delta" and f_dim < t_dim:
            # Pad missing dims. When appending to reach LIBERO's 7-dim (gripper is
            # the 7th dim), use -1.0 (open) so the robot can approach objects.
            # Any other pad dims (e.g. hypothetical 5→7) use 0.0.
            pad = [0.0] * (t_dim - f_dim)
            if t_dim == 7:
                pad[-1] = -1.0  # gripper = open; SmolVLA has no gripper output
            act = act + pad
        else:
            raise ValueError(
                f"No translation rule for {f_type} dim={f_dim} -> {t_type} dim={t_dim}."
            )

        # Clip gripper dimension only (normalized [-1, 1] open/close signal)
        if f_type == "eef_delta" and len(act) >= 7:
            act[6] = max(-1.0, min(1.0, act[6]))

        return act

    def _get_obs_image(self) -> Image.Image:
        """Fetch the current camera observation from the sim server."""
        resp = self._get("/obs")
        if "error" in resp:
            raise RuntimeError(f"get_obs failed: {resp['error']}")
        if "state" in resp:
            self._current_state = resp["state"]
        if "state_dict" in resp:
            self._current_state_dict = resp["state_dict"]
        self._parse_images_from_resp(resp)
        return self._current_images["primary"]

    def _get_vla_actions(
        self,
        image: Image.Image,
        instruction: str,
        state: list = None,
    ) -> list:
        """
        Query the VLA service for an action chunk.

        Encodes the PIL Image as a base64 PNG and POSTs to the VLA /predict
        endpoint. Returns the full action chunk (list of action vectors).

        Additional camera images (wrist, secondary) are read from
        ``self._current_images`` which is populated by ``_parse_images_from_resp()``.

        Args:
            image: Current primary observation as a PIL Image.
            instruction: Natural language instruction for the VLA.
            state: Optional robot proprioceptive state (joint positions etc.).

        Returns:
            List of action vectors, e.g. [[0.1, -0.2, ...], [0.3, 0.1, ...], ...]
        """
        buf = BytesIO()
        image.save(buf, format="PNG")

        images = {"primary": base64.b64encode(buf.getvalue()).decode("utf-8")}
        # Include any additional camera images from self._current_images
        for role in ("wrist", "secondary"):
            extra_img = self._current_images.get(role)
            if extra_img is not None:
                buf_extra = BytesIO()
                extra_img.save(buf_extra, format="PNG")
                images[role] = base64.b64encode(buf_extra.getvalue()).decode("utf-8")

        state_dict = {}
        if state is not None:
            state_dict["flat"] = state

        if getattr(self, "_current_state_dict", None) is not None:
            state_dict["structured"] = self._current_state_dict

        obs = {
            "instruction": instruction,
            "images": images,
            "state": state_dict,
        }

        url = f"{self.vla_server_url}/predict"
        r = requests.post(url, json={"obs": obs}, timeout=300)
        if r.status_code != 200:
            # Proxy 502/503 responses may not be valid JSON (e.g. nginx HTML)
            try:
                error_msg = r.json().get("error", f"HTTP {r.status_code}")
            except (ValueError, AttributeError):
                error_msg = r.text if r.text else f"HTTP {r.status_code}"
            raise ConnectionError(f"[SimWrapper] VLA proxy returned {r.status_code}: {error_msg}")
        resp = r.json()
        return resp["actions"]

    # ------------------------------------------------------------------
    # BaseWorldStub interface
    # ------------------------------------------------------------------

    def act(self, command: str):
        """
        Execute a natural-language subtask command in the simulator.

        Queries the VLA HTTP service at `vla_server_url` to translate the
        natural-language `command` into low-level action sequences. The VLA
        returns an action chunk (multiple actions at once); all actions in the
        chunk are executed before querying the VLA again with the updated
        observation. This continues until `max_steps` is reached or the
        environment signals done.

        Raises ConnectionError if the VLA service is not reachable, either at
        the start or during the rollout. This prevents silent corruption from
        random-action fallback.

        Args:
            command: Natural-language instruction describing the subtask.
        """
        logger.info("Executing subtask: '%s'", command)

        # Check VLA reachability once at the start; refresh policy info if not yet loaded
        try:
            requests.get(f"{self.vla_server_url}/health", timeout=5)
            if not self._policy_info:
                self._fetch_policy_info()
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"[SimWrapper] VLA server unreachable at {self.vla_server_url}. "
                f"Cannot execute subtask '{command}'. Aborting episode."
            ) from exc

        logger.info(
            "Using VLA service at %s (policy=%s, sim=%s, max_steps=%d)",
            self.vla_server_url,
            self._policy_action_space,
            self._sim_expected_action_space,
            self.max_steps,
        )

        frames = []

        # Clear the action buffer at the start of every subtask so that stale
        # actions from a previous subtask (different command) are not replayed.
        self._action_buffer.clear()

        # Use the reset observation directly so the first policy call sees the
        # exact scene returned by /reset instead of an extra round-trip /obs.
        current_obs = (
            self.current_image.copy() if self.current_image is not None else self._get_obs_image()
        )
        frames.append(_image_to_numpy(current_obs))

        step = 0
        done = False

        while step < self.max_steps and not done:
            # --- Refill buffer when empty ---
            if self._action_buffer.empty:
                try:
                    chunk = self._get_vla_actions(
                        current_obs,
                        command,
                        state=self._current_state or None,
                    )
                    self._validate_action_chunk(chunk, expected_dim=self.action_dim)
                except requests.exceptions.ConnectionError as exc:
                    raise ConnectionError(
                        f"[SimWrapper] VLA server became unreachable at "
                        f"{self.vla_server_url} during rollout (step {step}/{self.max_steps}). "
                        f"Aborting episode."
                    ) from exc
                self._action_buffer.push(chunk)

            # --- Pop one action from the buffer ---
            raw_action = self._action_buffer.pop()
            if raw_action is None:
                # Buffer is empty after push — shouldn't happen, but guard.
                logger.warning("Action buffer empty immediately after push; breaking.")
                break

            # Translate from policy action space to sim action space.
            # Clipping is policy-specific; only gripper dim is clipped (inside
            # _translate_action). Real-space EEF deltas must NOT be clipped here.
            action = self._translate_action(
                raw_action,
                self._policy_action_space,
                self._sim_expected_action_space,
            )

            resp = self._post("/step", {"action": action})
            if "error" in resp:
                logger.error("Step failed: %s", resp["error"])
                done = True
                break

            # Decode the observation frame and update proprioceptive state
            self._parse_images_from_resp(resp)
            current_obs = self._current_images["primary"]
            if "state" in resp:
                self._current_state = resp["state"]
            if "state_dict" in resp:
                self._current_state_dict = resp["state_dict"]
            frames.append(_image_to_numpy(current_obs))
            step += 1

            # Check for task success (early termination)
            done = resp.get("done", False)
            if done:
                logger.info("Environment signaled done at step %d", step)
                break

        # Update current image to final observation
        if frames:
            self.current_image = Image.fromarray(frames[-1])

        # Store frames as LITEN expects: list of (command, frames) tuples
        self.subtask_frame_tuples.append((command, frames))
        self.eval_len += len(frames)

        logger.info("Subtask done. Recorded %d frames.", len(frames))

    def _ensure_backend_initialized(self, force: bool = False) -> None:
        """Re-initialize the sim backend by calling /init again.

        Used for crash recovery when the backend signals it is not initialized
        (e.g. after a sim worker restart or OOM-induced backend teardown).
        """
        resp = self._post(
            "/init",
            {
                "sim": self.sim_name,
                "task": self.task_name,
                "camera_resolution": self.camera_resolution,
                "suite": self.suite,
                "headless": self._headless,
                "delta_actions": self._delta_actions,
                "sim_config": self._sim_config,
            },
        )
        if not resp.get("success"):
            raise RuntimeError(f"Failed to re-init sim env: {resp.get('error', 'unknown')}")

    def physical_reset(self, episode_index: int = None):
        """Reset the simulator environment via HTTP.

        Also calls POST /reset on the VLA policy server so it can clear any
        per-episode internal state (e.g. action queues). Native lerobot-eval
        always calls policy.reset() at episode start; we mirror that here.

        Args:
            episode_index: Which episode init_state to load. Forwarded to
                LiberoBackend.reset() so set_init_state() uses the correct
                configuration for the current episode (not always index 0).
                RoboCasa and RoboTwin ignore this field.
        """
        # Clear the local action buffer so no stale actions from the previous
        # episode bleed into the new one.
        self._action_buffer.clear()

        # Reset the VLA policy's internal state (best-effort; not all servers
        # implement /reset and it's a no-op when n_action_steps=1 anyway).
        try:
            requests.post(f"{self.vla_server_url}/reset", timeout=5)
        except requests.exceptions.ConnectionError:
            pass  # non-fatal — server may not expose /reset
        except Exception as e:
            logger.debug("VLA /reset call failed (non-fatal): %s", e)

        payload = {}
        if episode_index is not None:
            payload["episode_index"] = episode_index
        try:
            resp = self._post("/reset", payload)
        except (requests.exceptions.ConnectionError, RuntimeError) as e:
            raise ConnectionError(
                f"[SimWrapper] Failed to reach sim server at {self.sim_server_url}/reset: {e}"
            ) from e
        if "error" in resp:
            err_msg = resp.get("error", "")
            if "NoneType" in err_msg or "not initialized" in err_msg:
                logger.warning("[SimWrapper] Sim backend lost — re-initializing and retrying reset")
                self._ensure_backend_initialized(force=True)
                resp = self._post("/reset", payload)
            if "error" in resp:
                raise RuntimeError(f"reset failed: {resp.get('error', 'unknown')}")
        if "image" in resp or "images" in resp:
            self._parse_images_from_resp(resp)
            self.current_image = self._current_images.get("primary")
            if "state" in resp:
                self._current_state = resp["state"]
            if "state_dict" in resp:
                self._current_state_dict = resp["state_dict"]
        else:
            # Backend doesn't return an image on reset (e.g. RoboTwin) — fetch separately.
            self.current_image = self._get_obs_image()

    def get_obs(self) -> Image.Image:
        """Return the current camera observation as a PIL Image."""
        self.current_image = self._get_obs_image()
        return self.current_image

    def check_success(self) -> bool:
        """Query the simulator for task success."""
        try:
            resp = self._get("/success")
        except (requests.exceptions.ConnectionError, RuntimeError) as e:
            logger.warning("check_success network error (returning False): %s", e)
            return False
        if "error" in resp:
            logger.error("check_success failed: %s", resp["error"])
            return False
        return resp.get("success", False)

    def close(self):
        """Clean up (does NOT shut down the sim server so it can be reused)."""
        pass

    def shutdown_server(self):
        """Shut down the simulator server process."""
        try:
            self._post("/close")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Override reset to work with the sim
    # ------------------------------------------------------------------

    def reset(self, new_task: str = None, keep_frames: bool = False, **kwargs):
        """
        Reset the world state. Overrides BaseWorldStub.reset() to handle
        the simulator and avoid calling refresh_objects without an image argument.
        """
        if not keep_frames:
            self.subtask_frame_tuples = []
        if new_task is not None:
            self.task_instruction = new_task
        self.physical_reset()
        if not self._no_vlm:
            self.refresh_objects(self.current_image)
        return self.current_image
