"""Tests for the VLA / sim spec handshake.

Verifies that ``SimWrapper._validate_specs()`` correctly:
- Raises ``SpecMismatchError`` on HARD failures (image_transform conflict, dim
  mismatch, format mismatch).
- Logs a warning on WARN-only issues without raising.
- Accepts matching specs without error ("spec validation passed").
- Degrades gracefully when no spec keys are present (legacy server).

All tests use ``unittest.mock`` / ``monkeypatch`` — no real sims or VLAs.
"""

from __future__ import annotations

import logging
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sims.env_wrapper import SimWrapper, SpecMismatchError, _apply_image_transform
from robo_eval.specs import (
    ActionObsSpec,
    POSITION_DELTA,
    GRIPPER_CLOSE_NEG,
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _encode_image(img: Image.Image) -> str:
    import base64
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


_DUMMY_IMG_B64 = _encode_image(Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)))


def _pi05_action_spec() -> dict:
    """Canonical pi05 action spec dict (ActionObsSpec.to_dict() format)."""
    return {
        "position": {"name": "position", "dims": 3, "format": "delta_xyz", "range": [-1, 1]},
        "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle", "range": [-3.15, 3.15]},
        "gripper": {"name": "gripper", "dims": 1, "format": "binary_close_negative", "range": [-1, 1]},
    }


def _pi05_obs_spec() -> dict:
    """Canonical pi05 observation spec dict."""
    return {
        "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
        "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
        "state": {"name": "state", "dims": 8, "format": "libero_eef_pos3_aa3_grip2"},
        "instruction": {"name": "language", "dims": 0, "format": "language"},
    }


def _libero_action_spec() -> dict:
    """Canonical LIBERO sim action spec dict (what sim consumes, accepts delta_axisangle)."""
    return {
        "position": {"name": "position", "dims": 3, "format": "delta_xyz", "range": [-1, 1],
                     "accepts": ["delta_xyz", "delta_axisangle"]},
        "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle",
                     "range": [-3.15, 3.15],
                     "accepts": ["delta_axisangle", "axis_angle"]},
        "gripper": {"name": "gripper", "dims": 1, "format": "binary_close_negative",
                    "range": [-1, 1],
                    "accepts": ["binary_close_negative"]},
    }


def _libero_obs_spec() -> dict:
    """Canonical LIBERO sim observation spec dict (what sim provides)."""
    return {
        "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
        "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
        "state": {"name": "state", "dims": 8, "format": "libero_eef_pos3_aa3_grip2"},
        "instruction": {"name": "language", "dims": 0, "format": "language"},
    }


def _make_vla_info(
    image_transform="applied_in_sim",
    action_spec=None,
    observation_spec=None,
    action_dim=7,
    state_dim=8,
    cameras=None,
):
    info = {
        "name": "test-vla",
        "model_id": "test/model",
        "action_space": {"type": "eef_delta", "dim": action_dim},
        "state_dim": state_dim,
        "action_chunk_size": 1,
        "obs_requirements": {
            "cameras": cameras or ["primary", "wrist"],
            "state_dim": state_dim,
            "image_transform": image_transform,
        },
    }
    if action_spec is not None:
        info["action_spec"] = action_spec
    if observation_spec is not None:
        info["observation_spec"] = observation_spec
    return info


def _make_sim_info(
    image_transform="applied_in_sim",
    action_spec=None,
    observation_spec=None,
    action_dim=7,
    state_dim=8,
    cameras=None,
):
    cam_list = []
    for role in (cameras or ["primary", "wrist"]):
        cam_list.append({"key": f"{role}_image", "resolution": [256, 256], "role": role})
    info = {
        "sim": "test-sim",
        "action_space": {"type": "eef_delta", "dim": action_dim, "accepted_dims": [action_dim]},
        "obs_space": {
            "cameras": cam_list,
            "state": {"dim": state_dim, "format": "libero_eef_pos3_aa3_grip2"},
            "image_transform": image_transform,
        },
        "max_steps": 280,
    }
    if action_spec is not None:
        info["action_spec"] = action_spec
    if observation_spec is not None:
        info["observation_spec"] = observation_spec
    return info


def _make_wrapper(vla_info: dict, sim_info: dict, monkeypatch, strict: str = "1") -> SimWrapper:
    """Construct a SimWrapper with fully mocked HTTP calls.

    Parameters
    ----------
    vla_info:
        Mock VLA /info response.
    sim_info:
        Mock sim /info response.
    monkeypatch:
        pytest monkeypatch fixture.
    strict:
        Value for ROBO_EVAL_STRICT_SPECS env var ("1" = strict, "0" = lenient).
    """
    from world_stubs import BaseWorldStub

    monkeypatch.setenv("ROBO_EVAL_STRICT_SPECS", strict)

    def fake_base_init(self, initial_image=None, task_instruction=None):
        self.subtask_frame_tuples = []
        self.eval_len = 0
        self.current_image = initial_image
        self.task_instruction = task_instruction
        self.manipulable_object_uids = []
        self.execution_trace = None

    monkeypatch.setattr(BaseWorldStub, "__init__", fake_base_init)

    def fake_fetch_policy_info(self_):
        self_._policy_info = vla_info
        self_._policy_action_space = vla_info.get("action_space", {})
        model_chunk_size = vla_info.get("action_chunk_size", 1)
        self_._effective_chunk_size = self_._chunk_size_override or model_chunk_size
        from sims.env_wrapper import ActionChunkBuffer
        self_._action_buffer = ActionChunkBuffer(
            chunk_size=self_._effective_chunk_size,
            action_ensemble=self_._action_ensemble,
            ema_alpha=self_._ema_alpha,
        )

    monkeypatch.setattr(SimWrapper, "_fetch_policy_info", fake_fetch_policy_info)

    def fake_post(self_, path, json_data=None):
        if path == "/init":
            return {"success": True, "task_description": "test task"}
        if path == "/reset":
            return {"image": _DUMMY_IMG_B64}
        return {}

    monkeypatch.setattr(SimWrapper, "_post", fake_post)

    class _FakeGetResp:
        ok = True
        status_code = 200
        def __init__(self, payload):
            self._payload = payload
        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if "/info" in url:
            return _FakeGetResp(sim_info)
        if "/obs" in url:
            return _FakeGetResp({"image": _DUMMY_IMG_B64})
        return _FakeGetResp({"ready": True})

    import requests as req_mod
    monkeypatch.setattr(req_mod, "get", fake_get)

    return SimWrapper(
        sim_server_url="http://fake-sim:5001",
        sim_name="libero",
        task_name="0",
        no_vlm=True,
    )


# ===========================================================================
# 1. Positive tests — matching specs → no error
# ===========================================================================


class TestPositiveHandshake:
    """Matching VLA and sim specs should not raise."""

    def test_matching_specs_no_error(self, monkeypatch):
        """pi05 action/obs spec matches LIBERO sim spec → no SpecMismatchError."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7

    def test_matching_image_transform_applied_in_sim(self, monkeypatch):
        """Both sides say 'applied_in_sim' → no error."""
        vla = _make_vla_info(image_transform="applied_in_sim")
        sim = _make_sim_info(image_transform="applied_in_sim")
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._image_transform == "applied_in_sim"

    def test_vla_none_sim_applied_in_sim_ok(self, monkeypatch):
        """VLA says 'none', sim says 'applied_in_sim' → OK (sim handles flip, VLA doesn't care)."""
        vla = _make_vla_info(image_transform="none")
        sim = _make_sim_info(image_transform="applied_in_sim")
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._image_transform == "none"

    def test_both_none_transform_ok(self, monkeypatch):
        """Both sides say 'none' → OK (no flip involved)."""
        vla = _make_vla_info(image_transform="none")
        sim = _make_sim_info(image_transform="none")
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._image_transform == "none"

    def test_spec_validation_passed_logged(self, monkeypatch, caplog):
        """On clean match, 'spec validation passed' appears in the log."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with caplog.at_level(logging.INFO, logger="sims.env_wrapper"):
            _make_wrapper(vla, sim, monkeypatch)
        assert any("spec validation passed" in r.message for r in caplog.records)


# ===========================================================================
# 2. HARD failure — image_transform conflicts
# ===========================================================================


class TestHardImageTransformConflict:
    """image_transform conflicts → SpecMismatchError (HARD)."""

    def test_vla_flip_hw_sim_applied_in_sim_raises(self, monkeypatch):
        """VLA 'flip_hw' + sim 'applied_in_sim' → SpecMismatchError (double flip)."""
        vla = _make_vla_info(image_transform="flip_hw")
        sim = _make_sim_info(image_transform="applied_in_sim")
        with pytest.raises(SpecMismatchError, match="double flip"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_vla_flip_h_sim_applied_in_sim_raises(self, monkeypatch):
        """VLA 'flip_h' + sim 'applied_in_sim' → SpecMismatchError."""
        vla = _make_vla_info(image_transform="flip_h")
        sim = _make_sim_info(image_transform="applied_in_sim")
        with pytest.raises(SpecMismatchError):
            _make_wrapper(vla, sim, monkeypatch)

    def test_vla_flip_hw_sim_none_raises(self, monkeypatch):
        """VLA 'flip_hw' + sim 'none' → SpecMismatchError."""
        vla = _make_vla_info(image_transform="flip_hw")
        sim = _make_sim_info(image_transform="none")
        with pytest.raises(SpecMismatchError, match="image_transform conflict"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_vla_flip_h_sim_none_raises(self, monkeypatch):
        """VLA 'flip_h' + sim 'none' → SpecMismatchError."""
        vla = _make_vla_info(image_transform="flip_h")
        sim = _make_sim_info(image_transform="none")
        with pytest.raises(SpecMismatchError):
            _make_wrapper(vla, sim, monkeypatch)

    def test_strict_off_no_raise(self, monkeypatch):
        """ROBO_EVAL_STRICT_SPECS=0 demotes HARD to warning — no exception raised."""
        vla = _make_vla_info(image_transform="flip_hw")
        sim = _make_sim_info(image_transform="applied_in_sim")
        # Should not raise with strict=0
        w = _make_wrapper(vla, sim, monkeypatch, strict="0")
        # Wrapper still stores the VLA's transform value
        assert w._image_transform == "flip_hw"


# ===========================================================================
# 3. HARD failure — ActionObsSpec action dim mismatch
# ===========================================================================


class TestHardDimMismatch:
    """ActionObsSpec-level HARD failures (action dim mismatch etc.) → SpecMismatchError."""

    def test_action_dim_mismatch_raises(self, monkeypatch):
        """VLA declares 3-dim position, sim expects 6-dim → SpecMismatchError."""
        bad_vla_action = dict(_pi05_action_spec())
        bad_vla_action["position"] = {
            "name": "position", "dims": 6, "format": "delta_xyz", "range": [-1, 1]
        }
        vla = _make_vla_info(
            action_spec=bad_vla_action,
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_action_format_mismatch_raises(self, monkeypatch):
        """VLA gripper=binary_close_positive, sim expects binary_close_negative → HARD."""
        bad_vla_action = dict(_pi05_action_spec())
        bad_vla_action["gripper"] = {
            "name": "gripper", "dims": 1, "format": "binary_close_positive", "range": [-1, 1]
        }
        vla = _make_vla_info(
            action_spec=bad_vla_action,
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_missing_required_action_key_raises(self, monkeypatch):
        """VLA omits 'gripper' key, sim expects it → HARD."""
        vla_action = {k: v for k, v in _pi05_action_spec().items() if k != "gripper"}
        vla = _make_vla_info(
            action_spec=vla_action,
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_strict_off_dim_mismatch_no_raise(self, monkeypatch):
        """ROBO_EVAL_STRICT_SPECS=0 — action dim mismatch doesn't raise."""
        bad_vla_action = dict(_pi05_action_spec())
        bad_vla_action["position"] = {
            "name": "position", "dims": 6, "format": "delta_xyz", "range": [-1, 1]
        }
        vla = _make_vla_info(
            action_spec=bad_vla_action,
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        # strict=0 → no raise
        _make_wrapper(vla, sim, monkeypatch, strict="0")


# ===========================================================================
# 4. WARN-only — range mismatch → log, no raise
# ===========================================================================


class TestWarnOnlyMismatches:
    """WARN-severity issues log a warning but do not raise."""

    def test_range_mismatch_no_raise(self, monkeypatch, caplog):
        """Mismatched range is WARN: logged but no exception raised."""
        wide_range_vla_action = dict(_pi05_action_spec())
        wide_range_vla_action["position"] = {
            "name": "position", "dims": 3, "format": "delta_xyz", "range": [-2.0, 2.0]
        }
        vla = _make_vla_info(
            action_spec=wide_range_vla_action,
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with caplog.at_level(logging.WARNING, logger="sims.env_wrapper"):
            w = _make_wrapper(vla, sim, monkeypatch)
        # No exception raised
        assert w.action_dim == 7
        # Warning must appear in log
        warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("range" in m.lower() or "WARN" in m for m in warn_msgs), \
            f"Expected range WARN in log; got: {warn_msgs}"


# ===========================================================================
# 5. Legacy fallback — no spec keys → WARN once, no raise
# ===========================================================================


class TestLegacyFallback:
    """Legacy servers with no action_spec/observation_spec → WARN, no raise."""

    def test_no_spec_keys_no_raise(self, monkeypatch, caplog):
        """Neither side declares specs → legacy WARN, no exception."""
        vla = _make_vla_info()   # no action_spec / observation_spec
        sim = _make_sim_info()   # no action_spec / observation_spec
        with caplog.at_level(logging.WARNING, logger="sims.env_wrapper"):
            w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7
        warn_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("legacy" in m.lower() or "no action_spec" in m.lower() for m in warn_msgs), \
            f"Expected legacy WARN; got: {warn_msgs}"

    def test_only_vla_has_spec_raises_in_strict(self, monkeypatch):
        """VLA declares ActionObsSpec but sim does not → HARD failure (strict mode).

        Strict mode requires both sides of the handshake to declare compatible
        specs before the orchestrator starts.
        """
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info()  # no ActionObsSpec
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_only_sim_has_spec_raises_in_strict(self, monkeypatch):
        """Sim declares ActionObsSpec but VLA does not → HARD failure (strict mode)."""
        vla = _make_vla_info()  # no ActionObsSpec
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_one_sided_spec_strict_off_no_raise(self, monkeypatch):
        """ROBO_EVAL_STRICT_SPECS=0 demotes one-sided spec absence to WARN."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info()  # no ActionObsSpec
        w = _make_wrapper(vla, sim, monkeypatch, strict="0")
        assert w.action_dim == 7


# ===========================================================================
# 6. SpecMismatchError class properties
# ===========================================================================


class TestSpecMismatchErrorClass:
    """Verify SpecMismatchError is a RuntimeError subclass with expected message."""

    def test_is_runtime_error(self):
        err = SpecMismatchError("test message")
        assert isinstance(err, RuntimeError)

    def test_message_preserved(self):
        msg = "gripper convention clash"
        err = SpecMismatchError(msg)
        assert msg in str(err)
