"""
Tests for auto-discovery: action space negotiation, observation negotiation,
image transforms, and compatibility matrix.

Uses unittest.mock to mock HTTP responses — no real sims or VLAs needed.
"""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sims.env_wrapper import (
    SimWrapper,
    _apply_image_transform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vla_info(
    action_type="eef_delta",
    action_dim=7,
    state_dim=8,
    cameras=None,
    image_transform="none",
):
    """Build a mock VLA /info response with obs_requirements."""
    info = {
        "name": "test-vla",
        "model_id": "test/test-model",
        "action_space": {"type": action_type, "dim": action_dim},
        "state_dim": state_dim,
        "action_chunk_size": 1,
        "obs_requirements": {
            "cameras": cameras or ["primary"],
            "state_dim": state_dim,
            "image_transform": image_transform,
        },
    }
    return info


def _make_sim_info(
    action_type="eef_delta",
    action_dim=7,
    state_dim=8,
    cameras=None,
    accepted_dims=None,
):
    """Build a mock sim /info response.

    Args:
        accepted_dims: If provided, added to action_space dict.  None means
            the field is omitted (tests the fallback-to-strict-identity path).
    """
    cam_list = []
    for role in (cameras or ["primary"]):
        cam_list.append(
            {"key": f"{role}_image", "resolution": [256, 256], "role": role}
        )
    action_space = {"type": action_type, "dim": action_dim}
    if accepted_dims is not None:
        action_space["accepted_dims"] = accepted_dims
    return {
        "sim": "test-sim",
        "action_space": action_space,
        "obs_space": {
            "cameras": cam_list,
            "state": {"dim": state_dim, "format": "test"},
        },
    }


def _encode_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_wrapper(
    vla_info,
    sim_info,
    monkeypatch,
):
    """Construct a SimWrapper with fully mocked HTTP calls.

    Returns the wrapper instance.  Patches:
      - BaseWorldStub.__init__  (no VLM calls)
      - SimWrapper._fetch_policy_info  (uses vla_info)
      - SimWrapper._post  (fakes /init)
      - requests.get  (fakes sim /info and VLA /health)
    """
    from world_stubs import BaseWorldStub

    # Dummy base init
    def fake_base_init(self, initial_image=None, task_instruction=None):
        self.subtask_frame_tuples = []
        self.eval_len = 0
        self.current_image = initial_image
        self.task_instruction = task_instruction
        self.manipulable_object_uids = []
        self.execution_trace = None

    monkeypatch.setattr(BaseWorldStub, "__init__", fake_base_init)

    # VLA /info
    def fake_fetch_policy_info(self_):
        self_._policy_info = vla_info
        self_._policy_action_space = vla_info.get("action_space", {})

    monkeypatch.setattr(SimWrapper, "_fetch_policy_info", fake_fetch_policy_info)

    # /init POST
    dummy_img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    def fake_post(self_, path, json_data=None):
        if path == "/init":
            return {"success": True, "task_description": "test task"}
        if path == "/reset":
            return {"image": _encode_image(dummy_img)}
        return {}

    monkeypatch.setattr(SimWrapper, "_post", fake_post)

    # GET dispatcher: sim /info and vla /health
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
            return _FakeGetResp({"image": _encode_image(dummy_img)})
        # Default (health etc.)
        return _FakeGetResp({"ready": True})

    import requests as req_mod

    monkeypatch.setattr(req_mod, "get", fake_get)

    wrapper = SimWrapper(
        sim_server_url="http://fake-sim:5001",
        sim_name="libero",
        task_name="0",
        no_vlm=True,
    )
    return wrapper


# ======================================================================
# 1. TestSpaceNegotiation
# ======================================================================


class TestSpaceNegotiation:
    """Verify _negotiate_spaces() accepts identity matches and rejects mismatches."""

    def test_identity_match_succeeds(self, monkeypatch):
        """eef_delta/7 + eef_delta/7 → no error."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim = _make_sim_info(action_type="eef_delta", action_dim=7)
        w = _make_wrapper(vla, sim, monkeypatch)
        # If we got here, negotiation passed
        assert w._policy_action_space["type"] == "eef_delta"
        assert w._sim_action_space["type"] == "eef_delta"

    def test_type_mismatch_raises(self, monkeypatch):
        """eef_delta/7 + joint_pos/14 → ValueError."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim = _make_sim_info(action_type="joint_pos", action_dim=14)
        with pytest.raises(ValueError, match="Action space type mismatch"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_dim_mismatch_raises(self, monkeypatch):
        """eef_delta/7 + eef_delta/12 → ValueError."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim = _make_sim_info(action_type="eef_delta", action_dim=12)
        with pytest.raises(ValueError, match="Action space dim mismatch"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_identity_joint_pos(self, monkeypatch):
        """joint_pos/14 + joint_pos/14 → no error."""
        vla = _make_vla_info(action_type="joint_pos", action_dim=14, state_dim=14)
        sim = _make_sim_info(action_type="joint_pos", action_dim=14, state_dim=14)
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._policy_action_space == {"type": "joint_pos", "dim": 14}
        assert w._sim_action_space == {"type": "joint_pos", "dim": 14}


# ======================================================================
# 2. TestObsNegotiation
# ======================================================================


class TestObsNegotiation:
    """Verify _negotiate_obs() checks cameras and state dims."""

    def test_matching_cameras_succeeds(self, monkeypatch):
        """VLA wants [primary], sim has [primary] → OK."""
        vla = _make_vla_info(cameras=["primary"])
        sim = _make_sim_info(cameras=["primary"])
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._image_transform == "none"

    def test_missing_camera_raises(self, monkeypatch):
        """VLA wants [primary, wrist], sim has [primary] → ValueError."""
        vla = _make_vla_info(cameras=["primary", "wrist"])
        sim = _make_sim_info(cameras=["primary"])
        with pytest.raises(ValueError, match="Camera mismatch"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_state_dim_match_succeeds(self, monkeypatch):
        """VLA state_dim=8, sim state_dim=8 → OK."""
        vla = _make_vla_info(state_dim=8)
        sim = _make_sim_info(state_dim=8)
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7

    def test_state_dim_mismatch_raises(self, monkeypatch):
        """VLA state_dim=8, sim state_dim=14 → ValueError."""
        vla = _make_vla_info(state_dim=8)
        sim = _make_sim_info(state_dim=14)
        with pytest.raises(ValueError, match="State dim mismatch"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_state_dim_zero_always_ok(self, monkeypatch):
        """VLA state_dim=0 (ignores state) → OK regardless of sim state_dim."""
        vla = _make_vla_info(state_dim=0)
        # obs_requirements.state_dim = 0 → VLA doesn't need state
        sim = _make_sim_info(state_dim=14)
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._policy_info["state_dim"] == 0


# ======================================================================
# 3. TestImageTransform
# ======================================================================


class TestImageTransform:
    """Test _apply_image_transform() directly (no HTTP mocks needed)."""

    def _make_gradient_image(self, h=8, w=8):
        """Create a small gradient image where each pixel value is unique."""
        arr = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        return Image.fromarray(arr)

    def test_flip_hw_applied(self):
        """'flip_hw' flips both H and W axes (180° rotation)."""
        img = self._make_gradient_image()
        original = np.array(img)
        result = _apply_image_transform(img, "flip_hw")
        result_arr = np.array(result)
        expected = original[::-1, ::-1]
        np.testing.assert_array_equal(result_arr, expected)

    def test_no_transform(self):
        """'none' returns image unchanged."""
        img = self._make_gradient_image()
        original = np.array(img).copy()
        result = _apply_image_transform(img, "none")
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr, original)

    def test_default_no_transform(self):
        """Empty string / falsy value → image unchanged (same as 'none')."""
        img = self._make_gradient_image()
        original = np.array(img).copy()
        result = _apply_image_transform(img, "")
        result_arr = np.array(result)
        np.testing.assert_array_equal(result_arr, original)

    def test_unknown_transform_raises(self):
        """Unknown transform name → ValueError."""
        img = self._make_gradient_image()
        with pytest.raises(ValueError, match="Unknown image_transform"):
            _apply_image_transform(img, "rotate_90")

    def test_flip_hw_is_involution(self):
        """Applying flip_hw twice returns the original image."""
        img = self._make_gradient_image()
        original = np.array(img).copy()
        flipped = _apply_image_transform(img, "flip_hw")
        restored = _apply_image_transform(flipped, "flip_hw")
        np.testing.assert_array_equal(np.array(restored), original)

    def test_image_transform_applied_during_negotiation(self, monkeypatch):
        """VLA declaring 'flip_hw' → wrapper stores the transform."""
        vla = _make_vla_info(image_transform="flip_hw")
        sim = _make_sim_info()
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._image_transform == "flip_hw"


# ======================================================================
# 4. TestCompatibilityMatrix
# ======================================================================


class TestCompatibilityMatrix:
    """Integration tests for realistic VLA ↔ sim combinations."""

    @staticmethod
    def _pi05_info():
        """pi0.5 /info: eef_delta/7, state_dim=8, cameras=[primary, wrist]."""
        return _make_vla_info(
            action_type="eef_delta",
            action_dim=7,
            state_dim=8,
            cameras=["primary", "wrist"],
            image_transform="flip_hw",
        )

    @staticmethod
    def _libero_info():
        """LIBERO sim /info: eef_delta/7, cameras=[primary, wrist], state_dim=8."""
        return _make_sim_info(
            action_type="eef_delta",
            action_dim=7,
            state_dim=8,
            cameras=["primary", "wrist"],
        )

    @staticmethod
    def _robocasa_info():
        """RoboCasa sim /info (PandaOmron default): eef_delta/12, two cameras, state_dim=9.

        Reports accepted_dims=[7,12] to mirror the real RoboCasaBackend
        which pads 7-dim actions to 12-dim internally.
        """
        return _make_sim_info(
            action_type="eef_delta",
            action_dim=12,
            state_dim=9,
            cameras=["primary", "wrist"],
            accepted_dims=[7, 12],
        )

    @staticmethod
    def _internvla_info():
        """Hypothetical InternVLA: joint_pos/14, state_dim=14."""
        return _make_vla_info(
            action_type="joint_pos",
            action_dim=14,
            state_dim=14,
            cameras=["primary"],
        )

    @staticmethod
    def _robotwin_info():
        """RoboTwin sim /info: joint_pos/14, single camera, state_dim=14."""
        return _make_sim_info(
            action_type="joint_pos",
            action_dim=14,
            state_dim=14,
            cameras=["primary"],
        )

    def test_pi05_libero_compatible(self, monkeypatch):
        """pi0.5 (eef_delta/7) + LIBERO (eef_delta/7) → OK."""
        w = _make_wrapper(self._pi05_info(), self._libero_info(), monkeypatch)
        assert w.action_dim == 7
        assert w._image_transform == "flip_hw"

    def test_pi05_robocasa_compatible_via_accepted_dims(self, monkeypatch):
        """Cosmos-like VLA (dim=7) + RoboCasa PandaOmron (accepted_dims=[7,12]) → OK.

        With Option B (accepted_dims), the sim advertises that it can
        accept 7-dim actions (arm-only, padded to 12 internally).
        Uses state_dim=9 to match RoboCasa's obs space (pi0.5 uses 8).
        """
        # Use a Cosmos-like VLA that matches RoboCasa's state_dim=9
        vla = _make_vla_info(
            action_type="eef_delta", action_dim=7, state_dim=9,
            cameras=["primary", "wrist"],
        )
        w = _make_wrapper(vla, self._robocasa_info(), monkeypatch)
        assert w._policy_action_space["dim"] == 7
        assert w._sim_action_space["accepted_dims"] == [7, 12]

    def test_internvla_robotwin_compatible(self, monkeypatch):
        """InternVLA (joint_pos/14) + RoboTwin (joint_pos/14) → OK."""
        w = _make_wrapper(
            self._internvla_info(), self._robotwin_info(), monkeypatch
        )
        assert w.action_dim == 14

    def test_internvla_libero_incompatible(self, monkeypatch):
        """InternVLA (joint_pos/14) + LIBERO (eef_delta/7) → ValueError."""
        with pytest.raises(ValueError, match="type mismatch"):
            _make_wrapper(
                self._internvla_info(), self._libero_info(), monkeypatch
            )


# ======================================================================
# 5. TestAcceptedDims
# ======================================================================


class TestAcceptedDims:
    """Verify accepted_dims negotiation (Option B)."""

    def test_accepted_dims_match(self, monkeypatch):
        """VLA dim=7, sim accepted_dims=[7,12] → OK."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim = _make_sim_info(
            action_type="eef_delta", action_dim=12, accepted_dims=[7, 12],
        )
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._policy_action_space["dim"] == 7
        assert w._sim_action_space["accepted_dims"] == [7, 12]

    def test_accepted_dims_full_match(self, monkeypatch):
        """VLA dim=12, sim accepted_dims=[7,12] → OK."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=12, state_dim=9)
        sim = _make_sim_info(
            action_type="eef_delta", action_dim=12, state_dim=9,
            accepted_dims=[7, 12],
        )
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._policy_action_space["dim"] == 12
        assert w._sim_action_space["accepted_dims"] == [7, 12]

    def test_accepted_dims_reject(self, monkeypatch):
        """VLA dim=9, sim accepted_dims=[7,12] → ValueError."""
        vla = _make_vla_info(action_type="eef_delta", action_dim=9)
        sim = _make_sim_info(
            action_type="eef_delta", action_dim=12, accepted_dims=[7, 12],
        )
        with pytest.raises(ValueError, match="dim mismatch"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_no_accepted_dims_falls_back(self, monkeypatch):
        """No accepted_dims field → strict identity (dim must match exactly)."""
        # Match: dim==dim, no accepted_dims → OK
        vla = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim = _make_sim_info(action_type="eef_delta", action_dim=7)
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w._policy_action_space["dim"] == 7
        assert "accepted_dims" not in w._sim_action_space

        # Mismatch: dim!=dim, no accepted_dims → ValueError
        vla2 = _make_vla_info(action_type="eef_delta", action_dim=7)
        sim2 = _make_sim_info(action_type="eef_delta", action_dim=12)
        with pytest.raises(ValueError, match="dim mismatch"):
            _make_wrapper(vla2, sim2, monkeypatch)
