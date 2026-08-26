"""Tests for SimWrapper space negotiation and image transform.

VLA and sim action spaces must match exactly by type and dimension, with no
padding, truncation, or translation.
"""

import numpy as np
import pytest
from PIL import Image

from sims.env_wrapper import SimWrapper, _apply_image_transform

# ---- Helper: construct a minimal SimWrapper-like object for method tests ----


class _StubWrapper:
    """Minimal stub with the attributes that negotiation methods read."""

    def __init__(self, policy_action_space, sim_action_space, policy_info=None, sim_info=None):
        self._policy_action_space = policy_action_space
        self._sim_action_space = sim_action_space
        self._policy_info = policy_info or {}
        self._sim_info = sim_info or {}
        self._image_transform = "none"


# ==========================================================================
# _negotiate_spaces: strict identity matching
# ==========================================================================


class TestNegotiateSpaces:
    def test_identity_eef_delta_7(self):
        """Matching type + dim passes without error."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
        )
        # Should not raise
        SimWrapper._negotiate_spaces(w)

    def test_identity_joint_pos_14(self):
        """joint_pos dim=14 matching passes."""
        w = _StubWrapper(
            policy_action_space={"type": "joint_pos", "dim": 14},
            sim_action_space={"type": "joint_pos", "dim": 14},
        )
        SimWrapper._negotiate_spaces(w)

    def test_type_mismatch_raises(self):
        """Different action types raise ValueError."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "joint_pos", "dim": 7},
        )
        with pytest.raises(ValueError, match="type mismatch"):
            SimWrapper._negotiate_spaces(w)

    def test_dim_mismatch_raises(self):
        """Same type but different dims raise ValueError."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 12},
        )
        with pytest.raises(ValueError, match="dim mismatch"):
            SimWrapper._negotiate_spaces(w)

    def test_dim_mismatch_6_to_7_raises(self):
        """No automatic padding from 6 to 7 dimensions."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 6},
            sim_action_space={"type": "eef_delta", "dim": 7},
        )
        with pytest.raises(ValueError, match="dim mismatch"):
            SimWrapper._negotiate_spaces(w)


# ==========================================================================
# _negotiate_obs: camera and state checks
# ==========================================================================


class TestNegotiateObs:
    def test_no_obs_requirements_skips(self):
        """VLA without obs_requirements skips negotiation."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={},  # no obs_requirements
            sim_info={"obs_space": {"cameras": [{"role": "primary"}]}},
        )
        # Should not raise
        SimWrapper._negotiate_obs(w)

    def test_cameras_match(self):
        """VLA requires cameras that sim provides — passes."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary", "wrist"],
                    "state_dim": 8,
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}, {"role": "wrist"}],
                    "state": {"dim": 8},
                }
            },
        )
        SimWrapper._negotiate_obs(w)

    def test_missing_camera_raises(self):
        """VLA requires a camera the sim doesn't have — raises."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary", "wrist"],
                    "state_dim": 0,
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}],
                    "state": {"dim": 0},
                }
            },
        )
        with pytest.raises(ValueError, match="Camera mismatch"):
            SimWrapper._negotiate_obs(w)

    def test_state_dim_mismatch_raises(self):
        """VLA requires state_dim=8 but sim provides state_dim=14 — raises."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary"],
                    "state_dim": 8,
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}],
                    "state": {"dim": 14},
                }
            },
        )
        with pytest.raises(ValueError, match="State dim mismatch"):
            SimWrapper._negotiate_obs(w)

    def test_vla_needs_state_sim_has_none_raises(self):
        """VLA requires state but sim provides none — raises."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary"],
                    "state_dim": 8,
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}],
                    "state": {"dim": 0},
                }
            },
        )
        with pytest.raises(ValueError, match="State mismatch"):
            SimWrapper._negotiate_obs(w)

    def test_vla_ignores_state_passes(self):
        """VLA with state_dim=0 doesn't care about sim state — passes."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary"],
                    "state_dim": 0,
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}],
                    "state": {"dim": 8},
                }
            },
        )
        SimWrapper._negotiate_obs(w)

    def test_image_transform_stored(self):
        """VLA-declared image_transform is stored on the wrapper."""
        w = _StubWrapper(
            policy_action_space={"type": "eef_delta", "dim": 7},
            sim_action_space={"type": "eef_delta", "dim": 7},
            policy_info={
                "obs_requirements": {
                    "cameras": ["primary"],
                    "state_dim": 0,
                    "image_transform": "flip_hw",
                }
            },
            sim_info={
                "obs_space": {
                    "cameras": [{"role": "primary"}],
                    "state": {"dim": 0},
                }
            },
        )
        SimWrapper._negotiate_obs(w)
        assert w._image_transform == "flip_hw"


# ==========================================================================
# _apply_image_transform
# ==========================================================================


class TestImageTransform:
    def _make_gradient_image(self):
        """Create a small gradient image for transform testing."""
        arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        return Image.fromarray(arr)

    def test_flip_hw(self):
        """flip_hw reverses both H and W axes."""
        img = self._make_gradient_image()
        result = _apply_image_transform(img, "flip_hw")
        orig = np.array(img)
        expected = orig[::-1, ::-1]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_flip_h(self):
        """flip_h reverses only H axis."""
        img = self._make_gradient_image()
        result = _apply_image_transform(img, "flip_h")
        orig = np.array(img)
        expected = orig[::-1]
        np.testing.assert_array_equal(np.array(result), expected)

    def test_none_passthrough(self):
        """'none' returns the image unchanged."""
        img = self._make_gradient_image()
        result = _apply_image_transform(img, "none")
        np.testing.assert_array_equal(np.array(result), np.array(img))

    def test_empty_string_passthrough(self):
        """Empty string treated as 'none'."""
        img = self._make_gradient_image()
        result = _apply_image_transform(img, "")
        np.testing.assert_array_equal(np.array(result), np.array(img))

    def test_unknown_transform_raises(self):
        """Unknown transform name raises ValueError."""
        img = self._make_gradient_image()
        with pytest.raises(ValueError, match="Unknown image_transform"):
            _apply_image_transform(img, "rotate_90")
