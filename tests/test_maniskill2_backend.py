"""Tests for ManiSkill2Backend (sims/sim_worker.py).

These tests do not require mani_skill2 or sapien to be installed and cover:

1. Backend registration in BACKENDS dict.
2. get_info() returns a well-formed typed spec (importable without mani_skill2).
3. init() raises RuntimeError on aarch64 with a useful message.
4. Task resolution: numeric index, exact name, short name, invalid name.
5. _extract_image graceful fallback when obs is None or malformed.

They validate the stub and metadata contract only.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the backend directly from sim_worker.
# sim_worker has no top-level mani_skill2 import so this always succeeds.
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from sims.sim_worker import BACKENDS, ManiSkill2Backend

# ===========================================================================
# 1. Registration
# ===========================================================================


class TestRegistration:
    def test_registered_in_backends(self):
        """ManiSkill2Backend must be present in the BACKENDS registry."""
        assert "maniskill2" in BACKENDS

    def test_registered_class(self):
        assert BACKENDS["maniskill2"] is ManiSkill2Backend


# ===========================================================================
# 2. get_info — valid spec without any mani_skill2 import
# ===========================================================================


class TestGetInfo:
    def setup_method(self):
        self.backend = ManiSkill2Backend()

    def test_returns_dict(self):
        info = self.backend.get_info()
        assert isinstance(info, dict)

    def test_action_space_present(self):
        info = self.backend.get_info()
        assert "action_space" in info
        assert info["action_space"]["dim"] == 7
        assert info["action_space"]["type"] == "eef_delta"

    def test_action_spec_keys(self):
        info = self.backend.get_info()
        spec = info["action_spec"]
        for key in ("position", "rotation", "gripper"):
            assert key in spec, f"action_spec missing '{key}'"

    def test_action_spec_dims(self):
        info = self.backend.get_info()
        spec = info["action_spec"]
        assert spec["position"]["dims"] == 3
        assert spec["rotation"]["dims"] == 3
        assert spec["gripper"]["dims"] == 1

    def test_observation_spec_keys(self):
        info = self.backend.get_info()
        obs_spec = info["observation_spec"]
        for key in ("primary", "state", "instruction"):
            assert key in obs_spec, f"observation_spec missing '{key}'"

    def test_obs_spec_primary_format(self):
        info = self.backend.get_info()
        assert info["observation_spec"]["primary"]["format"] == "rgb_hwc_uint8"

    def test_delta_actions_true(self):
        info = self.backend.get_info()
        assert info["delta_actions"] is True

    def test_max_steps_positive(self):
        info = self.backend.get_info()
        assert isinstance(info["max_steps"], int)
        assert info["max_steps"] > 0

    def test_aarch64_blocked_flag(self):
        """get_info() must advertise the aarch64 blocker so tooling can surface it."""
        info = self.backend.get_info()
        assert "aarch64_blocked" in info
        assert info["aarch64_blocked"] is True
        assert "aarch64_note" in info
        assert len(info["aarch64_note"]) > 10

    def test_sim_field(self):
        info = self.backend.get_info()
        assert info.get("sim") == "maniskill2"


# ===========================================================================
# 3. init() on aarch64 raises RuntimeError
# ===========================================================================


class TestInitAarch64:
    """On aarch64, init() must raise RuntimeError with sapien diagnostics."""

    @pytest.mark.skipif(
        platform.machine() != "aarch64",
        reason="aarch64 blocker only applies on aarch64",
    )
    def test_raises_on_aarch64(self):
        backend = ManiSkill2Backend()
        with pytest.raises(RuntimeError) as exc_info:
            backend.init("PickCube-v0", camera_resolution=64, headless=True)
        msg = str(exc_info.value)
        assert "sapien" in msg.lower() or "aarch64" in msg.lower(), (
            f"Error message should mention sapien/aarch64 blocker; got: {msg}"
        )
        assert "mani_skill2" in msg.lower() or "maniskill2" in msg.lower()

    def test_aarch64_blocker_message_contains_workarounds(self):
        """The blocker message string must include actionable options."""
        msg = ManiSkill2Backend._aarch64_blocker_message()
        assert "sapien" in msg.lower()
        assert "aarch64" in msg
        # Must suggest at least one workaround
        assert "x86_64" in msg or "setup.sh" in msg or "aloha_gym" in msg


# ===========================================================================
# 4. Task resolution
# ===========================================================================


class TestTaskResolution:
    def setup_method(self):
        self.backend = ManiSkill2Backend()

    def test_numeric_index_0(self):
        assert self.backend._resolve_task("0") == "PickCube-v0"

    def test_numeric_index_1(self):
        assert self.backend._resolve_task("1") == "StackCube-v0"

    def test_numeric_index_2(self):
        assert self.backend._resolve_task("2") == "PegInsertionSide-v0"

    def test_exact_name(self):
        assert self.backend._resolve_task("PickCube-v0") == "PickCube-v0"
        assert self.backend._resolve_task("StackCube-v0") == "StackCube-v0"
        assert self.backend._resolve_task("PegInsertionSide-v0") == "PegInsertionSide-v0"

    def test_substring_match(self):
        assert self.backend._resolve_task("pickcube") == "PickCube-v0"
        assert self.backend._resolve_task("stackcube") == "StackCube-v0"
        assert self.backend._resolve_task("peginsertion") == "PegInsertionSide-v0"

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            self.backend._resolve_task("99")

    def test_unrecognised_name_raises(self):
        with pytest.raises(ValueError):
            self.backend._resolve_task("DoesNotExist-v0")


# ===========================================================================
# 5. _extract_image graceful fallback
# ===========================================================================


class TestExtractImage:
    def setup_method(self):
        self.backend = ManiSkill2Backend()
        self.backend._cam_res = 64

    def test_none_obs_returns_blank(self):
        img, img2 = self.backend._extract_image(None)
        assert img is not None
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert img2 is None

    def test_valid_obs_structure(self):
        """Simulate the mani_skill2 rgbd obs dict shape."""
        fake_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        fake_rgb[10, 10] = [255, 0, 0]  # red pixel
        fake_obs = {"image": {"base_camera": {"rgb": fake_rgb}}}
        img, img2 = self.backend._extract_image(fake_obs)
        assert img.shape == (64, 64, 3)
        assert img[10, 10, 0] == 255  # red channel preserved
        assert img2 is None

    def test_malformed_obs_returns_blank(self):
        """If the expected obs structure is missing, return a blank frame (no crash)."""
        img, img2 = self.backend._extract_image({"unexpected": "structure"})
        assert img.shape == (64, 64, 3)
        assert img2 is None

    def test_flat_rgb_fallback(self):
        """Flat 'rgb' key in obs is used as a fallback."""
        fake_rgb = np.ones((64, 64, 3), dtype=np.uint8) * 42
        fake_obs = {"rgb": fake_rgb}
        img, img2 = self.backend._extract_image(fake_obs)
        assert img is not None
        assert img[0, 0, 0] == 42


# ===========================================================================
# 6. TASKS list integrity
# ===========================================================================


class TestTasksList:
    def test_tasks_non_empty(self):
        assert len(ManiSkill2Backend.TASKS) >= 3

    def test_all_tasks_have_description(self):
        for task in ManiSkill2Backend.TASKS:
            assert task in ManiSkill2Backend._TASK_DESCRIPTIONS, (
                f"Task '{task}' has no description in _TASK_DESCRIPTIONS"
            )

    def test_all_tasks_have_max_steps(self):
        for task in ManiSkill2Backend.TASKS:
            assert task in ManiSkill2Backend._TASK_MAX_STEPS, (
                f"Task '{task}' has no entry in _TASK_MAX_STEPS"
            )

    def test_max_steps_positive(self):
        for task, steps in ManiSkill2Backend._TASK_MAX_STEPS.items():
            assert steps > 0, f"max_steps for '{task}' must be positive"
