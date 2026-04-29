"""Tests for GymPushTBackend (gym-pusht / PushT pushing benchmark).

These tests run without ``gym-pusht`` or ``gymnasium`` installed. The backend
is exercised via unittest.mock so the test suite passes in environments where
those optional packages are unavailable.

To run against a real install:
    .venvs/gym_pusht/bin/pytest tests/test_gym_pusht_backend.py -v
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module-level stubs — inserted before importing sims.sim_worker so that
# GymPushTBackend can be imported without the real packages.
# ---------------------------------------------------------------------------

def _inject_stubs() -> None:
    """Inject lightweight module stubs for gymnasium and gym_pusht."""
    if "gymnasium" not in sys.modules:
        gymnasium_stub = types.ModuleType("gymnasium")
        gymnasium_stub.make = MagicMock()  # overridden per-test
        sys.modules["gymnasium"] = gymnasium_stub
    if "gym_pusht" not in sys.modules:
        sys.modules["gym_pusht"] = types.ModuleType("gym_pusht")


_inject_stubs()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_env(reward: float = 0.5) -> MagicMock:
    """Return a MagicMock that behaves like a gym-pusht environment."""
    fake_env = MagicMock()
    fake_env.spec.max_episode_steps = 300
    _fake_obs = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "agent_pos": np.array([0.5, 0.5], dtype=np.float32),
    }
    fake_env.reset.return_value = (_fake_obs, {})
    fake_env.step.return_value = (_fake_obs, reward, False, False, {})
    fake_env.render.return_value = np.zeros((96, 96, 3), dtype=np.uint8)
    return fake_env


def _make_backend(reward: float = 0.5):
    """Create a GymPushTBackend with env pre-wired to a MagicMock."""
    from sims.sim_worker import GymPushTBackend  # noqa: PLC0415

    fake_env = _make_fake_env(reward)
    b = GymPushTBackend()
    # Set the state normally established by init().
    b._task_id = "gym_pusht/PushT-v0"
    b._cam_res = 96
    b._max_steps = 300
    b.env = fake_env
    return b, fake_env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def backend():
    """GymPushTBackend with a MagicMock environment."""
    return _make_backend()


# ---------------------------------------------------------------------------
# Task resolution
# ---------------------------------------------------------------------------

def test_resolve_task_by_full_id():
    from sims.sim_worker import GymPushTBackend
    b = GymPushTBackend()
    assert b._resolve_task("gym_pusht/PushT-v0") == "gym_pusht/PushT-v0"


def test_resolve_task_by_index_zero():
    from sims.sim_worker import GymPushTBackend
    b = GymPushTBackend()
    assert b._resolve_task("0") == "gym_pusht/PushT-v0"


def test_resolve_task_by_short_alias():
    from sims.sim_worker import GymPushTBackend
    b = GymPushTBackend()
    assert b._resolve_task("pusht") == "gym_pusht/PushT-v0"


def test_resolve_task_bad_index():
    from sims.sim_worker import GymPushTBackend
    b = GymPushTBackend()
    with pytest.raises(ValueError, match="only has task index 0"):
        b._resolve_task("5")


def test_resolve_task_unknown_name():
    from sims.sim_worker import GymPushTBackend
    b = GymPushTBackend()
    with pytest.raises(ValueError, match="not recognised"):
        b._resolve_task("CubeGrasp-v99")


# ---------------------------------------------------------------------------
# Init (stubbed via sys.modules)
# ---------------------------------------------------------------------------

def test_init_sets_task_id():
    """init() must store the resolved gymnasium task id."""
    from sims.sim_worker import GymPushTBackend  # noqa: PLC0415

    fake_env = _make_fake_env()
    sys.modules["gymnasium"].make = MagicMock(return_value=fake_env)

    b = GymPushTBackend()
    result = b.init(task_name="gym_pusht/PushT-v0", camera_resolution=96, headless=True)
    assert b._task_id == "gym_pusht/PushT-v0"
    assert "task_description" in result
    assert "T" in result["task_description"]


def test_init_task_description_by_index():
    """init() must also accept numeric index 0."""
    from sims.sim_worker import GymPushTBackend

    fake_env = _make_fake_env()
    sys.modules["gymnasium"].make = MagicMock(return_value=fake_env)

    b = GymPushTBackend()
    result = b.init(task_name="0", camera_resolution=96, headless=True)
    assert "task_description" in result


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_returns_image_tuple(backend):
    b, _ = backend
    img, img2 = b.reset(episode_index=0)
    assert img is not None
    assert img2 is None  # gym-pusht is single-camera


def test_reset_resets_step_count(backend):
    b, _ = backend
    b._step_count = 99
    b.reset()
    assert b._step_count == 0


def test_reset_resets_reward(backend):
    b, _ = backend
    b._last_reward = 1.0
    b.reset()
    assert b._last_reward == 0.0


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def test_step_returns_correct_structure(backend):
    b, _ = backend
    b.reset()
    img, img2, reward, done, info = b.step([0.5, 0.5])
    assert img is not None
    assert img2 is None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "success" in info


def test_step_increments_step_count(backend):
    b, _ = backend
    b.reset()
    b.step([0.0, 0.0])
    assert b._step_count == 1


def test_step_truncates_long_action(backend):
    b, fake_env = backend
    b.reset()
    # 7-dim input → must be clipped to 2-dim before calling env.step
    b.step([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    called_action = fake_env.step.call_args.args[0]
    assert called_action.shape == (2,)


def test_step_success_flag_when_high_reward(backend):
    b, fake_env = backend
    b.reset()
    fake_obs = {
        "pixels": np.zeros((96, 96, 3), dtype=np.uint8),
        "agent_pos": np.array([0.5, 0.5], dtype=np.float32),
    }
    fake_env.step.return_value = (fake_obs, 0.95, False, False, {})
    _, _, _, done, info = b.step([0.5, 0.5])
    assert info["success"] is True
    assert done is True


def test_step_default_reward_is_not_success(backend):
    """Default fake reward (0.5) must not trigger success."""
    b, _ = backend
    b.reset()
    _, _, reward, done, info = b.step([0.5, 0.5])
    assert info["success"] is False
    assert done is False


# ---------------------------------------------------------------------------
# check_success
# ---------------------------------------------------------------------------

def test_check_success_false_by_default(backend):
    b, _ = backend
    b.reset()
    assert b.check_success() is False


def test_check_success_true_after_high_reward(backend):
    b, _ = backend
    b._last_reward = 0.95
    assert b.check_success() is True


# ---------------------------------------------------------------------------
# get_info
# ---------------------------------------------------------------------------

def test_get_info_action_space(backend):
    b, _ = backend
    info = b.get_info()
    assert info["action_space"]["dim"] == 2
    assert info["action_space"]["type"] == "eef_xy"


def test_get_info_observation_spec(backend):
    b, _ = backend
    info = b.get_info()
    assert "primary" in info["observation_spec"]
    assert "state" in info["observation_spec"]


def test_get_info_delta_actions_false(backend):
    b, _ = backend
    info = b.get_info()
    assert info["delta_actions"] is False


# ---------------------------------------------------------------------------
# _extract_image
# ---------------------------------------------------------------------------

def test_extract_image_from_dict(backend):
    b, _ = backend
    obs = {"pixels": np.full((96, 96, 3), 42, dtype=np.uint8)}
    img = b._extract_image(obs)
    assert img.shape == (96, 96, 3)
    assert img[0, 0, 0] == 42


def test_extract_image_none_returns_zeros(backend):
    b, _ = backend
    img = b._extract_image(None)
    assert img.shape == (96, 96, 3)
    assert (img == 0).all()


# ---------------------------------------------------------------------------
# _extract_state
# ---------------------------------------------------------------------------

def test_extract_state_returns_two_floats(backend):
    b, _ = backend
    obs = {"agent_pos": np.array([0.3, 0.7], dtype=np.float32)}
    state = b._extract_state(obs)
    assert len(state) == 2
    assert abs(state[0] - 0.3) < 1e-5


def test_extract_state_none_returns_zeros(backend):
    b, _ = backend
    state = b._extract_state(None)
    assert state == [0.0, 0.0]


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

def test_close_sets_env_none(backend):
    b, fake_env = backend
    b.close()
    assert b.env is None
    fake_env.close.assert_called_once()


def test_close_idempotent(backend):
    b, _ = backend
    b.close()
    b.close()  # should not raise


# ---------------------------------------------------------------------------
# BACKENDS registry
# ---------------------------------------------------------------------------

def test_gym_pusht_in_backends():
    from sims.sim_worker import BACKENDS, GymPushTBackend

    assert "gym_pusht" in BACKENDS
    assert BACKENDS["gym_pusht"] is GymPushTBackend
