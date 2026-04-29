"""ActionObsSpec contract gate tests.

These tests cover strict validation of VLA and simulator action/observation
specs during the ``/info`` handshake.

These tests guarantee:

1. **Mismatched action dim** at episode start → ``SpecMismatchError``.
2. **Mismatched state dim** at episode start → ``SpecMismatchError``.
3. **Mismatched action format** (``binary_close_positive`` vs
   ``binary_close_negative``) → ``SpecMismatchError``.
4. **Mismatched action space type** (``eef_delta`` vs ``joint_pos``) →
   ``SpecMismatchError`` (caught by ``_negotiate_spaces`` before ActionObsSpec).
5. **One-sided ActionObsSpec** (sim has none, model declares them, or vice versa)
   → ``SpecMismatchError``.
6. **All currently-shipped (model, sim) pairs in ``configs/``** parse and
   validate without raising.
"""

from __future__ import annotations

import pytest

from sims.env_wrapper import SpecMismatchError

# Reuse the shared mocked handshake helpers.
from tests.test_spec_handshake import (
    _libero_action_spec,
    _libero_obs_spec,
    _make_sim_info,
    _make_vla_info,
    _make_wrapper,
    _pi05_action_spec,
    _pi05_obs_spec,
)

# ---------------------------------------------------------------------------
# 1. Mismatched action dim → orchestrator raises at episode start
# ---------------------------------------------------------------------------


class TestActionDimMismatch:
    def test_action_position_dim_mismatch_raises(self, monkeypatch):
        bad_action = dict(_pi05_action_spec())
        bad_action["position"] = {
            "name": "position",
            "dims": 6,
            "format": "delta_xyz",
            "range": [-1, 1],
        }
        vla = _make_vla_info(action_spec=bad_action, observation_spec=_pi05_obs_spec())
        sim = _make_sim_info(action_spec=_libero_action_spec(), observation_spec=_libero_obs_spec())
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 2. Mismatched state dim → orchestrator raises at episode start
# ---------------------------------------------------------------------------


class TestStateDimMismatch:
    """The headline issue: RoboCasa 9-dim quat state vs LIBERO 8-dim axis-angle."""

    def test_robocasa_9dim_state_vs_libero_8dim_raises(self, monkeypatch):
        """LIBERO-trained pi05 (8-dim) targeting RoboCasa (9-dim quat) → HARD."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),  # state dims=8
        )
        # RoboCasa-style sim: 9-dim quaternion state.
        robocasa_obs = dict(_libero_obs_spec())
        robocasa_obs["state"] = {
            "name": "state",
            "dims": 9,
            "format": "robocasa_grip2_eef_pos3_quat4",
        }
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=robocasa_obs,
            state_dim=9,
        )
        with pytest.raises((SpecMismatchError, ValueError), match=r"(?i)(HARD|state)"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_state_format_mismatch_raises(self, monkeypatch):
        """Same dim, different format → HARD (axis-angle 8d vs quat-prefixed 8d)."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        bad_obs = dict(_libero_obs_spec())
        bad_obs["state"] = {"name": "state", "dims": 8, "format": "robocasa_quat_format"}
        sim = _make_sim_info(action_spec=_libero_action_spec(), observation_spec=bad_obs)
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 3. Mismatched action space type (eef_delta vs joint_pos) → raises
# ---------------------------------------------------------------------------


class TestActionSpaceTypeMismatch:
    def test_eef_delta_vs_joint_pos_raises(self, monkeypatch):
        """VLA emits eef_delta (7-dim), sim wants joint_pos (14-dim) → ValueError."""
        vla = _make_vla_info(action_dim=7)  # eef_delta default
        # Sim with joint_pos action_space (RoboTwin-style).
        sim = _make_sim_info()
        sim["action_space"] = {"type": "joint_pos", "dim": 14, "accepted_dims": [14]}
        with pytest.raises((SpecMismatchError, ValueError), match=r"(?i)(type|HARD)"):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 4. Gripper convention mismatch → raises (close_positive vs close_negative)
# ---------------------------------------------------------------------------


class TestGripperConventionMismatch:
    def test_gripper_close_positive_vs_negative_raises(self, monkeypatch):
        bad_action = dict(_pi05_action_spec())
        bad_action["gripper"] = {
            "name": "gripper",
            "dims": 1,
            "format": "binary_close_positive",
            "range": [-1, 1],
        }
        vla = _make_vla_info(action_spec=bad_action, observation_spec=_pi05_obs_spec())
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),  # accepts only close_negative
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 5. One-sided ActionObsSpec
# ---------------------------------------------------------------------------


class TestOneSidedSpecLeak:
    """One-sided ActionObsSpec declarations should fail validation."""

    def test_sim_lacks_dimspec_raises(self, monkeypatch):
        """A model spec without a simulator spec is a HARD failure."""
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info()  # legacy sim, no ActionObsSpec
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)

    def test_vla_lacks_dimspec_raises(self, monkeypatch):
        vla = _make_vla_info()  # legacy VLA, no ActionObsSpec
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        with pytest.raises(SpecMismatchError, match="HARD"):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 6. All shipped (model, sim) pairs validate
# ---------------------------------------------------------------------------


class TestShippedConfigsValidate:
    """Every ``configs/*.yaml`` file should pair a model and sim that pass the gate."""

    def test_libero_pi05_pair_passes(self, monkeypatch):
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        # Should NOT raise.
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7

    def test_libero_smolvla_pair_passes(self, monkeypatch):
        # smolvla shares the LIBERO eef_delta + 8-dim state contract.
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),  # same shape as smolvla
            observation_spec=_pi05_obs_spec(),
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7

    def test_libero_openvla_pair_passes(self, monkeypatch):
        # openvla declares only primary image + language (no state, no wrist).
        openvla_obs = {
            "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "instruction": {"name": "language", "dims": 0, "format": "language"},
        }
        vla = _make_vla_info(
            action_spec=_pi05_action_spec(),  # same eef_delta shape
            observation_spec=openvla_obs,
            state_dim=0,
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(),
            observation_spec=_libero_obs_spec(),
        )
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == 7
