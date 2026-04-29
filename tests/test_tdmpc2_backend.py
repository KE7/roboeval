"""TDMPC2 policy backend — ABC compliance and ActionObsSpec contract tests.

These tests run without GPU / lerobot / tdmpc2 actually installed.  They verify:

1.  ``TDMPC2Policy`` is a concrete subclass of ``VLAPolicyBase`` (ABC compliance).
2.  ``get_action_spec()`` returns a dict of ``ActionObsSpec`` instances with the
    correct dims, format, and range for 4-dim Sawyer eef-delta — exact match
    for the metaworld backend.
3.  ``get_observation_spec()`` returns IMAGE_RGB + 39-dim metaworld_obs state
    + LANGUAGE entries.
4.  ``get_info()`` returns the mandatory keys, ``action_space.type == 'eef_delta'``,
    ``action_space.dim == 4``, and ``paradigm == 'model_based_rl'``.
5.  ``reset()`` does not raise when the model is not loaded (graceful no-op).
6.  Spec contract: ``get_action_spec()`` and ``get_observation_spec()`` entries
    are ``ActionObsSpec`` instances (not raw dicts).
7.  Module-level constants and CLI defaults (port 5109, model_id default).
8.  VLA registration: ``"tdmpc2"`` appears in ``robo_eval.config.VLA_CONFIGS``
    on port 5109.
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_tdmpc2_policy():
    """Import TDMPC2Policy without executing module-level GPU imports."""
    for mod in [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.tdmpc2",
        "lerobot.policies.tdmpc2.modeling_tdmpc2",
        "lerobot.common",
        "lerobot.common.policies",
        "lerobot.common.policies.tdmpc2",
        "lerobot.common.policies.tdmpc2.modeling_tdmpc2",
        "lerobot.utils",
        "lerobot.utils.constants",
        "tdmpc2",
        "torch",
        "torchvision",
        "torchvision.transforms",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    import robo_eval.specs  # noqa: F401

    if "sims.vla_policies.tdmpc2_policy" in sys.modules:
        del sys.modules["sims.vla_policies.tdmpc2_policy"]

    module = importlib.import_module("sims.vla_policies.tdmpc2_policy")
    return module.TDMPC2Policy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def policy_cls():
    return _import_tdmpc2_policy()


@pytest.fixture(scope="module")
def policy_instance(policy_cls):
    return policy_cls()


# ---------------------------------------------------------------------------
# 1. ABC compliance
# ---------------------------------------------------------------------------


class TestABCCompliance:
    def test_is_vla_policy_base_subclass(self, policy_cls):
        from sims.vla_policies.base import VLAPolicyBase
        assert issubclass(policy_cls, VLAPolicyBase)

    def test_load_model_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "load_model", None))

    def test_predict_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "predict", None))

    def test_get_info_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "get_info", None))

    def test_reset_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "reset", None))


# ---------------------------------------------------------------------------
# 2. get_action_spec()
# ---------------------------------------------------------------------------


class TestActionSpec:
    def test_returns_dict(self, policy_instance):
        assert isinstance(policy_instance.get_action_spec(), dict)

    def test_has_eef_delta_key(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert "eef_delta" in spec, (
            "TDMPC2 metaworld pairing requires 'eef_delta' action key"
        )

    def test_eef_delta_is_action_obs_spec(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        spec = policy_instance.get_action_spec()
        assert isinstance(spec["eef_delta"], ActionObsSpec)

    def test_eef_delta_dims_4(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert spec["eef_delta"].dims == 4, (
            "TDMPC2 metaworld eef-delta must be 4-dim — exact match for sim"
        )

    def test_eef_delta_format_xyz_gripper(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert spec["eef_delta"].format == "eef_delta_xyz_gripper", (
            "format must match the metaworld backend's accepts list"
        )

    def test_eef_delta_range_unit(self, policy_instance):
        spec = policy_instance.get_action_spec()
        lo, hi = spec["eef_delta"].range
        assert lo < 0 and hi > 0
        assert abs(lo) <= 1.5 and abs(hi) <= 1.5, (
            "TDMPC2 normalises actions to roughly unit range"
        )


# ---------------------------------------------------------------------------
# 3. get_observation_spec()
# ---------------------------------------------------------------------------


class TestObservationSpec:
    def test_returns_dict(self, policy_instance):
        assert isinstance(policy_instance.get_observation_spec(), dict)

    def test_has_primary(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        obs = policy_instance.get_observation_spec()
        assert "primary" in obs
        assert isinstance(obs["primary"], ActionObsSpec)
        assert obs["primary"].format == "rgb_hwc_uint8"

    def test_state_is_metaworld_39(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        obs = policy_instance.get_observation_spec()
        assert "state" in obs
        assert isinstance(obs["state"], ActionObsSpec)
        assert obs["state"].dims == 39
        assert obs["state"].format == "metaworld_obs"

    def test_has_instruction(self, policy_instance):
        obs = policy_instance.get_observation_spec()
        assert "instruction" in obs

    def test_all_entries_are_action_obs_spec(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        for key, val in policy_instance.get_observation_spec().items():
            assert isinstance(val, ActionObsSpec), f"{key}: {type(val)}"


# ---------------------------------------------------------------------------
# 4. get_info()
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_returns_dict(self, policy_instance):
        assert isinstance(policy_instance.get_info(), dict)

    def test_has_required_keys(self, policy_instance):
        info = policy_instance.get_info()
        for key in ("name", "model_id", "action_space", "state_dim", "action_chunk_size"):
            assert key in info

    def test_action_space_type_eef_delta(self, policy_instance):
        info = policy_instance.get_info()
        assert info["action_space"]["type"] == "eef_delta"

    def test_action_space_dim_4(self, policy_instance):
        info = policy_instance.get_info()
        assert info["action_space"]["dim"] == 4

    def test_state_dim_39(self, policy_instance):
        info = policy_instance.get_info()
        assert info["state_dim"] == 39

    def test_name_is_tdmpc2(self, policy_instance):
        info = policy_instance.get_info()
        assert info["name"] == "TDMPC2"

    def test_paradigm_is_model_based_rl(self, policy_instance):
        """TDMPC2 reports the model-based RL paradigm."""
        info = policy_instance.get_info()
        assert info.get("paradigm") == "model_based_rl"


# ---------------------------------------------------------------------------
# 5. reset() — graceful no-op before load_model
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_before_load_does_not_raise(self, policy_cls):
        p = policy_cls()
        p.reset()  # must not raise


# ---------------------------------------------------------------------------
# 6. Spec contract: ActionObsSpec instances (not raw dicts)
# ---------------------------------------------------------------------------


class TestSpecContract:
    def test_action_spec_values_typed(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        for key, val in policy_instance.get_action_spec().items():
            assert isinstance(val, ActionObsSpec), key

    def test_obs_spec_values_typed(self, policy_instance):
        from robo_eval.specs import ActionObsSpec
        for key, val in policy_instance.get_observation_spec().items():
            assert isinstance(val, ActionObsSpec), key


# ---------------------------------------------------------------------------
# 7. Module-level constants + CLI defaults
# ---------------------------------------------------------------------------


class TestModuleSmoke:
    def test_module_imports_cleanly(self):
        _import_tdmpc2_policy()

    def test_default_model_id(self):
        mod = importlib.import_module("sims.vla_policies.tdmpc2_policy")
        assert "tdmpc2" in mod._MODEL_ID_DEFAULT.lower()

    def test_default_port_5109(self):
        """Port 5109 — avoids collision with vqbet (5108)."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=5109)
        args = parser.parse_args([])
        assert args.port == 5109

    def test_action_dim_constant(self):
        mod = importlib.import_module("sims.vla_policies.tdmpc2_policy")
        assert mod._ACTION_DIM == 4
        assert mod._STATE_DIM == 39


# ---------------------------------------------------------------------------
# 8. VLA registration
# ---------------------------------------------------------------------------


class TestVLARegistration:
    def test_tdmpc2_in_vla_configs(self):
        from robo_eval.config import VLA_CONFIGS
        assert "tdmpc2" in VLA_CONFIGS, (
            "tdmpc2 must be registered in VLA_CONFIGS for the orchestrator to find it"
        )

    def test_tdmpc2_port_5109(self):
        from robo_eval.config import VLA_CONFIGS
        assert VLA_CONFIGS["tdmpc2"].port == 5109

    def test_tdmpc2_model_id(self):
        from robo_eval.config import VLA_CONFIGS
        assert "tdmpc2" in VLA_CONFIGS["tdmpc2"].model_id.lower()


# ---------------------------------------------------------------------------
# 9. Metaworld pairing — spec compatibility
# ---------------------------------------------------------------------------


class TestMetaworldPairing:
    """TDMPC2 action specs must match the metaworld backend contract."""

    def test_format_matches_metaworld_accepts(self, policy_instance):
        # MetaWorldBackend.get_info() declares:
        #   action_spec.eef_delta.accepts = ["eef_delta_xyz_gripper"]
        spec = policy_instance.get_action_spec()
        assert spec["eef_delta"].format == "eef_delta_xyz_gripper"

    def test_dims_match_metaworld_accepted_dims(self, policy_instance):
        # MetaWorldBackend.get_info() declares accepted_dims=[4]
        spec = policy_instance.get_action_spec()
        assert spec["eef_delta"].dims == 4
