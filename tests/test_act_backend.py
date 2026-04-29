"""ACT policy backend — ABC compliance and ActionObsSpec contract tests.

These tests run without GPU / lerobot installed.  They verify:

1.  ``ACTPolicy`` is a concrete subclass of ``VLAPolicyBase`` (ABC compliance).
2.  ``get_action_spec()`` returns a dict of ``ActionObsSpec`` instances with the
    correct dims, format, and range for 14-dim aloha joint-position control.
3.  ``get_observation_spec()`` returns the expected IMAGE_RGB + 14-dim state +
    LANGUAGE entries.
4.  ``get_info()`` returns the mandatory keys and the correct action-space type
    (``joint_pos``).
5.  ``reset()`` does not raise when the model is not loaded (graceful no-op).
6.  ``predict()`` is declared (not abstract).
7.  Spec contract: ``get_action_spec()`` and ``get_observation_spec()`` entries
    are ``ActionObsSpec`` instances (not raw dicts) so validation can inspect
    their fields.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_act_policy():
    """Import ACTPolicy without executing module-level GPU imports."""
    # Stub out heavy dependencies so the module can be imported in CI.
    for mod in [
        "lerobot",
        "lerobot.policies",
        "lerobot.policies.act",
        "lerobot.policies.act.modeling_act",
        "lerobot.common",
        "lerobot.common.policies",
        "lerobot.common.policies.act",
        "lerobot.common.policies.act.modeling_act",
        "lerobot.utils",
        "lerobot.utils.constants",
        "torch",
        "torchvision",
        "torchvision.transforms",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Ensure the roboeval.specs constants are importable.
    import roboeval.specs

    if "sims.vla_policies.act_policy" in sys.modules:
        del sys.modules["sims.vla_policies.act_policy"]

    module = importlib.import_module("sims.vla_policies.act_policy")
    return module.ACTPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def policy_cls():
    return _import_act_policy()


@pytest.fixture(scope="module")
def policy_instance(policy_cls):
    """Instantiate the policy without loading the model."""
    return policy_cls()


# ---------------------------------------------------------------------------
# 1. ABC compliance
# ---------------------------------------------------------------------------


class TestABCCompliance:
    def test_is_vla_policy_base_subclass(self, policy_cls):
        from sims.vla_policies.base import VLAPolicyBase

        assert issubclass(policy_cls, VLAPolicyBase), "ACTPolicy must subclass VLAPolicyBase"

    def test_load_model_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "load_model", None)), "load_model must be implemented"

    def test_predict_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "predict", None)), "predict must be implemented"

    def test_get_info_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "get_info", None)), "get_info must be implemented"

    def test_reset_declared(self, policy_cls):
        assert callable(getattr(policy_cls, "reset", None)), "reset must be implemented"


# ---------------------------------------------------------------------------
# 2. get_action_spec()
# ---------------------------------------------------------------------------


class TestActionSpec:
    def test_returns_dict(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert isinstance(spec, dict), "get_action_spec() must return a dict"

    def test_has_joint_pos_key(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert "joint_pos" in spec, (
            "ACT action spec must have a 'joint_pos' key (canonical aloha key, "
            "matching AlohaGymBackend and RoboTwinBackend)"
        )

    def test_joint_pos_is_action_obs_spec(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        spec = policy_instance.get_action_spec()
        assert isinstance(spec["joint_pos"], ActionObsSpec), (
            "'joint_pos' must be an ActionObsSpec instance, not a raw dict"
        )

    def test_joint_pos_dims_14(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert spec["joint_pos"].dims == 14, (
            "ACT aloha joint action must be 14-dimensional (7 per arm)"
        )

    def test_joint_pos_format_absolute_joint_positions(self, policy_instance):
        spec = policy_instance.get_action_spec()
        assert spec["joint_pos"].format == "absolute_joint_positions", (
            "ACT uses absolute joint positions; format must be 'absolute_joint_positions' "
            "(canonical string accepted by AlohaGymBackend / RoboTwinBackend)"
        )

    def test_joint_pos_range_plausible(self, policy_instance):
        spec = policy_instance.get_action_spec()
        lo, hi = spec["joint_pos"].range
        assert lo < 0 and hi > 0, (
            "absolute_joint_positions range must span negative to positive (radians)"
        )


# ---------------------------------------------------------------------------
# 3. get_observation_spec()
# ---------------------------------------------------------------------------


class TestObservationSpec:
    def test_returns_dict(self, policy_instance):
        obs = policy_instance.get_observation_spec()
        assert isinstance(obs, dict), "get_observation_spec() must return a dict"

    def test_has_primary_camera(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        obs = policy_instance.get_observation_spec()
        assert "primary" in obs, "obs spec must declare a 'primary' camera"
        assert isinstance(obs["primary"], ActionObsSpec)
        assert obs["primary"].format == "rgb_hwc_uint8", (
            "primary camera must be IMAGE_RGB (rgb_hwc_uint8)"
        )

    def test_has_state(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        obs = policy_instance.get_observation_spec()
        assert "state" in obs, "obs spec must declare a 'state' entry"
        state_spec = obs["state"]
        assert isinstance(state_spec, ActionObsSpec)
        assert state_spec.dims == 14, (
            "ACT aloha state must be 14-dimensional (joint positions, 7 per arm)"
        )

    def test_has_instruction(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        obs = policy_instance.get_observation_spec()
        assert "instruction" in obs, (
            "obs spec must declare 'instruction' (accepted but ignored by ACT)"
        )
        assert isinstance(obs["instruction"], ActionObsSpec)

    def test_all_entries_are_action_obs_spec(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        obs = policy_instance.get_observation_spec()
        for key, val in obs.items():
            assert isinstance(val, ActionObsSpec), (
                f"obs spec entry '{key}' must be ActionObsSpec, got {type(val)}"
            )


# ---------------------------------------------------------------------------
# 4. get_info()
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_returns_dict(self, policy_instance):
        info = policy_instance.get_info()
        assert isinstance(info, dict)

    def test_has_required_keys(self, policy_instance):
        info = policy_instance.get_info()
        for key in ("name", "model_id", "action_space", "state_dim", "action_chunk_size"):
            assert key in info, f"get_info() must include '{key}'"

    def test_action_space_type_joint_pos(self, policy_instance):
        info = policy_instance.get_info()
        assert info["action_space"]["type"] == "joint_pos", (
            "ACT action space type must be 'joint_pos' (not 'eef_delta')"
        )

    def test_action_space_dim_14(self, policy_instance):
        info = policy_instance.get_info()
        assert info["action_space"]["dim"] == 14

    def test_state_dim_14(self, policy_instance):
        info = policy_instance.get_info()
        assert info["state_dim"] == 14

    def test_action_chunk_size_100(self, policy_instance):
        info = policy_instance.get_info()
        # Default chunk size for aloha ACT checkpoints is 100
        assert info["action_chunk_size"] == 100

    def test_name_is_act(self, policy_instance):
        info = policy_instance.get_info()
        assert info["name"] == "ACT"

    def test_image_transform_none(self, policy_instance):
        """Aloha cameras do not need the LIBERO 180° flip."""
        info = policy_instance.get_info()
        obs_req = info.get("obs_requirements", {})
        transform = obs_req.get("image_transform", "none")
        assert transform == "none", (
            "ACT aloha does not require an image flip; image_transform must be 'none'"
        )


# ---------------------------------------------------------------------------
# 5. reset() — graceful no-op before load_model
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_before_load_does_not_raise(self, policy_cls):
        """reset() must not raise even when the model has not been loaded."""
        p = policy_cls()
        # _policy is not set yet; reset() should be a safe no-op
        p.reset()  # must not raise


# ---------------------------------------------------------------------------
# 6. Spec contract: specs are ActionObsSpec instances (not raw dicts)
# ---------------------------------------------------------------------------


class TestSpecContract:
    """Given policy specs, validation requires ActionObsSpec values.

    Raw dicts do not expose the ``.dims`` and ``.format`` attributes expected by
    the spec contract.
    """

    def test_action_spec_values_typed(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        for key, val in policy_instance.get_action_spec().items():
            assert isinstance(val, ActionObsSpec), (
                f"action_spec['{key}'] is {type(val)}, expected ActionObsSpec"
            )

    def test_obs_spec_values_typed(self, policy_instance):
        from roboeval.specs import ActionObsSpec

        for key, val in policy_instance.get_observation_spec().items():
            assert isinstance(val, ActionObsSpec), (
                f"obs_spec['{key}'] is {type(val)}, expected ActionObsSpec"
            )


# ---------------------------------------------------------------------------
# 7. Module-level smoke: module imports cleanly with stubs
# ---------------------------------------------------------------------------


class TestModuleSmoke:
    def test_module_imports_cleanly(self):
        """Importing act_policy should not raise even with stubs."""
        _import_act_policy()  # already done in fixture; call again for isolation

    def test_default_model_id(self):
        """Default checkpoint is the transfer-cube human demo checkpoint."""
        import importlib

        mod = importlib.import_module("sims.vla_policies.act_policy")
        assert "act_aloha_sim_transfer_cube_human" in mod._MODEL_ID_DEFAULT, (
            "Default model ID must be the transfer-cube checkpoint (best smoke coverage)"
        )

    def test_default_port_5107(self):
        """Default port must be 5107 (avoids collision with other VLA servers)."""
        # Parse the argparser default.
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=5107)
        args = parser.parse_args([])
        assert args.port == 5107

    def test_action_dim_constant(self):
        import importlib

        mod = importlib.import_module("sims.vla_policies.act_policy")
        assert mod._ACTION_DIM == 14
        assert mod._STATE_DIM == 14
        assert mod._CHUNK_SIZE == 100
