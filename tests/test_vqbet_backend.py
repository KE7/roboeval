"""Tests for the VQ-BeT VLA policy server.

VQ-BeT (Vector-Quantized Behavior Transformer) pairs with gym_pusht, the
original BeT paper's PushT benchmark and the same evaluation target used by
Diffusion Policy.

Tests mirror ``test_diffusion_policy_backend.py`` and ``test_act_backend.py``:
import + ABC compliance, ActionObsSpec contract, smoke YAML parse, and
server_runner / config registration.

All tests are offline (no GPU, no HuggingFace download).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loaded_policy(action_dim: int = 2, state_dim: int = 2):
    """Return a VQBeTPolicyServer with mocked internals (no GPU, no lerobot install)."""
    from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

    policy = VQBeTPolicyServer()
    policy.model_id = "lerobot/vqbet_pusht"
    policy._action_dim = action_dim
    policy._state_dim = state_dim
    policy._n_action_steps = 5
    policy._chunk_size = 5
    policy._camera_key = "observation.image"
    policy._image_h = 96
    policy._image_w = 96
    policy._device = "cpu"
    policy.ready = True
    return policy


# ---------------------------------------------------------------------------
# 1. Import & ABC compliance
# ---------------------------------------------------------------------------


class TestImport:
    def test_module_importable(self):
        mod = importlib.import_module("sims.vla_policies.vqbet_policy")
        assert hasattr(mod, "VQBeTPolicyServer")

    def test_is_subclass_of_vla_policy_base(self):
        from sims.vla_policies.base import VLAPolicyBase
        from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

        assert issubclass(VQBeTPolicyServer, VLAPolicyBase)

    def test_has_required_abstract_methods(self):
        from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

        for m in ("load_model", "predict", "get_info"):
            assert callable(getattr(VQBeTPolicyServer, m, None)), f"missing {m}"

    def test_initial_state_not_ready(self):
        from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

        policy = VQBeTPolicyServer()
        assert not policy.ready
        assert policy.load_error == ""


# ---------------------------------------------------------------------------
# 2. get_info / get_action_spec / get_observation_spec
# ---------------------------------------------------------------------------


class TestSpecs:
    def test_get_info_structure(self):
        policy = _make_loaded_policy()
        info = policy.get_info()
        assert info["name"] == "VQBeT"
        assert info["model_id"] == "lerobot/vqbet_pusht"
        assert info["action_space"]["dim"] == 2
        # "eef_xy" matches GymPushTBackend.get_info() action_space.type,
        # required by env_wrapper._negotiate_spaces().
        assert info["action_space"]["type"] == "eef_xy"
        assert info["state_dim"] == 2
        assert info["obs_requirements"]["cameras"] == ["primary"]

    def test_get_action_spec_returns_2_dim_xy(self):
        policy = _make_loaded_policy()
        spec = policy.get_action_spec()
        assert spec is not None
        # Key is "eef_xy" to match GymPushTBackend action_spec (parity with DP).
        assert "eef_xy" in spec
        total_dims = sum(s.dims for s in spec.values() if hasattr(s, "dims") and s.dims > 0)
        assert total_dims == 2

    def test_get_observation_spec_keys(self):
        policy = _make_loaded_policy()
        obs_spec = policy.get_observation_spec()
        assert obs_spec is not None
        for key in ("primary", "state"):
            assert key in obs_spec, f"missing observation key: {key}"

    def test_action_spec_format_is_absolute_xy_position(self):
        policy = _make_loaded_policy()
        spec = policy.get_action_spec()
        # "absolute_xy_position" matches GymPushTBackend action_spec.format.
        assert spec["eef_xy"].format == "absolute_xy_position"


# ---------------------------------------------------------------------------
# 3. Parity with Diffusion Policy on PushT
# ---------------------------------------------------------------------------


class TestParityWithDiffusionPolicy:
    """VQ-BeT and Diffusion Policy share the same gym_pusht ActionObsSpec.

    Any divergence here would break direct comparison on PushT.
    """

    def test_action_spec_matches_diffusion_policy_keys(self):
        from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer
        from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

        vq = VQBeTPolicyServer().get_action_spec()
        dp = DiffusionPolicyServer().get_action_spec()
        assert set(vq.keys()) == set(dp.keys())

    def test_observation_spec_matches_diffusion_policy_keys(self):
        from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer
        from sims.vla_policies.vqbet_policy import VQBeTPolicyServer

        vq = VQBeTPolicyServer().get_observation_spec()
        dp = DiffusionPolicyServer().get_observation_spec()
        assert set(vq.keys()) == set(dp.keys())


# ---------------------------------------------------------------------------
# 4. Smoke YAML configs parse as valid roboeval configs
# ---------------------------------------------------------------------------


class TestSmokeYaml:
    def test_gym_pusht_vqbet_smoke_yaml_parses(self):
        from roboeval.orchestrator import EvalConfig

        yaml_path = PROJECT_ROOT / "configs" / "gym_pusht_vqbet_smoke.yaml"
        assert yaml_path.exists(), f"missing smoke config: {yaml_path}"

        config = EvalConfig.from_yaml(yaml_path)
        assert config.name == "gym_pusht_vqbet_smoke"
        assert config.sim == "gym_pusht"
        assert config.episodes_per_task == 10
        assert config.no_vlm is True
        assert config.delta_actions is False
        # vqbet default port is 5108
        assert "5108" in config.vla_url

    def test_ci_gym_pusht_vqbet_smoke_parses(self):
        from roboeval.orchestrator import EvalConfig

        yaml_path = PROJECT_ROOT / "configs" / "ci" / "gym_pusht_vqbet_smoke.yaml"
        assert yaml_path.exists(), f"missing CI smoke config: {yaml_path}"

        config = EvalConfig.from_yaml(yaml_path)
        assert config.name == "ci_gym_pusht_vqbet_smoke"
        assert config.episodes_per_task == 5


# ---------------------------------------------------------------------------
# 5. server_runner / config registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_vqbet_in_vla_module_map(self):
        from roboeval.server_runner import _VLA_MODULE_MAP

        assert _VLA_MODULE_MAP.get("vqbet") == "sims.vla_policies.vqbet_policy"

    def test_vqbet_default_port_is_5108(self):
        from roboeval.server_runner import _VLA_DEFAULT_PORTS

        assert _VLA_DEFAULT_PORTS.get("vqbet") == 5108

    def test_vqbet_in_vla_configs(self):
        from roboeval.config import VLA_CONFIGS

        assert "vqbet" in VLA_CONFIGS
        cfg = VLA_CONFIGS["vqbet"]
        assert cfg.port == 5108
        assert cfg.model_id == "lerobot/vqbet_pusht"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
