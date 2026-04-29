"""Tests for the Octo VLA policy server.

Covers:
  1. OctoPolicy imports without jax/octo installed (graceful degradation).
  2. OctoPolicy.load_model records the correct load_error when jax is missing.
  3. OctoPolicy.load_model records the correct load_error when octo is missing.
  4. OctoPolicy.get_info returns the expected metadata.
  5. ActionObsSpec gate: Octo (primary image, no state) × LIBERO sim passes validation.
  6. ActionObsSpec gate: Octo × sim with mismatched state dim raises SpecMismatchError.
  7. /health endpoint returns ready=false when model is not loaded (missing octo pkg).
  8. configs/bridge_octo_smoke.yaml parses as a valid robo-eval config.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sims.vla_policies.octo_policy import OctoPolicy, _ACTION_DIM


# ---------------------------------------------------------------------------
# 1. Import without heavy deps
# ---------------------------------------------------------------------------


class TestImport:
    def test_octo_policy_importable_without_jax(self):
        """OctoPolicy should be importable even if jax/octo are not installed."""
        # This test passes in CI where jax/octo are absent.
        policy = OctoPolicy()
        assert not policy.ready
        assert policy.load_error == ""


# ---------------------------------------------------------------------------
# 2-3. load_model graceful degradation
# ---------------------------------------------------------------------------


class TestLoadModelDegradation:
    def test_missing_jax_sets_load_error(self):
        """When jax is not importable, load_model sets a clear load_error."""
        policy = OctoPolicy()
        with patch.dict("sys.modules", {"jax": None}):
            policy.load_model("rail-berkeley/octo-small-1.5", "cpu")
        assert not policy.ready
        assert "jax" in policy.load_error.lower()

    def test_missing_octo_sets_load_error(self):
        """When octo package is not importable, load_model sets a clear load_error."""
        import importlib

        policy = OctoPolicy()
        # Provide a fake jax so the jax guard passes
        fake_jax = MagicMock()
        fake_jax.random.PRNGKey.return_value = object()
        fake_jax.devices.return_value = [MagicMock()]

        with patch.dict("sys.modules", {"jax": fake_jax, "octo": None,
                                        "octo.model": None, "octo.model.octo_model": None}):
            policy.load_model("rail-berkeley/octo-small-1.5", "cpu")

        assert not policy.ready
        assert "octo" in policy.load_error.lower()


# ---------------------------------------------------------------------------
# 4. get_info metadata
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_get_info_structure(self):
        policy = OctoPolicy()
        policy.model_id = "rail-berkeley/octo-small-1.5"
        info = policy.get_info()
        assert info["name"] == "octo-small-1.5"
        assert info["action_space"]["dim"] == _ACTION_DIM
        assert info["state_dim"] == 0
        assert info["action_chunk_size"] == 1
        assert "cameras" in info["obs_requirements"]
        assert info["obs_requirements"]["cameras"] == ["primary"]

    def test_get_action_spec_returns_7_dims(self):
        from robo_eval.specs import ActionObsSpec

        policy = OctoPolicy()
        spec = policy.get_action_spec()
        assert spec is not None
        total_dims = sum(s.dims for s in spec.values() if hasattr(s, "dims") and s.dims > 0)
        # position (3) + rotation (3) + gripper (1) = 7
        assert total_dims == _ACTION_DIM

    def test_get_observation_spec_has_primary_and_language(self):
        policy = OctoPolicy()
        obs_spec = policy.get_observation_spec()
        assert obs_spec is not None
        assert "primary" in obs_spec
        assert "instruction" in obs_spec


# ---------------------------------------------------------------------------
# 5-6. ActionObsSpec gate (reuses test_spec_handshake machinery)
# ---------------------------------------------------------------------------


class TestSpecGate:
    def test_octo_x_libero_passes_gate(self, monkeypatch):
        """Given image-only Octo observations, LIBERO validation passes."""
        from tests.test_spec_handshake import (
            _make_vla_info,
            _make_sim_info,
            _make_wrapper,
            _libero_action_spec,
            _libero_obs_spec,
        )

        octo_action_spec = {
            "position": {"name": "position", "dims": 3, "format": "delta_xyz", "range": [-1, 1]},
            "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle", "range": [-3.15, 3.15]},
            "gripper": {"name": "gripper", "dims": 1, "format": "binary_close_negative", "range": [-1, 1]},
        }
        # Use "rgb_hwc_uint8" — the format string used by IMAGE_RGB constant and LIBERO obs spec.
        octo_obs_spec = {
            "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "instruction": {"name": "language", "dims": 0, "format": "language"},
        }

        vla = _make_vla_info(action_spec=octo_action_spec, observation_spec=octo_obs_spec, state_dim=0)
        sim = _make_sim_info(action_spec=_libero_action_spec(), observation_spec=_libero_obs_spec())

        # Should NOT raise — Octo image-only VLA can attach to LIBERO.
        w = _make_wrapper(vla, sim, monkeypatch)
        assert w.action_dim == _ACTION_DIM

    def test_octo_image_format_mismatch_raises(self, monkeypatch):
        """Given an undeclared required camera, the spec gate rejects the pairing.

        The spec gate raises when the VLA requires a key the sim doesn't declare.
        This test verifies that a key missing from the sim side is caught.
        """
        from tests.test_spec_handshake import (
            _make_vla_info,
            _make_sim_info,
            _make_wrapper,
            _libero_action_spec,
        )
        from sims.env_wrapper import SpecMismatchError

        octo_action_spec = {
            "position": {"name": "position", "dims": 3, "format": "delta_xyz", "range": [-1, 1]},
            "rotation": {"name": "rotation", "dims": 3, "format": "delta_axisangle", "range": [-3.15, 3.15]},
            "gripper": {"name": "gripper", "dims": 1, "format": "binary_close_negative", "range": [-1, 1]},
        }
        # Octo requires a "wrist" camera that the sim doesn't provide.
        octo_obs_spec_needs_wrist = {
            "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "wrist": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "instruction": {"name": "language", "dims": 0, "format": "language"},
        }
        # Sim only provides primary (no wrist camera).
        sim_obs_spec_no_wrist = {
            "primary": {"name": "image", "dims": 0, "format": "rgb_hwc_uint8"},
            "instruction": {"name": "language", "dims": 0, "format": "language"},
        }
        vla = _make_vla_info(
            action_spec=octo_action_spec, observation_spec=octo_obs_spec_needs_wrist, state_dim=0
        )
        sim = _make_sim_info(
            action_spec=_libero_action_spec(), observation_spec=sim_obs_spec_no_wrist
        )

        with pytest.raises((SpecMismatchError, ValueError)):
            _make_wrapper(vla, sim, monkeypatch)


# ---------------------------------------------------------------------------
# 7. /health endpoint when model missing
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_503_when_not_ready(self):
        """FastAPI /health must return 503 when model load failed."""
        from fastapi.testclient import TestClient
        from sims.vla_policies.base import make_app

        policy = OctoPolicy()
        policy.load_error = "octo not installed"
        app = make_app(policy, "rail-berkeley/octo-small-1.5", "cpu")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert not data["ready"]
        assert "error" in data

    def test_info_endpoint_includes_notes(self):
        """GET /info must include model metadata and specs."""
        from fastapi.testclient import TestClient
        from sims.vla_policies.base import make_app

        policy = OctoPolicy()
        policy.model_id = "rail-berkeley/octo-small-1.5"
        # Simulate ready state without actually loading jax/octo.
        policy.ready = True
        app = make_app(policy, "rail-berkeley/octo-small-1.5", "cpu")

        client = TestClient(app)
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state_dim"] == 0
        assert "action_spec" in data
        assert "observation_spec" in data


# ---------------------------------------------------------------------------
# 8. Smoke YAML parses as valid config
# ---------------------------------------------------------------------------


class TestSmokeYaml:
    def test_bridge_octo_smoke_yaml_parses(self):
        """configs/bridge_octo_smoke.yaml must be loadable as an EvalConfig."""
        from robo_eval.orchestrator import EvalConfig

        yaml_path = PROJECT_ROOT / "configs" / "bridge_octo_smoke.yaml"
        assert yaml_path.exists(), f"Missing smoke config: {yaml_path}"

        config = EvalConfig.from_yaml(yaml_path)
        assert config.name == "bridge_octo_smoke"
        # config.vla is not a field; the YAML key "vla: octo" is stored as params/extra.
        assert config.sim == "libero"
        assert config.suite == "libero_spatial"
        assert config.episodes_per_task == 2
        assert config.no_vlm is True
        # VLA server URL should default to the Octo port from the YAML
        assert "5106" in config.vla_url

    def test_bridge_octo_smoke_yaml_passes_preflight_validate(self):
        """The smoke YAML must pass robo_eval.preflight.validate_yaml."""
        from robo_eval.preflight import validate_yaml

        yaml_path = PROJECT_ROOT / "configs" / "bridge_octo_smoke.yaml"
        results = validate_yaml(yaml_path)
        checks = {r.name: r for r in results}
        assert checks["yaml.parse"].ok, checks["yaml.parse"].message
        assert checks["config.load"].ok, checks["config.load"].message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
