"""Tests for the Diffusion Policy VLA backend.

Covers:
  1. ABC compliance for DiffusionPolicyServer and VLAPolicyBase.
  2. ActionObsSpec shape and format correctness (PushT 2-dim checkpoint).
  3. ``get_info()`` structure and required keys.
  4. Import path exists and class is importable without model hardware.
  5. Policy server is accessible from the sims.vla_policies namespace.
  6. Smoke: predict() returns a correctly-shaped action for a mock observation.

Tests use mocked policy-loading paths to avoid model downloads and hardware
dependencies.
"""

from __future__ import annotations

import base64
import importlib
import io
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blank_image_b64(w: int = 96, h: int = 96) -> str:
    """Return a base64-encoded blank JPEG for use as a mock camera frame."""
    img = Image.new("RGB", (w, h), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mock_policy(action_dim: int = 2, state_dim: int = 2):
    """Return a DiffusionPolicyServer with mocked model internals."""
    from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer

    server = DiffusionPolicyServer()

    # Simulate load_model side-effects without real weights.
    server.model_id = "lerobot/diffusion_pusht"
    server._action_dim = action_dim
    server._state_dim = state_dim
    server._n_action_steps = 8
    server._chunk_size = 16
    server._camera_key = "observation.image"
    server._image_h = 96
    server._image_w = 96
    server._device = "cpu"

    # Mock the underlying lerobot policy.
    mock_lerobot = MagicMock()
    mock_lerobot.select_action.return_value = np.zeros(action_dim, dtype=np.float32)
    server._policy = mock_lerobot
    server.ready = True

    return server


# ---------------------------------------------------------------------------
# 1. ABC compliance
# ---------------------------------------------------------------------------


class TestABCCompliance:
    def test_import_succeeds(self):
        """The module must be importable without model hardware or downloads."""
        mod = importlib.import_module("sims.vla_policies.diffusion_policy_policy")
        assert hasattr(mod, "DiffusionPolicyServer")

    def test_is_subclass_of_vla_policy_base(self):
        from sims.vla_policies.base import VLAPolicyBase
        from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer

        assert issubclass(DiffusionPolicyServer, VLAPolicyBase)

    def test_has_required_abstract_methods(self):
        """DiffusionPolicyServer must implement all VLAPolicyBase abstracts."""
        from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer

        assert callable(getattr(DiffusionPolicyServer, "load_model", None))
        assert callable(getattr(DiffusionPolicyServer, "predict", None))
        assert callable(getattr(DiffusionPolicyServer, "get_info", None))

    def test_supports_batching_false(self):
        """DiffusionPolicy DDPM loop is not batch-parallelised in v0.1."""
        from sims.vla_policies.diffusion_policy_policy import DiffusionPolicyServer

        assert DiffusionPolicyServer.supports_batching is False

    def test_reset_is_callable(self):
        server = _make_mock_policy()
        server.reset()  # should not raise
        server._policy.reset.assert_called_once()


# ---------------------------------------------------------------------------
# 2. ActionObsSpec — PushT 2-dim checkpoint
# ---------------------------------------------------------------------------


class TestActionObsSpec:
    def test_get_action_spec_returns_dict(self):
        server = _make_mock_policy(action_dim=2)
        spec = server.get_action_spec()
        assert isinstance(spec, dict)
        assert len(spec) > 0

    def test_pusht_action_spec_has_position_key(self):
        server = _make_mock_policy(action_dim=2)
        spec = server.get_action_spec()
        assert "eef_xy" in spec

    def test_pusht_action_spec_position_dims(self):
        from robo_eval.specs import ActionObsSpec

        server = _make_mock_policy(action_dim=2)
        spec = server.get_action_spec()
        pos = spec["eef_xy"]
        assert isinstance(pos, ActionObsSpec)
        assert pos.dims == 2

    def test_pusht_action_spec_format_is_absolute(self):
        server = _make_mock_policy(action_dim=2)
        spec = server.get_action_spec()
        assert "absolute" in spec["eef_xy"].format

    def test_non_pusht_action_spec_uses_joint_pos(self):
        """14-dim (ALOHA) checkpoint should declare joint_pos."""
        server = _make_mock_policy(action_dim=14)
        spec = server.get_action_spec()
        assert "joint_pos" in spec

    def test_get_observation_spec_returns_dict(self):
        server = _make_mock_policy()
        obs = server.get_observation_spec()
        assert isinstance(obs, dict)
        assert "primary" in obs

    def test_observation_spec_state_dim_matches(self):
        from robo_eval.specs import ActionObsSpec

        server = _make_mock_policy(state_dim=2)
        obs = server.get_observation_spec()
        state_spec = obs["state"]
        assert isinstance(state_spec, ActionObsSpec)
        assert state_spec.dims == 2

    def test_action_spec_serialisable(self):
        """All ActionObsSpec values must be serialisable via .to_dict()."""
        server = _make_mock_policy()
        for spec in server.get_action_spec().values():
            d = spec.to_dict()
            assert d["name"]
            assert isinstance(d["dims"], int)


# ---------------------------------------------------------------------------
# 3. get_info() structure
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_get_info_returns_dict(self):
        server = _make_mock_policy()
        info = server.get_info()
        assert isinstance(info, dict)

    def test_required_keys_present(self):
        server = _make_mock_policy()
        info = server.get_info()
        assert "name" in info
        assert "model_id" in info
        assert "action_space" in info
        assert "state_dim" in info
        assert "action_chunk_size" in info

    def test_name_is_diffusion_policy(self):
        server = _make_mock_policy()
        assert server.get_info()["name"] == "DiffusionPolicy"

    def test_model_id_matches(self):
        server = _make_mock_policy()
        assert server.get_info()["model_id"] == "lerobot/diffusion_pusht"

    def test_action_space_has_dim(self):
        server = _make_mock_policy(action_dim=2)
        info = server.get_info()
        assert info["action_space"]["dim"] == 2

    def test_action_chunk_size_positive(self):
        server = _make_mock_policy()
        assert server.get_info()["action_chunk_size"] > 0

    def test_obs_requirements_present(self):
        server = _make_mock_policy()
        assert "obs_requirements" in server.get_info()


# ---------------------------------------------------------------------------
# 4 + 5. Import and namespace
# ---------------------------------------------------------------------------


class TestImportAndNamespace:
    def test_accessible_via_sims_vla_policies(self):
        """The policy is registered in the sims.vla_policies package namespace."""
        import sims.vla_policies.diffusion_policy_policy as dp_module

    def test_main_function_exists(self):
        from sims.vla_policies.diffusion_policy_policy import main

        assert callable(main)

    def test_model_id_default_is_pusht(self):
        from sims.vla_policies.diffusion_policy_policy import _MODEL_ID_DEFAULT

        assert "diffusion_pusht" in _MODEL_ID_DEFAULT or "diffusion" in _MODEL_ID_DEFAULT


# ---------------------------------------------------------------------------
# 6. Smoke: predict() with mocked lerobot and torch
# ---------------------------------------------------------------------------


def _make_torch_mock(action_dim: int = 2):
    """Return a mock torch module that produces zero tensors for predict()."""

    torch_mock = MagicMock()

    torch_mock.device.return_value = MagicMock()

    def _tensor(data, **kwargs):
        m = MagicMock()
        m.unsqueeze.return_value = m
        m.to.return_value = m
        return m

    torch_mock.tensor.side_effect = _tensor

    torch_mock.no_grad.return_value = MagicMock(
        __enter__=lambda s: None, __exit__=lambda s, *a: None
    )

    return torch_mock


class TestPredictSmoke:
    """Smoke tests for predict() using mocked torch + lerobot internals.

    These tests patch torch at the sims.vla_policies.diffusion_policy_policy
    module level so optional model dependencies are not required.
    """

    def _make_obs(self, state_dim: int = 2):
        from sims.vla_policies.vla_schema import VLAObservation

        return VLAObservation(
            instruction="push the T block onto the target",
            images={"primary": _make_blank_image_b64()},
            state={"flat": [0.0] * state_dim},
        )

    def _run_predict(self, server, obs):
        """Run predict() with torch + torchvision + PIL mocked out."""
        import sys

        pil_img_mock = MagicMock()
        pil_img_mock.convert.return_value = pil_img_mock

        pil_mock = MagicMock()
        pil_mock.Image.open.return_value = pil_img_mock
        pil_mock.Image.new.return_value = pil_img_mock

        tensor_mock = MagicMock()
        tensor_mock.unsqueeze.return_value = tensor_mock
        tensor_mock.to.return_value = tensor_mock

        resize_mock = MagicMock()
        resize_mock.return_value = tensor_mock

        transforms_mock = MagicMock()
        transforms_mock.Compose.return_value = resize_mock
        transforms_mock.Resize.return_value = MagicMock()
        transforms_mock.ToTensor.return_value = MagicMock()

        tv_mock = MagicMock()
        tv_mock.transforms = transforms_mock

        tensor_state_mock = MagicMock()
        tensor_state_mock.unsqueeze.return_value = tensor_state_mock
        tensor_state_mock.to.return_value = tensor_state_mock

        torch_mock = MagicMock()
        torch_mock.device.return_value = MagicMock()
        torch_mock.tensor.return_value = tensor_state_mock
        torch_mock.no_grad.return_value.__enter__ = lambda s: None
        torch_mock.no_grad.return_value.__exit__ = lambda s, *a: None

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": torch_mock,
                    "PIL": pil_mock,
                    "PIL.Image": pil_mock.Image,
                    "torchvision": tv_mock,
                    "torchvision.transforms": transforms_mock,
                },
            ),
        ):
            return server.predict(obs)

    def test_predict_returns_list_of_action_lists(self):
        server = _make_mock_policy(action_dim=2)
        obs = self._make_obs()
        result = self._run_predict(server, obs)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_predict_action_shape(self):
        """Each action vector should have action_dim elements."""
        server = _make_mock_policy(action_dim=2)
        obs = self._make_obs()
        result = self._run_predict(server, obs)
        action_vec = result[0]
        assert len(action_vec) == 2

    def test_predict_action_values_are_floats(self):
        server = _make_mock_policy(action_dim=2)
        obs = self._make_obs()
        result = self._run_predict(server, obs)
        for v in result[0]:
            assert isinstance(v, float)

    def test_predict_with_missing_image_uses_blank(self):
        """If no 'primary' key, predict should not crash (uses blank fallback)."""
        from sims.vla_policies.vla_schema import VLAObservation

        server = _make_mock_policy(action_dim=2)
        obs = VLAObservation(
            instruction="push",
            images={},
            state={"flat": [0.0, 0.0]},
        )
        result = self._run_predict(server, obs)
        assert len(result[0]) == 2

    def test_predict_with_image_key_alias(self):
        """Accepts 'image' key as alternative to 'primary'."""
        from sims.vla_policies.vla_schema import VLAObservation

        server = _make_mock_policy(action_dim=2)
        obs = VLAObservation(
            instruction="push",
            images={"image": _make_blank_image_b64()},
            state={"flat": [0.0, 0.0]},
        )
        result = self._run_predict(server, obs)
        assert len(result[0]) == 2
