"""Tests for ``AlohaGymBackend`` (gym-aloha bimanual sim).

These tests run without requiring gym-aloha itself to be importable.
We exercise the parts of the backend that don't touch a live env: ABC
compliance, ``BACKENDS`` registry, default ``get_info()`` shape (with
ActionObsSpec block), task-name resolution, and image extraction behaviour on
a synthetic obs dict.
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _stub_optional_modules() -> None:
    """Stub heavyweight imports so ``sims.sim_worker`` imports cleanly here.

    The test environment has core dependencies but may not include robosuite,
    libero, gym_aloha, etc.
    """
    for mod in [
        "robosuite", "robosuite.wrappers",
        "libero", "libero.envs", "libero.envs.libero_envs",
        "libero.envs.bddl_base_domain",
        "libero.libero", "libero.libero.benchmark",
        "bddl",
        "gymnasium", "gym_aloha",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    rw = sys.modules["robosuite.wrappers"]
    if not hasattr(rw, "GymWrapper"):
        rw.GymWrapper = object
    bm = sys.modules["libero.libero.benchmark"]
    if not hasattr(bm, "get_benchmark_dict"):
        bm.get_benchmark_dict = lambda: {}
    envs = sys.modules["libero.envs"]
    if not hasattr(envs, "OffScreenRenderEnv"):
        envs.OffScreenRenderEnv = object


_stub_optional_modules()

from sims.sim_worker import (  # noqa: E402  (after stubbing)
    BACKENDS,
    AlohaGymBackend,
    SimBackendBase,
)


class TestAlohaGymBackendABCCompliance(unittest.TestCase):
    """``AlohaGymBackend`` must satisfy the ``SimBackendBase`` ABC contract."""

    def test_subclass_of_base(self):
        self.assertTrue(issubclass(AlohaGymBackend, SimBackendBase))

    def test_instantiable_without_args(self):
        b = AlohaGymBackend()
        self.assertIsNone(b.env)
        self.assertEqual(b._task_id, "")

    def test_implements_all_abstract_methods(self):
        """No abstract methods leak through."""
        b = AlohaGymBackend()
        for name in ("init", "reset", "step", "get_obs", "check_success",
                     "close", "get_info"):
            self.assertTrue(callable(getattr(b, name)),
                            f"missing method: {name}")
        # Instantiation would have raised TypeError if any abstract method
        # were left unimplemented; reaching here confirms ABC compliance.

    def test_close_is_idempotent_when_uninitialised(self):
        b = AlohaGymBackend()
        # Must not raise even though env was never created.
        b.close()
        b.close()


class TestAlohaGymBackendsRegistry(unittest.TestCase):
    """``BACKENDS["aloha_gym"]`` must point at ``AlohaGymBackend``."""

    def test_registered_under_canonical_name(self):
        self.assertIn("aloha_gym", BACKENDS)
        self.assertIs(BACKENDS["aloha_gym"], AlohaGymBackend)

    def test_registered_class_is_subclass_of_base(self):
        self.assertTrue(issubclass(BACKENDS["aloha_gym"], SimBackendBase))


class TestAlohaGymBackendGetInfo(unittest.TestCase):
    """``get_info()`` must declare ActionObsSpec contracts."""

    def setUp(self):
        self.backend = AlohaGymBackend()

    def test_info_top_level_keys(self):
        info = self.backend.get_info()
        for key in ("action_space", "obs_space", "max_steps", "delta_actions",
                    "action_spec", "observation_spec"):
            self.assertIn(key, info, f"get_info() missing key: {key}")

    def test_action_space_is_14dim_joint_pos(self):
        info = self.backend.get_info()
        self.assertEqual(info["action_space"]["type"], "joint_pos")
        self.assertEqual(info["action_space"]["dim"], 14)
        # Must accept 32-dim padded chunks too.
        self.assertIn(32, info["action_space"]["accepted_dims"])

    def test_action_spec_declared_and_well_formed(self):
        spec = self.backend.get_info()["action_spec"]
        self.assertIn("joint_pos", spec)
        joint = spec["joint_pos"]
        self.assertEqual(joint["dims"], 14)
        self.assertEqual(joint["format"], "absolute_joint_positions")
        # ``accepts`` must be present so the spec gate can validate emitted
        # action formats.
        self.assertIn("accepts", joint)
        self.assertIn("absolute_joint_positions", joint["accepts"])

    def test_observation_spec_declares_state_and_images(self):
        spec = self.backend.get_info()["observation_spec"]
        for key in ("primary", "wrist", "state", "instruction"):
            self.assertIn(key, spec, f"obs_spec missing role: {key}")
        self.assertEqual(spec["state"]["dims"], 14)

    def test_image_transform_is_none(self):
        """gym-aloha images are already upright; no transform is applied."""
        info = self.backend.get_info()
        self.assertEqual(info["obs_space"]["image_transform"], "none")

    def test_delta_actions_false(self):
        # gym-aloha is absolute-joint, never delta.
        self.assertFalse(self.backend.get_info()["delta_actions"])


class TestAlohaGymBackendTaskResolution(unittest.TestCase):
    """``_resolve_task`` must accept gym ids, short forms, and indices."""

    def setUp(self):
        self.backend = AlohaGymBackend()

    def test_full_gym_id(self):
        self.assertEqual(
            self.backend._resolve_task("AlohaTransferCube-v0"),
            "AlohaTransferCube-v0",
        )

    def test_short_form_transfer_cube(self):
        self.assertEqual(
            self.backend._resolve_task("transfer_cube"),
            "AlohaTransferCube-v0",
        )

    def test_short_form_insertion(self):
        self.assertEqual(
            self.backend._resolve_task("insertion"),
            "AlohaInsertion-v0",
        )

    def test_numeric_index_zero(self):
        # Index 0 must resolve to the first registered task; we don't pin the
        # ordering here (TASKS list is the source of truth) but we do verify
        # that _resolve_task accepts numeric indices at all.
        resolved = self.backend._resolve_task("0")
        self.assertIn(resolved, AlohaGymBackend.TASKS)

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            self.backend._resolve_task("nonexistent_task")

    def test_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.backend._resolve_task("999")


class TestAlohaGymBackendImageExtraction(unittest.TestCase):
    """``_extract_image`` must return gym-aloha images without flipping them."""

    def setUp(self):
        self.backend = AlohaGymBackend()

    def _synthetic_obs(self, h: int = 8, w: int = 8) -> dict:
        top = np.zeros((h, w, 3), dtype=np.uint8)
        top[0, :, 0] = 200          # gradient on row 0
        angle = np.zeros((h, w, 3), dtype=np.uint8)
        angle[0, :, 1] = 100
        return {
            "pixels": {"top": top, "angle": angle},
            "agent_pos": np.zeros(14, dtype=np.float32),
        }

    def test_primary_not_flipped(self):
        obs = self._synthetic_obs()
        raw = obs["pixels"]["top"].copy()
        primary, _wrist = self.backend._extract_image(obs)
        np.testing.assert_array_equal(primary, raw)

    def test_wrist_not_flipped(self):
        obs = self._synthetic_obs()
        raw = obs["pixels"]["angle"].copy()
        _primary, wrist = self.backend._extract_image(obs)
        np.testing.assert_array_equal(wrist, raw)

    def test_extract_state_returns_14dim(self):
        obs = self._synthetic_obs()
        state = self.backend._extract_state(obs)
        self.assertEqual(len(state), 14)

    def test_empty_obs_returns_black_placeholder(self):
        primary, wrist = self.backend._extract_image({})
        self.assertIsNotNone(primary)
        self.assertEqual(primary.shape, (480, 480, 3))
        self.assertEqual(primary.dtype, np.uint8)
        self.assertFalse(primary.any())
        self.assertIsNone(wrist)


class TestAlohaGymBackendCheckSuccess(unittest.TestCase):
    """``check_success`` must reflect the cached last reward."""

    def test_success_below_threshold(self):
        b = AlohaGymBackend()
        b._last_reward = 3.0
        self.assertFalse(b.check_success())

    def test_success_at_threshold(self):
        b = AlohaGymBackend()
        b._last_reward = 4.0
        self.assertTrue(b.check_success())

    def test_success_above_threshold(self):
        b = AlohaGymBackend()
        b._last_reward = 4.5
        self.assertTrue(b.check_success())


if __name__ == "__main__":
    unittest.main()
