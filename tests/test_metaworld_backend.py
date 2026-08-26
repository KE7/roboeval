"""Tests for ``MetaWorldBackend`` (Meta-World ~50 single-arm manipulation tasks).

These tests must not require metaworld itself to be importable.
They exercise the parts of the backend that don't touch a live env: ABC
compliance, ``BACKENDS`` registry, default ``get_info()`` shape (including
ActionObsSpec block), task-name resolution, and the spec-gate-blocked verdict
that the orchestrator must produce when pairing a 7-dim VLA with this 4-dim sim.
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


def _stub_optional_modules() -> None:
    """Stub heavyweight imports so ``sims.sim_worker`` imports cleanly here.

    The default test environment may not include metaworld, robosuite,
    libero, gym_aloha, etc.  Stub those optional packages so this file can
    import the backend module and exercise metadata-only behavior.
    """
    for mod in [
        "robosuite",
        "robosuite.wrappers",
        "libero",
        "libero.envs",
        "libero.envs.libero_envs",
        "libero.envs.bddl_base_domain",
        "libero.libero",
        "libero.libero.benchmark",
        "bddl",
        "gymnasium",
        "gym_aloha",
        "metaworld",
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

from sims.sim_worker import (  # after stubbing
    _MW_TASK_DESCRIPTIONS,
    BACKENDS,
    MetaWorldBackend,
    SimBackendBase,
)


class TestMetaWorldBackendABCCompliance(unittest.TestCase):
    """``MetaWorldBackend`` must satisfy the ``SimBackendBase`` ABC contract."""

    def test_subclass_of_base(self):
        self.assertTrue(issubclass(MetaWorldBackend, SimBackendBase))

    def test_instantiable_without_args(self):
        b = MetaWorldBackend()
        self.assertIsNone(b.env)
        self.assertEqual(b._task_name, "")

    def test_implements_all_abstract_methods(self):
        """No abstract methods leak through."""
        b = MetaWorldBackend()
        for name in ("init", "reset", "step", "get_obs", "check_success", "close", "get_info"):
            self.assertTrue(callable(getattr(b, name)), f"missing method: {name}")

    def test_close_is_idempotent_when_uninitialised(self):
        b = MetaWorldBackend()
        b.close()
        b.close()


class TestMetaWorldBackendRegistry(unittest.TestCase):
    """``BACKENDS["metaworld"]`` must point at ``MetaWorldBackend``."""

    def test_registered_under_canonical_name(self):
        self.assertIn("metaworld", BACKENDS)
        self.assertIs(BACKENDS["metaworld"], MetaWorldBackend)

    def test_registered_class_is_subclass_of_base(self):
        self.assertTrue(issubclass(BACKENDS["metaworld"], SimBackendBase))


class TestMetaWorldBackendGetInfo(unittest.TestCase):
    """``get_info()`` must declare the 4-dim ActionObsSpec contracts."""

    def setUp(self):
        self.backend = MetaWorldBackend()

    def test_info_top_level_keys(self):
        info = self.backend.get_info()
        for key in (
            "action_space",
            "obs_space",
            "max_steps",
            "delta_actions",
            "action_spec",
            "observation_spec",
        ):
            self.assertIn(key, info, f"get_info() missing key: {key}")

    def test_action_space_is_4dim_eef_delta(self):
        info = self.backend.get_info()
        self.assertEqual(info["action_space"]["type"], "eef_delta")
        self.assertEqual(info["action_space"]["dim"], 4)

    def test_accepted_dims_excludes_7(self):
        """4-dim spec must NOT accept 7 — that is what triggers the spec gate."""
        info = self.backend.get_info()
        self.assertNotIn(7, info["action_space"]["accepted_dims"])
        self.assertIn(4, info["action_space"]["accepted_dims"])

    def test_action_spec_declared_4dim(self):
        spec = self.backend.get_info()["action_spec"]
        self.assertIn("eef_delta", spec)
        eef = spec["eef_delta"]
        self.assertEqual(eef["dims"], 4)
        self.assertEqual(eef["format"], "eef_delta_xyz_gripper")
        self.assertIn("accepts", eef)
        self.assertIn("eef_delta_xyz_gripper", eef["accepts"])

    def test_observation_spec_declares_cameras_and_state(self):
        spec = self.backend.get_info()["observation_spec"]
        for key in ("primary", "wrist", "state", "instruction"):
            self.assertIn(key, spec, f"obs_spec missing role: {key}")
        self.assertEqual(spec["state"]["dims"], 39)

    def test_image_transform_is_applied_in_sim(self):
        info = self.backend.get_info()
        self.assertEqual(info["obs_space"]["image_transform"], "applied_in_sim")

    def test_delta_actions_true(self):
        # MetaWorld uses end-effector position delta control.
        self.assertTrue(self.backend.get_info()["delta_actions"])

    def test_cameras_declared_as_corner_and_behindgripper(self):
        cameras = self.backend.get_info()["obs_space"]["cameras"]
        keys = [c["key"] for c in cameras]
        self.assertIn("corner", keys)
        self.assertIn("behindGripper", keys)


class TestMetaWorldBackendTaskResolution(unittest.TestCase):
    """``_resolve_task`` must accept MT50 names, short forms, and indices."""

    def setUp(self):
        self.backend = MetaWorldBackend()

    def test_canonical_name(self):
        self.assertEqual(
            self.backend._resolve_task("button-press-v3"),
            "button-press-v3",
        )

    def test_underscore_form(self):
        # 'button_press_v3' normalises to 'button-press-v3'.
        resolved = self.backend._resolve_task("button_press_v3")
        self.assertEqual(resolved, "button-press-v3")

    def test_numeric_index_zero(self):
        resolved = self.backend._resolve_task("0")
        self.assertIn(resolved, MetaWorldBackend.TASKS)

    def test_numeric_index_last(self):
        last_idx = str(len(MetaWorldBackend.TASKS) - 1)
        resolved = self.backend._resolve_task(last_idx)
        self.assertIn(resolved, MetaWorldBackend.TASKS)

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            self.backend._resolve_task("nonexistent_task_xyz")

    def test_index_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.backend._resolve_task("999")

    def test_tasks_list_has_50_entries(self):
        # Meta-World MT50 has exactly 50 tasks.
        self.assertEqual(len(MetaWorldBackend.TASKS), 50)

    def test_all_tasks_have_descriptions(self):
        """Every task in TASKS must have an entry in _MW_TASK_DESCRIPTIONS."""
        for task in MetaWorldBackend.TASKS:
            self.assertIn(
                task, _MW_TASK_DESCRIPTIONS, f"_MW_TASK_DESCRIPTIONS missing entry for task: {task}"
            )


class TestMetaWorldBackendSpecGateContract(unittest.TestCase):
    """Given a 4-dim sim spec, 7-dim VLA actions are rejected by the spec gate.

    The orchestrator's ActionObsSpec gate blocks a VLA/sim pairing when the
    VLA's declared action dims are not in the sim's ``accepted_dims``.  This
    test suite documents and verifies that contract from the sim side.
    """

    def setUp(self):
        self.backend = MetaWorldBackend()
        self.info = self.backend.get_info()

    def test_action_spec_dims_is_4(self):
        eef = self.info["action_spec"]["eef_delta"]
        self.assertEqual(eef["dims"], 4, "Changing dims from 4 breaks the spec-gate-blocked test.")

    def test_accepted_dims_does_not_contain_7(self):
        """If 7 is added to accepted_dims, the spec gate won't fire for 7-dim VLAs."""
        accepted = self.info["action_space"]["accepted_dims"]
        self.assertNotIn(
            7, accepted, "accepted_dims must NOT include 7 in v0.1 — adding it bypasses the gate."
        )

    def test_action_space_dim_is_4(self):
        self.assertEqual(self.info["action_space"]["dim"], 4)


class TestMetaWorldBackendStepActionAdaptation(unittest.TestCase):
    """``step()`` 7→4 truncation must work without a live env (unit test).

    The backend performs defensive 7→4 adaptation in step().  This test
    verifies the mapping logic by patching out the env so we can test the
    adaptation in isolation.
    """

    def setUp(self):
        self.backend = MetaWorldBackend()
        # Install a mock env so step() can call self.env.step()
        self._actions_received = []

        class _MockEnv:
            def step(inner_self, action):  # noqa: N805
                self._actions_received.append(action.copy())
                fake_obs = np.zeros(39, dtype=np.float32)
                return fake_obs, 0.0, False, False, {"success": False}

        self.backend.env = _MockEnv()
        # Suppress renderer calls
        self.backend._mj_renderer = None
        self.backend._mj_renderer_wrist = None

    def _step(self, action):
        """Call backend.step() and return the action forwarded to the env."""
        self._actions_received.clear()
        self.backend.step(action)
        return self._actions_received[0]

    def test_4dim_passthrough(self):
        action_in = [0.1, 0.2, 0.3, 0.9]
        forwarded = self._step(action_in)
        np.testing.assert_allclose(forwarded, [0.1, 0.2, 0.3, 0.9], atol=1e-5)

    def test_7dim_truncation_keeps_xyz_and_gripper(self):
        # 7-dim: [dx, dy, dz, drx, dry, drz, gripper]
        action_in = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
        forwarded = self._step(action_in)
        # Expected: [dx=0.1, dy=0.2, dz=0.3, gripper=0.9]
        np.testing.assert_allclose(forwarded, [0.1, 0.2, 0.3, 0.9], atol=1e-5)

    def test_3dim_padded_to_4(self):
        action_in = [0.1, 0.2, 0.3]
        forwarded = self._step(action_in)
        self.assertEqual(len(forwarded), 4)
        np.testing.assert_allclose(forwarded[:3], [0.1, 0.2, 0.3], atol=1e-5)

    def test_step_returns_5tuple(self):
        result = self.backend.step([0.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(result), 5, "step() must return (img, img2, reward, done, info)")


class TestMetaWorldBackendCheckSuccess(unittest.TestCase):
    """``check_success`` must reflect the cached last success flag."""

    def test_initially_false(self):
        b = MetaWorldBackend()
        self.assertFalse(b.check_success())

    def test_true_when_set(self):
        b = MetaWorldBackend()
        b._last_success = True
        self.assertTrue(b.check_success())


class TestMetaWorldTaskDescriptions(unittest.TestCase):
    """``_MW_TASK_DESCRIPTIONS`` sanity checks."""

    def test_has_50_entries(self):
        self.assertEqual(len(_MW_TASK_DESCRIPTIONS), 50)

    def test_all_keys_end_with_v3(self):
        for name in _MW_TASK_DESCRIPTIONS:
            self.assertTrue(name.endswith("-v3"), f"Unexpected task name format: {name}")

    def test_all_descriptions_non_empty(self):
        for name, desc in _MW_TASK_DESCRIPTIONS.items():
            self.assertTrue(desc.strip(), f"Empty description for task: {name}")


if __name__ == "__main__":
    unittest.main()
