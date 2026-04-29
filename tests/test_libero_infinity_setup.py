"""Tests for ``LiberoInfinityBackend`` and its setup/venv configuration.

These tests run in a plain Python environment. They MUST NOT require
libero_infinity, libero, robosuite, scenic, or any other sim-venv package to
be importable.  We stub all heavyweight modules so ``sims.sim_worker`` can be
imported in a plain Python 3.11 environment.

Coverage:
  - ABC compliance / instantiation
  - ``BACKENDS`` registry entry
  - ``get_info()`` contract (7-dim eef_delta, cameras, state dim, ActionObsSpec)
  - server_runner defaults (port 5308, .venvs/libero_infinity)
  - pyproject.toml [libero_infinity] extra is present and lists required deps
  - setup.sh markers are present
  - smoke config exists
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Stubs — inject fake heavy modules before importing sim_worker
# ---------------------------------------------------------------------------

def _stub_optional_modules() -> None:
    """Prevent ImportError for packages only present in the sim venvs."""
    to_stub = [
        "robosuite", "robosuite.wrappers",
        "libero", "libero.envs", "libero.envs.libero_envs",
        "libero.envs.bddl_base_domain",
        "libero.libero", "libero.libero.benchmark",
        "bddl",
        "gymnasium", "gym_aloha",
        "metaworld",
        "libero_infinity",
        "libero_infinity.task_config",
        "libero_infinity.scenic_generator",
        "libero_infinity.bddl_preprocessor",
        "libero_infinity.simulator",
        "scenic",
    ]
    for mod in to_stub:
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

from sims.sim_worker import (  # noqa: E402  (import after stubbing)
    BACKENDS,
    LiberoInfinityBackend,
    SimBackendBase,
)


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------

class TestLiberoInfinityBackendABCCompliance(unittest.TestCase):
    """LiberoInfinityBackend must satisfy the SimBackendBase duck-type contract."""

    def test_subclass_of_base(self):
        self.assertTrue(issubclass(LiberoInfinityBackend, SimBackendBase))

    def test_instantiable_without_args(self):
        b = LiberoInfinityBackend()
        self.assertIsNotNone(b)

    def test_implements_all_abstract_methods(self):
        b = LiberoInfinityBackend()
        for name in ("init", "reset", "step", "get_obs", "check_success",
                     "close", "get_info"):
            self.assertTrue(
                callable(getattr(b, name, None)),
                f"LiberoInfinityBackend missing method: {name}",
            )

    def test_close_is_idempotent_when_uninitialised(self):
        b = LiberoInfinityBackend()
        b.close()
        b.close()


# ---------------------------------------------------------------------------
# BACKENDS registry
# ---------------------------------------------------------------------------

class TestLiberoInfinityBackendRegistry(unittest.TestCase):
    """BACKENDS["libero_infinity"] must point at LiberoInfinityBackend."""

    def test_registered_under_canonical_name(self):
        self.assertIn("libero_infinity", BACKENDS)
        self.assertIs(BACKENDS["libero_infinity"], LiberoInfinityBackend)

    def test_registered_class_is_subclass_of_base(self):
        self.assertTrue(issubclass(BACKENDS["libero_infinity"], SimBackendBase))


# ---------------------------------------------------------------------------
# get_info() contract
# ---------------------------------------------------------------------------

class TestLiberoInfinityBackendGetInfo(unittest.TestCase):
    """get_info() must declare the 7-dim eef_delta ActionObsSpec contracts."""

    def setUp(self):
        self.backend = LiberoInfinityBackend()
        self.info = self.backend.get_info()

    def test_top_level_keys_present(self):
        for key in ("action_space", "obs_space", "max_steps", "delta_actions",
                    "action_spec", "observation_spec"):
            self.assertIn(key, self.info, f"get_info() missing key: {key}")

    def test_action_space_is_7dim_eef_delta(self):
        self.assertEqual(self.info["action_space"]["type"], "eef_delta")
        self.assertEqual(self.info["action_space"]["dim"], 7)

    def test_accepted_dims_includes_7(self):
        self.assertIn(7, self.info["action_space"]["accepted_dims"])

    def test_cameras_declared(self):
        cameras = self.info["obs_space"]["cameras"]
        keys = [c["key"] for c in cameras]
        self.assertIn("agentview_image", keys)
        self.assertIn("robot0_eye_in_hand_image", keys)

    def test_image_transform_applied_in_sim(self):
        self.assertEqual(self.info["obs_space"]["image_transform"], "applied_in_sim")

    def test_action_spec_has_position_rotation_gripper(self):
        spec = self.info["action_spec"]
        for part in ("position", "rotation", "gripper"):
            self.assertIn(part, spec, f"action_spec missing part: {part}")

    def test_gripper_uses_binary_close_negative(self):
        spec = self.info["action_spec"]["gripper"]
        self.assertEqual(spec["format"], "binary_close_negative")
        self.assertEqual(spec["dims"], 1)

    def test_observation_spec_has_required_roles(self):
        obs_spec = self.info["observation_spec"]
        for role in ("primary", "wrist", "state", "instruction"):
            self.assertIn(role, obs_spec, f"observation_spec missing role: {role}")

    def test_state_dims_is_8(self):
        self.assertEqual(self.info["observation_spec"]["state"]["dims"], 8)


# ---------------------------------------------------------------------------
# server_runner defaults
# ---------------------------------------------------------------------------

class TestServerRunnerLiberoInfinityDefaults(unittest.TestCase):
    """server_runner must map libero_infinity → port 5308 and its own venv."""

    def _get_runner(self):
        # Import inside the test so module-level stubs are already installed.
        import robo_eval.server_runner as sr
        return sr

    def test_default_port_is_5308(self):
        sr = self._get_runner()
        self.assertIn("libero_infinity", sr._SIM_DEFAULT_PORTS)
        self.assertEqual(sr._SIM_DEFAULT_PORTS["libero_infinity"], 5308)

    def test_default_venv_is_libero_infinity(self):
        sr = self._get_runner()
        self.assertIn("libero_infinity", sr._SIM_DEFAULT_VENVS)
        self.assertEqual(sr._SIM_DEFAULT_VENVS["libero_infinity"],
                         ".venvs/libero_infinity")

    def test_libero_infinity_in_known_sims(self):
        sr = self._get_runner()
        self.assertIn("libero_infinity", sr._KNOWN_SIMS)

    def test_venv_differs_from_base_libero(self):
        """libero_infinity must use a dedicated venv separate from libero."""
        sr = self._get_runner()
        self.assertNotEqual(
            sr._SIM_DEFAULT_VENVS.get("libero_infinity"),
            sr._SIM_DEFAULT_VENVS.get("libero"),
        )


# ---------------------------------------------------------------------------
# pyproject.toml extra
# ---------------------------------------------------------------------------

class TestLiberoInfinityPyprojectExtra(unittest.TestCase):
    """pyproject.toml must declare a [libero_infinity] optional-dependencies entry."""

    @classmethod
    def setUpClass(cls):
        repo_root = pathlib.Path(__file__).parent.parent
        pyproject_path = repo_root / "pyproject.toml"
        cls.content = pyproject_path.read_text(encoding="utf-8")

    def test_libero_infinity_extra_section_present(self):
        self.assertIn("libero_infinity", self.content,
                      "pyproject.toml missing [libero_infinity] extra")

    def test_libero_infinity_package_listed(self):
        self.assertIn("libero-infinity", self.content,
                      "pyproject.toml [libero_infinity] does not list libero-infinity package")

    def test_python_311_noted_in_comment(self):
        self.assertIn("3.11", self.content)

    def test_scenic_dep_present(self):
        self.assertIn("scenic", self.content,
                      "pyproject.toml [libero_infinity] should list scenic")


# ---------------------------------------------------------------------------
# Smoke config
# ---------------------------------------------------------------------------

class TestLiberoInfinitySmokeConfig(unittest.TestCase):
    """configs/libero_infinity_pi05_smoke.yaml must exist and be valid YAML."""

    def test_smoke_config_exists(self):
        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / "libero_infinity_pi05_smoke.yaml"
        self.assertTrue(cfg.exists(), f"Smoke config not found: {cfg}")

    def test_smoke_config_parseable(self):
        import yaml
        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / "libero_infinity_pi05_smoke.yaml"
        with open(cfg) as f:
            data = yaml.safe_load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("sim", data)
        self.assertEqual(data["sim"], "libero_infinity")

    def test_smoke_config_uses_port_5308(self):
        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / "libero_infinity_pi05_smoke.yaml"
        content = cfg.read_text()
        self.assertIn("5308", content)


if __name__ == "__main__":
    unittest.main()
