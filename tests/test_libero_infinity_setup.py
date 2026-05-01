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

import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

# ---------------------------------------------------------------------------
# Stubs — inject fake heavy modules before importing sim_worker
# ---------------------------------------------------------------------------


def _stub_optional_modules() -> None:
    """Prevent ImportError for packages only present in the sim venvs."""
    to_stub = [
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
        "libero_infinity",
        "libero_infinity.compiler",
        "libero_infinity.planner",
        "libero_infinity.planner.composition",
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

    composition = sys.modules["libero_infinity.planner.composition"]
    if not hasattr(composition, "AXIS_PRESETS"):
        composition.AXIS_PRESETS = {
            "combined": frozenset(
                ["position", "object", "camera", "lighting", "distractor", "background", "robot"]
            ),
            "full": frozenset(
                [
                    "position",
                    "object",
                    "camera",
                    "lighting",
                    "texture",
                    "distractor",
                    "articulation",
                    "background",
                    "robot",
                ]
            ),
        }


_stub_optional_modules()

from sims.sim_worker import (  # import after stubbing
    BACKENDS,
    LiberoInfinityBackend,
    SimBackendBase,
)
from sims.env_wrapper import SimWrapper, normalize_libero_infinity_sim_config
from roboeval.world_stubs import BaseWorldStub

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
        for name in ("init", "reset", "step", "get_obs", "check_success", "close", "get_info"):
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
        for key in (
            "action_space",
            "obs_space",
            "max_steps",
            "delta_actions",
            "action_spec",
            "observation_spec",
        ):
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
# sim_config perturbation forwarding
# ---------------------------------------------------------------------------


class TestLiberoInfinityPerturbationConfig(unittest.TestCase):
    """Perturbation values from sim_config are forwarded to the Scenic generator."""

    FULL_AXIS_STRING = "position,object,robot,camera,lighting,texture,distractor,background,articulation"
    COMBINED_AXIS_STRING = "position,object,robot,camera,lighting,distractor,background"

    def _init_with_perturbation(self, perturbation):
        seen = {}
        scenic_path = pathlib.Path(tempfile.gettempdir()) / "roboeval_li_perturb_test.scenic"
        scenic_path.write_text("# test scenic\n", encoding="utf-8")
        bddl_path = pathlib.Path(tempfile.gettempdir()) / "roboeval_li_perturb_test.bddl"
        bddl_path.write_text("(define (problem test))\n", encoding="utf-8")

        class FakeTaskConfig:
            language = "test task"

            @classmethod
            def from_bddl(cls, path):
                seen["bddl_path"] = path
                return cls()

        def fake_generate_scenic_file(cfg, perturbation=None, **kwargs):
            seen["perturbation"] = perturbation
            seen["kwargs"] = kwargs
            return str(scenic_path)

        fake_scenario = types.SimpleNamespace(generate=lambda **kwargs: (object(), None))

        task_config_mod = sys.modules["libero_infinity.task_config"]
        compiler_mod = sys.modules["libero_infinity.compiler"]
        bddl_preprocessor_mod = sys.modules["libero_infinity.bddl_preprocessor"]
        scenic_mod = sys.modules["scenic"]

        with (
            mock.patch.object(LiberoInfinityBackend, "_resolve_bddl_path", return_value=str(bddl_path)),
            mock.patch.object(LiberoInfinityBackend, "reset", return_value=(None, None)),
            mock.patch.object(task_config_mod, "TaskConfig", FakeTaskConfig, create=True),
            mock.patch.object(
                compiler_mod,
                "generate_scenic_file",
                fake_generate_scenic_file,
                create=True,
            ),
            mock.patch.object(
                bddl_preprocessor_mod,
                "parse_object_classes",
                return_value={},
                create=True,
            ),
            mock.patch.object(scenic_mod, "scenarioFromFile", return_value=fake_scenario, create=True),
        ):
            backend = LiberoInfinityBackend()
            backend.init(
                task_name="0",
                camera_resolution=128,
                suite="libero_infinity_spatial",
                sim_config={"perturbation": perturbation, "max_distractors": 3},
            )

        scenic_path.unlink(missing_ok=True)
        bddl_path.unlink(missing_ok=True)
        return seen

    def test_scalar_axis_position_is_forwarded(self):
        seen = self._init_with_perturbation("position")
        self.assertEqual(seen["perturbation"], "position")

    def test_multi_axis_list_is_forwarded(self):
        axes = ["position", "camera", "lighting"]
        seen = self._init_with_perturbation(axes)
        self.assertEqual(seen["perturbation"], "position,camera,lighting")

    def test_multi_axis_list_normalizes_without_heavy_imports(self):
        self.assertEqual(
            LiberoInfinityBackend._normalize_perturbation(["position", "camera", "lighting"]),
            "position,camera,lighting",
        )

    def test_full_axis_preset_is_forwarded(self):
        seen = self._init_with_perturbation("full")
        self.assertEqual(seen["perturbation"], self.FULL_AXIS_STRING)

    def test_full_axis_preset_normalizes_without_heavy_imports(self):
        self.assertEqual(
            LiberoInfinityBackend._normalize_perturbation("full"),
            self.FULL_AXIS_STRING,
        )

    def test_combined_legacy_preset_is_forwarded(self):
        seen = self._init_with_perturbation("combined")
        self.assertEqual(seen["perturbation"], self.COMBINED_AXIS_STRING)
        self.assertEqual(seen["kwargs"]["max_distractors"], 3)

    def test_combined_legacy_preset_normalizes_without_heavy_imports(self):
        self.assertEqual(
            LiberoInfinityBackend._normalize_perturbation("combined"),
            self.COMBINED_AXIS_STRING,
        )

    def test_unknown_axis_rejected_without_heavy_imports(self):
        with self.assertRaisesRegex(ValueError, "Unknown LIBERO-Infinity perturbation"):
            LiberoInfinityBackend._normalize_perturbation(["position", "not_an_axis"])


# ---------------------------------------------------------------------------
# wrapper-level sim_config normalization / forwarding
# ---------------------------------------------------------------------------


class TestLiberoInfinitySimConfigForwarding(unittest.TestCase):
    """SimWrapper normalizes selectors and forwards sim_config over /init."""

    def test_normalize_single_axis(self):
        cfg = normalize_libero_infinity_sim_config(
            "libero_infinity", {"perturbation": " Camera ", "seed": 7}
        )
        self.assertEqual(cfg["perturbation"], "camera")
        self.assertEqual(cfg["seed"], 7)

    def test_normalize_custom_axis_list(self):
        cfg = normalize_libero_infinity_sim_config(
            "libero_infinity", {"perturbation": ["position", "lighting", "distractor"]}
        )
        self.assertEqual(cfg["perturbation"], ["position", "lighting", "distractor"])

    def test_normalize_all_axes_alias(self):
        cfg = normalize_libero_infinity_sim_config(
            "libero_infinity", {"perturbation": "all_axes"}
        )
        self.assertEqual(cfg["perturbation"], "full")

    def test_forwards_normalized_sim_config_to_init_without_sim_deps(self):
        captured = {}

        def fake_post(wrapper, path, json_data=None):
            captured[path] = json_data
            return {"success": True, "task_description": "demo task"}

        def fake_fetch_policy_info(wrapper):
            wrapper._policy_info = {"model_id": "fake"}
            wrapper._policy_action_space = {"type": "eef_delta", "dim": 7}

        def fake_fetch_sim_info(wrapper):
            wrapper._sim_info = {
                "action_space": {"type": "eef_delta", "dim": 7},
                "obs_space": {"cameras": [], "state": {"dim": 8}},
            }
            wrapper._sim_action_space = wrapper._sim_info["action_space"]

        with (
            mock.patch.object(SimWrapper, "_post", fake_post),
            mock.patch.object(SimWrapper, "_fetch_policy_info", fake_fetch_policy_info),
            mock.patch.object(SimWrapper, "_fetch_sim_info", fake_fetch_sim_info),
            mock.patch.object(SimWrapper, "_negotiate_spaces", return_value=None),
            mock.patch.object(SimWrapper, "_negotiate_obs", return_value=None),
            mock.patch.object(SimWrapper, "_validate_specs", return_value=None),
            mock.patch.object(SimWrapper, "_get_obs_image", return_value=None),
            mock.patch.object(BaseWorldStub, "__init__", return_value=None),
        ):
            SimWrapper(
                sim_server_url="http://sim.example",
                sim_name="libero_infinity",
                task_name="0",
                suite="libero_infinity_spatial",
                no_vlm=True,
                sim_config={"perturbation": "all_axes", "seed": 123},
            )

        self.assertEqual(captured["/init"]["sim_config"]["perturbation"], "full")
        self.assertEqual(captured["/init"]["sim_config"]["seed"], 123)


# ---------------------------------------------------------------------------
# server_runner defaults
# ---------------------------------------------------------------------------


class TestServerRunnerLiberoInfinityDefaults(unittest.TestCase):
    """server_runner must map libero_infinity → port 5308 and its own venv."""

    def _get_runner(self):
        # Import inside the test so module-level stubs are already installed.
        import roboeval.server_runner as sr

        return sr

    def test_default_port_is_5308(self):
        sr = self._get_runner()
        self.assertIn("libero_infinity", sr._SIM_DEFAULT_PORTS)
        self.assertEqual(sr._SIM_DEFAULT_PORTS["libero_infinity"], 5308)

    def test_default_venv_is_libero_infinity(self):
        sr = self._get_runner()
        self.assertIn("libero_infinity", sr._SIM_DEFAULT_VENVS)
        self.assertEqual(sr._SIM_DEFAULT_VENVS["libero_infinity"], ".venvs/libero_infinity")

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
        self.assertIn(
            "libero_infinity", self.content, "pyproject.toml missing [libero_infinity] extra"
        )

    def test_libero_infinity_package_listed(self):
        self.assertIn(
            "libero-infinity",
            self.content,
            "pyproject.toml [libero_infinity] does not list libero-infinity package",
        )

    def test_python_311_noted_in_comment(self):
        self.assertIn("3.11", self.content)

    def test_scenic_dep_present(self):
        self.assertIn("scenic", self.content, "pyproject.toml [libero_infinity] should list scenic")


# ---------------------------------------------------------------------------
# Smoke config
# ---------------------------------------------------------------------------


class TestLiberoInfinitySmokeConfig(unittest.TestCase):
    """LIBERO-Infinity smoke/example configs must exist and be valid YAML."""

    CONFIGS = {
        "libero_infinity_pi05_smoke.yaml": "camera",
        "libero_infinity_pi05_liten_smoke.yaml": [
            "position",
            "lighting",
            "distractor",
        ],
        "libero_infinity_pi05_position_perturb_smoke.yaml": "position",
        "libero_infinity_pi05_multi_axis_perturb_smoke.yaml": [
            "position",
            "lighting",
            "distractor",
        ],
        "libero_infinity_pi05_full_perturb_smoke.yaml": "full",
    }

    def _load_config(self, name):
        import yaml

        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / name
        with open(cfg, encoding="utf-8") as f:
            return cfg, yaml.safe_load(f)

    def test_smoke_config_exists(self):
        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / "libero_infinity_pi05_smoke.yaml"
        self.assertTrue(cfg.exists(), f"Smoke config not found: {cfg}")

    def test_configs_parseable(self):
        for name in self.CONFIGS:
            with self.subTest(config=name):
                _cfg, data = self._load_config(name)
                self.assertIsInstance(data, dict)
                self.assertIn("sim", data)
                self.assertEqual(data["sim"], "libero_infinity")

    def test_smoke_config_uses_port_5308(self):
        repo_root = pathlib.Path(__file__).parent.parent
        cfg = repo_root / "configs" / "libero_infinity_pi05_smoke.yaml"
        content = cfg.read_text()
        self.assertIn("5308", content)

    def test_perturbation_example_configs_use_expected_shapes(self):
        for name, expected in self.CONFIGS.items():
            with self.subTest(config=name):
                _cfg, data = self._load_config(name)
                self.assertEqual(data["sim_config"]["perturbation"], expected)

    def test_examples_cover_suite_and_task_subset_selection(self):
        _cfg, spatial = self._load_config("libero_infinity_pi05_smoke.yaml")
        self.assertEqual(spatial["suite"], "libero_infinity_spatial")
        self.assertEqual(spatial["task"], "")
        self.assertEqual(spatial["max_tasks"], 1)

        _cfg, multi_suite = self._load_config("libero_infinity_pi05_liten_smoke.yaml")
        self.assertEqual(multi_suite["suite"], "libero_infinity_spatial,libero_infinity_goal")
        self.assertEqual(multi_suite["max_tasks"], 2)

        _cfg, full = self._load_config("libero_infinity_pi05_full_perturb_smoke.yaml")
        self.assertEqual(full["suite"], "libero_infinity_goal")
        self.assertEqual(full["task"], "bowl")
        self.assertEqual(full["max_tasks"], 2)


# ---------------------------------------------------------------------------
# Docs coverage
# ---------------------------------------------------------------------------


class TestLiberoInfinityDocs(unittest.TestCase):
    """Docs must mention full-axis support and subset selection."""

    @classmethod
    def setUpClass(cls):
        repo_root = pathlib.Path(__file__).parent.parent
        cls.libero_doc = (repo_root / "docs" / "libero_infinity.md").read_text(
            encoding="utf-8"
        )
        cls.pairs_doc = (repo_root / "docs" / "supported_pairs.md").read_text(
            encoding="utf-8"
        )

    def test_libero_doc_lists_all_nine_axes(self):
        for axis in (
            "position",
            "object",
            "robot",
            "camera",
            "lighting",
            "texture",
            "distractor",
            "background",
            "articulation",
        ):
            self.assertIn(f"`{axis}`", self.libero_doc)

    def test_docs_mention_full_axis_and_subset_support(self):
        for text in (self.libero_doc, self.pairs_doc):
            normalized = " ".join(text.split())
            self.assertIn("full", normalized)
            self.assertIn("all nine axes", normalized)
            self.assertIn("not forced to run all four", normalized)


if __name__ == "__main__":
    unittest.main()
