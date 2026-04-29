"""Tests for robo_eval.specs — ActionObsSpec interface specification utilities."""

from __future__ import annotations

import numpy as np
import pytest

from robo_eval.specs import (
    GRIPPER_CLOSE_NEG,
    GRIPPER_CLOSE_POS,
    HARD,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_ABS,
    POSITION_DELTA,
    ROTATION_AA,
    ROTATION_EULER,
    ROTATION_EULER_ACCEPTS_AA,
    ROTATION_QUAT,
    STATE_EEF_POS_AA_GRIP,
    WARN,
    ActionObsSpec,
    check_specs,
)

# ---------------------------------------------------------------------------
# ActionObsSpec construction
# ---------------------------------------------------------------------------


class TestActionObsSpecConstruction:
    def test_basic_fields(self):
        spec = ActionObsSpec("position", 3, "delta_xyz", (-1.0, 1.0))
        assert spec.name == "position"
        assert spec.dims == 3
        assert spec.format == "delta_xyz"
        assert spec.range == (-1.0, 1.0)
        assert spec.accepts is None
        assert spec.description == ""

    def test_with_accepts(self):
        spec = ActionObsSpec(
            "rotation", 3, "euler_xyz", accepts=frozenset({"euler_xyz", "axis_angle"})
        )
        assert spec.accepts == frozenset({"euler_xyz", "axis_angle"})

    def test_frozen(self):
        """ActionObsSpec is frozen — mutation must raise."""
        spec = ActionObsSpec("pos", 3, "delta_xyz")
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "other"  # type: ignore[misc]

    def test_zero_dims_image(self):
        assert IMAGE_RGB.dims == 0
        assert IMAGE_RGB.format == "rgb_hwc_uint8"

    def test_language_constant(self):
        assert LANGUAGE.dims == 0

    def test_predefined_constants_importable(self):
        """Smoke-test that all canonical constants exist and are ActionObsSpec instances."""
        for name, obj in [
            ("POSITION_DELTA", POSITION_DELTA),
            ("POSITION_ABS", POSITION_ABS),
            ("ROTATION_AA", ROTATION_AA),
            ("ROTATION_EULER", ROTATION_EULER),
            ("ROTATION_QUAT", ROTATION_QUAT),
            ("GRIPPER_CLOSE_POS", GRIPPER_CLOSE_POS),
            ("GRIPPER_CLOSE_NEG", GRIPPER_CLOSE_NEG),
            ("IMAGE_RGB", IMAGE_RGB),
            ("LANGUAGE", LANGUAGE),
            ("STATE_EEF_POS_AA_GRIP", STATE_EEF_POS_AA_GRIP),
        ]:
            assert isinstance(obj, ActionObsSpec), f"{name} is not a ActionObsSpec"


# ---------------------------------------------------------------------------
# ActionObsSpec.validate()
# ---------------------------------------------------------------------------


class TestActionObsSpecValidate:
    def test_valid_in_range(self):
        errors = POSITION_DELTA.validate(np.array([0.0, 0.5, -0.5]))
        assert errors == []

    def test_out_of_range(self):
        errors = POSITION_DELTA.validate(np.array([2.0, 0.0, 0.0]))
        assert len(errors) == 1
        assert "outside" in errors[0]

    def test_nan_detected(self):
        errors = POSITION_DELTA.validate(np.array([float("nan"), 0.0, 0.0]))
        assert any("NaN" in e or "nan" in e.lower() for e in errors)

    def test_inf_detected(self):
        errors = POSITION_DELTA.validate(np.array([0.0, float("inf"), 0.0]))
        assert any(
            "NaN" in e or "Inf" in e or "nan" in e.lower() or "inf" in e.lower() for e in errors
        )

    def test_too_few_dims(self):
        errors = POSITION_DELTA.validate(np.array([0.0, 0.5]))  # only 2D, expects 3
        assert any("2D" in e or "2" in e for e in errors)

    def test_no_range_no_error(self):
        """Spec without a range constraint should not flag any value."""
        spec = ActionObsSpec("raw", 3, "raw")
        errors = spec.validate(np.array([999.0, -999.0, 0.0]))
        assert errors == []

    def test_zero_dims_skips_range_check(self):
        """dims=0 specs (images, language) skip range validation."""
        errors = IMAGE_RGB.validate(np.zeros(10))  # irrelevant data
        assert errors == []

    def test_boundary_values_pass(self):
        """Values at the boundary (within tolerance) should pass."""
        errors = POSITION_DELTA.validate(np.array([1.0, -1.0, 0.5]))
        assert errors == []

    def test_at_tolerance_boundary(self):
        """Values just within the 0.01 tolerance should not raise an error."""
        errors = POSITION_DELTA.validate(np.array([1.005, 0.0, 0.0]))
        assert errors == []

    def test_just_over_tolerance_fails(self):
        errors = POSITION_DELTA.validate(np.array([1.02, 0.0, 0.0]))
        assert len(errors) == 1


# ---------------------------------------------------------------------------
# ActionObsSpec.to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------


class TestActionObsSpecRoundTrip:
    def _roundtrip(self, spec: ActionObsSpec) -> ActionObsSpec:
        return ActionObsSpec.from_dict(spec.to_dict())

    def test_basic(self):
        spec = ActionObsSpec("position", 3, "delta_xyz", (-1.0, 1.0))
        rt = self._roundtrip(spec)
        assert rt == spec

    def test_with_accepts(self):
        spec = ActionObsSpec(
            "rotation",
            3,
            "euler_xyz",
            (-3.15, 3.15),
            accepts=frozenset({"euler_xyz", "axis_angle"}),
        )
        rt = self._roundtrip(spec)
        assert rt == spec
        assert isinstance(rt.accepts, frozenset)
        assert "euler_xyz" in rt.accepts
        assert "axis_angle" in rt.accepts

    def test_without_range(self):
        spec = ActionObsSpec("raw", 0, "raw")
        rt = self._roundtrip(spec)
        assert rt.range is None
        assert rt == spec

    def test_with_description(self):
        spec = ActionObsSpec(
            "gripper", 1, "binary_close_positive", (-1.0, 1.0), description="pi05 convention"
        )
        rt = self._roundtrip(spec)
        assert rt.description == "pi05 convention"
        assert rt == spec

    def test_accepts_is_frozenset(self):
        """from_dict must restore accepts as frozenset, not list."""
        spec = ROTATION_EULER_ACCEPTS_AA
        d = spec.to_dict()
        assert isinstance(d["accepts"], list)  # wire format is list
        rt = ActionObsSpec.from_dict(d)
        assert isinstance(rt.accepts, frozenset)

    def test_no_accepts_key_if_none(self):
        spec = POSITION_DELTA
        d = spec.to_dict()
        assert "accepts" not in d

    def test_no_range_key_if_none(self):
        spec = ActionObsSpec("lang", 0, "language")
        d = spec.to_dict()
        assert "range" not in d

    def test_predefined_constants_roundtrip(self):
        for spec in [POSITION_DELTA, ROTATION_AA, ROTATION_QUAT, GRIPPER_CLOSE_POS, IMAGE_RGB]:
            assert self._roundtrip(spec) == spec


# ---------------------------------------------------------------------------
# is_compatible() / accepts logic
# ---------------------------------------------------------------------------


class TestIsCompatible:
    def test_exact_match(self):
        ok, msg = ROTATION_AA.is_compatible(ROTATION_AA)
        assert ok
        assert msg == ""

    def test_format_mismatch_no_accepts(self):
        ok, msg = ROTATION_AA.is_compatible(ROTATION_EULER)
        assert not ok
        assert "axis_angle" in msg

    def test_dims_mismatch(self):
        prod = ActionObsSpec("rotation", 3, "axis_angle")
        cons = ActionObsSpec("rotation", 4, "axis_angle")
        ok, msg = prod.is_compatible(cons)
        assert not ok
        assert "3D" in msg or "4D" in msg

    def test_accepts_allows_compatible_format(self):
        """ROTATION_EULER_ACCEPTS_AA should accept axis_angle producer."""
        ok, msg = ROTATION_AA.is_compatible(ROTATION_EULER_ACCEPTS_AA)
        assert ok, msg

    def test_accepts_rejects_unknown_format(self):
        prod = ActionObsSpec("rotation", 4, "quaternion_xyzw")
        ok, msg = prod.is_compatible(ROTATION_EULER_ACCEPTS_AA)
        assert not ok

    def test_zero_dims_skips_dim_check(self):
        """dims=0 on either side should not trigger a dims mismatch."""
        prod = ActionObsSpec("image", 0, "rgb_hwc_uint8")
        cons = ActionObsSpec("image", 0, "rgb_hwc_uint8")
        ok, _ = prod.is_compatible(cons)
        assert ok


# ---------------------------------------------------------------------------
# check_specs() — severity classification
# ---------------------------------------------------------------------------


class TestCheckSpecs:
    """Unit tests for check_specs() severity rules."""

    # Helpers
    @staticmethod
    def _severities(results):
        return [sev for sev, _ in results]

    @staticmethod
    def _messages(results):
        return [msg for _, msg in results]

    # ── HARD: missing required component ─────────────────────────────────

    def test_hard_missing_required_obs_component(self):
        """Server requires wrist image, sim doesn't provide it → HARD."""
        server_obs = {"wrist": ActionObsSpec("wrist", 0, "rgb_hwc_uint8")}
        bench_obs = {}  # sim doesn't provide wrist
        results = check_specs({}, {}, server_obs, bench_obs)
        sevs = self._severities(results)
        assert HARD in sevs

    def test_hard_missing_action_component_benchmark_expects(self):
        """Benchmark expects 'position', server declares other keys but not 'position' → HARD."""
        # Server declares something (rotation) but NOT position — so check fires
        server_action = {"rotation": ROTATION_AA}
        bench_action = {"position": POSITION_DELTA, "rotation": ROTATION_AA}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert HARD in sevs

    def test_hard_action_dims_mismatch(self):
        """Server returns 4-dim position, sim expects 3-dim → HARD."""
        server_action = {"position": ActionObsSpec("position", 4, "delta_xyz")}
        bench_action = {"position": ActionObsSpec("position", 3, "delta_xyz")}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert HARD in sevs

    def test_hard_format_mismatch_no_accepts(self):
        """Server produces axis_angle, sim expects euler_xyz (no accepts) → HARD."""
        server_action = {"rotation": ROTATION_AA}
        bench_action = {"rotation": ROTATION_EULER}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert HARD in sevs

    def test_hard_no_overlapping_action_keys(self):
        """Server and benchmark have completely disjoint action keys → HARD."""
        server_action = {"x": ActionObsSpec("x", 3, "fmt_a")}
        bench_action = {"y": ActionObsSpec("y", 3, "fmt_b")}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert HARD in sevs

    # ── HARD: image_transform mismatch (via observation spec) ────────────

    def test_hard_image_transform_mismatch(self):
        """Sim provides raw images, server expects flip_hw → HARD."""
        bench_obs = {"image": ActionObsSpec("image", 0, "rgb_hwc_uint8_raw")}
        server_obs = {"image": ActionObsSpec("image", 0, "rgb_hwc_uint8_flip_hw")}
        results = check_specs({}, {}, server_obs, bench_obs)
        sevs = self._severities(results)
        assert HARD in sevs

    # ── WARN: range mismatch ──────────────────────────────────────────────

    def test_warn_range_mismatch(self):
        """Server's range is (−1,1) but benchmark expects (0,1) → WARN."""
        server_action = {
            "gripper": ActionObsSpec("gripper", 1, "binary_close_positive", (-1.0, 1.0))
        }
        bench_action = {"gripper": ActionObsSpec("gripper", 1, "binary_close_positive", (0.0, 1.0))}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert WARN in sevs
        assert HARD not in sevs

    # ── WARN: legacy missing spec ─────────────────────────────────────────

    def test_warn_legacy_no_specs(self):
        """No specs declared at all → WARN (legacy server)."""
        results = check_specs({}, {}, {}, {})
        sevs = self._severities(results)
        assert WARN in sevs
        assert HARD not in sevs

    # ── PASS: consumer accepts produces format ────────────────────────────

    def test_pass_when_accepts_includes_format(self):
        """ROTATION_EULER_ACCEPTS_AA accepts axis_angle → no HARD, no WARN."""
        server_action = {"rotation": ROTATION_AA}
        bench_action = {"rotation": ROTATION_EULER_ACCEPTS_AA}
        results = check_specs(server_action, bench_action, {}, {})
        sevs = self._severities(results)
        assert HARD not in sevs
        assert WARN not in sevs

    # ── No issues → empty results ─────────────────────────────────────────

    def test_pass_fully_compatible(self):
        server_action = {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }
        bench_action = {
            "position": POSITION_DELTA,
            "rotation": ROTATION_AA,
            "gripper": GRIPPER_CLOSE_POS,
        }
        server_obs = {"image": IMAGE_RGB, "language": LANGUAGE}
        bench_obs = {"image": IMAGE_RGB, "language": LANGUAGE}
        results = check_specs(server_action, bench_action, server_obs, bench_obs)
        assert results == [], f"Expected no issues, got: {results}"

    # ── Symmetry: empty bench/server specs don't produce false positives ──

    def test_no_false_positives_empty_bench_action(self):
        """If bench declares no action spec, nothing to cross-check."""
        server_action = {"position": POSITION_DELTA}
        results = check_specs(server_action, {}, {}, {})
        # No HARD entries just because bench has no spec
        sevs = self._severities(results)
        assert HARD not in sevs


# ---------------------------------------------------------------------------
# Registry smoke test (imported here to avoid a separate file for 3 lines)
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_resolve_dimspec(self):
        from robo_eval.registry import resolve_import_string

        cls = resolve_import_string("robo_eval.specs:ActionObsSpec")
        assert cls is ActionObsSpec

    def test_resolve_constant(self):
        from robo_eval.registry import resolve_import_string

        obj = resolve_import_string("robo_eval.specs:POSITION_DELTA")
        assert obj is POSITION_DELTA

    def test_resolve_invalid_path(self):
        from robo_eval.registry import resolve_import_string

        with pytest.raises(ValueError):
            resolve_import_string("no_colon_here")

    def test_resolve_missing_module(self):
        from robo_eval.registry import resolve_import_string

        with pytest.raises((ImportError, ModuleNotFoundError)):
            resolve_import_string("nonexistent_module_xyz:SomeClass")

    def test_resolve_missing_attr(self):
        from robo_eval.registry import resolve_import_string

        with pytest.raises(AttributeError):
            resolve_import_string("robo_eval.specs:NonExistentClass")
