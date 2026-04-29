"""
Unit tests for the most critical CLI functions.

Covers:
- resolve_suites() from config.py
- validate_port() from config.py

Run with: pytest tests/test_cli_critical.py -v
"""

import pytest

# ---------------------------------------------------------------------------
# resolve_suites
# ---------------------------------------------------------------------------


class TestResolveSuites:
    """Tests for roboeval.config.resolve_suites()."""

    def test_single_suite(self):
        from roboeval.config import resolve_suites

        assert resolve_suites("libero_spatial") == ["libero_spatial"]

    def test_preset_libero(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("libero")
        assert "libero_spatial" in result
        assert "libero_object" in result
        assert "libero_goal" in result
        assert "libero_10" in result
        assert len(result) == 4

    def test_preset_libero_pro(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("libero_pro")
        assert "libero_pro_spatial_object" in result
        assert "libero_pro_goal_swap" in result
        assert "libero_pro_spatial_with_mug" in result
        assert len(result) == 3

    def test_comma_separated(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("libero_spatial, libero_goal")
        assert result == ["libero_spatial", "libero_goal"]

    def test_deduplication(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("libero_spatial,libero_spatial")
        assert result == ["libero_spatial"]

    def test_preset_plus_individual(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("libero,libero_spatial_object")
        assert "libero_spatial" in result
        assert "libero_spatial_object" in result
        assert result.count("libero_spatial") == 1

    def test_unknown_suite_passthrough(self):
        from roboeval.config import resolve_suites

        result = resolve_suites("custom_suite_xyz")
        assert result == ["custom_suite_xyz"]


# ---------------------------------------------------------------------------
# validate_port
# ---------------------------------------------------------------------------


class TestValidatePort:
    """Tests for roboeval.config.validate_port()."""

    def test_valid_port(self):
        from roboeval.config import validate_port

        assert validate_port(5100) == 5100
        assert validate_port(8080) == 8080
        assert validate_port(65535) == 65535
        assert validate_port(1024) == 1024

    def test_privileged_port_rejected(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError, match="privileged"):
            validate_port(80)
        with pytest.raises(ValueError, match="privileged"):
            validate_port(443)

    def test_zero_rejected(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError):
            validate_port(0)

    def test_negative_rejected(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError):
            validate_port(-1)

    def test_too_large_rejected(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError, match="65535"):
            validate_port(70000)

    def test_custom_name_in_error(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError, match="--sim-base-port"):
            validate_port(0, "--sim-base-port")
