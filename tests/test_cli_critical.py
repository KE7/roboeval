"""
Unit tests for the most critical CLI functions.

Covers:
- count_successes() from results.py
- resolve_suites() from config.py
- validate_port() from config.py
- BackendPool.next_healthy() from proxy.py

Run with: pytest tests/test_cli_critical.py -v
"""

import asyncio
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# count_successes
# ---------------------------------------------------------------------------

class TestCountSuccesses:
    """Tests for robo_eval.results.count_successes()."""

    def test_empty_file(self, tmp_path):
        from robo_eval.results import count_successes
        log = tmp_path / "empty.log"
        log.write_text("")
        assert count_successes(log) == (0, 0, [])

    def test_missing_file(self, tmp_path):
        from robo_eval.results import count_successes
        log = tmp_path / "nonexistent.log"
        assert count_successes(log) == (0, 0, [])

    def test_all_success(self, tmp_path):
        from robo_eval.results import count_successes
        log = tmp_path / "task.log"
        lines = [
            "Episode 1/10",
            "Simulator reports success: True",
            "Episode 2/10",
            "Simulator reports success: True",
            "Episode 3/10",
            "Simulator reports success: True",
        ]
        log.write_text("\n".join(lines))
        assert count_successes(log) == (3, 3, [1, 1, 1])

    def test_mixed_results(self, tmp_path):
        from robo_eval.results import count_successes
        log = tmp_path / "task.log"
        lines = [
            "Episode 1/5",
            "Simulator reports success: True",
            "Episode 2/5",
            "Simulator reports success: False",
            "Episode 3/5",
            "Simulator reports success: True",
            "Episode 4/5",
            "Simulator reports success: False",
            "Episode 5/5",
            "Simulator reports success: False",
        ]
        log.write_text("\n".join(lines))
        assert count_successes(log) == (2, 5, [1, 0, 1, 0, 0])

    def test_all_failures(self, tmp_path):
        from robo_eval.results import count_successes
        log = tmp_path / "task.log"
        lines = [
            "Simulator reports success: False",
            "Simulator reports success: False",
        ]
        log.write_text("\n".join(lines))
        assert count_successes(log) == (0, 2, [0, 0])


# ---------------------------------------------------------------------------
# resolve_suites
# ---------------------------------------------------------------------------

class TestResolveSuites:
    """Tests for robo_eval.config.resolve_suites()."""

    def test_single_suite(self):
        from robo_eval.config import resolve_suites
        assert resolve_suites("libero_spatial") == ["libero_spatial"]

    def test_preset_libero(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("libero")
        assert "libero_spatial" in result
        assert "libero_object" in result
        assert "libero_goal" in result
        assert "libero_10" in result
        assert len(result) == 4

    def test_preset_libero_pro(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("libero_pro")
        assert "libero_pro_spatial_object" in result
        assert "libero_pro_goal_swap" in result
        assert "libero_pro_spatial_with_mug" in result
        assert len(result) == 3

    def test_comma_separated(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("libero_spatial, libero_goal")
        assert result == ["libero_spatial", "libero_goal"]

    def test_deduplication(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("libero_spatial,libero_spatial")
        assert result == ["libero_spatial"]

    def test_preset_plus_individual(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("libero,libero_spatial_object")
        assert "libero_spatial" in result
        assert "libero_spatial_object" in result
        # libero_spatial should not be duplicated
        assert result.count("libero_spatial") == 1

    def test_unknown_suite_passthrough(self):
        from robo_eval.config import resolve_suites
        result = resolve_suites("custom_suite_xyz")
        assert result == ["custom_suite_xyz"]


# ---------------------------------------------------------------------------
# validate_port
# ---------------------------------------------------------------------------

class TestValidatePort:
    """Tests for robo_eval.config.validate_port()."""

    def test_valid_port(self):
        from robo_eval.config import validate_port
        assert validate_port(5100) == 5100
        assert validate_port(8080) == 8080
        assert validate_port(65535) == 65535
        assert validate_port(1024) == 1024

    def test_privileged_port_rejected(self):
        from robo_eval.config import validate_port
        with pytest.raises(ValueError, match="privileged"):
            validate_port(80)
        with pytest.raises(ValueError, match="privileged"):
            validate_port(443)

    def test_zero_rejected(self):
        from robo_eval.config import validate_port
        with pytest.raises(ValueError):
            validate_port(0)

    def test_negative_rejected(self):
        from robo_eval.config import validate_port
        with pytest.raises(ValueError):
            validate_port(-1)

    def test_too_large_rejected(self):
        from robo_eval.config import validate_port
        with pytest.raises(ValueError, match="65535"):
            validate_port(70000)

    def test_custom_name_in_error(self):
        from robo_eval.config import validate_port
        with pytest.raises(ValueError, match="--sim-base-port"):
            validate_port(0, "--sim-base-port")


# ---------------------------------------------------------------------------
# BackendPool.next_healthy
# ---------------------------------------------------------------------------

class TestBackendPoolNextHealthy:
    """Tests for robo_eval.proxy.BackendPool.next_healthy()."""

    def _run(self, coro):
        """Helper to run async code in tests."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_no_backends(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool([])
        result = self._run(pool.next_healthy())
        assert result is None

    def test_all_unhealthy(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool(["http://localhost:9999"])
        # Backends start unhealthy by default
        result = self._run(pool.next_healthy())
        assert result is None

    def test_one_healthy(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool(["http://localhost:5100"])
        pool.backends[0].healthy = True
        result = self._run(pool.next_healthy())
        assert result is not None
        assert result.url == "http://localhost:5100"

    def test_round_robin(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool([
            "http://localhost:5100",
            "http://localhost:5101",
            "http://localhost:5102",
        ])
        for b in pool.backends:
            b.healthy = True

        first = self._run(pool.next_healthy())
        second = self._run(pool.next_healthy())
        third = self._run(pool.next_healthy())

        # Should round-robin through all 3
        urls = [first.url, second.url, third.url]
        assert len(set(urls)) == 3  # All different

    def test_skips_unhealthy(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool([
            "http://localhost:5100",
            "http://localhost:5101",
        ])
        # Only second backend is healthy
        pool.backends[0].healthy = False
        pool.backends[1].healthy = True

        result = self._run(pool.next_healthy())
        assert result.url == "http://localhost:5101"

    def test_requests_served_increments(self):
        from robo_eval.proxy import BackendPool
        pool = BackendPool(["http://localhost:5100"])
        pool.backends[0].healthy = True

        assert pool.backends[0].requests_served == 0
        self._run(pool.next_healthy())
        assert pool.backends[0].requests_served == 1
        self._run(pool.next_healthy())
        assert pool.backends[0].requests_served == 2
