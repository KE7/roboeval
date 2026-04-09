"""
Tests for robo_eval/proxy.py — Backend/BackendPool additional coverage.

Covers:
- Backend dataclass fields and defaults
- BackendPool.healthy_backends / any_healthy properties
- BackendPool.mark_unhealthy
- BackendPool info caching
- Round-robin wrapping with unhealthy nodes
"""

import asyncio
import time

import pytest

from robo_eval.proxy import Backend, BackendPool


def _run(coro):
    """Helper to run async code in tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Backend dataclass
# ---------------------------------------------------------------------------

class TestBackend:
    def test_defaults(self):
        b = Backend(url="http://localhost:5100")
        assert b.url == "http://localhost:5100"
        assert b.healthy is False
        assert b.last_check == 0.0
        assert b.last_error == ""
        assert b.requests_served == 0

    def test_grace_until_in_future(self):
        b = Backend(url="http://localhost:5100")
        assert b.grace_until > time.time() - 1  # Should be ~30s in the future

    def test_url_trailing_slash_stripped_by_pool(self):
        pool = BackendPool(["http://localhost:5100/"])
        assert pool.backends[0].url == "http://localhost:5100"


# ---------------------------------------------------------------------------
# BackendPool properties
# ---------------------------------------------------------------------------

class TestBackendPoolProperties:
    def test_healthy_backends_none_initially(self):
        pool = BackendPool(["http://localhost:5100", "http://localhost:5101"])
        assert pool.healthy_backends == []

    def test_healthy_backends_after_marking(self):
        pool = BackendPool(["http://localhost:5100", "http://localhost:5101"])
        pool.backends[0].healthy = True
        assert len(pool.healthy_backends) == 1
        assert pool.healthy_backends[0].url == "http://localhost:5100"

    def test_any_healthy_false_initially(self):
        pool = BackendPool(["http://localhost:5100"])
        assert pool.any_healthy is False

    def test_any_healthy_true_when_one_healthy(self):
        pool = BackendPool(["http://localhost:5100", "http://localhost:5101"])
        pool.backends[1].healthy = True
        assert pool.any_healthy is True


# ---------------------------------------------------------------------------
# BackendPool.mark_unhealthy
# ---------------------------------------------------------------------------

class TestMarkUnhealthy:
    def test_marks_backend_unhealthy(self):
        pool = BackendPool(["http://localhost:5100"])
        pool.backends[0].healthy = True
        _run(pool.mark_unhealthy(pool.backends[0], "connection refused"))
        assert pool.backends[0].healthy is False
        assert pool.backends[0].last_error == "connection refused"

    def test_mark_already_unhealthy(self):
        pool = BackendPool(["http://localhost:5100"])
        _run(pool.mark_unhealthy(pool.backends[0], "error"))
        assert pool.backends[0].healthy is False


# ---------------------------------------------------------------------------
# BackendPool info cache
# ---------------------------------------------------------------------------

class TestInfoCache:
    def test_cache_empty_initially(self):
        pool = BackendPool(["http://localhost:5100"])
        assert pool.get_cached_info() is None

    def test_set_and_get_cache(self):
        pool = BackendPool(["http://localhost:5100"])
        info = {"model_id": "test-model", "action_space": {"type": "eef_delta", "dim": 7}}
        pool.set_cached_info(info)
        cached = pool.get_cached_info()
        assert cached is not None
        assert cached["model_id"] == "test-model"

    def test_cache_expires(self):
        pool = BackendPool(["http://localhost:5100"])
        pool._info_cache_ttl = 0.01  # 10ms TTL
        pool.set_cached_info({"model_id": "test"})
        time.sleep(0.02)
        assert pool.get_cached_info() is None


# ---------------------------------------------------------------------------
# Round-robin advanced scenarios
# ---------------------------------------------------------------------------

class TestRoundRobinAdvanced:
    def test_wraps_around(self):
        pool = BackendPool([
            "http://localhost:5100",
            "http://localhost:5101",
        ])
        for b in pool.backends:
            b.healthy = True

        # Exhaust both and wrap
        b1 = _run(pool.next_healthy())
        b2 = _run(pool.next_healthy())
        b3 = _run(pool.next_healthy())
        # b3 should wrap around to the same as b1
        assert b3.url == b1.url

    def test_all_become_unhealthy_mid_iteration(self):
        pool = BackendPool(["http://localhost:5100"])
        pool.backends[0].healthy = True
        b = _run(pool.next_healthy())
        assert b is not None

        # Now mark unhealthy
        _run(pool.mark_unhealthy(pool.backends[0]))
        b2 = _run(pool.next_healthy())
        assert b2 is None

    def test_requests_served_tracks_per_backend(self):
        pool = BackendPool([
            "http://localhost:5100",
            "http://localhost:5101",
        ])
        for b in pool.backends:
            b.healthy = True

        for _ in range(6):
            _run(pool.next_healthy())

        # Each should have served 3 requests
        assert pool.backends[0].requests_served == 3
        assert pool.backends[1].requests_served == 3

    def test_skips_multiple_unhealthy(self):
        pool = BackendPool([
            "http://localhost:5100",
            "http://localhost:5101",
            "http://localhost:5102",
            "http://localhost:5103",
        ])
        # Only last one is healthy
        pool.backends[3].healthy = True
        b = _run(pool.next_healthy())
        assert b.url == "http://localhost:5103"
