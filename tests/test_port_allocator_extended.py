"""
Extended tests for robo_eval/port_allocator.py.

Covers:
- reserve_block() with preferred_start, search, exact
- reserve_port() — single port reservation
- release() — lease file removal
- Stale lease cleanup
- PortReservation context manager
- Edge cases: invalid count, no available ports
"""

import json
import os

import pytest

import robo_eval.config as config_mod
from robo_eval.port_allocator import PortAllocator, PortReservation


@pytest.fixture
def allocator(tmp_path, monkeypatch):
    """Provide a PortAllocator with TMPDIR set to tmp_path and ports always available."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)
    return PortAllocator(owner_pid=os.getpid())


# ---------------------------------------------------------------------------
# reserve_block
# ---------------------------------------------------------------------------

class TestReserveBlock:
    def test_preferred_start(self, allocator):
        res = allocator.reserve_block(kind="sim", count=3, preferred_start=6000)
        try:
            assert res.ports == [6000, 6001, 6002]
            assert res.kind == "sim"
        finally:
            res.release()

    def test_search_when_preferred_taken(self, allocator, tmp_path, monkeypatch):
        # First reservation takes 6000-6001
        res1 = allocator.reserve_block(kind="sim", count=2, preferred_start=6000)
        # Second should search and find 6002-6003
        res2 = allocator.reserve_block(kind="sim", count=2, preferred_start=6000)
        try:
            assert res2.ports == [6002, 6003]
        finally:
            res2.release()
            res1.release()

    def test_exact_succeeds(self, allocator):
        res = allocator.reserve_block(kind="vla", count=2, preferred_start=7000, exact=True)
        try:
            assert res.ports == [7000, 7001]
        finally:
            res.release()

    def test_exact_fails_when_taken(self, allocator):
        res1 = allocator.reserve_block(kind="vla", count=1, preferred_start=7000, exact=True)
        try:
            with pytest.raises(RuntimeError, match="already reserved"):
                allocator.reserve_block(kind="vla", count=1, preferred_start=7000, exact=True)
        finally:
            res1.release()

    def test_count_zero_raises(self, allocator):
        with pytest.raises(ValueError, match="count must be >= 1"):
            allocator.reserve_block(kind="sim", count=0)

    def test_no_available_block_raises(self, allocator):
        # Reserve the entire tiny range
        r1 = allocator.reserve_block(kind="sim", count=3, preferred_start=8000, search_start=8000, search_end=8002)
        try:
            with pytest.raises(RuntimeError, match="No available"):
                allocator.reserve_block(kind="sim", count=1, search_start=8000, search_end=8002)
        finally:
            r1.release()


# ---------------------------------------------------------------------------
# reserve_port
# ---------------------------------------------------------------------------

class TestReservePort:
    def test_single_port(self, allocator):
        res = allocator.reserve_port(kind="proxy", preferred_port=9000)
        try:
            assert res.ports == [9000]
        finally:
            res.release()

    def test_search_when_preferred_taken(self, allocator):
        res1 = allocator.reserve_port(kind="proxy", preferred_port=9000)
        res2 = allocator.reserve_port(kind="proxy", preferred_port=9000)
        try:
            assert res1.ports == [9000]
            assert res2.ports == [9001]
        finally:
            res2.release()
            res1.release()


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_removes_lease_file(self, allocator, tmp_path):
        res = allocator.reserve_block(kind="sim", count=2, preferred_start=10000)
        # Lease files should exist
        leases_dir = tmp_path / "robo-eval-ports"
        assert (leases_dir / "10000.json").exists()
        assert (leases_dir / "10001.json").exists()

        res.release()
        assert not (leases_dir / "10000.json").exists()
        assert not (leases_dir / "10001.json").exists()

    def test_double_release_is_safe(self, allocator):
        res = allocator.reserve_block(kind="sim", count=1, preferred_start=11000)
        res.release()
        res.release()  # Should not raise


# ---------------------------------------------------------------------------
# PortReservation context manager
# ---------------------------------------------------------------------------

class TestPortReservationContextManager:
    def test_context_manager_releases(self, allocator, tmp_path):
        leases_dir = tmp_path / "robo-eval-ports"

        with allocator.reserve_block(kind="sim", count=1, preferred_start=12000) as res:
            assert res.ports == [12000]
            assert (leases_dir / "12000.json").exists()

        # After exiting context, lease should be released
        assert not (leases_dir / "12000.json").exists()

    def test_context_manager_releases_on_exception(self, allocator, tmp_path):
        leases_dir = tmp_path / "robo-eval-ports"
        try:
            with allocator.reserve_block(kind="sim", count=1, preferred_start=13000) as res:
                assert (leases_dir / "13000.json").exists()
                raise ValueError("test error")
        except ValueError:
            pass

        assert not (leases_dir / "13000.json").exists()


# ---------------------------------------------------------------------------
# Stale lease cleanup
# ---------------------------------------------------------------------------

class TestStaleLeaseCleanup:
    def test_cleans_dead_pid_leases(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)

        leases_dir = tmp_path / "robo-eval-ports"
        leases_dir.mkdir(parents=True)

        # Create a stale lease with a dead PID
        stale_lease = {
            "port": 14000,
            "kind": "sim",
            "owner_pid": 999999999,  # Very unlikely to be alive
            "token": "stale-token",
            "created_at": 0,
        }
        (leases_dir / "14000.json").write_text(json.dumps(stale_lease))

        # New allocator should clean up stale lease during reserve
        alloc = PortAllocator(owner_pid=os.getpid())
        res = alloc.reserve_block(kind="sim", count=1, preferred_start=14000)
        try:
            assert res.ports == [14000]  # Stale lease was cleaned up
        finally:
            res.release()

    def test_cleans_corrupt_lease_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)

        leases_dir = tmp_path / "robo-eval-ports"
        leases_dir.mkdir(parents=True)

        # Write corrupt JSON
        (leases_dir / "15000.json").write_text("not-json")

        alloc = PortAllocator(owner_pid=os.getpid())
        res = alloc.reserve_block(kind="sim", count=1, preferred_start=15000)
        try:
            assert res.ports == [15000]
        finally:
            res.release()


# ---------------------------------------------------------------------------
# Concurrent reservations don't overlap
# ---------------------------------------------------------------------------

class TestConcurrentReservations:
    def test_sequential_blocks_dont_overlap(self, allocator):
        reservations = []
        all_ports = set()
        for i in range(5):
            res = allocator.reserve_block(
                kind="sim", count=3, preferred_start=20000, search_start=20000
            )
            reservations.append(res)
            for p in res.ports:
                assert p not in all_ports, f"Port {p} already reserved!"
                all_ports.add(p)

        for res in reservations:
            res.release()

    def test_lease_file_content_matches(self, allocator, tmp_path):
        res = allocator.reserve_block(kind="vla", count=1, preferred_start=21000)
        try:
            lease_path = tmp_path / "robo-eval-ports" / "21000.json"
            data = json.loads(lease_path.read_text())
            assert data["port"] == 21000
            assert data["kind"] == "vla"
            assert data["owner_pid"] == os.getpid()
            assert data["token"] == res.token
        finally:
            res.release()

    def test_different_owner_pid(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)

        alloc1 = PortAllocator(owner_pid=os.getpid())
        alloc2 = PortAllocator(owner_pid=os.getpid() + 99999)

        res1 = alloc1.reserve_port(kind="sim", preferred_port=22000)
        # alloc2 can't get 22000 (taken by alloc1) — the PID of alloc2 is
        # not dead since we set it to current+99999 which doesn't exist either,
        # but the lease belongs to alloc1's real PID which IS alive.
        res2 = alloc2.reserve_port(kind="sim", preferred_port=22000)
        try:
            assert res2.ports[0] != 22000  # Should have moved to next available
        finally:
            res2.release()
            res1.release()
