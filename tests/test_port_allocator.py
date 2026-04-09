import os

import robo_eval.config as config_mod
from robo_eval.port_allocator import PortAllocator


def test_allocator_reserves_distinct_blocks(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)

    allocator = PortAllocator(owner_pid=os.getpid())
    res1 = allocator.reserve_block(
        kind="sim",
        count=2,
        preferred_start=41000,
        search_start=41000,
    )
    res2 = allocator.reserve_block(
        kind="sim",
        count=2,
        preferred_start=41000,
        search_start=41000,
    )

    try:
        assert res1.ports == [41000, 41001]
        assert res2.ports == [41002, 41003]
    finally:
        res2.release()
        res1.release()


def test_allocator_reuses_ports_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(config_mod, "is_port_available", lambda port: True)

    allocator = PortAllocator(owner_pid=os.getpid())
    res1 = allocator.reserve_block(
        kind="sim",
        count=2,
        preferred_start=42000,
        search_start=42000,
    )
    res1.release()

    res2 = allocator.reserve_block(
        kind="sim",
        count=2,
        preferred_start=42000,
        search_start=42000,
    )
    try:
        assert res2.ports == [42000, 42001]
    finally:
        res2.release()
