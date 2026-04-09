"""Cooperative port reservations for concurrent robo-eval runs.

This prevents two managed `robo-eval run` invocations from racing on the same
ports. Reservations are machine-local lease files under ${TMPDIR:-/tmp}.
External processes can still steal a port after reservation; server startup
and health checks remain the final guard against those non-cooperative cases.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _tmp_root() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp"))


def _lock_path() -> Path:
    return _tmp_root() / "robo-eval-port-allocator.lock"


def _leases_dir() -> Path:
    return _tmp_root() / "robo-eval-ports"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@dataclass
class PortReservation:
    """A held reservation for one or more ports."""

    ports: list[int]
    kind: str
    owner_pid: int
    token: str

    def release(self) -> None:
        PortAllocator(owner_pid=self.owner_pid)._release(self)

    def __enter__(self) -> "PortReservation":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class PortAllocator:
    """Cooperative allocator backed by lock + lease files in tmp."""

    def __init__(self, owner_pid: Optional[int] = None):
        self.owner_pid = owner_pid or os.getpid()

    def reserve_port(
        self,
        *,
        kind: str,
        preferred_port: Optional[int] = None,
        search_start: int = 1024,
        search_end: int = 65535,
        exact: bool = False,
    ) -> PortReservation:
        return self.reserve_block(
            kind=kind,
            count=1,
            preferred_start=preferred_port,
            search_start=search_start,
            search_end=search_end,
            exact=exact,
        )

    def reserve_block(
        self,
        *,
        kind: str,
        count: int,
        preferred_start: Optional[int] = None,
        search_start: int = 1024,
        search_end: int = 65535,
        exact: bool = False,
    ) -> PortReservation:
        from .config import is_port_available, validate_port

        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        if preferred_start is not None:
            validate_port(preferred_start, "preferred_start")
            validate_port(preferred_start + count - 1, "preferred_start + count - 1")

        with self._locked():
            self._cleanup_stale_locked()

            if preferred_start is not None and exact:
                ports = [preferred_start + i for i in range(count)]
                if not self._ports_free_locked(ports, is_port_available):
                    raise RuntimeError(
                        f"Requested {kind} port block {ports[0]}-{ports[-1]} is already reserved or in use"
                    )
                return self._reserve_ports_locked(ports, kind)

            start = preferred_start if preferred_start is not None else search_start
            max_start = search_end - count + 1
            for base_port in range(start, max_start + 1):
                ports = [base_port + i for i in range(count)]
                if self._ports_free_locked(ports, is_port_available):
                    return self._reserve_ports_locked(ports, kind)

            raise RuntimeError(
                f"No available {kind} port block of size {count} in range "
                f"{start}-{search_end}"
            )

    def _release(self, reservation: PortReservation) -> None:
        with self._locked():
            for port in reservation.ports:
                lease_path = self._lease_path(port)
                if not lease_path.exists():
                    continue
                try:
                    data = json.loads(lease_path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    data.get("owner_pid") == reservation.owner_pid
                    and data.get("token") == reservation.token
                ):
                    lease_path.unlink(missing_ok=True)

    def _ports_free_locked(self, ports: list[int], is_port_available_fn) -> bool:
        return all(
            not self._lease_path(port).exists() and is_port_available_fn(port)
            for port in ports
        )

    def _reserve_ports_locked(self, ports: list[int], kind: str) -> PortReservation:
        token = uuid.uuid4().hex
        reservation = PortReservation(
            ports=ports,
            kind=kind,
            owner_pid=self.owner_pid,
            token=token,
        )
        now = time.time()
        for port in ports:
            payload = {
                "port": port,
                "kind": kind,
                "owner_pid": self.owner_pid,
                "token": token,
                "created_at": now,
            }
            self._lease_path(port).write_text(json.dumps(payload))
        return reservation

    def _cleanup_stale_locked(self) -> None:
        for lease_path in _leases_dir().glob("*.json"):
            try:
                data = json.loads(lease_path.read_text())
            except (OSError, json.JSONDecodeError):
                lease_path.unlink(missing_ok=True)
                continue

            pid = data.get("owner_pid")
            if not isinstance(pid, int) or not _pid_alive(pid):
                lease_path.unlink(missing_ok=True)

    def _lease_path(self, port: int) -> Path:
        return _leases_dir() / f"{port}.json"

    def _locked(self):
        _leases_dir().mkdir(parents=True, exist_ok=True)
        _lock_path().parent.mkdir(parents=True, exist_ok=True)
        lock_file = _lock_path().open("a+")

        # Try to acquire lock with timeout (30s max) using non-blocking mode.
        # Wrapped in try/except to ensure file handle is closed on any error
        # (prevents FD leak under contention or unexpected exceptions).
        deadline = time.time() + 30
        try:
            while True:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() >= deadline:
                        raise TimeoutError("Could not acquire port allocator lock within 30s")
                    time.sleep(0.1)
        except BaseException:
            lock_file.close()
            raise

        class _LockCtx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                finally:
                    lock_file.close()

        return _LockCtx()
