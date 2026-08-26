"""Regression tests covering known failure modes.

See docs/failure_modes.md for the expected behavior each test covers.
"""

from __future__ import annotations

import json
import socket
import sys
import time

import pytest

from roboeval import server_runner
from sims import env_wrapper, litellm_vlm

# ---------------------------------------------------------------------------
# Health polling returns readiness and surfaces load errors without masking
# the original server response.
# ---------------------------------------------------------------------------


def test_poll_health_returns_ready_when_status_ok(monkeypatch):
    """A healthy JSON response should report ready without an error message."""

    class _Resp:
        ok = True
        headers = {"content-type": "application/json"}

        def json(self):
            return {"status": "ok", "ready": True}

    monkeypatch.setattr(server_runner.requests, "get", lambda *a, **k: _Resp())
    ready, err = server_runner._poll_health("http://x", timeout=1.0, interval=0.01)
    assert ready is True
    assert err == ""


def test_poll_health_surfaces_load_error_fast(monkeypatch):
    class _Resp:
        ok = False
        status_code = 503
        headers = {"content-type": "application/json"}

        def json(self):
            return {"status": "error", "ready": False, "error": "CUDA out of memory"}

    monkeypatch.setattr(server_runner.requests, "get", lambda *a, **k: _Resp())
    t0 = time.time()
    ready, err = server_runner._poll_health("http://x", timeout=10.0, interval=0.01)
    assert ready is False
    assert "CUDA out of memory" in err
    assert time.time() - t0 < 1.0  # fail fast instead of waiting for timeout


# ---------------------------------------------------------------------------
# Port checks fail before starting a server on an occupied port.
# ---------------------------------------------------------------------------


def test_assert_port_free_raises_when_bound():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    try:
        port = s.getsockname()[1]
        with pytest.raises(RuntimeError, match=f"Port {port} on .* is already in use"):
            server_runner._assert_port_free("127.0.0.1", port)
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Missing explicit venv paths are treated as hard errors.
# ---------------------------------------------------------------------------


def test_resolve_python_must_exist_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="venv .* not found"):
        server_runner._resolve_python(
            venv_path=str(tmp_path / "missing_venv"),
            project_root=tmp_path,
            must_exist=True,
        )


def test_resolve_python_must_exist_false_falls_back(tmp_path):
    py = server_runner._resolve_python(
        venv_path=str(tmp_path / "missing_venv"),
        project_root=tmp_path,
        must_exist=False,
    )
    assert py == sys.executable


# ---------------------------------------------------------------------------
# Unknown simulator backends fail before launching a subprocess.
# ---------------------------------------------------------------------------


def test_start_sim_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown sim backend: 'liber0'"):
        server_runner.start_sim(backend="liber0")


# ---------------------------------------------------------------------------
# NaN and wrong-shape actions are rejected before /step.
# ---------------------------------------------------------------------------


def test_validate_action_chunk_rejects_nan():
    w = object.__new__(env_wrapper.SimWrapper)  # bypass __init__
    with pytest.raises(ValueError, match="NaN or inf"):
        env_wrapper.SimWrapper._validate_action_chunk(
            w, [[0.0, float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]], expected_dim=7
        )


def test_validate_action_chunk_rejects_wrong_shape():
    w = object.__new__(env_wrapper.SimWrapper)
    with pytest.raises(ValueError, match=r"dim=5 but .* negotiated action_dim is 7"):
        env_wrapper.SimWrapper._validate_action_chunk(w, [[0.0] * 5], expected_dim=7)


def test_validate_action_chunk_accepts_valid_chunk():
    w = object.__new__(env_wrapper.SimWrapper)
    env_wrapper.SimWrapper._validate_action_chunk(
        w, [[0.0] * 7, [0.1] * 7], expected_dim=7
    )  # no raise


# ---------------------------------------------------------------------------
# An unreachable litellm proxy fails at startup with remediation text.
# ---------------------------------------------------------------------------


def test_assert_litellm_reachable_raises_with_remediation():
    # localhost:1 is reserved/closed.
    with pytest.raises(RuntimeError, match=r"(?s)litellm proxy not reachable.*--no-vlm"):
        litellm_vlm._assert_litellm_reachable("127.0.0.1", 1)


# ---------------------------------------------------------------------------
# Result writes remain atomic if replacing the output file fails.
# ---------------------------------------------------------------------------


def test_atomic_write_json_preserves_existing_on_oserror(tmp_path, monkeypatch):
    from roboeval.orchestrator import _atomic_write_json

    out = tmp_path / "result.json"
    out.write_text(json.dumps({"old": True}))

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("roboeval.orchestrator.os.replace", boom)

    with pytest.raises(OSError):
        _atomic_write_json(out, {"new": True})

    # Original file is intact; tmp is gone.
    assert json.loads(out.read_text()) == {"old": True}
    assert not (tmp_path / "result.json.tmp").exists()


# ---------------------------------------------------------------------------
# Concurrent non-shard runs do not collide on output filenames.
# ---------------------------------------------------------------------------


def test_claim_output_path_unique_under_concurrency(tmp_path):
    from roboeval.orchestrator import EvalConfig, Orchestrator

    paths = set()
    for _ in range(20):
        cfg = EvalConfig.from_dict({"name": "race", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg)
        p = orch._claim_output_path("race")
        assert p not in paths
        paths.add(p)
