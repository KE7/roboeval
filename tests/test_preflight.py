"""Unit tests for robo_eval.preflight.

Tests:
    - validate_yaml: YAML parse, config load, ActionObsSpec round-trip
    - check_server: /health, /info, spec validation (with stubbed FastAPI servers)
    - check_benchmark: /health, /info, /reset, /step (with stubbed servers)
    - run_preflight: end-to-end with mocked servers
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_eval.preflight import (
    validate_yaml,
    check_server,
    check_benchmark,
    PreflightConfig,
    _check,
    print_results,
    run_preflight,
)


# ---------------------------------------------------------------------------
# Fixtures: stubbed server responses
# ---------------------------------------------------------------------------


class _MockResponse:
    """Minimal mock for requests.Response."""

    def __init__(self, json_data: dict, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code
        self.ok = status_code < 400

    def raise_for_status(self):
        if not self.ok:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def _make_requests_mock(responses: dict[str, dict]):
    """Return a mock requests module where get/post return the given responses."""

    def fake_get(url, **kwargs):
        path = url.split("localhost")[1].lstrip("0123456789").lstrip(":").lstrip("/")
        # Extract the path suffix (/health, /info, etc.)
        for suffix, resp_data in responses.items():
            if url.endswith(suffix):
                if isinstance(resp_data, Exception):
                    raise resp_data
                return _MockResponse(resp_data)
        return _MockResponse({}, status_code=404)

    def fake_post(url, **kwargs):
        for suffix, resp_data in responses.items():
            if url.endswith(suffix):
                if isinstance(resp_data, Exception):
                    raise resp_data
                return _MockResponse(resp_data)
        return _MockResponse({}, status_code=404)

    mock = MagicMock()
    mock.get.side_effect = fake_get
    mock.post.side_effect = fake_post
    mock.RequestException = Exception
    return mock


# ---------------------------------------------------------------------------
# validate_yaml tests
# ---------------------------------------------------------------------------


class TestValidateYaml:
    def test_valid_yaml_passes(self, tmp_path):
        yaml_content = """
name: test_run
suite: libero_spatial
episodes_per_task: 2
no_vlm: true
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        results = validate_yaml(p)
        checks = {r.name: r for r in results}
        assert checks["yaml.parse"].ok, checks["yaml.parse"].message
        assert checks["config.load"].ok, checks["config.load"].message
        assert checks["specs.dimspec_roundtrip"].ok

    def test_invalid_yaml_fails(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: {\n  unclosed_brace: [")
        results = validate_yaml(p)
        assert any(not r.ok for r in results)

    def test_nonexistent_file_fails(self, tmp_path):
        results = validate_yaml(tmp_path / "does_not_exist.yaml")
        assert any(not r.ok for r in results)

    def test_benchmark_registry_resolves(self, tmp_path):
        yaml_content = "name: test\nbenchmark: 'robo_eval.specs:ActionObsSpec'\n"
        p = tmp_path / "bench.yaml"
        p.write_text(yaml_content)
        results = validate_yaml(p)
        checks = {r.name: r for r in results}
        # registry.resolve should be OK since ActionObsSpec exists
        assert checks["registry.resolve"].ok

    def test_bad_benchmark_registry_fails(self, tmp_path):
        yaml_content = "name: test\nbenchmark: 'no_such_module:NoSuchClass'\n"
        p = tmp_path / "bad_bench.yaml"
        p.write_text(yaml_content)
        results = validate_yaml(p)
        checks = {r.name: r for r in results}
        assert not checks["registry.resolve"].ok

    def test_dimspec_round_trip_check(self, tmp_path):
        yaml_content = "name: rt_test\n"
        p = tmp_path / "rt.yaml"
        p.write_text(yaml_content)
        results = validate_yaml(p)
        checks = {r.name: r for r in results}
        assert checks["specs.dimspec_roundtrip"].ok


# ---------------------------------------------------------------------------
# check_server tests
# ---------------------------------------------------------------------------


class TestCheckServer:
    def test_healthy_server_with_specs(self):
        server = {"url": "http://localhost:9999", "name": "test_vla"}
        responses = {
            "/health": {"ready": True, "status": "ok"},
            "/info": {
                "name": "pi05",
                "action_spec": {
                    "position": {"name": "position", "dims": 3, "format": "delta_xyz"}
                },
                "observation_spec": {
                    "primary": {"name": "primary", "dims": 0, "format": "image_rgb_hwc_uint8"}
                },
            },
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_server(server)

        checks = {r.name: r for r in results}
        assert checks["test_vla.health"].ok
        assert checks["test_vla.info"].ok
        assert checks["test_vla.spec_validation"].ok

    def test_health_failure_stops_early(self):
        server = {"url": "http://localhost:9999", "name": "vla_x"}
        responses = {
            "/health": ConnectionError("refused"),
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_server(server)

        checks = {r.name: r for r in results}
        assert not checks["vla_x.health"].ok
        # info check should not be present
        assert "vla_x.info" not in checks

    def test_health_not_ready(self):
        server = {"url": "http://localhost:9999", "name": "vla_y"}
        responses = {
            "/health": {"ready": False},
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_server(server)

        checks = {r.name: r for r in results}
        # ready=False means health check fails
        assert not checks["vla_y.health"].ok

    def test_legacy_server_no_specs(self):
        """Server without action_spec/observation_spec still passes."""
        server = {"url": "http://localhost:9999", "name": "legacy_vla"}
        responses = {
            "/health": {"ready": True},
            "/info": {"name": "legacy_model", "action_chunk_size": 50},
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_server(server)

        checks = {r.name: r for r in results}
        assert checks["legacy_vla.health"].ok
        assert checks["legacy_vla.spec_validation"].ok

    def test_invalid_spec_dict_fails(self):
        server = {"url": "http://localhost:9999", "name": "bad_spec_vla"}
        responses = {
            "/health": {"ready": True},
            "/info": {
                "action_spec": {
                    "position": {"not_a_valid_spec": True}
                }
            },
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_server(server)

        checks = {r.name: r for r in results}
        assert not checks["bad_spec_vla.spec_validation"].ok


# ---------------------------------------------------------------------------
# check_benchmark tests
# ---------------------------------------------------------------------------


class TestCheckBenchmark:
    def test_healthy_sim_passes(self):
        sim = {"url": "http://localhost:9998", "name": "libero"}
        responses = {
            "/health": {"ready": True},
            "/info": {"action_dim": 7, "backend": "libero"},
            "/reset": {"obs": {}, "task_instruction": "pick up"},
            "/step": {"obs": {}, "success": False, "done": False},
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_benchmark(sim)

        checks = {r.name: r for r in results}
        assert checks["libero.health"].ok
        assert checks["libero.info"].ok
        assert checks["libero.reset"].ok
        assert checks["libero.step"].ok

    def test_health_failure_stops_early(self):
        sim = {"url": "http://localhost:9998", "name": "sim_x"}
        responses = {
            "/health": Exception("refused"),
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_benchmark(sim)

        checks = {r.name: r for r in results}
        assert not checks["sim_x.health"].ok
        assert "sim_x.reset" not in checks

    def test_reset_failure_stops_step(self):
        sim = {"url": "http://localhost:9998", "name": "sim_y"}
        responses = {
            "/health": {"ready": True},
            "/info": {"action_dim": 7},
            "/reset": Exception("sim crashed"),
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            results = check_benchmark(sim)

        checks = {r.name: r for r in results}
        assert not checks["sim_y.reset"].ok
        assert "sim_y.step" not in checks


# ---------------------------------------------------------------------------
# PreflightConfig tests
# ---------------------------------------------------------------------------


class TestPreflightConfig:
    def test_from_dict_defaults(self):
        cfg = PreflightConfig.from_dict({})
        assert len(cfg.servers) == 1
        assert len(cfg.sims) == 1

    def test_from_dict_explicit_servers(self):
        d = {
            "servers": [
                {"url": "http://localhost:5100", "name": "pi05"},
                {"url": "http://localhost:5101", "name": "smolvla"},
            ],
            "sims": [
                {"url": "http://localhost:5300", "name": "libero"},
            ],
        }
        cfg = PreflightConfig.from_dict(d)
        assert len(cfg.servers) == 2
        assert len(cfg.sims) == 1

    def test_from_yaml(self, tmp_path):
        yaml_content = """
vla_url: http://localhost:9876
sim_url: http://localhost:9877
"""
        p = tmp_path / "pf.yaml"
        p.write_text(yaml_content)
        cfg = PreflightConfig.from_yaml(p)
        assert cfg.vla_url == "http://localhost:9876"


# ---------------------------------------------------------------------------
# print_results tests
# ---------------------------------------------------------------------------


class TestPrintResults:
    def test_all_ok_returns_0(self, capsys):
        results = [_check("a.check", True), _check("b.check", True)]
        rc = print_results(results)
        assert rc == 0

    def test_one_fail_returns_1(self, capsys):
        results = [_check("a.check", True), _check("b.check", False, "broken")]
        rc = print_results(results)
        assert rc == 1

    def test_all_fail_returns_1(self, capsys):
        results = [_check("a.check", False), _check("b.check", False)]
        rc = print_results(results)
        assert rc == 1


# ---------------------------------------------------------------------------
# run_preflight integration test (mocked)
# ---------------------------------------------------------------------------


class TestRunPreflightIntegration:
    def test_validate_only(self, tmp_path):
        yaml_content = "name: pf_test\nsuite: libero_spatial\nepisodes_per_task: 1\n"
        p = tmp_path / "pf.yaml"
        p.write_text(yaml_content)

        rc = run_preflight(p, validate=True, server=False, benchmark=False)
        assert rc == 0

    def test_server_check_with_mock(self, tmp_path):
        yaml_content = "name: pf_server_test\nvla_url: http://localhost:9999\n"
        p = tmp_path / "pf_server.yaml"
        p.write_text(yaml_content)

        responses = {
            "/health": {"ready": True},
            "/info": {"name": "pi05"},
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            rc = run_preflight(p, validate=False, server=True, benchmark=False)
        assert rc == 0

    def test_benchmark_check_with_mock(self, tmp_path):
        yaml_content = "name: pf_bench_test\nsim_url: http://localhost:9998\n"
        p = tmp_path / "pf_bench.yaml"
        p.write_text(yaml_content)

        responses = {
            "/health": {"ready": True},
            "/info": {"action_dim": 7},
            "/reset": {"obs": {}},
            "/step": {"obs": {}, "success": False},
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            rc = run_preflight(p, validate=False, server=False, benchmark=True)
        assert rc == 0

    def test_server_unreachable_returns_nonzero(self, tmp_path):
        yaml_content = "name: fail_test\nvla_url: http://localhost:9999\n"
        p = tmp_path / "fail.yaml"
        p.write_text(yaml_content)

        responses = {
            "/health": Exception("Connection refused"),
        }
        mock_requests = _make_requests_mock(responses)
        with patch("robo_eval.preflight.requests", mock_requests):
            rc = run_preflight(p, validate=False, server=True, benchmark=False)
        assert rc != 0

    def test_all_checks_dry_run_mocked(self, tmp_path):
        """--all: runs validate + server + benchmark + 1-episode dry run (mocked)."""
        yaml_content = """
name: all_test
vla_url: http://localhost:9999
sim_url: http://localhost:9998
suite: libero_spatial
episodes_per_task: 1
max_tasks: 1
no_vlm: true
output_dir: {}
""".format(str(tmp_path / "dry_run"))
        p = tmp_path / "all.yaml"
        p.write_text(yaml_content)

        responses = {
            "/health": {"ready": True},
            "/info": {"action_dim": 7},
            "/reset": {"obs": {}},
            "/step": {"obs": {}, "success": False},
        }
        mock_requests = _make_requests_mock(responses)

        # Mock the orchestrator run to keep this test self-contained.
        mock_orch_result = {
            "benchmark": "all_test",
            "tasks": [{"task": "task_0", "episodes": [
                {"episode_id": 0, "metrics": {"success": True}, "steps": 50}
            ], "num_episodes": 1, "avg_steps": 50.0}],
            "mean_success": 1.0,
        }
        with patch("robo_eval.preflight.requests", mock_requests):
            with patch("robo_eval.preflight.Orchestrator") as MockOrch:
                instance = MockOrch.return_value
                instance.run.return_value = mock_orch_result
                rc = run_preflight(
                    p,
                    validate=True,
                    server=True,
                    benchmark=True,
                    all_checks=True,
                    results_dir=str(tmp_path / "dry_run"),
                )
        assert rc == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
