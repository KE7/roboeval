from typer.testing import CliRunner

import robo_eval.cli as cli_mod
import robo_eval.runner as runner_mod
from robo_eval.cli import app


runner = CliRunner()


def test_dry_run_infers_sim_workers_from_requested_concurrency():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial,object,goal",
            "--episodes",
            "1",
            "--max-tasks",
            "4",
            "--tasks-parallel",
            "2",
            "--suites-parallel",
            "3",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Sim workers:     6 on ports " in result.stdout


def test_dry_run_auto_selects_free_ports(monkeypatch):
    def fake_find_available_port(preferred_port=None, **kwargs):
        assert preferred_port == 5200
        return 6200

    def fake_find_available_port_block(count, preferred_start=None, **kwargs):
        if preferred_start == 5102:
            return 6100
        if preferred_start == 5300:
            return 6300
        raise AssertionError(f"unexpected preferred_start={preferred_start}, count={count}")

    monkeypatch.setattr(cli_mod, "find_available_port", fake_find_available_port)
    monkeypatch.setattr(cli_mod, "find_available_port_block", fake_find_available_port_block)

    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--episodes",
            "1",
            "--max-tasks",
            "2",
            "--tasks-parallel",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Ports:         6100-6100" in result.stdout
    assert "Proxy:           port 6200 -> 1 backend(s)" in result.stdout
    assert "Sim workers:     2 on ports 6300-6301" in result.stdout


def test_explicit_port_override_errors_when_occupied(monkeypatch):
    monkeypatch.setattr(
        cli_mod,
        "find_available_port_block",
        lambda count, preferred_start=None, **kwargs: preferred_start,
    )
    monkeypatch.setattr(cli_mod, "is_port_available", lambda port: port != 5400)

    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--episodes",
            "1",
            "--max-tasks",
            "1",
            "--sim-base-port",
            "5400",
            "--dry-run",
        ],
    )

    assert result.exit_code == 1
    assert "Requested ports are already in use: sim worker port 5400" in result.stdout


def test_parallel_suites_reuse_worker_slices_by_batch(tmp_path, monkeypatch):
    calls = []

    class DummyPool:
        def __init__(self):
            self.processes = {5300 + i: object() for i in range(6)}

    def fake_run_single_suite_parallel(**kwargs):
        calls.append((kwargs["suite"], kwargs.get("managed_ports")))

    monkeypatch.setattr(runner_mod, "_run_single_suite_parallel", fake_run_single_suite_parallel)
    monkeypatch.setattr(runner_mod, "collect_results", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner_mod, "write_scores_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_mod, "write_summary", lambda *args, **kwargs: "ok")

    runner_mod.run_eval_parallel(
        suites=[
            "libero_infinity_spatial",
            "libero_infinity_object",
            "libero_infinity_goal",
            "libero_infinity_10",
            "libero_infinity_spatial",
        ],
        vla_url="http://localhost:5200",
        results_dir=tmp_path,
        max_episodes=1,
        num_tasks=4,
        suites_parallel=3,
        managed_sim_pool=DummyPool(),
        max_workers=2,
    )

    assert calls == [
        ("libero_infinity_spatial", [5300, 5301]),
        ("libero_infinity_object", [5302, 5303]),
        ("libero_infinity_goal", [5304, 5305]),
        ("libero_infinity_10", [5300, 5301]),
        ("libero_infinity_spatial", [5302, 5303]),
    ]
