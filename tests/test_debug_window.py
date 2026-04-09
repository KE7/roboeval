from pathlib import Path

from typer.testing import CliRunner

from robo_eval.cli import app
from robo_eval.runner import _build_eval_cmd
from robo_eval.servers import SimWorkerPool


runner = CliRunner()


def test_debug_window_dry_run_valid():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial",
            "--episodes",
            "1",
            "--max-tasks",
            "1",
            "--sequential",
            "--tasks-parallel",
            "1",
            "--debug-window",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Rendering:       windowed GLFW debug" in result.stdout


def test_debug_window_rejects_multiple_suites():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial,goal",
            "--debug-window",
        ],
    )

    assert result.exit_code == 1
    assert "--debug-window supports exactly one suite" in result.stdout


def test_debug_window_rejects_parallel():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial",
            "--debug-window",
            "--parallel",
        ],
    )

    assert result.exit_code == 1
    assert "--debug-window requires --sequential" in result.stdout


def test_debug_window_rejects_multiple_workers():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial",
            "--sequential",
            "--tasks-parallel",
            "2",
            "--debug-window",
        ],
    )

    assert result.exit_code == 1
    assert "--debug-window requires --tasks-parallel 1" in result.stdout


def test_debug_window_rejects_suite_parallelism():
    result = runner.invoke(
        app,
        [
            "run",
            "--benchmark",
            "libero_infinity",
            "--vla",
            "smolvla",
            "--suites",
            "spatial",
            "--sequential",
            "--tasks-parallel",
            "1",
            "--suites-parallel",
            "2",
            "--debug-window",
        ],
    )

    assert result.exit_code == 1
    assert "does not support suite-level parallelism" in result.stdout


def test_build_eval_cmd_headless_default(tmp_path: Path):
    cmd = _build_eval_cmd(
        task_idx=0,
        suite="libero_infinity_spatial",
        sim_type="libero_infinity",
        sim_port=5300,
        max_episodes=1,
        experience_dir=tmp_path,
    )

    assert "--headless" in cmd


def test_build_eval_cmd_windowed_omits_headless(tmp_path: Path):
    cmd = _build_eval_cmd(
        task_idx=0,
        suite="libero_infinity_spatial",
        sim_type="libero_infinity",
        sim_port=5300,
        max_episodes=1,
        experience_dir=tmp_path,
        headless=False,
    )

    assert "--headless" not in cmd


def test_sim_worker_pool_headless_start_includes_flag(tmp_path: Path, monkeypatch):
    recorded = {}

    class DummyProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return DummyProc()

    monkeypatch.setattr("robo_eval.servers.subprocess.Popen", fake_popen)

    pool = SimWorkerPool(
        sim_type="libero_infinity",
        base_port=5300,
        num_workers=1,
        headless=True,
        logs_dir=tmp_path,
    )
    pool.start_worker(0)

    assert "--headless" in recorded["cmd"]


def test_sim_worker_pool_windowed_start_omits_flag(tmp_path: Path, monkeypatch):
    recorded = {}

    class DummyProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        recorded["kwargs"] = kwargs
        return DummyProc()

    monkeypatch.setattr("robo_eval.servers.subprocess.Popen", fake_popen)

    pool = SimWorkerPool(
        sim_type="libero_infinity",
        base_port=5300,
        num_workers=1,
        headless=False,
        logs_dir=tmp_path,
    )
    pool.start_worker(0)

    assert "--headless" not in recorded["cmd"]
