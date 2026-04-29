"""Unit tests for roboeval.orchestrator.

Tests:
    - Sharding: round-robin partition correctness across multiple shard_id/num_shards.
    - Result file naming: non-shard and shard variants.
    - Atomic progress file write.
    - File-lock claim/release.
    - Subprocess command building (no real eval launched).
    - Episode success/failure recording.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roboeval.orchestrator import (  # tests adjust sys.path before importing local package.
    EvalConfig,
    Orchestrator,
    _parse_steps_from_stdout,
    _parse_success_from_stdout,
)

# ---------------------------------------------------------------------------
# EvalConfig tests
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.no_vlm is True
        assert cfg.delta_actions is True
        assert cfg.episodes_per_task == 10

    def test_from_dict_minimal(self):
        cfg = EvalConfig.from_dict({"name": "test_run"})
        assert cfg.name == "test_run"
        assert cfg.no_vlm is True

    def test_from_dict_full(self):
        d = {
            "name": "full_run",
            "vla_url": "http://localhost:9999",
            "sim_url": "http://localhost:9998",
            "sim": "robocasa",
            "suite": "libero_object",
            "tasks": [0, 1, 2],
            "max_tasks": 3,
            "episodes_per_task": 5,
            "no_vlm": False,
            "vlm_model": "gpt-4",
            "delta_actions": False,
            "output_dir": "/tmp/test_out",
            "params": {"seed": 42},
        }
        cfg = EvalConfig.from_dict(d)
        assert cfg.vla_url == "http://localhost:9999"
        assert cfg.tasks == [0, 1, 2]
        assert cfg.episodes_per_task == 5
        assert cfg.no_vlm is False
        assert cfg.vlm_model == "gpt-4"
        assert cfg.params == {"seed": 42}

    def test_to_dict_round_trip(self):
        cfg = EvalConfig.from_dict({"name": "rt", "episodes_per_task": 7})
        d = cfg.to_dict()
        cfg2 = EvalConfig.from_dict(d)
        assert cfg2.episodes_per_task == 7
        assert cfg2.name == "rt"

    def test_from_yaml(self, tmp_path):
        yaml_content = """
name: yaml_test
suite: libero_spatial
episodes_per_task: 3
max_tasks: 2
no_vlm: true
"""
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        cfg = EvalConfig.from_yaml(p)
        assert cfg.name == "yaml_test"
        assert cfg.episodes_per_task == 3
        assert cfg.max_tasks == 2

    def test_from_dict_preserves_single_named_task(self):
        cfg = EvalConfig.from_dict(
            {"name": "mw", "sim": "metaworld", "suite": "", "task": "button-press-v2"}
        )
        assert cfg.task == "button-press-v2"
        assert cfg.tasks == []


# ---------------------------------------------------------------------------
# Sharding tests
# ---------------------------------------------------------------------------


class TestSharding:
    """Verify round-robin sharding partitions work items correctly."""

    def _make_orch(self, shard_id, num_shards, episodes_per_task=6, max_tasks=2):
        cfg = EvalConfig.from_dict(
            {
                "name": "shard_test",
                "episodes_per_task": episodes_per_task,
                "max_tasks": max_tasks,
                "suite": "libero_spatial",
                "output_dir": "/tmp/shard_test",
            }
        )
        return Orchestrator(config=cfg, shard_id=shard_id, num_shards=num_shards)

    def _get_work_items(self, orch: Orchestrator) -> list[tuple[int, int]]:
        """Simulate what Orchestrator.run() does to build work items."""
        cfg = orch.config
        tasks = orch._build_task_list()
        all_items = [(t, ep) for t in tasks for ep in range(cfg.episodes_per_task)]
        if orch.num_shards is not None and orch.shard_id is not None:
            all_items = [w for i, w in enumerate(all_items) if i % orch.num_shards == orch.shard_id]
        return all_items

    def test_2shards_partition_coverage(self):
        """2 shards cover all items exactly once."""
        items_0 = self._get_work_items(self._make_orch(0, 2))
        items_1 = self._get_work_items(self._make_orch(1, 2))
        all_items = items_0 + items_1

        # No duplicates
        assert len(all_items) == len(set(all_items))
        # Full coverage (2 tasks × 6 episodes = 12)
        assert len(all_items) == 12

    def test_4shards_partition_coverage(self):
        """4 shards cover 2 tasks × 6 episodes = 12 items exactly."""
        all_items = []
        for sid in range(4):
            all_items.extend(self._get_work_items(self._make_orch(sid, 4)))
        assert len(all_items) == len(set(all_items)), "Duplicate items found"
        assert len(all_items) == 12

    def test_shard_sizes_approximately_equal(self):
        """Shards have roughly equal size for 4 shards × 12 total items."""
        sizes = [len(self._get_work_items(self._make_orch(sid, 4))) for sid in range(4)]
        assert max(sizes) - min(sizes) <= 1, f"Unbalanced shards: {sizes}"

    def test_no_shard_all_items(self):
        """Non-sharded orchestrator processes all items."""
        cfg = EvalConfig.from_dict(
            {"name": "x", "episodes_per_task": 3, "max_tasks": 2, "output_dir": "/tmp"}
        )
        orch = Orchestrator(config=cfg)
        items = self._get_work_items(orch)
        assert len(items) == 6  # 2 tasks × 3 episodes

    def test_shard_id_0_gets_first_item(self):
        """Shard 0 gets item 0 (round-robin starts at 0)."""
        items = self._get_work_items(self._make_orch(0, 3, episodes_per_task=3, max_tasks=1))
        # 1 task × 3 episodes = items (0,0),(0,1),(0,2); shard 0 gets 0,2 in order? No:
        # item 0 -> shard 0, item 1 -> shard 1, item 2 -> shard 2
        assert (0, 0) in items

    def test_single_shard(self):
        """1-of-1 sharding is the same as no sharding."""
        cfg = EvalConfig.from_dict(
            {"name": "s", "episodes_per_task": 4, "max_tasks": 2, "output_dir": "/tmp"}
        )
        orch = Orchestrator(config=cfg, shard_id=0, num_shards=1)
        items = self._get_work_items(orch)
        assert len(items) == 8

    def test_named_task_config_builds_single_named_work_item(self):
        cfg = EvalConfig.from_dict(
            {
                "name": "metaworld",
                "sim": "metaworld",
                "suite": "",
                "task": "button-press-v2",
                "episodes_per_task": 2,
                "output_dir": "/tmp",
            }
        )
        orch = Orchestrator(config=cfg)
        items = self._get_work_items(orch)
        assert items == [("button-press-v2", 0), ("button-press-v2", 1)]


# ---------------------------------------------------------------------------
# Result file naming
# ---------------------------------------------------------------------------


class TestResultFileNaming:
    def test_non_shard_name_contains_timestamp(self):
        cfg = EvalConfig.from_dict({"name": "my run!", "output_dir": "/tmp"})
        orch = Orchestrator(config=cfg)
        stem = orch._shard_stem("my_run_")
        # Non-shard: stem is returned unchanged
        assert "shard" not in stem

    def test_shard_name_format(self):
        cfg = EvalConfig.from_dict({"name": "eval", "output_dir": "/tmp"})
        orch = Orchestrator(config=cfg, shard_id=2, num_shards=8)
        stem = orch._shard_stem("eval")
        assert stem == "eval_shard2of8"

    def test_shard_name_zero_padded_not_required(self):
        cfg = EvalConfig.from_dict({"name": "x", "output_dir": "/tmp"})
        orch = Orchestrator(config=cfg, shard_id=0, num_shards=4)
        assert orch._shard_stem("x") == "x_shard0of4"

    def test_claim_output_path_shard(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "claim_test", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg, shard_id=1, num_shards=4)
        output_path = orch._claim_output_path("claim_test")
        assert "shard1of4" in output_path.name
        orch._release_lock()

    def test_claim_output_path_non_shard(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "claim_test", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg)
        output_path = orch._claim_output_path("claim_test")
        assert "shard" not in output_path.name
        assert output_path.suffix == ".json"
        orch._release_lock()


# ---------------------------------------------------------------------------
# Atomic progress file
# ---------------------------------------------------------------------------


class TestAtomicProgress:
    def test_progress_written_atomically(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "prog", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg)
        orch._progress_path = tmp_path / "prog.progress"

        orch._update_progress(3, 10, 1)

        assert orch._progress_path.exists()
        data = json.loads(orch._progress_path.read_text())
        assert data == {"completed": 3, "total": 10, "errors": 1}

    def test_progress_overwritten(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "prog", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg)
        orch._progress_path = tmp_path / "prog.progress"

        orch._update_progress(1, 5, 0)
        orch._update_progress(2, 5, 1)

        data = json.loads(orch._progress_path.read_text())
        assert data["completed"] == 2
        assert data["errors"] == 1

    def test_progress_no_path_no_error(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "prog", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg)
        # _progress_path is None by default
        orch._update_progress(0, 5, 0)  # Should not raise


# ---------------------------------------------------------------------------
# File lock claim/release
# ---------------------------------------------------------------------------


class TestFileLock:
    def test_lock_and_release(self, tmp_path):
        cfg = EvalConfig.from_dict({"name": "lock_test", "output_dir": str(tmp_path)})
        orch = Orchestrator(config=cfg, shard_id=0, num_shards=2)
        path = orch._claim_output_path("lock_test")
        # Lock file should exist while lock is held
        lock_path = Path(str(path) + ".lock")
        assert lock_path.exists()
        orch._release_lock()
        # Lock file removed after release
        assert not lock_path.exists()

    def test_double_lock_fails(self, tmp_path):
        """A second orchestrator cannot claim the same shard path."""
        cfg1 = EvalConfig.from_dict({"name": "lock_test2", "output_dir": str(tmp_path)})
        orch1 = Orchestrator(config=cfg1, shard_id=0, num_shards=2)
        orch2 = Orchestrator(
            config=EvalConfig.from_dict({"name": "lock_test2", "output_dir": str(tmp_path)}),
            shard_id=0,
            num_shards=2,
        )

        orch1._claim_output_path("lock_test2")
        try:
            with pytest.raises((FileExistsError, OSError)):
                orch2._claim_output_path("lock_test2")
        finally:
            orch1._release_lock()


# ---------------------------------------------------------------------------
# Subprocess command building
# ---------------------------------------------------------------------------


class TestSubprocessCmd:
    def test_cmd_contains_sim_and_suite(self):
        cfg = EvalConfig.from_dict(
            {
                "name": "x",
                "sim": "libero",
                "suite": "libero_spatial",
                "sim_url": "http://localhost:5300",
                "no_vlm": True,
                "delta_actions": True,
                "output_dir": "/tmp",
            }
        )
        orch = Orchestrator(config=cfg)
        cmd = orch._build_subprocess_cmd(task_id=3, episode=7)
        assert "--sim" in cmd
        assert "libero" in cmd
        assert "--suite" in cmd
        assert "libero_spatial" in cmd
        assert "--task" in cmd
        assert "3" in cmd
        assert "--episode" in cmd
        assert "7" in cmd

    def test_cmd_no_vlm_flag(self):
        cfg = EvalConfig.from_dict({"name": "x", "no_vlm": True, "output_dir": "/tmp"})
        orch = Orchestrator(config=cfg)
        cmd = orch._build_subprocess_cmd(0, 0)
        assert "--no-vlm" in cmd

    def test_cmd_with_vlm_model(self):
        cfg = EvalConfig.from_dict(
            {"name": "x", "no_vlm": False, "vlm_model": "gpt-4o", "output_dir": "/tmp"}
        )
        orch = Orchestrator(config=cfg)
        cmd = orch._build_subprocess_cmd(0, 0)
        assert "--no-vlm" not in cmd
        assert "--vlm-model" in cmd
        assert "gpt-4o" in cmd

    def test_cmd_vla_url_in_env(self):
        cfg = EvalConfig.from_dict(
            {"name": "x", "vla_url": "http://localhost:9876", "output_dir": "/tmp"}
        )
        orch = Orchestrator(config=cfg)
        env = orch._build_subprocess_env()
        assert env["VLA_URL"] == "http://localhost:9876"

    def test_cmd_extra_env_overrides_vla_url(self):
        cfg = EvalConfig.from_dict(
            {"name": "x", "vla_url": "http://localhost:9876", "output_dir": "/tmp"}
        )
        orch = Orchestrator(config=cfg, extra_env={"VLA_URL": "http://localhost:1234"})
        env = orch._build_subprocess_env()
        assert env["VLA_URL"] == "http://localhost:1234"


# ---------------------------------------------------------------------------
# Orchestrator.run() with mocked subprocess
# ---------------------------------------------------------------------------


class TestOrchestratorRunMocked:
    """Test Orchestrator.run() without launching real subprocesses."""

    def _make_fake_episode_result(self, success: bool, steps: int = 50) -> dict:
        return {
            "episode_id": 0,
            "metrics": {"success": success},
            "steps": steps,
            "elapsed_sec": 1.0,
        }

    def test_run_records_success(self, tmp_path):
        cfg = EvalConfig.from_dict(
            {
                "name": "mock_run",
                "episodes_per_task": 2,
                "max_tasks": 1,
                "output_dir": str(tmp_path),
            }
        )
        orch = Orchestrator(config=cfg)

        # Patch _run_episode to return a success result
        call_count = 0

        def fake_run_episode(task_id, episode):
            nonlocal call_count
            call_count += 1
            return {
                "episode_id": episode,
                "metrics": {"success": True},
                "steps": 42,
                "elapsed_sec": 0.5,
            }

        orch._run_episode = fake_run_episode

        result = orch.run()

        assert call_count == 2  # 2 episodes, 1 task
        assert result.get("mean_success") == 1.0

    def test_run_handles_episode_exception(self, tmp_path):
        """One episode exception does not abort the run."""
        cfg = EvalConfig.from_dict(
            {
                "name": "err_run",
                "episodes_per_task": 2,
                "max_tasks": 1,
                "output_dir": str(tmp_path),
            }
        )
        orch = Orchestrator(config=cfg)

        def failing_run_episode(task_id, episode):
            if episode == 0:
                raise RuntimeError("Simulated crash")
            return {
                "episode_id": episode,
                "metrics": {"success": True},
                "steps": 10,
            }

        orch._run_episode = failing_run_episode
        result = orch.run()

        # Should complete with partial results
        assert result is not None
        # One success out of 2 episodes
        tasks = result.get("tasks", [])
        total_eps = sum(len(t.get("episodes", [])) for t in tasks)
        assert total_eps == 2

    def test_run_writes_result_json(self, tmp_path):
        cfg = EvalConfig.from_dict(
            {
                "name": "json_run",
                "episodes_per_task": 1,
                "max_tasks": 1,
                "output_dir": str(tmp_path),
            }
        )
        orch = Orchestrator(config=cfg)
        orch._run_episode = lambda t, e: {
            "episode_id": e,
            "metrics": {"success": True},
            "steps": 5,
        }

        orch.run()

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["benchmark"] == "json_run"

    def test_shard_run_writes_shard_json(self, tmp_path):
        cfg = EvalConfig.from_dict(
            {
                "name": "shard_run",
                "episodes_per_task": 2,
                "max_tasks": 1,
                "output_dir": str(tmp_path),
            }
        )
        orch = Orchestrator(config=cfg, shard_id=0, num_shards=2)
        orch._run_episode = lambda t, e: {
            "episode_id": e,
            "metrics": {"success": True},
            "steps": 5,
        }

        orch.run()

        shard_files = list(tmp_path.glob("*shard0of2.json"))
        assert len(shard_files) == 1
        data = json.loads(shard_files[0].read_text())
        assert data["shard"]["id"] == 0
        assert data["shard"]["total"] == 2


# ---------------------------------------------------------------------------
# Stdout parsers
# ---------------------------------------------------------------------------


class TestStdoutParsers:
    def test_parse_success_true(self):
        stdout = "Simulator reports success: True\nEpisode complete."
        assert _parse_success_from_stdout(stdout) is True

    def test_parse_success_false(self):
        stdout = "Simulator reports success: False\nEpisode complete."
        assert _parse_success_from_stdout(stdout) is False

    def test_parse_success_empty(self):
        assert _parse_success_from_stdout("") is False

    def test_parse_steps(self):
        stdout = "total steps=147 time=2.3s"
        assert _parse_steps_from_stdout(stdout) == 147

    def test_parse_steps_not_found(self):
        assert _parse_steps_from_stdout("no steps here") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
