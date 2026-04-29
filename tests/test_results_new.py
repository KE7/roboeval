"""Unit tests for robo_eval.results (collector and merge).

Tests:
    - ResultCollector: record, aggregate, print_summary
    - merge_shards: single shard, all shards, missing shard, duplicate episode_id
    - round-trip JSON serialization
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_eval.results.collector import (
    ResultCollector,
    EpisodeResult,
    _build_task_result,
    _aggregate_metrics,
)
from robo_eval.results.merge import merge_shards, load_shard_files, find_shard_files


# ---------------------------------------------------------------------------
# ResultCollector tests
# ---------------------------------------------------------------------------


class TestResultCollector:
    def _make_episode(self, ep_id: int, success: bool, steps: int = 50) -> EpisodeResult:
        return EpisodeResult(
            episode_id=ep_id,
            metrics={"success": success},
            steps=steps,
            elapsed_sec=1.0,
        )

    def test_record_and_get_task_result(self):
        col = ResultCollector("test_bench", metric_keys={"success": "mean"})
        col.record("task_0", self._make_episode(0, True))
        col.record("task_0", self._make_episode(1, False))
        tr = col.get_task_result("task_0")
        assert tr["num_episodes"] == 2
        assert tr["mean_success"] == pytest.approx(0.5)

    def test_error_count(self):
        col = ResultCollector("test_bench")
        col.record("task_0", EpisodeResult(episode_id=0, metrics={"success": False},
                                            failure_reason="crash", failure_detail="oops"))
        col.record("task_0", self._make_episode(1, True))
        assert col.error_count == 1

    def test_multiple_tasks(self):
        col = ResultCollector("multi_task_bench", metric_keys={"success": "mean"})
        for t in range(3):
            for ep in range(4):
                col.record(f"task_{t}", self._make_episode(ep, ep % 2 == 0))
        result = col.get_benchmark_result()
        assert len(result["tasks"]) == 3
        assert result.get("mean_success") is not None

    def test_to_json_serializable(self):
        col = ResultCollector("json_bench", metric_keys={"success": "mean"})
        col.record("task_0", self._make_episode(0, True, steps=30))
        js = col.to_json()
        data = json.loads(js)
        assert data["benchmark"] == "json_bench"
        assert len(data["tasks"]) == 1

    def test_metric_keys_mean_success(self):
        col = ResultCollector("x", metric_keys={"success": "mean"})
        for i in range(5):
            col.record("task_0", self._make_episode(i, i < 4))  # 4/5 success
        result = col.get_benchmark_result()
        assert result["mean_success"] == pytest.approx(0.8)

    def test_numpy_scalar_normalization(self):
        """Numpy scalars in metrics are JSON-serializable."""
        import numpy as np
        col = ResultCollector("np_bench", metric_keys={"success": "mean"})
        ep = EpisodeResult(
            episode_id=0,
            metrics={"success": np.bool_(True)},  # type: ignore[dict-item]
            steps=5,
        )
        col.record("task_0", ep)
        js = col.to_json()
        data = json.loads(js)
        assert data["tasks"][0]["episodes"][0]["metrics"]["success"] is True

    def test_print_summary_no_error(self):
        col = ResultCollector("print_test", metric_keys={"success": "mean"})
        col.record("task_0", self._make_episode(0, True))
        # Should not raise (rich may or may not be installed)
        col.print_summary()


# ---------------------------------------------------------------------------
# merge_shards tests
# ---------------------------------------------------------------------------


def _make_shard(
    shard_id: int,
    total: int,
    benchmark: str = "test_bench",
    task_name: str = "task_0",
    episodes: list | None = None,
    partial: bool = False,
    metric_keys: dict | None = None,
) -> dict:
    """Helper to construct a minimal shard dict."""
    if episodes is None:
        ep_id = shard_id  # deterministic per shard
        episodes = [
            {"episode_id": ep_id, "metrics": {"success": True}, "steps": 50}
        ]
    result = {
        "benchmark": benchmark,
        "shard": {"id": shard_id, "total": total},
        "tasks": [
            {
                "task": task_name,
                "episodes": episodes,
                "num_episodes": len(episodes),
                "avg_steps": 50.0,
            }
        ],
        "mode": "sync",
        "harness_version": "0.1.0",
        "metric_keys": metric_keys or {"success": "mean"},
    }
    if partial:
        result["partial"] = True
    return result


class TestMergeShards:
    def test_single_shard_merge(self):
        shard = _make_shard(0, 1, episodes=[
            {"episode_id": 0, "metrics": {"success": True}, "steps": 50},
        ])
        merged = merge_shards([shard])
        assert merged["benchmark"] == "test_bench"
        assert merged.get("partial") is None or merged.get("partial") is False
        info = merged["merge_info"]
        assert info["total_episodes"] == 1
        assert info["shards_missing"] == []

    def test_all_shards_present(self):
        shards = []
        for i in range(4):
            shards.append(_make_shard(i, 4, episodes=[
                {"episode_id": i, "metrics": {"success": True}, "steps": 50}
            ]))
        merged = merge_shards(shards)
        assert merged.get("partial") is None or not merged.get("partial")
        assert merged["merge_info"]["total_episodes"] == 4
        assert len(merged["merge_info"]["shards_missing"]) == 0

    def test_missing_shard_marks_partial(self):
        """Missing shard sets partial=True."""
        shards = [
            _make_shard(0, 3, episodes=[
                {"episode_id": 0, "metrics": {"success": True}, "steps": 50}
            ]),
            _make_shard(2, 3, episodes=[
                {"episode_id": 2, "metrics": {"success": True}, "steps": 50}
            ]),
            # shard 1 is missing
        ]
        merged = merge_shards(shards)
        assert merged.get("partial") is True
        assert 1 in merged["merge_info"]["shards_missing"]

    def test_duplicate_episode_id_last_write_wins(self):
        """Duplicate episode_id across shards keeps the last shard's entry."""
        # Two shards both contain episode_id=0 but with different success values
        shard_a = _make_shard(0, 2, episodes=[
            {"episode_id": 0, "metrics": {"success": False}, "steps": 10}
        ])
        shard_b = _make_shard(1, 2, episodes=[
            {"episode_id": 0, "metrics": {"success": True}, "steps": 50}  # wins
        ])
        merged = merge_shards([shard_a, shard_b])
        task_episodes = merged["tasks"][0]["episodes"]
        ep0 = next(e for e in task_episodes if e["episode_id"] == 0)
        assert ep0["metrics"]["success"] is True

    def test_benchmark_mismatch_raises(self):
        shards = [
            _make_shard(0, 2, benchmark="bench_a"),
            _make_shard(1, 2, benchmark="bench_b"),
        ]
        with pytest.raises(ValueError, match="Benchmark mismatch"):
            merge_shards(shards)

    def test_total_mismatch_raises(self):
        shards = [
            _make_shard(0, 2),
            _make_shard(1, 3),  # total mismatch
        ]
        with pytest.raises(ValueError, match="total mismatch"):
            merge_shards(shards)

    def test_duplicate_shard_id_raises(self):
        shards = [
            _make_shard(0, 2),
            _make_shard(0, 2),  # duplicate id
        ]
        with pytest.raises(ValueError, match="Duplicate shard IDs"):
            merge_shards(shards)

    def test_empty_shards_raises(self):
        with pytest.raises(ValueError, match="No shard files"):
            merge_shards([])

    def test_metric_aggregates_recomputed(self):
        """Mean success is correctly aggregated across merged shards."""
        shards = []
        # 4 shards, each with 1 episode; 2 succeed, 2 fail
        for i in range(4):
            shards.append(_make_shard(i, 4, episodes=[
                {"episode_id": i, "metrics": {"success": i < 2}, "steps": 50}
            ]))
        merged = merge_shards(shards)
        assert merged["mean_success"] == pytest.approx(0.5)

    def test_config_snapshot_preserved(self):
        shard = _make_shard(0, 1)
        shard["config"] = {"name": "test_snap", "suite": "libero_spatial"}
        merged = merge_shards([shard])
        assert merged["config"]["name"] == "test_snap"

    def test_merge_info_fields(self):
        shards = [_make_shard(i, 3, episodes=[
            {"episode_id": i, "metrics": {"success": True}, "steps": 10}
        ]) for i in range(3)]
        merged = merge_shards(shards)
        info = merged["merge_info"]
        assert info["num_shards"] == 3
        assert sorted(info["shards_found"]) == [0, 1, 2]
        assert info["shards_missing"] == []
        assert info["total_episodes"] == 3


class TestLoadShardFiles:
    def test_load_valid_shard_files(self, tmp_path):
        shards = [_make_shard(i, 2) for i in range(2)]
        paths = []
        for i, s in enumerate(shards):
            p = tmp_path / f"shard_{i}.json"
            p.write_text(json.dumps(s))
            paths.append(p)

        loaded = load_shard_files(paths)
        assert len(loaded) == 2
        assert loaded[0]["shard"]["id"] == 0

    def test_load_missing_shard_key_raises(self, tmp_path):
        p = tmp_path / "not_a_shard.json"
        p.write_text(json.dumps({"benchmark": "x", "tasks": []}))
        with pytest.raises(ValueError, match="missing 'shard' field"):
            load_shard_files([p])

    def test_find_shard_files_glob(self, tmp_path):
        for i in range(3):
            (tmp_path / f"result_shard{i}of3.json").write_text(
                json.dumps(_make_shard(i, 3))
            )
        paths = find_shard_files(str(tmp_path / "*shard*.json"))
        assert len(paths) == 3

    def test_find_shard_files_no_match_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_shard_files(str(tmp_path / "no_match_*.json"))


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_collector_to_json_and_back(self):
        col = ResultCollector("roundtrip_bench", metric_keys={"success": "mean"})
        for ep in range(3):
            col.record("task_0", EpisodeResult(
                episode_id=ep, metrics={"success": ep > 0}, steps=100
            ))
        js = col.to_json()
        data = json.loads(js)
        assert data["tasks"][0]["num_episodes"] == 3
        # Use abs tolerance to handle 4-decimal rounding.
        assert data["mean_success"] == pytest.approx(2 / 3, abs=1e-3)

    def test_merge_and_json_serializable(self):
        shards = [_make_shard(i, 2, episodes=[
            {"episode_id": i, "metrics": {"success": True}, "steps": 50}
        ]) for i in range(2)]
        merged = merge_shards(shards)
        # Should be fully JSON serializable
        js = json.dumps(merged, default=str)
        data = json.loads(js)
        assert data["benchmark"] == "test_bench"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
