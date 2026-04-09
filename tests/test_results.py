"""
Tests for robo_eval/results.py — result collection, aggregation, and reporting.

Covers:
- wilson_ci() — confidence interval computation
- collect_results() — aggregation from logs and episodes
- _collect_task_from_episodes() — episode JSON collection
- write_scores_json() — JSON output
- write_summary() — human-readable summary
"""

import json
import math

import pytest

from robo_eval.results import (
    collect_results,
    count_successes,
    wilson_ci,
    write_scores_json,
    write_summary,
)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_all_success(self):
        lower, upper = wilson_ci(10, 10)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0
        assert upper >= lower
        # All 10 successes -> upper should be 1.0 or close
        assert upper > 0.9

    def test_all_failure(self):
        lower, upper = wilson_ci(0, 10)
        assert lower >= 0.0
        assert upper <= 1.0
        assert lower < 0.05  # Should be close to 0

    def test_half_success(self):
        lower, upper = wilson_ci(50, 100)
        assert 0.3 < lower < 0.5
        assert 0.5 < upper < 0.7

    def test_single_trial_success(self):
        lower, upper = wilson_ci(1, 1)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_single_trial_failure(self):
        lower, upper = wilson_ci(0, 1)
        assert lower >= 0.0
        assert upper <= 1.0

    def test_custom_z_score(self):
        # z=1.645 for 90% CI (narrower than 95%)
        lower_90, upper_90 = wilson_ci(50, 100, z=1.645)
        lower_95, upper_95 = wilson_ci(50, 100, z=1.96)
        assert (upper_90 - lower_90) < (upper_95 - lower_95)

    def test_bounds_clipped(self):
        # Even with extreme values, bounds should be in [0, 1]
        lower, upper = wilson_ci(0, 1000)
        assert lower >= 0.0
        lower, upper = wilson_ci(1000, 1000)
        assert upper <= 1.0


# ---------------------------------------------------------------------------
# collect_results — log-based
# ---------------------------------------------------------------------------

class TestCollectResults:
    def _write_log(self, logs_dir, suite, task_idx, successes, total):
        log_file = logs_dir / f"{suite}_task{task_idx}.log"
        lines = []
        for i in range(total):
            result = "True" if i < successes else "False"
            lines.append(f"Episode {i+1}/{total}")
            lines.append(f"Simulator reports success: {result}")
        log_file.write_text("\n".join(lines))

    def test_single_suite_single_task(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        self._write_log(logs_dir, "test_suite", 0, 7, 10)

        results = collect_results(tmp_path, ["test_suite"], num_tasks=1, max_episodes=10)
        assert results["overall"]["success"] == 7
        assert results["overall"]["total"] == 10
        assert results["overall"]["rate"] == 0.7

    def test_multiple_tasks(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        self._write_log(logs_dir, "s1", 0, 8, 10)
        self._write_log(logs_dir, "s1", 1, 6, 10)

        results = collect_results(tmp_path, ["s1"], num_tasks=2, max_episodes=10)
        assert results["overall"]["success"] == 14
        assert results["overall"]["total"] == 20

    def test_multiple_suites(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        self._write_log(logs_dir, "s1", 0, 10, 10)
        self._write_log(logs_dir, "s2", 0, 5, 10)

        results = collect_results(tmp_path, ["s1", "s2"], num_tasks=1, max_episodes=10)
        assert results["overall"]["success"] == 15
        assert results["overall"]["total"] == 20
        assert results["suites"]["s1"]["success"] == 10
        assert results["suites"]["s2"]["success"] == 5

    def test_missing_log_counted(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        # Only task 0 exists, task 1 missing
        self._write_log(logs_dir, "s1", 0, 5, 10)

        results = collect_results(tmp_path, ["s1"], num_tasks=2, max_episodes=10)
        # Missing task should count as 0/max_episodes
        assert results["overall"]["total"] == 20  # 10 + 10 (default for missing)

    def test_per_task_rate(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        self._write_log(logs_dir, "s1", 0, 3, 10)

        results = collect_results(tmp_path, ["s1"], num_tasks=1, max_episodes=10)
        task = results["tasks"][0]
        assert task["rate"] == 0.3
        assert task["n_success"] == 3
        assert task["n_episodes"] == 10

    def test_confidence_intervals_present(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        self._write_log(logs_dir, "s1", 0, 5, 10)

        results = collect_results(tmp_path, ["s1"], num_tasks=1, max_episodes=10)
        suite = results["suites"]["s1"]
        assert "ci_lower" in suite
        assert "ci_upper" in suite
        overall = results["overall"]
        assert "ci_lower" in overall
        assert "ci_upper" in overall

    def test_task_std_dev(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        # Mixed results: should have nonzero std
        self._write_log(logs_dir, "s1", 0, 5, 10)

        results = collect_results(tmp_path, ["s1"], num_tasks=1, max_episodes=10)
        task = results["tasks"][0]
        assert task["std"] > 0


# ---------------------------------------------------------------------------
# collect_results — episode JSON-based
# ---------------------------------------------------------------------------

class TestCollectResultsFromEpisodes:
    def test_prefers_episode_json_over_log(self, tmp_path):
        from robo_eval.episode_logger import EpisodeResult, save_episode_result

        # Write log file with 3/5 success
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        lines = ["Simulator reports success: True"] * 3 + ["Simulator reports success: False"] * 2
        (logs_dir / "test_suite_task0.log").write_text("\n".join(lines))

        # Write episode JSONs with 4/5 success (should be preferred)
        for i in range(5):
            ep = EpisodeResult(task=0, episode=i, success=(i < 4), steps=100, duration_s=10.0, vla_calls=50)
            save_episode_result(tmp_path, "test_suite", 0, i, ep)

        results = collect_results(tmp_path, ["test_suite"], num_tasks=1, max_episodes=5)
        assert results["overall"]["success"] == 4  # From episodes, not log


# ---------------------------------------------------------------------------
# write_scores_json
# ---------------------------------------------------------------------------

class TestWriteScoresJson:
    def test_writes_valid_json(self, tmp_path):
        results = {"overall": {"success": 10, "total": 20, "rate": 0.5}}
        output = tmp_path / "scores.json"
        write_scores_json(results, output)
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["overall"]["success"] == 10


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------

class TestWriteSummary:
    def test_writes_summary_file(self, tmp_path):
        results = {
            "tasks": [
                {"suite": "s1", "task": 0, "n_success": 8, "n_episodes": 10,
                 "rate": 0.8, "std": 0.1, "complete": True},
            ],
            "suites": {
                "s1": {"success": 8, "total": 10, "rate": 0.8, "ci_lower": 0.5, "ci_upper": 0.95}
            },
            "overall": {"success": 8, "total": 10, "rate": 0.8, "ci_lower": 0.5, "ci_upper": 0.95},
        }
        output = tmp_path / "summary.txt"
        text = write_summary(results, output)
        assert output.exists()
        assert "s1" in text
        assert "80.0%" in text

    def test_includes_metadata(self, tmp_path):
        results = {
            "tasks": [],
            "suites": {},
            "overall": {"success": 0, "total": 0, "rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0},
        }
        metadata = {"timestamp": "2026-03-01T00:00:00", "vla": "pi05"}
        output = tmp_path / "summary.txt"
        text = write_summary(results, output, metadata)
        assert "2026-03-01" in text
        assert "pi05" in text
