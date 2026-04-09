"""
Tests for robo_eval/episode_logger.py — structured episode result logging.

Covers:
- EpisodeResult dataclass creation and defaults
- save_episode_result() writes correct JSON
- load_episode_results() reads and sorts
- episode_results_for_task() filters by task
- Edge cases: missing dir, corrupt files
"""

import json

import pytest

from robo_eval.episode_logger import (
    EpisodeResult,
    episode_results_for_task,
    load_episode_results,
    save_episode_result,
)


# ---------------------------------------------------------------------------
# EpisodeResult dataclass
# ---------------------------------------------------------------------------

class TestEpisodeResult:
    def test_create_basic(self):
        ep = EpisodeResult(task=0, episode=1, success=True, steps=100, duration_s=12.5, vla_calls=50)
        assert ep.task == 0
        assert ep.episode == 1
        assert ep.success is True
        assert ep.steps == 100
        assert ep.duration_s == 12.5
        assert ep.vla_calls == 50

    def test_timestamp_auto_set(self):
        ep = EpisodeResult(task=0, episode=0, success=False, steps=10, duration_s=1.0, vla_calls=5)
        assert ep.timestamp != ""
        # Should be ISO format
        assert "T" in ep.timestamp

    def test_timestamp_explicit(self):
        ep = EpisodeResult(
            task=0, episode=0, success=True, steps=10,
            duration_s=1.0, vla_calls=5, timestamp="2026-01-01T00:00:00"
        )
        assert ep.timestamp == "2026-01-01T00:00:00"

    def test_subtasks_default_empty(self):
        ep = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        assert ep.subtasks == []

    def test_subtasks_populated(self):
        ep = EpisodeResult(
            task=0, episode=0, success=True, steps=10,
            duration_s=1.0, vla_calls=5,
            subtasks=["pick bowl", "place on counter"]
        )
        assert len(ep.subtasks) == 2
        assert ep.subtasks[0] == "pick bowl"


# ---------------------------------------------------------------------------
# save_episode_result
# ---------------------------------------------------------------------------

class TestSaveEpisodeResult:
    def test_writes_json_file(self, tmp_path):
        ep = EpisodeResult(task=3, episode=2, success=True, steps=150, duration_s=20.0, vla_calls=75)
        filepath = save_episode_result(tmp_path, "libero_spatial", 3, 2, ep)
        assert filepath.exists()
        assert filepath.name == "libero_spatial_task3_ep2.json"

    def test_json_content_correct(self, tmp_path):
        ep = EpisodeResult(
            task=0, episode=1, success=False, steps=280, duration_s=30.0, vla_calls=140,
            subtasks=["pick", "place"], timestamp="2026-03-01T12:00:00"
        )
        filepath = save_episode_result(tmp_path, "libero_goal", 0, 1, ep)
        data = json.loads(filepath.read_text())
        assert data["task"] == 0
        assert data["episode"] == 1
        assert data["success"] is False
        assert data["steps"] == 280
        assert data["duration_s"] == 30.0
        assert data["vla_calls"] == 140
        assert data["subtasks"] == ["pick", "place"]
        assert data["timestamp"] == "2026-03-01T12:00:00"

    def test_creates_episodes_subdir(self, tmp_path):
        ep = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        save_episode_result(tmp_path, "suite_a", 0, 0, ep)
        assert (tmp_path / "episodes").is_dir()

    def test_multiple_episodes(self, tmp_path):
        for i in range(5):
            ep = EpisodeResult(task=0, episode=i, success=(i % 2 == 0), steps=100, duration_s=10.0, vla_calls=50)
            save_episode_result(tmp_path, "libero_spatial", 0, i, ep)
        files = list((tmp_path / "episodes").glob("*.json"))
        assert len(files) == 5


# ---------------------------------------------------------------------------
# load_episode_results
# ---------------------------------------------------------------------------

class TestLoadEpisodeResults:
    def test_load_empty_dir(self, tmp_path):
        assert load_episode_results(tmp_path, "libero_spatial") == []

    def test_load_no_episodes_dir(self, tmp_path):
        assert load_episode_results(tmp_path, "libero_spatial") == []

    def test_load_sorted_by_task_and_episode(self, tmp_path):
        # Save in reverse order
        for task in [1, 0]:
            for ep in [2, 1, 0]:
                result = EpisodeResult(task=task, episode=ep, success=True, steps=10, duration_s=1.0, vla_calls=5)
                save_episode_result(tmp_path, "libero_spatial", task, ep, result)

        loaded = load_episode_results(tmp_path, "libero_spatial")
        assert len(loaded) == 6
        # Should be sorted by (task, episode)
        assert loaded[0]["task"] == 0 and loaded[0]["episode"] == 0
        assert loaded[1]["task"] == 0 and loaded[1]["episode"] == 1
        assert loaded[2]["task"] == 0 and loaded[2]["episode"] == 2
        assert loaded[3]["task"] == 1 and loaded[3]["episode"] == 0

    def test_load_filters_by_suite(self, tmp_path):
        ep_a = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        ep_b = EpisodeResult(task=0, episode=0, success=False, steps=10, duration_s=1.0, vla_calls=5)
        save_episode_result(tmp_path, "suite_a", 0, 0, ep_a)
        save_episode_result(tmp_path, "suite_b", 0, 0, ep_b)

        results_a = load_episode_results(tmp_path, "suite_a")
        assert len(results_a) == 1
        assert results_a[0]["success"] is True

    def test_load_skips_corrupt_json(self, tmp_path):
        # Write a valid episode
        ep = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        save_episode_result(tmp_path, "suite_x", 0, 0, ep)
        # Write a corrupt file
        corrupt = tmp_path / "episodes" / "suite_x_task0_ep1.json"
        corrupt.write_text("{invalid json")

        results = load_episode_results(tmp_path, "suite_x")
        assert len(results) == 1  # Only the valid one


# ---------------------------------------------------------------------------
# episode_results_for_task
# ---------------------------------------------------------------------------

class TestEpisodeResultsForTask:
    def test_filter_specific_task(self, tmp_path):
        for task in range(3):
            for ep in range(2):
                result = EpisodeResult(task=task, episode=ep, success=True, steps=10, duration_s=1.0, vla_calls=5)
                save_episode_result(tmp_path, "test_suite", task, ep, result)

        task1_results = episode_results_for_task(tmp_path, "test_suite", 1)
        assert len(task1_results) == 2
        assert all(r["task"] == 1 for r in task1_results)

    def test_nonexistent_task(self, tmp_path):
        ep = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        save_episode_result(tmp_path, "test_suite", 0, 0, ep)
        assert episode_results_for_task(tmp_path, "test_suite", 99) == []

    def test_no_episodes_dir(self, tmp_path):
        assert episode_results_for_task(tmp_path, "test_suite", 0) == []

    def test_sorted_by_episode(self, tmp_path):
        for ep in [4, 2, 0, 3, 1]:
            result = EpisodeResult(task=5, episode=ep, success=True, steps=10, duration_s=1.0, vla_calls=5)
            save_episode_result(tmp_path, "test_suite", 5, ep, result)

        results = episode_results_for_task(tmp_path, "test_suite", 5)
        episodes = [r["episode"] for r in results]
        assert episodes == [0, 1, 2, 3, 4]
