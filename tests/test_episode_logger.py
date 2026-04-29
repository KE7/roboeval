"""
Tests for roboeval/episode_logger.py — structured episode result logging.

Covers:
- EpisodeResult dataclass creation and defaults
- save_episode_result() writes correct JSON
"""

import json

from roboeval.episode_logger import (
    EpisodeResult,
    save_episode_result,
)

# ---------------------------------------------------------------------------
# EpisodeResult dataclass
# ---------------------------------------------------------------------------


class TestEpisodeResult:
    def test_create_basic(self):
        ep = EpisodeResult(
            task=0, episode=1, success=True, steps=100, duration_s=12.5, vla_calls=50
        )
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
            task=0,
            episode=0,
            success=True,
            steps=10,
            duration_s=1.0,
            vla_calls=5,
            timestamp="2026-01-01T00:00:00",
        )
        assert ep.timestamp == "2026-01-01T00:00:00"

    def test_subtasks_default_empty(self):
        ep = EpisodeResult(task=0, episode=0, success=True, steps=10, duration_s=1.0, vla_calls=5)
        assert ep.subtasks == []

    def test_subtasks_populated(self):
        ep = EpisodeResult(
            task=0,
            episode=0,
            success=True,
            steps=10,
            duration_s=1.0,
            vla_calls=5,
            subtasks=["pick bowl", "place on counter"],
        )
        assert len(ep.subtasks) == 2
        assert ep.subtasks[0] == "pick bowl"


# ---------------------------------------------------------------------------
# save_episode_result
# ---------------------------------------------------------------------------


class TestSaveEpisodeResult:
    def test_writes_json_file(self, tmp_path):
        ep = EpisodeResult(
            task=3, episode=2, success=True, steps=150, duration_s=20.0, vla_calls=75
        )
        filepath = save_episode_result(tmp_path, "libero_spatial", 3, 2, ep)
        assert filepath.exists()
        assert filepath.name == "libero_spatial_task3_ep2.json"

    def test_json_content_correct(self, tmp_path):
        ep = EpisodeResult(
            task=0,
            episode=1,
            success=False,
            steps=280,
            duration_s=30.0,
            vla_calls=140,
            subtasks=["pick", "place"],
            timestamp="2026-03-01T12:00:00",
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
            ep = EpisodeResult(
                task=0, episode=i, success=(i % 2 == 0), steps=100, duration_s=10.0, vla_calls=50
            )
            save_episode_result(tmp_path, "libero_spatial", 0, i, ep)
        files = list((tmp_path / "episodes").glob("*.json"))
        assert len(files) == 5
