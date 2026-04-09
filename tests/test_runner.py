"""
Tests for robo_eval/runner.py — pure functions that don't need running sims.

Covers:
- _build_eval_cmd() — correct CLI args generated
- save_eval_config() — writes valid YAML
- _get_git_commit() — handles missing git gracefully
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# _build_eval_cmd
# ---------------------------------------------------------------------------

class TestBuildEvalCmd:
    def _build(self, **kwargs):
        from robo_eval.runner import _build_eval_cmd
        defaults = dict(
            task_idx=0,
            suite="libero_spatial",
            sim_type="libero",
            sim_port=5300,
            max_episodes=10,
            experience_dir=Path("/tmp/exp"),
        )
        defaults.update(kwargs)
        return _build_eval_cmd(**defaults)

    def test_basic_command(self):
        cmd = self._build()
        assert "eval" in cmd
        assert "--sim" in cmd
        idx = cmd.index("--sim")
        assert cmd[idx + 1] == "libero"

    def test_task_idx(self):
        cmd = self._build(task_idx=5)
        idx = cmd.index("--task")
        assert cmd[idx + 1] == "5"

    def test_suite_name(self):
        cmd = self._build(suite="libero_goal")
        idx = cmd.index("--suite")
        assert cmd[idx + 1] == "libero_goal"

    def test_sim_url(self):
        cmd = self._build(sim_port=5555)
        idx = cmd.index("--sim-url")
        assert cmd[idx + 1] == "http://localhost:5555"

    def test_max_episodes(self):
        cmd = self._build(max_episodes=50)
        idx = cmd.index("--max-episodes")
        assert cmd[idx + 1] == "50"

    def test_max_steps_auto(self):
        # When max_steps not provided, should use suite-specific default
        cmd = self._build(suite="libero_spatial")
        idx = cmd.index("--max-steps")
        assert cmd[idx + 1] == "280"  # libero_spatial default

    def test_max_steps_override(self):
        cmd = self._build(max_steps=999)
        idx = cmd.index("--max-steps")
        assert cmd[idx + 1] == "999"

    def test_headless_default(self):
        cmd = self._build()
        assert "--headless" in cmd

    def test_headless_false(self):
        cmd = self._build(headless=False)
        assert "--headless" not in cmd

    def test_delta_actions(self):
        cmd = self._build(delta_actions=True)
        assert "--delta-actions" in cmd

    def test_no_delta_actions(self):
        cmd = self._build(delta_actions=False)
        assert "--delta-actions" not in cmd

    def test_no_vlm(self):
        cmd = self._build(no_vlm=True)
        assert "--no-vlm" in cmd

    def test_no_think(self):
        cmd = self._build(no_think=True)
        assert "--no-think" in cmd

    def test_vlm_model(self):
        cmd = self._build(vlm_model="vertex_ai/gemini-3-flash-preview")
        idx = cmd.index("--vlm-model")
        assert cmd[idx + 1] == "vertex_ai/gemini-3-flash-preview"

    def test_vlm_endpoint(self):
        cmd = self._build(vlm_endpoint="localhost:4000")
        idx = cmd.index("--vlm-endpoint")
        assert cmd[idx + 1] == "localhost:4000"

    def test_seed(self):
        cmd = self._build(seed=42)
        idx = cmd.index("--seed")
        assert cmd[idx + 1] == "42"

    def test_record_video(self):
        cmd = self._build(record_video=True, record_video_n=5)
        assert "--record-video" in cmd
        idx = cmd.index("--record-video-n")
        assert cmd[idx + 1] == "5"

    def test_no_record_video(self):
        cmd = self._build(record_video=False)
        assert "--record-video" not in cmd

    def test_results_dir(self):
        cmd = self._build(results_dir=Path("/tmp/results"))
        idx = cmd.index("--results-dir")
        assert cmd[idx + 1] == "/tmp/results"

    def test_experience_dir(self):
        cmd = self._build(experience_dir=Path("/tmp/exp/task0"))
        idx = cmd.index("--experience-dir")
        assert cmd[idx + 1] == "/tmp/exp/task0"


# ---------------------------------------------------------------------------
# save_eval_config
# ---------------------------------------------------------------------------

class TestSaveEvalConfig:
    def test_writes_yaml(self, tmp_path):
        from robo_eval.runner import save_eval_config
        config_path = save_eval_config(
            tmp_path,
            suites=["libero_spatial", "libero_goal"],
            episodes_per_task=10,
            mode="direct",
            sim_base_port=5300,
            seed=42,
        )
        assert config_path.exists()
        assert config_path.name == "eval_config.yaml"

    def test_yaml_content(self, tmp_path):
        from robo_eval.runner import save_eval_config
        save_eval_config(
            tmp_path,
            vla="pi05",
            suites=["libero_spatial"],
            episodes_per_task=50,
            mode="planner",
            sim_base_port=5300,
            seed=123,
            vla_replicas=4,
            sim_workers=10,
        )
        text = (tmp_path / "eval_config.yaml").read_text()
        data = yaml.safe_load(text)
        assert data["vla"] == "pi05"
        assert data["suites"] == ["libero_spatial"]
        assert data["episodes_per_task"] == 50
        assert data["mode"] == "planner"
        assert data["seed"] == 123
        assert data["vla_replicas"] == 4
        assert data["sim_workers"] == 10

    def test_git_commit_present(self, tmp_path):
        from robo_eval.runner import save_eval_config
        save_eval_config(
            tmp_path,
            suites=["libero_spatial"],
            episodes_per_task=10,
        )
        text = (tmp_path / "eval_config.yaml").read_text()
        data = yaml.safe_load(text)
        assert "git_commit" in data

    def test_timestamp_present(self, tmp_path):
        from robo_eval.runner import save_eval_config
        save_eval_config(
            tmp_path,
            suites=["libero_spatial"],
            episodes_per_task=10,
        )
        text = (tmp_path / "eval_config.yaml").read_text()
        data = yaml.safe_load(text)
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# _get_git_commit
# ---------------------------------------------------------------------------

class TestGetGitCommit:
    def test_returns_string(self):
        from robo_eval.runner import _get_git_commit
        result = _get_git_commit()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_handles_failure(self):
        from robo_eval.runner import _get_git_commit
        with patch("subprocess.run", side_effect=Exception("git not found")):
            result = _get_git_commit()
            assert result == "unknown"
