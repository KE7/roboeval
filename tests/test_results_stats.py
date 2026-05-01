from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboeval.results.stats import (
    find_result_files,
    rows_for_summaries,
    summarize_result_file,
    suite_mean_ci,
    wilson_ci,
    wilson_difference_ci,
)


def test_wilson_ci_matches_known_center_case():
    ci = wilson_ci(50, 100)
    assert ci.rate == pytest.approx(0.5)
    assert ci.low == pytest.approx(0.4038, abs=5e-4)
    assert ci.high == pytest.approx(0.5962, abs=5e-4)


def test_wilson_ci_handles_extreme_proportions():
    ci = wilson_ci(0, 10)
    assert ci.rate == 0.0
    assert ci.low == 0.0
    assert ci.high == pytest.approx(0.2775, abs=5e-4)


def test_wilson_difference_uses_quadrature_half_widths():
    diff = wilson_difference_ci(70, 100, 50, 100)
    ci_a = wilson_ci(70, 100)
    ci_b = wilson_ci(50, 100)
    expected_half_width = (ci_a.half_width**2 + ci_b.half_width**2) ** 0.5
    assert diff.delta == pytest.approx(0.2)
    assert diff.half_width == pytest.approx(expected_half_width)


def test_suite_mean_does_not_pool_episodes(tmp_path):
    result_path = _write_result(
        tmp_path / "run.json",
        tasks=[
            _task("easy", [True] * 90 + [False] * 10),
            _task("hard", [False]),
        ],
    )
    summary = summarize_result_file(result_path)

    pooled_rate = 90 / 101
    assert summary.suite_rate == pytest.approx(0.45)
    assert summary.suite_rate != pytest.approx(pooled_rate)

    suite_ci = suite_mean_ci(summary.tasks)
    assert suite_ci.rate == pytest.approx(summary.suite_rate)


def test_summarize_result_file_derives_model_condition_and_rows(tmp_path):
    result_path = _write_result(
        tmp_path / "libero_infinity_pi05_position_perturb.json",
        benchmark="libero_infinity_pi05_position_perturb",
        config={
            "name": "libero_infinity_pi05_position_perturb",
            "suite": "libero_infinity_spatial",
            "sim_config": {"perturbation": "position"},
        },
        tasks=[_task("task_0", [True, False, True])],
    )
    summary = summarize_result_file(result_path)
    rows = rows_for_summaries([summary])

    assert summary.model == "pi05"
    assert summary.condition == "position"
    assert summary.suite == "libero_infinity_spatial"
    assert len(rows) == 5
    assert rows[-1]["level"] == "suite"


def test_rows_include_verified_video_paths_when_present(tmp_path):
    result_path = _write_result(
        tmp_path / "run.json",
        config={"name": "run", "suite": "libero_infinity_spatial"},
        tasks=[_task("task_0", [True, False])],
    )
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_path = videos_dir / "libero_infinity_spatial_task0_ep0_demo.mp4"
    video_path.write_bytes(b"fake mp4")

    summary = summarize_result_file(result_path)
    rows = rows_for_summaries([summary])
    task_row = next(row for row in rows if row["level"] == "task")
    episode_rows = [row for row in rows if row["level"] == "episode"]
    suite_row = next(row for row in rows if row["level"] == "suite")

    assert task_row["video_paths"] == str(video_path)
    assert task_row["video_verification_status"] == "partial:1/2"
    assert episode_rows[0]["video_path"] == str(video_path)
    assert episode_rows[0]["video_verification_status"] == "verified"
    assert episode_rows[1]["video_path"] == ""
    assert episode_rows[1]["video_verification_status"] == "not_found"
    assert suite_row["video_verification_status"] == "partial:1/2"


def test_find_result_files_skips_episode_json(tmp_path):
    _write_result(tmp_path / "run.json")
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    (episode_dir / "ep.json").write_text(json.dumps({"success": True}))

    paths = find_result_files(tmp_path)
    assert paths == [tmp_path / "run.json"]


def _write_result(
    path: Path,
    *,
    benchmark: str = "bench",
    config: dict | None = None,
    tasks: list[dict] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "created_at": "2026-05-01T00:00:00+00:00",
                "config": config or {"name": benchmark, "suite": "suite"},
                "tasks": tasks or [_task("task_0", [True])],
            }
        )
    )
    return path


def _task(name: str, outcomes: list[bool]) -> dict:
    return {
        "task": name,
        "episodes": [
            {"episode_id": i, "metrics": {"success": success}} for i, success in enumerate(outcomes)
        ],
    }
