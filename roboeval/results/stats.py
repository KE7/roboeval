"""Statistics and table helpers for benchmark result JSON files.

The reporting helpers here intentionally aggregate suite-level performance as
the mean of task-level success rates. They do not pool episodes across tasks.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ProportionCI:
    """A binomial proportion estimate with a confidence interval."""

    successes: int
    n: int
    rate: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0


@dataclass(frozen=True)
class DifferenceCI:
    """Approximate confidence interval for a difference of two rates."""

    delta: float
    low: float
    high: float

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0


@dataclass(frozen=True)
class EpisodeSummary:
    """Episode-level success summary with optional video artifact paths."""

    episode_id: int
    success: bool
    video_path: str
    video_verification_status: str


@dataclass(frozen=True)
class TaskSummary:
    """Task-level success summary extracted from a result file."""

    task: str
    successes: int
    n: int
    rate: float
    ci_low: float
    ci_high: float
    ci_half_width: float
    video_paths: tuple[str, ...]
    videos_verified: int
    video_verification_status: str
    episodes: tuple[EpisodeSummary, ...]


@dataclass(frozen=True)
class ResultSummary:
    """A report row for one completed result file."""

    source_file: str
    benchmark: str
    created_at: str
    model: str
    condition: str
    suite: str
    partial: bool
    tasks: tuple[TaskSummary, ...]
    suite_rate: float
    suite_ci_low: float
    suite_ci_high: float
    suite_ci_half_width: float

    @property
    def total_successes(self) -> int:
        return sum(t.successes for t in self.tasks)

    @property
    def total_episodes(self) -> int:
        return sum(t.n for t in self.tasks)


def wilson_ci(successes: int, n: int, z: float = Z_95) -> ProportionCI:
    """Compute a Wilson score confidence interval for ``successes / n``."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if successes < 0 or successes > n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    if n == 0:
        return ProportionCI(successes=0, n=0, rate=0.0, low=0.0, high=0.0)

    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))) / denom
    return ProportionCI(
        successes=successes,
        n=n,
        rate=phat,
        low=max(0.0, center - margin),
        high=min(1.0, center + margin),
    )


def wilson_difference_ci(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    z: float = Z_95,
) -> DifferenceCI:
    """Approximate a Newcombe/Wilson CI for ``p_a - p_b``.

    This uses independent Wilson intervals for the two proportions and combines
    their half-widths in quadrature, matching the approximation specified in the
    NeurIPS LIBERO-Infinity experiment plan.
    """
    ci_a = wilson_ci(successes_a, n_a, z=z)
    ci_b = wilson_ci(successes_b, n_b, z=z)
    delta = ci_a.rate - ci_b.rate
    half_width = math.sqrt(ci_a.half_width**2 + ci_b.half_width**2)
    return DifferenceCI(delta=delta, low=delta - half_width, high=delta + half_width)


def suite_mean_ci(tasks: Iterable[TaskSummary]) -> ProportionCI:
    """Aggregate task summaries as an unweighted suite mean.

    The returned ``successes`` and ``n`` are informational totals only. The
    ``rate`` and interval are computed from per-task rates and Wilson half
    widths, not from total successes divided by total episodes.
    """
    usable_tasks = [t for t in tasks if t.n > 0]
    if not usable_tasks:
        return ProportionCI(successes=0, n=0, rate=0.0, low=0.0, high=0.0)

    total_successes = sum(t.successes for t in usable_tasks)
    total_n = sum(t.n for t in usable_tasks)
    task_count = len(usable_tasks)
    rate = sum(t.rate for t in usable_tasks) / task_count
    half_width = math.sqrt(sum(t.ci_half_width**2 for t in usable_tasks)) / task_count
    return ProportionCI(
        successes=total_successes,
        n=total_n,
        rate=rate,
        low=max(0.0, rate - half_width),
        high=min(1.0, rate + half_width),
    )


def summarize_result_file(path: str | Path) -> ResultSummary:
    """Load one result JSON and compute per-task and suite-level CIs."""
    p = Path(path)
    data = json.loads(p.read_text())
    config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
    result_dir = p.parent
    suite = str(config.get("suite") or data.get("benchmark") or "")
    tasks = tuple(
        _summarize_task(t, result_dir=result_dir, suite=suite) for t in data.get("tasks", [])
    )
    suite_ci = suite_mean_ci(tasks)
    return ResultSummary(
        source_file=str(p),
        benchmark=str(data.get("benchmark", p.stem)),
        created_at=str(data.get("created_at", "")),
        model=_derive_model(data, config),
        condition=_derive_condition(data, config),
        suite=suite,
        partial=bool(data.get("partial")),
        tasks=tasks,
        suite_rate=suite_ci.rate,
        suite_ci_low=suite_ci.low,
        suite_ci_high=suite_ci.high,
        suite_ci_half_width=suite_ci.half_width,
    )


def find_result_files(root: str | Path) -> list[Path]:
    """Return completed benchmark JSON files below ``root``."""
    root_path = Path(root)
    if root_path.is_file():
        return [root_path]
    if not root_path.exists():
        return []
    paths: list[Path] = []
    for path in sorted(root_path.rglob("*.json")):
        if "episodes" in path.parts or path.name.endswith(".progress"):
            continue
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tasks = data.get("tasks") if isinstance(data, dict) else None
        if (
            isinstance(data, dict)
            and isinstance(tasks, list)
            and all(isinstance(task, dict) and isinstance(task.get("episodes"), list) for task in tasks)
        ):
            paths.append(path)
    return paths


def latest_summaries_per_cell(summaries: Iterable[ResultSummary]) -> list[ResultSummary]:
    """Keep only the latest result for each model/condition/suite/benchmark cell."""
    latest: dict[tuple[str, str, str, str], ResultSummary] = {}
    for summary in summaries:
        key = (summary.benchmark, summary.model, summary.condition, summary.suite)
        current = latest.get(key)
        if current is None or (summary.created_at, summary.source_file) > (
            current.created_at,
            current.source_file,
        ):
            latest[key] = summary
    return sorted(latest.values(), key=lambda s: (s.benchmark, s.model, s.condition, s.suite))


def rows_for_summaries(summaries: Iterable[ResultSummary]) -> list[dict[str, Any]]:
    """Flatten summaries into task rows plus one suite row per result file."""
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        base = {
            "source_file": summary.source_file,
            "benchmark": summary.benchmark,
            "created_at": summary.created_at,
            "model": summary.model,
            "condition": summary.condition,
            "suite": summary.suite,
            "partial": summary.partial,
        }
        for task in summary.tasks:
            rows.append(
                {
                    **base,
                    "level": "task",
                    "task": task.task,
                    "successes": task.successes,
                    "episodes": task.n,
                    "success_rate": task.rate,
                    "ci_low": task.ci_low,
                    "ci_high": task.ci_high,
                    "ci_half_width": task.ci_half_width,
                    "video_path": "",
                    "video_paths": ";".join(task.video_paths),
                    "video_verification_status": task.video_verification_status,
                    "videos_verified": task.videos_verified,
                }
            )
            for episode in task.episodes:
                rows.append(
                    {
                        **base,
                        "level": "episode",
                        "task": task.task,
                        "successes": int(episode.success),
                        "episodes": 1,
                        "success_rate": float(episode.success),
                        "ci_low": "",
                        "ci_high": "",
                        "ci_half_width": "",
                        "video_path": episode.video_path,
                        "video_paths": episode.video_path,
                        "video_verification_status": episode.video_verification_status,
                        "videos_verified": int(episode.video_verification_status == "verified"),
                    }
                )
        total_videos_verified = sum(task.videos_verified for task in summary.tasks)
        total_episodes = summary.total_episodes
        rows.append(
            {
                **base,
                "level": "suite",
                "task": "SUITE_MEAN",
                "successes": summary.total_successes,
                "episodes": summary.total_episodes,
                "success_rate": summary.suite_rate,
                "ci_low": summary.suite_ci_low,
                "ci_high": summary.suite_ci_high,
                "ci_half_width": summary.suite_ci_half_width,
                "video_path": "",
                "video_paths": ";".join(
                    video_path for task in summary.tasks for video_path in task.video_paths
                ),
                "video_verification_status": _video_status(total_videos_verified, total_episodes),
                "videos_verified": total_videos_verified,
            }
        )
    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render report rows as a compact Markdown table."""
    headers = [
        "benchmark",
        "model",
        "condition",
        "suite",
        "level",
        "task",
        "success",
        "rate",
        "95% CI",
        "video_path(s)",
        "video status",
        "partial",
        "source_file",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        success = f"{row['successes']}/{row['episodes']}"
        rate = _format_percent(row["success_rate"])
        ci = (
            f"[{_format_percent(row['ci_low'])}, {_format_percent(row['ci_high'])}]"
            if isinstance(row["ci_low"], (int, float)) and isinstance(row["ci_high"], (int, float))
            else ""
        )
        video_paths = str(row.get("video_paths") or row.get("video_path") or "")
        video_status = str(row.get("video_verification_status") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    str(row["model"]),
                    str(row["condition"]),
                    str(row["suite"]),
                    str(row["level"]),
                    str(row["task"]),
                    success,
                    rate,
                    ci,
                    video_paths,
                    video_status,
                    "yes" if row["partial"] else "no",
                    str(row["source_file"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write report rows as CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "source_file",
        "benchmark",
        "created_at",
        "model",
        "condition",
        "suite",
        "partial",
        "level",
        "task",
        "successes",
        "episodes",
        "success_rate",
        "ci_low",
        "ci_high",
        "ci_half_width",
        "video_path",
        "video_paths",
        "video_verification_status",
        "videos_verified",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_task(task: dict[str, Any], *, result_dir: Path, suite: str) -> TaskSummary:
    episodes = task.get("episodes", [])
    successes = sum(1 for ep in episodes if ep.get("metrics", {}).get("success") is True)
    n = len(episodes)
    ci = wilson_ci(successes, n)
    episode_summaries = tuple(
        _summarize_episode_video(
            ep,
            idx=idx,
            result_dir=result_dir,
            suite=suite,
            task_name=str(task.get("task", "")),
        )
        for idx, ep in enumerate(episodes)
    )
    video_paths = tuple(ep.video_path for ep in episode_summaries if ep.video_path)
    videos_verified = len(video_paths)
    return TaskSummary(
        task=str(task.get("task", "")),
        successes=successes,
        n=n,
        rate=ci.rate,
        ci_low=ci.low,
        ci_high=ci.high,
        ci_half_width=ci.half_width,
        video_paths=video_paths,
        videos_verified=videos_verified,
        video_verification_status=_video_status(videos_verified, n),
        episodes=episode_summaries,
    )


def _summarize_episode_video(
    ep: dict[str, Any],
    *,
    idx: int,
    result_dir: Path,
    suite: str,
    task_name: str,
) -> EpisodeSummary:
    episode_id = int(ep.get("episode_id", idx))
    video_path = _find_episode_video(
        result_dir=result_dir,
        suite=suite,
        task_name=task_name,
        episode_id=episode_id,
    )
    return EpisodeSummary(
        episode_id=episode_id,
        success=bool(ep.get("metrics", {}).get("success") is True),
        video_path=video_path,
        video_verification_status="verified" if video_path else "not_found",
    )


def _video_status(videos_verified: int, episodes: int) -> str:
    if episodes == 0:
        return "no_episodes"
    if videos_verified == episodes:
        return "verified"
    if videos_verified == 0:
        return "not_found"
    return f"partial:{videos_verified}/{episodes}"


def _find_episode_video(
    *,
    result_dir: Path,
    suite: str,
    task_name: str,
    episode_id: int,
) -> str:
    videos_dir = result_dir / "videos"
    if not videos_dir.is_dir():
        return ""
    task_id = _task_id_for_video(task_name)
    patterns = [
        f"{suite}_task{task_id}_ep{episode_id}_*.mp4",
        f"*task{task_id}_ep{episode_id}_*.mp4",
        f"*{task_name}*ep{episode_id}*.mp4",
    ]
    for pattern in patterns:
        matches = sorted(videos_dir.glob(pattern))
        if matches:
            return str(matches[0])
    return ""


def _task_id_for_video(task_name: str) -> str:
    if task_name.startswith("task_"):
        suffix = task_name.removeprefix("task_")
        if suffix.isdigit():
            return suffix
    digits = "".join(ch for ch in task_name if ch.isdigit())
    return digits or task_name


def _derive_model(data: dict[str, Any], config: dict[str, Any]) -> str:
    model = config.get("vla") or config.get("model") or config.get("policy")
    if model:
        return str(model)
    name = str(config.get("name") or data.get("benchmark") or "").lower()
    for candidate in ("pi05", "openvla", "smolvla", "groot", "cosmos", "internvla", "octo"):
        if candidate in name:
            return candidate
    return "unknown"


def _derive_condition(data: dict[str, Any], config: dict[str, Any]) -> str:
    sim_config = config.get("sim_config")
    if isinstance(sim_config, dict) and "perturbation" in sim_config:
        perturbation = sim_config["perturbation"]
        if isinstance(perturbation, list):
            return "+".join(str(p) for p in perturbation)
        return str(perturbation)

    name = str(config.get("name") or data.get("benchmark") or "").lower()
    for candidate in (
        "combined",
        "full",
        "articulation",
        "position",
        "object",
        "lighting",
        "camera",
        "distractor",
        "liten",
    ):
        if candidate in name:
            return candidate
    return "baseline"


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"
