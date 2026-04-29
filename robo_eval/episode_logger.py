"""Structured JSON episode result logging."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    """Structured result for a single evaluation episode."""

    task: int
    episode: int
    success: bool
    steps: int
    duration_s: float
    vla_calls: int
    subtasks: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat(timespec="seconds")


def save_episode_result(
    results_dir: str | Path,
    suite: str,
    task_idx: int,
    episode_idx: int,
    result: EpisodeResult,
) -> Path:
    """Write an episode result to a JSON file.

    File path: <results_dir>/episodes/<suite>_task<N>_ep<M>.json

    Args:
        results_dir: Root results directory (e.g. results/my_run/).
        suite: Benchmark suite name (e.g. "libero_spatial").
        task_idx: Task index within the suite.
        episode_idx: Episode index within the task.
        result: The EpisodeResult to serialize.

    Returns:
        Path to the written JSON file.
    """
    results_dir = Path(results_dir)
    episodes_dir = results_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{suite}_task{task_idx}_ep{episode_idx}.json"
    filepath = episodes_dir / filename

    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return filepath


def load_episode_results(
    results_dir: str | Path,
    suite: str,
) -> List[Dict]:
    """Load all episode result JSONs for a given suite.

    Reads all files matching <results_dir>/episodes/<suite>_task*_ep*.json
    and returns them sorted by (task, episode).

    Args:
        results_dir: Root results directory.
        suite: Benchmark suite name to filter by.

    Returns:
        List of episode result dicts, sorted by (task, episode).
    """
    results_dir = Path(results_dir)
    episodes_dir = results_dir / "episodes"

    if not episodes_dir.exists():
        return []

    results = []
    for filepath in episodes_dir.glob(f"{suite}_task*_ep*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", filepath, e)

    # Sort by (task, episode) for deterministic ordering
    results.sort(key=lambda r: (r.get("task", 0), r.get("episode", 0)))
    return results


def episode_results_for_task(
    results_dir: str | Path,
    suite: str,
    task_idx: int,
) -> List[Dict]:
    """Load episode results for a specific task.

    Args:
        results_dir: Root results directory.
        suite: Benchmark suite name.
        task_idx: Task index.

    Returns:
        List of episode result dicts for this task, sorted by episode.
    """
    results_dir = Path(results_dir)
    episodes_dir = results_dir / "episodes"

    if not episodes_dir.exists():
        return []

    results = []
    for filepath in episodes_dir.glob(f"{suite}_task{task_idx}_ep*.json"):
        try:
            with open(filepath) as f:
                data = json.load(f)
            results.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read %s: %s", filepath, e)

    results.sort(key=lambda r: r.get("episode", 0))
    return results
