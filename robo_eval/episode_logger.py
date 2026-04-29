"""Structured JSON episode result logging."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List


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
