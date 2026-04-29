"""Merge shard result files produced by ``--shard-id`` / ``--num-shards`` runs.

Merge behavior:
    - All shards must share the same ``benchmark`` name and ``shard.total``.
    - Missing shards are allowed — the result is marked ``"partial": True``.
    - Duplicate ``episode_id`` across shards: **last file wins** (dict overwrite,
      logged as warning).
    - Metric aggregates are recomputed from the merged episode set.
    - Config metadata is preserved in the merged JSON.

Expected input format:
    Each shard file is a JSON object with at minimum::

        {
            "benchmark": "...",
            "shard": {"id": 0, "total": 4},
            "tasks": [{"task": "...", "episodes": [...]}]
        }
"""

from __future__ import annotations

import glob
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboeval.results.collector import (
    _aggregate_metrics,
    _build_task_result,
    _extract_seed,
)

logger = logging.getLogger(__name__)

__version__ = "0.1.0"


def load_shard_files(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and validate shard JSON files."""
    shards = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise ValueError(f"Failed to load shard file {p}: {e}") from e
        if "shard" not in data:
            raise ValueError(f"{p}: not a shard result file (missing 'shard' field)")
        shards.append(data)
    return shards


def find_shard_files(pattern: str) -> list[Path]:
    """Expand a glob pattern and return matching paths sorted."""
    paths = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No shard files matched pattern: {pattern!r}")
    return paths


def merge_shards(shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge shard results into a single BenchmarkResult dict.

    Parameters
    ----------
    shards:
        List of shard result dicts (output of ``load_shard_files``).

    Returns
    -------
    dict
        Merged result with coverage metadata.  ``partial`` key is set to
        ``True`` when any expected shard is missing.
    """
    if not shards:
        raise ValueError("No shard files to merge")

    # Validate consistency across shards
    benchmark_name = shards[0]["benchmark"]
    expected_total = shards[0]["shard"]["total"]
    for s in shards:
        if s["benchmark"] != benchmark_name:
            raise ValueError(f"Benchmark mismatch: {s['benchmark']!r} vs {benchmark_name!r}")
        if s["shard"]["total"] != expected_total:
            raise ValueError(f"Shard total mismatch: {s['shard']['total']} vs {expected_total}")

    # Detect missing/duplicate shards
    found_ids = sorted(s["shard"]["id"] for s in shards)
    expected_ids = list(range(expected_total))
    missing_ids = sorted(set(expected_ids) - set(found_ids))

    id_counts = Counter(found_ids)
    duplicate_ids = [sid for sid, count in id_counts.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate shard IDs found: {sorted(duplicate_ids)}")

    # Merge episodes by task; later shards replace duplicate episode IDs.
    all_episodes: dict[str, dict[int, dict[str, Any]]] = {}  # task -> {ep_id -> ep}
    for shard in shards:
        shard_id = shard.get("shard", {}).get("id", "?")
        for task_result in shard.get("tasks", []):
            task_name = task_result["task"]
            if task_name not in all_episodes:
                all_episodes[task_name] = {}
            for ep in task_result.get("episodes", []):
                ep_id = ep.get("episode_id", 0)
                if ep_id in all_episodes[task_name]:
                    logger.warning(
                        "Duplicate episode_id %r in task %r (shard %s overwrites previous)",
                        ep_id,
                        task_name,
                        shard_id,
                    )
                all_episodes[task_name][ep_id] = ep

    # Build merged task results
    metric_keys: dict[str, str] = shards[0].get("metric_keys", {"success": "mean"})
    tasks = []
    all_episodes_flat: list[dict] = []
    for task_name in sorted(all_episodes.keys()):
        episodes = list(all_episodes[task_name].values())
        tasks.append(_build_task_result(task_name, episodes, metric_keys))
        all_episodes_flat.extend(episodes)

    is_partial = bool(missing_ids) or any(s.get("partial") for s in shards)

    config = shards[0].get("config", {})
    merged: dict[str, Any] = {
        "benchmark": benchmark_name,
        "mode": shards[0].get("mode", "sync"),
        "harness_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "tasks": tasks,
        "config": config,
        "merge_info": {
            "num_shards": expected_total,
            "shards_found": found_ids,
            "shards_missing": missing_ids,
            "total_episodes": len(all_episodes_flat),
        },
    }
    if is_partial:
        merged["partial"] = True

    # Preserve server metadata from the first shard.
    server_info = shards[0].get("server_info")
    if server_info is not None:
        merged["server_info"] = server_info

    # Preserve source shard metadata.
    merged["shard_harness_version"] = shards[0].get("harness_version")
    shard_dates = sorted(s.get("created_at", "") for s in shards if s.get("created_at"))
    if shard_dates:
        merged["shard_created_at"] = {"first": shard_dates[0], "last": shard_dates[-1]}

    seed = _extract_seed(config)
    if seed is not None:
        merged["seed"] = seed

    if metric_keys:
        merged["metric_keys"] = metric_keys
        _aggregate_metrics(merged, all_episodes_flat, metric_keys)

    return merged


def print_merge_report(merged: dict[str, Any]) -> None:
    """Print a human-readable merge report to stderr."""
    try:
        from rich.console import Console

        con = Console(stderr=True, highlight=False)
        _print_merge_report_rich(con, merged)
    except ImportError:
        _print_merge_report_plain(merged)


def _print_merge_report_rich(con: Any, merged: dict[str, Any]) -> None:
    """Rich-formatted merge report."""
    from roboeval.results.collector import print_task_table

    info = merged["merge_info"]
    total_shards = info["num_shards"]
    found = info["shards_found"]
    missing = info["shards_missing"]
    total_eps = info["total_episodes"]
    rate = merged.get("mean_success", 0.0)
    rate_color = "green" if rate >= 0.5 else "red"

    if missing:
        con.print(f"\n[yellow]Missing shards: {missing} (expected 0..{total_shards - 1})[/yellow]")
        con.print(f"Coverage: {total_eps} episodes (shards {len(found)}/{total_shards})")
        con.print(
            f"Merged result ([yellow]PARTIAL[/yellow]): [{rate_color}]{rate:.1%}[/{rate_color}]"
        )
    else:
        con.print(f"\n[green]All {total_shards} shards complete.[/green] {total_eps} episodes.")
        con.print(f"Overall: [{rate_color}]{rate:.1%}[/{rate_color}]")

    con.print(f"\n{'=' * 60}")
    con.print(f"[bold]Benchmark: {merged['benchmark']}[/bold]")
    con.print(f"{'=' * 60}")
    print_task_table(con, merged["tasks"], rate, rate_color)
    con.print(f"{'=' * 60}\n")


def _print_merge_report_plain(merged: dict[str, Any]) -> None:
    """Fallback plain-text merge report."""
    info = merged["merge_info"]
    rate = merged.get("mean_success", 0.0)
    total_eps = info["total_episodes"]
    missing = info["shards_missing"]

    print(f"\nMerge report: {merged['benchmark']}")
    if missing:
        print(f"  PARTIAL: missing shards {missing}")
    print(f"  Total episodes: {total_eps}")
    print(f"  Overall success: {rate:.1%}")
    for task in merged.get("tasks", []):
        n = task["num_episodes"]
        successes = round(task.get("mean_success", 0.0) * n)
        print(f"  {task['task']}: {task.get('mean_success', 0.0):.1%} ({successes}/{n})")
