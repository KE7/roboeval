"""
Result collection, aggregation, and reporting.

Parses eval log files to extract success counts and generates
scores.json and summary.txt files matching the existing format.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def wilson_ci(
    n_success: int, n_total: int, z: float = 1.96
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Args:
        n_success: Number of successes.
        n_total: Total number of trials.
        z: Z-score for desired confidence level (1.96 = 95%).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n_total == 0:
        return (0.0, 0.0)

    p_hat = n_success / n_total
    z2 = z * z
    denom = 1 + z2 / n_total
    center = p_hat + z2 / (2 * n_total)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z2 / (4 * n_total * n_total))
    lower = (center - spread) / denom
    upper = (center + spread) / denom

    return (max(0.0, lower), min(1.0, upper))


def count_successes(log_file: Path) -> Tuple[int, int, List[int]]:
    """Count successes and total episodes from a task log file.

    Returns:
        (n_success, n_total, episode_results) tuple where episode_results
        is a list of 1s (success) and 0s (failure) per episode.
    """
    n_success = 0
    n_total = 0
    episode_results: List[int] = []
    if not log_file.exists():
        return 0, 0, []

    text = log_file.read_text()
    # Extract per-episode success/failure in order
    for match in re.finditer(r"Simulator reports success: (True|False)", text):
        result = 1 if match.group(1) == "True" else 0
        episode_results.append(result)

    n_success = sum(episode_results)
    n_total = len(episode_results)

    return n_success, n_total, episode_results


def _collect_task_from_episodes(
    results_dir: Path,
    suite: str,
    task_idx: int,
) -> Optional[Tuple[int, int, List[int]]]:
    """Try to collect task results from episode JSON files.

    Episode JSONs are written by robo_eval.episode_logger and are more
    reliable than regex-parsing log files.

    Returns:
        (n_success, n_total, episode_results) if episode JSONs exist,
        or None if no episode JSONs found for this task.
    """
    episodes_dir = results_dir / "episodes"
    if not episodes_dir.exists():
        return None

    episode_files = sorted(episodes_dir.glob(f"{suite}_task{task_idx}_ep*.json"))
    if not episode_files:
        return None

    episode_results: List[int] = []
    for filepath in episode_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            episode_results.append(1 if data.get("success") else 0)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping corrupted episode JSON %s: %s", filepath, e)
            continue

    if not episode_results:
        return None

    n_success = sum(episode_results)
    n_total = len(episode_results)
    return n_success, n_total, episode_results


def collect_results(
    results_dir: Path,
    suites: List[str],
    num_tasks: int = 10,
    max_episodes: int = 10,
) -> Dict:
    """Collect results into a structured dict.

    Prefers structured episode JSON files (from episode_logger) when
    available; falls back to regex parsing of log files.

    Args:
        results_dir: Root results directory containing logs/ and/or episodes/
        suites: List of suite names to collect
        num_tasks: Number of tasks per suite
        max_episodes: Expected episodes per task

    Returns:
        Dict with tasks, suites, and overall results.
    """
    logs_dir = results_dir / "logs"
    tasks = []
    suite_results = {}
    overall_success = 0
    overall_total = 0

    for suite in suites:
        suite_success = 0
        suite_total = 0

        for task_idx in range(num_tasks):
            # Try episode JSONs first (more reliable), fall back to log parsing
            ep_data = _collect_task_from_episodes(results_dir, suite, task_idx)
            if ep_data is not None:
                n_success, n_total, episode_results = ep_data
            else:
                log_file = logs_dir / f"{suite}_task{task_idx}.log"
                n_success, n_total, episode_results = count_successes(log_file)

            # Use actual total if available, else use max_episodes
            actual_total = n_total if n_total > 0 else max_episodes
            complete = n_total >= max_episodes

            # Per-task success rate and std dev
            task_rate = n_success / actual_total if actual_total > 0 else 0.0
            if len(episode_results) > 1:
                mean = sum(episode_results) / len(episode_results)
                variance = sum((x - mean) ** 2 for x in episode_results) / (len(episode_results) - 1)
                task_std = math.sqrt(variance)
            else:
                task_std = 0.0

            tasks.append({
                "suite": suite,
                "task": task_idx,
                "n_success": n_success,
                "n_episodes": actual_total,
                "rate": round(task_rate, 4),
                "std": round(task_std, 4),
                "complete": complete,
            })

            suite_success += n_success
            suite_total += actual_total

        suite_rate = suite_success / suite_total if suite_total > 0 else 0.0
        ci_lower, ci_upper = wilson_ci(suite_success, suite_total)
        suite_results[suite] = {
            "success": suite_success,
            "total": suite_total,
            "rate": round(suite_rate, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
        }
        overall_success += suite_success
        overall_total += suite_total

    overall_rate = overall_success / overall_total if overall_total > 0 else 0.0
    overall_ci_lower, overall_ci_upper = wilson_ci(overall_success, overall_total)

    return {
        "tasks": tasks,
        "suites": suite_results,
        "overall": {
            "success": overall_success,
            "total": overall_total,
            "rate": round(overall_rate, 4),
            "ci_lower": round(overall_ci_lower, 4),
            "ci_upper": round(overall_ci_upper, 4),
        },
    }


def write_scores_json(results: Dict, output_path: Path):
    """Write scores.json in the format compatible with existing scripts."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Scores written to %s", output_path)


def write_summary(
    results: Dict,
    output_path: Path,
    metadata: Optional[Dict] = None,
):
    """Write human-readable summary.txt."""
    lines = []

    if metadata:
        lines.append(f"Benchmark run at {metadata.get('timestamp', datetime.now().isoformat())}")
        for k, v in metadata.items():
            if k != "timestamp":
                lines.append(f"  {k}: {v}")
        lines.append("")

    for suite_name, suite_data in results["suites"].items():
        lines.append(f"=== Suite: {suite_name} ===")
        # Per-task breakdown
        for task in results["tasks"]:
            if task["suite"] == suite_name:
                status = "DONE" if task["complete"] else "PENDING"
                task_rate_pct = task.get("rate", 0) * 100
                task_std = task.get("std", 0)
                lines.append(
                    f"  Task {task['task']}: "
                    f"{task['n_success']}/{task['n_episodes']} "
                    f"({task_rate_pct:.1f}%, std: {task_std:.3f}) "
                    f"[{status}]"
                )
        rate_pct = suite_data["rate"] * 100
        ci_lower = suite_data.get("ci_lower", 0) * 100
        ci_upper = suite_data.get("ci_upper", 0) * 100
        lines.append(
            f"  Suite total: {suite_data['success']}/{suite_data['total']} "
            f"({rate_pct:.1f}%, 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])"
        )
        lines.append("")

    overall = results["overall"]
    rate_pct = overall["rate"] * 100
    ci_lower = overall.get("ci_lower", 0) * 100
    ci_upper = overall.get("ci_upper", 0) * 100
    lines.append("=== OVERALL ===")
    lines.append(
        f"Total: {overall['success']} / {overall['total']} "
        f"(rate: {rate_pct:.1f}%, 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%])"
    )
    lines.append("")
    lines.append(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    text = "\n".join(lines)
    output_path.write_text(text)
    logger.info("Summary written to %s", output_path)
    return text


def print_status(results_dir: Path, suites: Optional[List[str]] = None):
    """Print current progress of a benchmark run.

    Auto-discovers suites from log files if not specified.
    """
    logs_dir = results_dir / "logs"
    if not logs_dir.exists():
        logger.info("No logs directory found at %s", logs_dir)
        return

    # Auto-discover suites from log filenames
    if not suites:
        suite_set = set()
        for f in logs_dir.glob("*_task*.log"):
            name = f.stem
            # Strip _taskN suffix
            match = re.match(r"(.+)_task\d+$", name)
            if match:
                suite_set.add(match.group(1))
        suites = sorted(suite_set)

    if not suites:
        logger.info("No eval logs found.")
        return

    total_success = 0
    total_episodes = 0
    total_complete = 0
    total_tasks = 0

    for suite in suites:
        logger.info("\n  Suite: %s", suite)
        suite_success = 0
        suite_episodes = 0
        suite_complete = 0

        # Detect actual task count from log files instead of hardcoding 10.
        suite_task_indices = set()
        for f in logs_dir.glob(f"{suite}_task*.log"):
            m = re.match(rf"{re.escape(suite)}_task(\d+)\.log$", f.name)
            if m:
                suite_task_indices.add(int(m.group(1)))
        num_tasks = max(suite_task_indices) + 1 if suite_task_indices else 10

        for task_idx in range(num_tasks):
            log_file = logs_dir / f"{suite}_task{task_idx}.log"
            if log_file.exists():
                n_success, n_total, _episode_results = count_successes(log_file)
                # Check if eval is still running (file exists but no completion marker)
                text = log_file.read_text()
                is_running = n_total == 0 and len(text) > 0
                is_complete = "All " in text and "episodes complete" in text

                if is_running:
                    # Count how many episodes have started
                    n_started = len(re.findall(r"Episode \d+/\d+", text))
                    logger.info("    Task %d: %d/? (%d eps started) [RUNNING]", task_idx, n_success, n_started)
                elif is_complete or n_total > 0:
                    logger.info("    Task %d: %d/%d [DONE]", task_idx, n_success, n_total)
                    suite_complete += 1
                else:
                    logger.info("    Task %d: %d/%d [PARTIAL]", task_idx, n_success, n_total)

                suite_success += n_success
                suite_episodes += n_total
            else:
                logger.info("    Task %d: [NOT STARTED]", task_idx)

            total_tasks += 1

        total_success += suite_success
        total_episodes += suite_episodes
        total_complete += suite_complete

        if suite_episodes > 0:
            rate = suite_success / suite_episodes * 100
            ci_lo, ci_hi = wilson_ci(suite_success, suite_episodes)
            ci_str = f", 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]" if suite_episodes >= 2 else ""
            logger.info("    Suite: %d/%d (%.1f%%%s) - %d/%d tasks complete", suite_success, suite_episodes, rate, ci_str, suite_complete, num_tasks)

    logger.info("\n  Overall: %d/%d tasks, %d/%d complete", total_success, total_episodes, total_complete, total_tasks)
    if total_episodes > 0:
        rate = total_success / total_episodes * 100
        ci_lo, ci_hi = wilson_ci(total_success, total_episodes)
        ci_str = f", 95% CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]" if total_episodes >= 2 else ""
        logger.info("  Success rate: %.1f%%%s", rate, ci_str)
