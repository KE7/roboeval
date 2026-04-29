"""Orchestrator: coordinates sharded benchmark evaluation runs via subprocess.

Execution flow:
    1. Load EvalConfig from a flat YAML file.
    2. Build a (task, episode) work list from tasks × episodes_per_task.
    3. Optionally shard via round-robin: ``item_index % num_shards == shard_id``.
    4. For each work item, launch ``run_sim_eval.py eval`` as a subprocess,
       capturing per-episode JSON already written by robo_eval.episode_logger.
    5. Collect results into a ResultCollector; write shard JSON + atomic .progress.
    6. File-lock the shard output to prevent two shards writing the same file.

Environment forwarding:
    - ``VLA_URL`` is forwarded to each subprocess (required by SimWrapper).
    - Full parent env is inherited; extra env vars can be passed via ``extra_env``.

Error recovery:
    - Per-episode try/except — one subprocess failure does not abort the shard.
    - Non-zero exit code is treated as episode failure with the stderr as detail.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^\w\-.]")

# ---------------------------------------------------------------------------
# EvalConfig dataclass
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Configuration for a VLA or sim HTTP server."""

    url: str = "http://localhost:5100"
    timeout: float = 60.0
    name: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ServerConfig":
        if not d:
            return cls()
        return cls(
            url=d.get("url", "http://localhost:5100"),
            timeout=float(d.get("timeout", 60.0)),
            name=d.get("name", ""),
        )


@dataclass
class EvalConfig:
    """Flat evaluation configuration loaded from YAML.

    All fields have sensible defaults so configs can be minimal.
    """

    # Required (no defaults)
    name: str = "eval"

    # VLA server
    vla_url: str = "http://localhost:5100"

    # Sim server
    sim_url: str = "http://localhost:5300"
    sim: str = "libero"

    # Task selection
    suite: str = "libero_spatial"
    tasks: list[int] = field(default_factory=list)
    max_tasks: int | None = None

    # Episodes
    episodes_per_task: int = 10
    # Per-episode wall-clock cap for simulator evaluation subprocesses.
    # Planner-assisted episodes can need many minutes, so the default is
    # generous; override per-config when needed.
    episode_timeout_seconds: int = 1800

    # VLM
    no_vlm: bool = True
    vlm_model: str | None = None
    vlm_endpoint: str = "localhost:4000"

    # Action
    delta_actions: bool = True

    # Python executable for subprocess launch
    eval_python: str = ""

    # Output
    output_dir: str = "./results"

    # Extra params forwarded to run_sim_eval.py
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalConfig":
        """Construct from a flat dict (e.g. from YAML)."""
        cfg = cls()
        cfg.name = d.get("name", "eval")
        cfg.vla_url = d.get("vla_url", os.environ.get("VLA_URL", "http://localhost:5100"))
        cfg.sim_url = d.get("sim_url", "http://localhost:5300")
        cfg.sim = d.get("sim", "libero")
        cfg.suite = d.get("suite", "libero_spatial")
        tasks_raw = d.get("tasks", [])
        cfg.tasks = [int(t) for t in tasks_raw] if tasks_raw else []
        cfg.max_tasks = d.get("max_tasks", None)
        cfg.episodes_per_task = int(d.get("episodes_per_task", 10))
        cfg.episode_timeout_seconds = int(d.get("episode_timeout_seconds", 1800))
        cfg.no_vlm = bool(d.get("no_vlm", True))
        cfg.vlm_model = d.get("vlm_model", None)
        cfg.vlm_endpoint = d.get("vlm_endpoint", "localhost:4000")
        cfg.delta_actions = bool(d.get("delta_actions", True))
        cfg.eval_python = d.get("eval_python", "")
        cfg.output_dir = d.get("output_dir", "./results")
        cfg.params = d.get("params", {})
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        """Load config from a YAML file."""
        import yaml  # type: ignore
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = cls.from_dict(d)
        # Store the raw dict for snapshot
        cfg._raw = d  # type: ignore[attr-defined]
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "vla_url": self.vla_url,
            "sim_url": self.sim_url,
            "sim": self.sim,
            "suite": self.suite,
            "tasks": self.tasks,
            "max_tasks": self.max_tasks,
            "episodes_per_task": self.episodes_per_task,
            "episode_timeout_seconds": self.episode_timeout_seconds,
            "no_vlm": self.no_vlm,
            "vlm_model": self.vlm_model,
            "vlm_endpoint": self.vlm_endpoint,
            "delta_actions": self.delta_actions,
            "eval_python": self.eval_python,
            "output_dir": self.output_dir,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Coordinates sharded evaluation runs via subprocess.

    Parameters
    ----------
    config:
        Flat config dict or EvalConfig instance.
    shard_id:
        Zero-based shard index, or None for non-sharded run.
    num_shards:
        Total number of shards, or None for non-sharded run.
    results_dir:
        Override output directory (defaults to config.output_dir).
    extra_env:
        Extra environment variables forwarded to subprocess.
    """

    def __init__(
        self,
        config: EvalConfig | dict[str, Any],
        shard_id: int | None = None,
        num_shards: int | None = None,
        results_dir: str | Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        if isinstance(config, dict):
            self.config = EvalConfig.from_dict(config)
        else:
            self.config = config
        self.shard_id = shard_id
        self.num_shards = num_shards
        self._results_dir_override = Path(results_dir) if results_dir else None
        self.extra_env = extra_env or {}
        self._progress_path: Path | None = None
        self._lock_fd: int | None = None
        self._lock_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the evaluation. Returns the benchmark result dict."""
        cfg = self.config
        safe_name = _SAFE_NAME_RE.sub("_", cfg.name)

        # Build task list
        tasks = self._build_task_list()
        if not tasks:
            logger.warning("No tasks to evaluate (suite=%r, tasks=%r)", cfg.suite, cfg.tasks)
            return {}

        # Build flat (task_id, episode) work items
        work_items: list[tuple[int, int]] = []
        for task_id in tasks:
            for ep in range(cfg.episodes_per_task):
                work_items.append((task_id, ep))

        # Shard round-robin
        if self.num_shards is not None and self.shard_id is not None:
            work_items = [
                w for i, w in enumerate(work_items)
                if i % self.num_shards == self.shard_id
            ]
            logger.info(
                "Shard %d/%d: %d work items assigned",
                self.shard_id,
                self.num_shards,
                len(work_items),
            )

        # Claim output path via file lock (shard mode prevents duplicate writes)
        output_path = self._claim_output_path(safe_name)
        self._progress_path = output_path.with_suffix(".progress")

        # Set up collector
        from robo_eval.results.collector import ResultCollector
        collector = ResultCollector(
            benchmark_name=cfg.name,
            mode="sync",
            metric_keys={"success": "mean"},
        )

        total_items = len(work_items)
        self._update_progress(0, total_items, 0)

        # Run episodes
        for item_idx, (task_id, ep) in enumerate(work_items):
            task_name = f"task_{task_id}"
            try:
                ep_result = self._run_episode(task_id, ep)
                collector.record(task_name, ep_result)
                status = "SUCCESS" if ep_result.get("metrics", {}).get("success") else "FAIL"
                logger.info(
                    "  [%d/%d] %s ep%d: %s (steps=%d)",
                    item_idx + 1,
                    total_items,
                    task_name,
                    ep,
                    status,
                    ep_result.get("steps", 0),
                )
            except Exception:
                logger.exception(
                    "  [%d/%d] %s ep%d: ERROR",
                    item_idx + 1,
                    total_items,
                    task_name,
                    ep,
                )
                from robo_eval.results.collector import EpisodeResult
                failed: EpisodeResult = {
                    "episode_id": ep,
                    "metrics": {"success": False},
                    "failure_reason": "exception",
                    "failure_detail": traceback.format_exc(),
                }
                collector.record(task_name, failed)
            finally:
                self._update_progress(item_idx + 1, total_items, collector.error_count)

        return self._save_results(collector, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_task_list(self) -> list[int]:
        """Build the list of task IDs to evaluate."""
        cfg = self.config
        if cfg.tasks:
            tasks = list(cfg.tasks)
        else:
            # Default: tasks 0..max_tasks-1 or all tasks in suite
            # Fall back to a reasonable default for libero suites
            default_n = _SUITE_TASK_COUNTS.get(cfg.suite, 10)
            tasks = list(range(default_n))

        if cfg.max_tasks is not None:
            tasks = tasks[: cfg.max_tasks]

        return tasks

    @property
    def _output_dir(self) -> Path:
        if self._results_dir_override is not None:
            d = self._results_dir_override
        else:
            d = Path(self.config.output_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _shard_stem(self, safe_name: str) -> str:
        if self.num_shards is not None and self.shard_id is not None:
            return f"{safe_name}_shard{self.shard_id}of{self.num_shards}"
        return safe_name

    def _claim_output_path(self, safe_name: str) -> Path:
        """Determine and file-lock the output path. Returns the path."""
        if self.num_shards is not None and self.shard_id is not None:
            output_path = self._output_dir / f"{self._shard_stem(safe_name)}.json"
        else:
            # Include pid+random suffix so two near-simultaneous runs don't collide.
            ts = int(time.time())
            tag = f"{ts}_{os.getpid()}_{secrets.token_hex(3)}"
            output_path = self._output_dir / f"{safe_name}_{tag}.json"

        # Try to acquire an exclusive lock using os.O_EXCL on a .lock file.
        # We use the same lock path for both the O_EXCL and filelock approaches
        # so that concurrent callers always contend on the same resource.
        if self.num_shards is not None and self.shard_id is not None:
            lock_path = Path(str(output_path) + ".lock")
            try:
                from filelock import FileLock, Timeout
                fl = FileLock(str(lock_path), timeout=0)
                fl.acquire()
                self._filelock = fl
                self._lock_path = lock_path
            except ImportError:
                # filelock not installed — fall back to O_EXCL
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    self._lock_fd = fd
                    self._lock_path = lock_path
                except FileExistsError:
                    raise FileExistsError(
                        f"Another eval is already writing to {output_path}. "
                        "Remove the .lock file or use a different output_dir."
                    )
            except Exception:
                raise FileExistsError(
                    f"Another eval is already writing to {output_path}. "
                    "Remove the .lock file or use a different output_dir."
                )

        return output_path

    def _release_lock(self) -> None:
        if hasattr(self, "_filelock") and self._filelock is not None:
            try:
                self._filelock.release()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._filelock = None  # type: ignore[attr-defined]
        if self._lock_fd is not None:
            try:
                os.close(self._lock_fd)
            except OSError:
                pass
            self._lock_fd = None
        if self._lock_path is not None:
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._lock_path = None

    def _update_progress(self, completed: int, total: int, errors: int) -> None:
        """Write atomic .progress file for live monitoring."""
        if self._progress_path is None:
            return
        tmp = self._progress_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {"completed": completed, "total": total, "errors": errors}
                )
            )
            tmp.replace(self._progress_path)  # atomic on POSIX
        except OSError as e:
            logger.debug("Could not write progress file: %s", e)

    def _build_subprocess_cmd(self, task_id: int, episode: int) -> list[str]:
        """Build the run_sim_eval.py subprocess command line."""
        cfg = self.config

        # Resolve Python executable
        python = cfg.eval_python or sys.executable

        # Find run_sim_eval.py relative to this file
        project_root = Path(__file__).resolve().parent.parent
        run_script = project_root / "run_sim_eval.py"

        cmd = [
            python, str(run_script), "eval",
            "--sim", cfg.sim,
            "--task", str(task_id),
            "--suite", cfg.suite,
            "--sim-url", cfg.sim_url,
            "--episode", str(episode),
            # run_sim_eval.py saves episodes to <results-dir>/episodes/
            # so pass _output_dir directly (not _output_dir/episodes)
            "--results-dir", str(self._output_dir),
            "--headless",
        ]

        if cfg.no_vlm:
            cmd.append("--no-vlm")
        elif cfg.vlm_model:
            cmd += ["--vlm-model", cfg.vlm_model]

        if cfg.delta_actions:
            cmd.append("--delta-actions")

        if cfg.vlm_endpoint:
            cmd += ["--vlm-endpoint", cfg.vlm_endpoint]

        # Forward any extra params
        for k, v in cfg.params.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]

        return cmd

    def _build_subprocess_env(self) -> dict[str, str]:
        """Build the environment dict for subprocess, forwarding VLA_URL."""
        env = os.environ.copy()
        # Ensure VLA_URL is set (required by SimWrapper)
        vla_url = self.extra_env.get("VLA_URL") or self.config.vla_url
        env["VLA_URL"] = vla_url
        env.update(self.extra_env)
        return env

    def _run_episode(self, task_id: int, episode: int) -> dict[str, Any]:
        """Run a single episode via subprocess and collect the result.

        Returns an EpisodeResult dict.
        """
        from robo_eval.results.collector import EpisodeResult

        cmd = self._build_subprocess_cmd(task_id, episode)
        env = self._build_subprocess_env()
        cfg = self.config

        logger.debug("Launching: %s", " ".join(cmd))
        t_start = time.time()

        ep_timeout = int(getattr(cfg, "episode_timeout_seconds", 1800))
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=ep_timeout,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t_start
            result: EpisodeResult = {
                "episode_id": episode,
                "metrics": {"success": False},
                "steps": 0,
                "elapsed_sec": elapsed,
                "failure_reason": "timeout",
                "failure_detail": f"subprocess exceeded {ep_timeout}s timeout",
            }
            return result

        elapsed = time.time() - t_start

        if proc.returncode != 0:
            logger.warning(
                "Episode task=%d ep=%d exited with code %d",
                task_id,
                episode,
                proc.returncode,
            )
            logger.warning("subprocess stderr (last 2000 chars):\n%s", proc.stderr[-2000:] if proc.stderr else "<empty>")
            result = {
                "episode_id": episode,
                "metrics": {"success": False},
                "steps": 0,
                "elapsed_sec": round(elapsed, 2),
                "failure_reason": f"nonzero_exit_{proc.returncode}",
                "failure_detail": proc.stderr[-1000:] if proc.stderr else "",
            }
            return result

        # Try to read the episode JSON written by episode_logger
        ep_json = self._read_episode_json(cfg.suite, task_id, episode)
        if ep_json is not None:
            steps = ep_json.get("steps", 0)
            success = bool(ep_json.get("success", False))
            result = {
                "episode_id": episode,
                "metrics": {"success": success},
                "steps": steps,
                "elapsed_sec": round(elapsed, 2),
            }
            return result

        # Fallback: parse success from stdout
        success = _parse_success_from_stdout(proc.stdout)
        steps = _parse_steps_from_stdout(proc.stdout)
        result = {
            "episode_id": episode,
            "metrics": {"success": success},
            "steps": steps,
            "elapsed_sec": round(elapsed, 2),
        }
        return result

    def _read_episode_json(
        self, suite: str, task_id: int, episode: int
    ) -> dict[str, Any] | None:
        """Read the episode JSON file written by episode_logger.

        run_sim_eval.py saves to <results-dir>/episodes/<suite>_task<N>_ep<M>.json.
        """
        # The orchestrator passes _output_dir as --results-dir, so episode files
        # are at _output_dir/episodes/
        episodes_dir = self._output_dir / "episodes"
        json_path = episodes_dir / f"{suite}_task{task_id}_ep{episode}.json"
        if json_path.exists():
            try:
                return json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _save_results(
        self, collector: "ResultCollector", output_path: Path  # type: ignore[name-defined]
    ) -> dict[str, Any]:
        """Save benchmark result to disk and return the result dict."""
        try:
            collector.print_summary()
        except Exception:
            pass

        result = collector.get_benchmark_result(config=self.config.to_dict())

        # Add shard metadata
        if self.num_shards is not None and self.shard_id is not None:
            result["shard"] = {"id": self.shard_id, "total": self.num_shards}

        try:
            _atomic_write_json(output_path, result)
            logger.info("Results saved to %s", output_path)
        except OSError as e:
            logger.error(
                "Failed to save results to %s: %s. "
                "Original file (if any) preserved; orphan .tmp removed.",
                output_path, e,
            )
        finally:
            self._release_lock()
            # Remove progress file — result JSON replaces it
            if self._progress_path and self._progress_path.exists():
                try:
                    self._progress_path.unlink()
                except OSError:
                    pass

        return result


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write *payload* as JSON via tmp + os.replace so disk-full doesn't
    corrupt an existing result file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default task counts per suite (used when tasks list is not specified)
_SUITE_TASK_COUNTS: dict[str, int] = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
    "robocasa": 20,
    "robotwin": 20,
}


def _parse_success_from_stdout(stdout: str) -> bool:
    """Parse success flag from run_sim_eval.py stdout."""
    if not stdout:
        return False
    # Look for "Simulator reports success: True" pattern
    for line in reversed(stdout.splitlines()):
        if "success: True" in line or "success=True" in line:
            return True
        if "success: False" in line or "success=False" in line:
            return False
    return False


def _parse_steps_from_stdout(stdout: str) -> int:
    """Parse total step count from stdout."""
    import re as _re
    if not stdout:
        return 0
    match = _re.search(r"steps[=:\s]+(\d+)", stdout, _re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def run_from_yaml(
    yaml_path: str | Path,
    shard_id: int | None = None,
    num_shards: int | None = None,
    results_dir: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convenience entry point: load config from YAML and run the orchestrator."""
    config = EvalConfig.from_yaml(yaml_path)
    orch = Orchestrator(
        config=config,
        shard_id=shard_id,
        num_shards=num_shards,
        results_dir=results_dir,
        extra_env=extra_env,
    )
    return orch.run()
