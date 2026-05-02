"""Orchestrator: coordinates sharded benchmark evaluation runs via subprocess.

Execution flow:
    1. Load EvalConfig from a flat YAML file.
    2. Build a (task, episode) work list from tasks × episodes_per_task.
    3. Optionally shard via round-robin: ``item_index % num_shards == shard_id``.
    4. For each work item, launch ``python -m roboeval.run_sim_eval eval`` as a subprocess,
       capturing per-episode JSON already written by roboeval.episode_logger.
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
from typing import TYPE_CHECKING, Any

from roboeval.config import resolve_suites

if TYPE_CHECKING:
    from roboeval.results.collector import ResultCollector

logger = logging.getLogger(__name__)

_SAFE_NAME_RE = re.compile(r"[^\w\-.]")


class ResetResamplingExhausted(RuntimeError):
    """Raised when all bounded reset-resampling candidates fail before rollout."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit

# ---------------------------------------------------------------------------
# EvalConfig dataclass
# ---------------------------------------------------------------------------


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
    task: str | None = None
    tasks: list[int | str] = field(default_factory=list)
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
    policy_instruction_override: str | None = None

    # Video capture
    record_video: bool = False
    record_video_n: int = 3

    # Python executable for subprocess launch
    eval_python: str = ""

    # Output
    output_dir: str = "./results"

    # Extra params forwarded to run_sim_eval.py
    params: dict[str, Any] = field(default_factory=dict)
    # Extra simulator configuration forwarded via --sim-config.
    sim_config: dict[str, Any] | str | None = None
    # Opt-in policy for generated-scene reset/startup failures. When enabled,
    # zero-step reset/startup candidates are excluded from completed eval
    # episodes and retried with deterministic later scene indices.
    reset_resample_on_failure: bool = False
    reset_resample_max_attempts: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalConfig:
        """Construct from a flat dict (e.g. from YAML)."""
        cfg = cls()
        cfg.name = d.get("name", "eval")
        cfg.vla_url = d.get("vla_url", os.environ.get("VLA_URL", "http://localhost:5100"))
        cfg.sim_url = d.get("sim_url", "http://localhost:5300")
        cfg.sim = d.get("sim", "libero")
        cfg.suite = d.get("suite", "libero_spatial")
        task_raw = d.get("task", None)
        cfg.task = str(task_raw) if task_raw not in (None, "") else None
        tasks_raw = d.get("tasks", [])
        cfg.tasks = [int(t) if str(t).isdigit() else str(t) for t in tasks_raw] if tasks_raw else []
        cfg.max_tasks = d.get("max_tasks", None)
        cfg.episodes_per_task = int(d.get("episodes_per_task", 10))
        cfg.episode_timeout_seconds = int(d.get("episode_timeout_seconds", 1800))
        cfg.no_vlm = bool(d.get("no_vlm", True))
        cfg.vlm_model = d.get("vlm_model", None)
        cfg.vlm_endpoint = d.get("vlm_endpoint", "localhost:4000")
        cfg.delta_actions = bool(d.get("delta_actions", True))
        cfg.policy_instruction_override = d.get("policy_instruction_override", None)
        cfg.record_video = bool(d.get("record_video", False))
        cfg.record_video_n = int(d.get("record_video_n", 3))
        cfg.eval_python = d.get("eval_python", "")
        cfg.output_dir = d.get("output_dir", "./results")
        cfg.params = d.get("params", {})
        cfg.sim_config = d.get("sim_config", None)
        cfg.reset_resample_on_failure = bool(d.get("reset_resample_on_failure", False))
        cfg.reset_resample_max_attempts = max(1, int(d.get("reset_resample_max_attempts", 1)))
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
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
            "task": self.task,
            "tasks": self.tasks,
            "max_tasks": self.max_tasks,
            "episodes_per_task": self.episodes_per_task,
            "episode_timeout_seconds": self.episode_timeout_seconds,
            "no_vlm": self.no_vlm,
            "vlm_model": self.vlm_model,
            "vlm_endpoint": self.vlm_endpoint,
            "delta_actions": self.delta_actions,
            "policy_instruction_override": self.policy_instruction_override,
            "record_video": self.record_video,
            "record_video_n": self.record_video_n,
            "eval_python": self.eval_python,
            "output_dir": self.output_dir,
            "params": self.params,
            "sim_config": self.sim_config,
            "reset_resample_on_failure": self.reset_resample_on_failure,
            "reset_resample_max_attempts": self.reset_resample_max_attempts,
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
        self._blockers: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the evaluation. Returns the benchmark result dict."""
        cfg = self.config
        safe_name = _SAFE_NAME_RE.sub("_", cfg.name)

        suites = self._build_suite_list()
        if not suites:
            logger.warning("No suites to evaluate (suite=%r)", cfg.suite)
            return {}

        # Build flat (suite, task_id, episode) work items.
        work_items: list[tuple[str, int | str, int]] = []
        for suite in suites:
            tasks = self._build_task_list(suite)
            if not tasks:
                logger.warning("No tasks to evaluate (suite=%r, tasks=%r)", suite, cfg.tasks)
                continue
            for task_id in tasks:
                for ep in range(cfg.episodes_per_task):
                    work_items.append((suite, task_id, ep))

        # Shard round-robin
        if self.num_shards is not None and self.shard_id is not None:
            work_items = [
                w for i, w in enumerate(work_items) if i % self.num_shards == self.shard_id
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
        from roboeval.results.collector import ResultCollector

        collector = ResultCollector(
            benchmark_name=cfg.name,
            mode="sync",
            metric_keys={"success": "mean"},
        )

        total_items = len(work_items)
        self._update_progress(0, total_items, 0)

        # Run episodes
        multi_suite = len(suites) > 1
        for item_idx, (suite, task_id, ep) in enumerate(work_items):
            task_name = f"{suite}_task_{task_id}" if multi_suite else f"task_{task_id}"
            try:
                if multi_suite or suite != cfg.suite:
                    ep_result = self._run_episode(task_id, ep, suite=suite)
                else:
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
            except ResetResamplingExhausted as exc:
                logger.error(
                    "  [%d/%d] %s ep%d: RESET RESAMPLING EXHAUSTED",
                    item_idx + 1,
                    total_items,
                    task_name,
                    ep,
                )
                self._blockers.append(exc.audit)
                break
            except Exception:
                logger.exception(
                    "  [%d/%d] %s ep%d: ERROR",
                    item_idx + 1,
                    total_items,
                    task_name,
                    ep,
                )
                from roboeval.results.collector import EpisodeResult

                failed: EpisodeResult = {
                    "episode_id": ep,
                    "metrics": {"success": False},
                    "failure_reason": "exception",
                    "failure_detail": traceback.format_exc(),
                }
                collector.record(task_name, failed)
            finally:
                self._update_progress(
                    item_idx + 1,
                    total_items,
                    collector.error_count + len(self._blockers),
                )

        result = self._save_results(collector, output_path)
        if self._blockers:
            raise RuntimeError(
                "reset resampling exhausted before completing all requested eval episodes; "
                f"see {self._output_dir / 'reset_resampling_audit.jsonl'}"
            )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_suite_list(self) -> list[str]:
        """Build the list of fully-qualified suites to evaluate."""
        from roboeval.config import get_suites_for_benchmark, qualify_suite

        cfg = self.config
        suites: list[str] = []
        for suite in resolve_suites(cfg.suite):
            try:
                benchmark_suites = get_suites_for_benchmark(cfg.sim)
            except ValueError:
                benchmark_suites = []
            if suite in benchmark_suites:
                suite = qualify_suite(cfg.sim, suite)
            suites.append(suite)
        return suites

    def _build_task_list(self, suite: str | None = None) -> list[int | str]:
        """Build the list of task IDs to evaluate."""
        cfg = self.config
        if cfg.tasks:
            tasks = list(cfg.tasks)
        elif cfg.task:
            tasks = [cfg.task]
        else:
            # Default: tasks 0..max_tasks-1 or all tasks in suite
            # Fall back to a reasonable default for libero suites
            default_n = _SUITE_TASK_COUNTS.get(suite or cfg.suite, 10)
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
                from filelock import FileLock

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
                except FileExistsError as _exist_err:
                    raise FileExistsError(
                        f"Another eval is already writing to {output_path}. "
                        "Remove the .lock file or use a different output_dir."
                    ) from _exist_err
            except Exception as _lock_err:
                raise FileExistsError(
                    f"Another eval is already writing to {output_path}. "
                    "Remove the .lock file or use a different output_dir."
                ) from _lock_err

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
            tmp.write_text(json.dumps({"completed": completed, "total": total, "errors": errors}))
            tmp.replace(self._progress_path)  # atomic on POSIX
        except OSError as e:
            logger.debug("Could not write progress file: %s", e)

    def _build_subprocess_cmd(
        self,
        task_id: int | str,
        episode: int,
        suite: str | None = None,
        result_episode: int | None = None,
    ) -> list[str]:
        """Build the roboeval.run_sim_eval subprocess command line."""
        cfg = self.config

        # Resolve Python executable
        python = cfg.eval_python or sys.executable

        cmd = [
            python,
            "-m",
            "roboeval.run_sim_eval",
            "eval",
            "--sim",
            cfg.sim,
            "--task",
            str(task_id),
            "--suite",
            suite or cfg.suite,
            "--sim-url",
            cfg.sim_url,
            "--episode",
            str(episode),
            # roboeval.run_sim_eval saves episodes to <results-dir>/episodes/
            # so pass _output_dir directly (not _output_dir/episodes)
            "--results-dir",
            str(self._output_dir),
            "--headless",
        ]
        if result_episode is not None and result_episode != episode:
            cmd += ["--result-episode", str(result_episode)]

        if cfg.no_vlm:
            cmd.append("--no-vlm")
        elif cfg.vlm_model:
            cmd += ["--vlm-model", cfg.vlm_model]

        if cfg.delta_actions:
            cmd.append("--delta-actions")

        if cfg.policy_instruction_override is not None:
            cmd += ["--policy-instruction-override", cfg.policy_instruction_override]

        if cfg.record_video:
            cmd += ["--record-video", "--record-video-n", str(cfg.record_video_n)]

        if cfg.vlm_endpoint:
            cmd += ["--vlm-endpoint", cfg.vlm_endpoint]

        if cfg.sim_config:
            if isinstance(cfg.sim_config, str):
                sim_config_path = cfg.sim_config
            else:
                import yaml  # type: ignore

                sim_config_path = str(self._output_dir / "sim_config.yaml")
                with open(sim_config_path, "w") as f:
                    yaml.safe_dump(cfg.sim_config, f, sort_keys=True)
            cmd += ["--sim-config", sim_config_path]

        # Forward any extra params. Boolean true values map to Typer flags such
        # as --record-video; false/null values are omitted.
        for k, v in cfg.params.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
                continue
            if v is None:
                continue
            cmd += [flag, str(v)]

        return cmd

    def _build_subprocess_env(self) -> dict[str, str]:
        """Build the environment dict for subprocess, forwarding VLA_URL."""
        env = os.environ.copy()
        # Ensure VLA_URL is set (required by SimWrapper)
        vla_url = self.extra_env.get("VLA_URL") or self.config.vla_url
        env["VLA_URL"] = vla_url
        env.update(self.extra_env)
        return env

    def _run_episode(
        self,
        task_id: int | str,
        episode: int,
        suite: str | None = None,
    ) -> dict[str, Any]:
        """Run a single episode via subprocess and collect the result.

        Returns an EpisodeResult dict.
        """
        cfg = self.config
        max_attempts = cfg.reset_resample_max_attempts if cfg.reset_resample_on_failure else 1
        resample_attempts: list[dict[str, Any]] = []

        last_reset_audit: dict[str, Any] | None = None
        for attempt in range(max_attempts):
            scene_episode = episode + attempt * max(1, cfg.episodes_per_task)
            result, proc_stdout, proc_stderr = self._run_episode_once(
                task_id,
                scene_episode,
                suite=suite,
                result_episode=episode if scene_episode != episode else None,
            )
            is_reset_failure = (
                cfg.reset_resample_on_failure
                and self._is_reset_startup_failure(result, proc_stdout, proc_stderr)
            )
            if not is_reset_failure:
                if resample_attempts:
                    result["reset_resampling"] = {
                        "policy": "exclude zero-step reset/startup failures; retry deterministic scene index",
                        "logical_episode_id": episode,
                        "accepted_scene_episode_id": scene_episode,
                        "attempts_before_accept": resample_attempts,
                        "max_attempts": max_attempts,
                    }
                    self._append_reset_resampling_audit(
                        {
                            "event": "accepted_resampled_scene",
                            "task_id": task_id,
                            "suite": suite or cfg.suite,
                            "logical_episode_id": episode,
                            "scene_episode_id": scene_episode,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "result": result,
                        }
                    )
                return result

            audit = self._reset_attempt_audit_record(
                task_id=task_id,
                suite=suite or cfg.suite,
                logical_episode=episode,
                scene_episode=scene_episode,
                attempt=attempt,
                max_attempts=max_attempts,
                result=result,
                stdout=proc_stdout,
                stderr=proc_stderr,
            )
            last_reset_audit = audit
            resample_attempts.append(audit)
            self._append_reset_resampling_audit(audit)
            if attempt + 1 >= max_attempts:
                exhausted = {
                    "event": "reset_resampling_exhausted",
                    "policy": (
                        "zero-step reset/startup failures are excluded from completed "
                        "eval episodes; bounded candidate budget exhausted"
                    ),
                    "task_id": task_id,
                    "suite": suite or cfg.suite,
                    "logical_episode_id": episode,
                    "max_attempts": max_attempts,
                    "attempts": resample_attempts,
                    "last_failure": last_reset_audit,
                }
                self._append_reset_resampling_audit(exhausted)
                raise ResetResamplingExhausted(
                    (
                        f"reset/startup failures exhausted {max_attempts} candidates "
                        f"for task={task_id} logical_ep={episode}"
                    ),
                    exhausted,
                )
            logger.warning(
                "Reset/startup failure excluded from eval episode task=%s ep=%d "
                "scene_ep=%d attempt=%d/%d; retrying",
                task_id,
                episode,
                scene_episode,
                attempt + 1,
                max_attempts,
            )

        # The loop always returns from its final iteration.
        return result

    def _run_episode_once(
        self,
        task_id: int | str,
        episode: int,
        suite: str | None = None,
        result_episode: int | None = None,
    ) -> tuple[dict[str, Any], str, str]:
        """Run one scene candidate and return result plus captured process output."""
        from roboeval.results.collector import EpisodeResult

        output_episode = result_episode if result_episode is not None else episode
        cmd = self._build_subprocess_cmd(
            task_id,
            episode,
            suite=suite,
            result_episode=result_episode,
        )
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
                "episode_id": output_episode,
                "metrics": {"success": False},
                "steps": 0,
                "elapsed_sec": elapsed,
                "failure_reason": "timeout",
                "failure_detail": f"subprocess exceeded {ep_timeout}s timeout",
            }
            return result, "", ""

        elapsed = time.time() - t_start

        if proc.returncode != 0:
            logger.warning(
                "Episode task=%s ep=%d exited with code %d",
                task_id,
                episode,
                proc.returncode,
            )
            logger.warning(
                "subprocess output (last 2000 chars):\n%s",
                ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-2000:] or "<empty>",
            )
            result = {
                "episode_id": output_episode,
                "metrics": {"success": False},
                "steps": 0,
                "elapsed_sec": round(elapsed, 2),
                "failure_reason": f"nonzero_exit_{proc.returncode}",
                "failure_detail": ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-4000:],
            }
            return result, proc.stdout or "", proc.stderr or ""

        # Try to read the episode JSON written by episode_logger
        ep_json = self._read_episode_json(suite or cfg.suite, task_id, output_episode)
        if ep_json is not None:
            steps = ep_json.get("steps", 0)
            success = bool(ep_json.get("success", False))
            result = {
                "episode_id": output_episode,
                "metrics": {"success": success},
                "steps": steps,
                "elapsed_sec": round(elapsed, 2),
            }
            self._attach_video_artifacts(result, suite or cfg.suite, task_id, output_episode)
            return result, proc.stdout or "", proc.stderr or ""

        # Fallback: parse success from stdout
        success = _parse_success_from_stdout(proc.stdout)
        steps = _parse_steps_from_stdout(proc.stdout)
        result = {
            "episode_id": output_episode,
            "metrics": {"success": success},
            "steps": steps,
            "elapsed_sec": round(elapsed, 2),
        }
        self._attach_video_artifacts(result, suite or cfg.suite, task_id, output_episode)
        return result, proc.stdout or "", proc.stderr or ""

    def _is_reset_startup_failure(self, result: dict[str, Any], stdout: str, stderr: str) -> bool:
        """Return True for zero-step scene reset/startup failures eligible for resampling."""
        if result.get("steps", 0) != 0 or not result.get("failure_reason"):
            return False
        text = f"{result.get('failure_detail', '')}\n{stdout}\n{stderr}".lower()
        reset_markers = (
            "/reset",
            "resetting simulator",
            "failed to reach sim server",
            "internal server error",
            "too many contacts",
            "ncon = 5000",
            "invalid scenic sample after mujoco settling",
            "failed to reset libero-infinity scene",
        )
        init_visibility_markers = (
            "failed to initialize simulator",
            "visibility check",
            "out of frame or fully occluded",
            "only weakly visible in agentview",
        )
        init_markers = ("/init", "post /init", "http 500")
        if (any(marker in text for marker in init_markers) and any(
            marker in text for marker in init_visibility_markers
        )):
            return True
        return any(marker in text for marker in reset_markers)

    def _reset_attempt_audit_record(
        self,
        *,
        task_id: int | str,
        suite: str,
        logical_episode: int,
        scene_episode: int,
        attempt: int,
        max_attempts: int,
        result: dict[str, Any],
        stdout: str,
        stderr: str,
    ) -> dict[str, Any]:
        seed = self._sim_seed()
        return {
            "event": "excluded_reset_startup_failure",
            "task_id": task_id,
            "suite": suite,
            "logical_episode_id": logical_episode,
            "scene_episode_id": scene_episode,
            "resample_attempt": attempt,
            "max_attempts": max_attempts,
            "run_seed": seed,
            "candidate_scenic_seeds": self._candidate_scenic_seeds(seed, scene_episode),
            "failure_reason": result.get("failure_reason"),
            "failure_detail_tail": str(result.get("failure_detail", ""))[-2000:],
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

    def _append_reset_resampling_audit(self, record: dict[str, Any]) -> None:
        audit_path = self._output_dir / "reset_resampling_audit.jsonl"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _sim_seed(self) -> int | None:
        sim_config = self.config.sim_config
        if isinstance(sim_config, dict) and "seed" in sim_config:
            try:
                return int(sim_config["seed"])
            except (TypeError, ValueError):
                return None
        return None

    def _candidate_scenic_seeds(self, run_seed: int | None, scene_episode: int) -> list[dict[str, Any]]:
        if run_seed is None or not isinstance(self.config.sim_config, dict):
            return []
        max_reset_attempts = int(self.config.sim_config.get("max_reset_attempts", 1))
        import hashlib

        return [
            {
                "reset_attempt": reset_attempt,
                "scenic_seed": int(
                    hashlib.sha256(f"{run_seed}:{scene_episode}:{reset_attempt}".encode()).hexdigest(),
                    16,
                )
                % (2**31),
                "formula": "sha256(run_seed:scene_episode_index:reset_attempt) % 2**31",
            }
            for reset_attempt in range(max(1, max_reset_attempts))
        ]

    def _attach_video_artifacts(
        self,
        result: dict[str, Any],
        suite: str,
        task_id: int | str,
        episode: int,
    ) -> None:
        """Attach recorded video artifact metadata to an episode result."""
        if not self.config.record_video:
            return
        videos_dir = self._output_dir / "videos"
        paths = sorted(videos_dir.glob(f"{suite}_task{task_id}_ep{episode}_*.mp4"))
        result["video"] = {
            "expected": True,
            "paths": [str(path) for path in paths],
            "artifact_present": any(path.is_file() and path.stat().st_size > 0 for path in paths),
        }

    def _read_episode_json(
        self, suite: str, task_id: int | str, episode: int
    ) -> dict[str, Any] | None:
        """Read the episode JSON file written by episode_logger.

        roboeval.run_sim_eval saves to <results-dir>/episodes/<suite>_task<N>_ep<M>.json.
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

    def _save_results(self, collector: ResultCollector, output_path: Path) -> dict[str, Any]:
        """Save benchmark result to disk and return the result dict."""
        try:
            collector.print_summary()
        except Exception:
            pass

        result = collector.get_benchmark_result(config=self.config.to_dict())
        if self._blockers:
            result["reset_resampling_blockers"] = self._blockers

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
                output_path,
                e,
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
    "libero_infinity_spatial": 10,
    "libero_infinity_object": 10,
    "libero_infinity_goal": 10,
    "libero_infinity_10": 10,
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
