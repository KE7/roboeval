#!/usr/bin/env python3
"""Pilot and conservatively queue LIBERO-Infinity runs on GB10.

The scheduler intentionally uses system MemAvailable plus process PSS/RSS
instead of GPU memory from nvidia-smi, because GB10 reports N/A for those GPU
memory fields. Run pilots first, inspect the produced metrics JSON, then raise
--max-parallel only when the projected memory budget is comfortable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gb10_mem_probe import sample as mem_sample  # noqa: E402


DEFAULT_CONFIG = "configs/libero_infinity_pi05_smoke.yaml"


@dataclass
class Cell:
    name: str
    config: str
    perturbation: Any | None
    suite: str | None
    task: str | None
    episodes_per_task: int | None
    max_tasks: int | None


def _git_provenance(path: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    return {
        "worktree": str(path),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--short")),
    }


def _build_provenance(
    manifest_provenance: dict[str, Any],
    libero_infinity_worktree: str | None,
) -> dict[str, Any]:
    provenance = {
        "privacy": "local-only; no push/tag/publication",
        "ownership": {
            "roboeval": "harness/config/orchestration/video/result plumbing",
            "libero_infinity": "generator/compiler/Scenic/runtime perturbation fixes",
        },
        "roboeval": _git_provenance(ROOT),
        "manifest": manifest_provenance,
    }
    li_path = (
        libero_infinity_worktree
        or manifest_provenance.get("libero_infinity_worktree")
        or os.environ.get("LIBERO_INFINITY_ROOT")
    )
    if li_path:
        path = Path(str(li_path)).expanduser()
        provenance["libero_infinity"] = (
            _git_provenance(path) if path.exists() else {"worktree": str(path), "missing": True}
        )
    return provenance


def _load_manifest(path: str | None) -> tuple[list[Cell], dict[str, Any]]:
    if not path:
        return (
            [
                Cell(
                    name="libero_infinity_pi05_camera",
                    config=DEFAULT_CONFIG,
                    perturbation=None,
                    suite=None,
                    task=None,
                    episodes_per_task=None,
                    max_tasks=None,
                )
            ],
            {},
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cells = raw.get("cells", raw if isinstance(raw, list) else [])
    manifest_provenance = raw.get("provenance", {}) if isinstance(raw, dict) else {}
    return (
        [
            Cell(
                name=str(item["name"]),
                config=str(item.get("config", DEFAULT_CONFIG)),
                perturbation=item.get("perturbation"),
                suite=item.get("suite"),
                task=item.get("task"),
                episodes_per_task=item.get("episodes_per_task"),
                max_tasks=item.get("max_tasks"),
            )
            for item in cells
        ],
        manifest_provenance or {},
    )


def _write_cell_config(
    cell: Cell,
    *,
    output_root: Path,
    vla_port: int,
    sim_port: int,
    pilot: bool,
    provenance: dict[str, Any],
) -> Path:
    with open(ROOT / cell.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = copy.deepcopy(cfg)
    cfg["name"] = cell.name + ("_pilot" if pilot else "")
    cfg["vla_url"] = f"http://localhost:{vla_port}"
    cfg["sim_url"] = f"http://localhost:{sim_port}"
    cfg["output_dir"] = str(output_root / cell.name / ("pilot" if pilot else "run"))
    if cell.suite is not None:
        cfg["suite"] = cell.suite
    if cell.task is not None:
        cfg["task"] = cell.task
    if cell.perturbation is not None:
        cfg.setdefault("sim_config", {})["perturbation"] = cell.perturbation
    if pilot:
        cfg["episodes_per_task"] = 1
        cfg["max_tasks"] = 1
        cfg["no_vlm"] = True
    else:
        if cell.episodes_per_task is not None:
            cfg["episodes_per_task"] = int(cell.episodes_per_task)
        if cell.max_tasks is not None:
            cfg["max_tasks"] = int(cell.max_tasks)
    cfg.setdefault("params", {})
    cfg["record_video"] = True
    cfg["record_video_n"] = int(cfg.get("episodes_per_task", 1))
    cfg["params"]["record_video"] = True
    cfg["params"]["record_video_n"] = int(cfg.get("episodes_per_task", 1))
    cfg["x_local_provenance"] = provenance

    out_dir = output_root / "_generated_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{cfg['name']}.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def _video_artifacts(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "videos").glob("*.mp4"))


def _probe_video(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "playable": False,
        "frame_count": None,
        "width": None,
        "height": None,
        "backend": None,
        "error": None,
    }
    if not result["exists"] or result["size_bytes"] == 0:
        result["error"] = "missing or empty file"
        return result
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        try:
            opened = cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            ok, _frame = cap.read()
            result.update(
                {
                    "playable": bool(opened and ok and width > 0 and height > 0),
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "backend": "cv2",
                }
            )
        finally:
            cap.release()
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def _verify_videos(output_dir: Path, expected_min: int) -> dict[str, Any]:
    videos = _video_artifacts(output_dir)
    probes = [_probe_video(path) for path in videos]
    playable = [probe for probe in probes if probe.get("playable")]
    return {
        "output_dir": str(output_dir),
        "expected_min": expected_min,
        "found": len(videos),
        "playable": len(playable),
        "ok": len(playable) >= expected_min,
        "artifacts": probes,
    }


def _run(cmd: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        stdout=log_f,
        stderr=log_f,
        start_new_session=True,
    )


def _wait_health(port: int, timeout: float) -> None:
    import requests

    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            if resp.ok and (data.get("ready") or data.get("status") in ("ok", "ready")):
                return
            last_error = str(data.get("error") or resp.status_code)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"port {port} did not become healthy: {last_error}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:  # noqa: BLE001
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass


def _peak_mem(samples: list[dict[str, Any]], baseline_available_kb: int | None) -> dict[str, Any]:
    min_available = min((s.get("mem_available_kb") or 0 for s in samples), default=0)
    max_pss = max((s.get("matched_process_total_pss_kb") or 0 for s in samples), default=0)
    max_rss = max((s.get("matched_process_total_rss_kb") or 0 for s in samples), default=0)
    delta = None
    if baseline_available_kb is not None and min_available:
        delta = max(0, baseline_available_kb - min_available)
    return {
        "min_mem_available_kb": min_available,
        "peak_matched_pss_kb": max_pss,
        "peak_matched_rss_kb": max_rss,
        "mem_available_delta_kb": delta,
    }


def _run_cell(
    cell: Cell,
    *,
    index: int,
    args: argparse.Namespace,
    pilot: bool,
) -> dict[str, Any]:
    output_root = Path(args.output_root)
    vla_port = args.vla_port_base + index * 10
    sim_port = args.sim_port_base + index * 10
    cfg_path = _write_cell_config(
        cell,
        output_root=output_root,
        vla_port=vla_port,
        sim_port=sim_port,
        pilot=pilot,
        provenance=args.provenance,
    )
    logs = output_root / "_logs" / cell.name
    serve_cmd = [
        args.roboeval,
        "serve",
        "--vla",
        "pi05",
        "--sim",
        "libero_infinity",
        "--vla-port",
        str(vla_port),
        "--sim-port",
        str(sim_port),
        "--headless",
        "--health-timeout",
        str(args.health_timeout),
        "--logs-dir",
        str(logs / "servers"),
    ]
    run_cmd = [args.roboeval, "run", "--config", str(cfg_path), "--vla-url", f"http://localhost:{vla_port}"]
    baseline = mem_sample(args.mem_pattern, [])
    baseline_available = baseline.get("mem_available_kb")
    proc = _run(serve_cmd, log_path=logs / ("serve_pilot.log" if pilot else "serve.log"))
    samples: list[dict[str, Any]] = [baseline]
    rc = 1
    try:
        _wait_health(vla_port, args.health_timeout)
        _wait_health(sim_port, args.health_timeout)
        run_proc = _run(run_cmd, log_path=logs / ("run_pilot.log" if pilot else "run.log"))
        while run_proc.poll() is None:
            samples.append(mem_sample(args.mem_pattern, [proc.pid, run_proc.pid]))
            time.sleep(args.sample_interval)
        samples.append(mem_sample(args.mem_pattern, [proc.pid, run_proc.pid]))
        rc = run_proc.returncode
    finally:
        _terminate(proc)
    metrics = {
        "cell": cell.name,
        "pilot": pilot,
        "config": str(cfg_path),
        "returncode": rc,
        "ports": {"vla": vla_port, "sim": sim_port},
        "memory": _peak_mem(samples, int(baseline_available) if baseline_available else None),
        "logs": str(logs),
        "provenance": args.provenance,
    }
    cell_output_dir = output_root / cell.name / ("pilot" if pilot else "run")
    metrics["videos"] = _verify_videos(cell_output_dir, expected_min=1)
    if rc == 0 and not metrics["videos"]["ok"]:
        metrics["returncode"] = 3
        metrics["video_error"] = "required playable video artifact was not produced"
    metrics_path = output_root / cell.name / ("pilot_metrics.json" if pilot else "run_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return metrics


def _memory_allows_start(args: argparse.Namespace) -> bool:
    data = mem_sample(args.mem_pattern, [])
    available_gib = (data.get("mem_available_kb") or 0) / 1024 / 1024
    return available_gib >= args.mem_available_floor_gib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="YAML file with a top-level cells list.")
    parser.add_argument("--mode", choices=("pilot", "run"), default="pilot")
    parser.add_argument("--output-root", default="results/libero_infinity_gb10_queue")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--mem-available-floor-gib", type=float, default=32.0)
    parser.add_argument("--vla-port-base", type=int, default=5510)
    parser.add_argument("--sim-port-base", type=int, default=5710)
    parser.add_argument("--health-timeout", type=float, default=240.0)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument(
        "--mem-pattern",
        default=(
            r"(^|/)(python|python3)[^ ]* .*("
            r"-m roboeval|-m sims\.sim_worker|-m sims\.vla_policies|sims/sim_worker\.py"
            r")|(^|/)(vllm|uvicorn)( |$)|/usr/local/bin/vllm"
        ),
    )
    parser.add_argument("--roboeval", default="roboeval")
    parser.add_argument(
        "--libero-infinity-worktree",
        help="Optional local private libero-infinity worktree for branch/commit provenance.",
    )
    args = parser.parse_args()

    if args.max_parallel != 1:
        print(
            "error: full/concurrent VLA servers are blocked by current GB10 policy. "
            "Keep --max-parallel 1 until pilot_metrics.json establishes peak memory.",
            file=sys.stderr,
        )
        return 2
    cells, manifest_provenance = _load_manifest(args.manifest)
    args.provenance = _build_provenance(manifest_provenance, args.libero_infinity_worktree)
    failures = 0
    # Intentionally sequential by default. This keeps pilot measurement clean and
    # avoids duplicating Pi0.5 model memory until the PM has real peak data.
    for i, cell in enumerate(cells):
        if not _memory_allows_start(args):
            print(
                f"refusing to start {cell.name}: MemAvailable below "
                f"{args.mem_available_floor_gib:.1f} GiB floor",
                file=sys.stderr,
            )
            failures += 1
            continue
        metrics = _run_cell(cell, index=i, args=args, pilot=args.mode == "pilot")
        print(json.dumps(metrics, indent=2, sort_keys=True, default=str))
        if metrics["returncode"] != 0:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
