#!/usr/bin/env python3
"""Sample GB10 unified memory pressure and selected process usage.

GB10 currently reports GPU utilization through nvidia-smi but returns N/A for
memory.total and memory.used. For scheduling roboeval jobs, process PSS/RSS and
system MemAvailable are the most useful conservative counters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable


DEFAULT_PATTERN = (
    r"(^|/)(python|python3)[^ ]* .*("
    r"-m roboeval|-m sims\.sim_worker|-m sims\.vla_policies|sims/sim_worker\.py"
    r")|(^|/)(vllm|uvicorn)( |$)|/usr/local/bin/vllm"
)


def _read_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts and parts[0].isdigit():
                out[key] = int(parts[0])
    return out


def _read_status_kb(pid: int, key: str) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except OSError:
        return None
    return None


def _read_smaps_rollup_kb(pid: int, key: str) -> int | None:
    try:
        with open(f"/proc/{pid}/smaps_rollup", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key + ":"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1])
    except OSError:
        return None
    return None


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _process_sample(pattern: re.Pattern[str], explicit_pids: set[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        cmdline = _cmdline(pid)
        if pid not in explicit_pids and not pattern.search(cmdline):
            continue
        rss_kb = _read_status_kb(pid, "VmRSS") or 0
        pss_kb = _read_smaps_rollup_kb(pid, "Pss")
        swap_kb = _read_smaps_rollup_kb(pid, "Swap") or _read_status_kb(pid, "VmSwap") or 0
        rows.append(
            {
                "pid": pid,
                "rss_kb": rss_kb,
                "pss_kb": pss_kb,
                "swap_kb": swap_kb,
                "cmdline": cmdline,
            }
        )
    return sorted(rows, key=lambda row: int(row["rss_kb"]), reverse=True)


def _nvidia_smi() -> dict[str, object]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip() or proc.stdout.strip()}
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "name": parts[0],
                    "utilization_gpu_percent": _int_or_none(parts[1]),
                    "memory_total_mib": _int_or_none(parts[2]),
                    "memory_used_mib": _int_or_none(parts[3]),
                }
            )
    return {"available": True, "gpus": rows}


def _int_or_none(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def sample(pattern: str, pids: Iterable[int]) -> dict[str, object]:
    meminfo = _read_meminfo()
    proc_rows = _process_sample(re.compile(pattern), set(pids))
    total_pss_kb = sum(int(row["pss_kb"] or row["rss_kb"]) for row in proc_rows)
    total_rss_kb = sum(int(row["rss_kb"]) for row in proc_rows)
    return {
        "timestamp": time.time(),
        "mem_total_kb": meminfo.get("MemTotal"),
        "mem_available_kb": meminfo.get("MemAvailable"),
        "swap_free_kb": meminfo.get("SwapFree"),
        "nvidia_smi": _nvidia_smi(),
        "matched_process_total_pss_kb": total_pss_kb,
        "matched_process_total_rss_kb": total_rss_kb,
        "processes": proc_rows,
    }


def _print_human(data: dict[str, object]) -> None:
    def gib(kb: object) -> str:
        return "n/a" if kb is None else f"{int(kb) / 1024 / 1024:.2f} GiB"

    print(f"time={data['timestamp']:.0f}")
    print(
        "memory: "
        f"available={gib(data.get('mem_available_kb'))} "
        f"total={gib(data.get('mem_total_kb'))} "
        f"swap_free={gib(data.get('swap_free_kb'))}"
    )
    print(
        "matched processes: "
        f"pss={gib(data.get('matched_process_total_pss_kb'))} "
        f"rss={gib(data.get('matched_process_total_rss_kb'))}"
    )
    nvsmi = data.get("nvidia_smi")
    if isinstance(nvsmi, dict):
        print(f"nvidia_smi: {json.dumps(nvsmi, sort_keys=True)}")
    for row in data["processes"]:  # type: ignore[index]
        pss = row.get("pss_kb") or row.get("rss_kb")  # type: ignore[union-attr]
        print(
            f"  pid={row['pid']} pss={gib(pss)} rss={gib(row['rss_kb'])} "
            f"swap={gib(row['swap_kb'])} {row['cmdline']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Regex matched against cmdline.")
    parser.add_argument("--pid", action="append", type=int, default=[], help="PID to always include.")
    parser.add_argument("--interval", type=float, default=0.0, help="Seconds between samples.")
    parser.add_argument("--count", type=int, default=1, help="Number of samples; 0 means forever.")
    parser.add_argument("--jsonl", action="store_true", help="Emit JSON Lines instead of human text.")
    args = parser.parse_args()

    i = 0
    while args.count == 0 or i < args.count:
        data = sample(args.pattern, args.pid)
        if args.jsonl:
            print(json.dumps(data, sort_keys=True), flush=True)
        else:
            _print_human(data)
        i += 1
        if args.interval <= 0 or (args.count and i >= args.count):
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
