"""Preflight smoke tests for roboeval.

Implements ``roboeval test``:

    --validate (default, fast):
        YAML parses, registry resolves, ActionObsSpec round-trips, no network.

    --server:
        Probe each declared VLA's /health + /info, validate spec dicts.

    --benchmark:
        Probe each declared sim's /health + /info, send /reset + zero /step.

    --all:
        All of the above + 1-episode end-to-end smoke run.

Returns nonzero exit on any failure.  Output is a status list:
    [OK]   vla.health
    [FAIL] sim.spec_validation: <reason>
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from roboeval.orchestrator import EvalConfig, Orchestrator

logger = logging.getLogger(__name__)

# ANSI colours for status output (disabled on non-TTY)
_GREEN = "\033[32m" if sys.stdout.isatty() else ""
_RED = "\033[31m" if sys.stdout.isatty() else ""
_YELLOW = "\033[33m" if sys.stdout.isatty() else ""
_RESET = "\033[0m" if sys.stdout.isatty() else ""


@dataclass
class CheckResult:
    name: str
    ok: bool
    message: str = ""


@dataclass
class PreflightConfig:
    """Minimal config needed by preflight checks."""

    vla_url: str = "http://localhost:5100"
    sim_url: str = "http://localhost:5300"
    sim: str = "libero"
    suite: str = "libero_spatial"
    name: str = "preflight"
    vla_name: str = "pi05"
    servers: list[dict[str, Any]] = field(default_factory=list)
    sims: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PreflightConfig:
        cfg = cls()
        cfg.vla_url = d.get("vla_url", "http://localhost:5100")
        cfg.sim_url = d.get("sim_url", "http://localhost:5300")
        cfg.sim = d.get("sim", "libero")
        cfg.suite = d.get("suite", "libero_spatial")
        cfg.name = d.get("name", "preflight")
        cfg.vla_name = d.get("vla", "pi05")

        # Allow declaring a list of servers/sims
        servers = d.get("servers", [])
        if not servers and cfg.vla_url:
            servers = [{"url": cfg.vla_url, "name": cfg.vla_name}]
        cfg.servers = servers

        sims = d.get("sims", [])
        if not sims and cfg.sim_url:
            sims = [{"url": cfg.sim_url, "name": cfg.sim}]
        cfg.sims = sims

        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreflightConfig:
        import yaml  # type: ignore

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)


def _check(name: str, ok: bool, message: str = "") -> CheckResult:
    return CheckResult(name=name, ok=ok, message=message)


def print_results(results: list[CheckResult]) -> int:
    """Print status list and return exit code (0 = all OK)."""
    failed = 0
    for r in results:
        if r.ok:
            tag = f"{_GREEN}[OK]  {_RESET}"
        else:
            tag = f"{_RED}[FAIL]{_RESET}"
            failed += 1
        suffix = f": {r.message}" if r.message else ""
        print(f"  {tag} {r.name}{suffix}")
    return 1 if failed > 0 else 0


def validate_yaml(config_path: str | Path) -> list[CheckResult]:
    """--validate: check that YAML parses and config is valid. No network."""
    results: list[CheckResult] = []
    path = Path(config_path)

    # 1. YAML parses
    try:
        import yaml  # type: ignore

        with open(path) as f:
            d = yaml.safe_load(f) or {}
        results.append(_check("yaml.parse", True))
    except Exception as e:
        results.append(_check("yaml.parse", False, str(e)))
        return results

    # 2. EvalConfig loads
    try:
        from roboeval.orchestrator import EvalConfig

        EvalConfig.from_dict(d)
        results.append(_check("config.load", True))
    except Exception as e:
        results.append(_check("config.load", False, str(e)))
        return results

    # 3. Registry resolves (if benchmark key present)
    benchmark_cls_path = d.get("benchmark")
    if benchmark_cls_path:
        try:
            from roboeval.registry import resolve_import_string

            resolve_import_string(benchmark_cls_path)
            results.append(_check("registry.resolve", True))
        except Exception as e:
            results.append(_check("registry.resolve", False, str(e)))
    else:
        results.append(_check("registry.resolve", True, "(no benchmark key — skipped)"))

    # 4. ActionObsSpec round-trip
    try:
        from roboeval.specs import GRIPPER_CLOSE_POS, POSITION_DELTA, ROTATION_AA, ActionObsSpec

        for spec in [POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS]:
            d2 = spec.to_dict()
            spec2 = ActionObsSpec.from_dict(d2)
            assert spec2 == spec, f"Round-trip failed for {spec.name}"
        results.append(_check("specs.dimspec_roundtrip", True))
    except Exception as e:
        results.append(_check("specs.dimspec_roundtrip", False, str(e)))

    return results


def check_server(server: dict[str, Any]) -> list[CheckResult]:
    """--server: probe a single VLA server's /health + /info."""
    results: list[CheckResult] = []
    url = server.get("url", "http://localhost:5100")
    name = server.get("name", url)

    # /health
    try:
        resp = requests.get(url + "/health", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        # Accept: ready=True, status="ok", or any 200 response with no explicit not-ready
        if "ready" in data:
            ready = bool(data["ready"])
        elif "status" in data:
            ready = data["status"] in ("ok", "ready")
        else:
            ready = True  # 200 with no ready field → assume ready
        results.append(_check(f"{name}.health", ready))
    except Exception as e:
        results.append(_check(f"{name}.health", False, str(e)))
        return results  # No point checking /info if health fails

    # /info
    try:
        resp = requests.get(url + "/info", timeout=5.0)
        resp.raise_for_status()
        info = resp.json()
        results.append(_check(f"{name}.info", True))
    except Exception as e:
        results.append(_check(f"{name}.info", False, str(e)))
        return results

    # Spec validation
    try:
        from roboeval.specs import ActionObsSpec

        action_spec_raw = info.get("action_spec", {})
        obs_spec_raw = info.get("observation_spec", {})
        for _key, raw in action_spec_raw.items():
            ActionObsSpec.from_dict(raw)
        for _key, raw in obs_spec_raw.items():
            ActionObsSpec.from_dict(raw)
        spec_count = len(action_spec_raw) + len(obs_spec_raw)
        if spec_count > 0:
            results.append(_check(f"{name}.spec_validation", True, f"{spec_count} specs"))
        else:
            results.append(_check(f"{name}.spec_validation", True, "(legacy — no specs declared)"))
    except Exception as e:
        results.append(_check(f"{name}.spec_validation", False, str(e)))

    return results


def check_benchmark(sim: dict[str, Any]) -> list[CheckResult]:
    """--benchmark: probe a simulator /health + /info, send /reset + zero /step."""
    results: list[CheckResult] = []
    url = sim.get("url", "http://localhost:5300")
    name = sim.get("name", url)

    # /health
    try:
        resp = requests.get(url + "/health", timeout=5.0)
        resp.raise_for_status()
        results.append(_check(f"{name}.health", True))
    except Exception as e:
        results.append(_check(f"{name}.health", False, str(e)))
        return results

    # /info
    try:
        resp = requests.get(url + "/info", timeout=5.0)
        resp.raise_for_status()
        info = resp.json()
        results.append(_check(f"{name}.info", True))
    except Exception as e:
        results.append(_check(f"{name}.info", False, str(e)))
        return results

    # /reset
    try:
        payload = {
            "task_name": sim.get("task_name", "0"),
            "suite": sim.get("suite", "libero_spatial"),
            "episode_index": 0,
        }
        resp = requests.post(url + "/reset", json=payload, timeout=30.0)
        resp.raise_for_status()
        results.append(_check(f"{name}.reset", True))
    except Exception as e:
        results.append(_check(f"{name}.reset", False, str(e)))
        return results

    # /step with zero action
    try:
        action_dim = info.get("action_dim", 7)
        zero_action = [0.0] * action_dim
        resp = requests.post(
            url + "/step",
            json={"action": zero_action},
            timeout=10.0,
        )
        resp.raise_for_status()
        results.append(_check(f"{name}.step", True))
    except Exception as e:
        results.append(_check(f"{name}.step", False, str(e)))

    return results


def run_preflight(
    config_path: str | Path,
    validate: bool = True,
    server: bool = False,
    benchmark: bool = False,
    all_checks: bool = False,
    results_dir: str | Path | None = None,
) -> int:
    """Run the selected preflight checks.

    Parameters
    ----------
    config_path:
        Path to the eval YAML config.
    validate:
        Run --validate checks (YAML, registry, ActionObsSpec).
    server:
        Probe each declared VLA server.
    benchmark:
        Probe each declared sim.
    all_checks:
        All of the above + 1-episode end-to-end smoke run.
    results_dir:
        Where to write the smoke-run results (--all only).

    Returns
    -------
    int
        Exit code: 0 = all OK, 1 = any failure.
    """
    if all_checks:
        validate = server = benchmark = True

    all_results: list[CheckResult] = []

    # --validate
    if validate:
        print("-- validate --")
        vr = validate_yaml(config_path)
        all_results.extend(vr)
        print_results(vr)
        print()
        if any(not r.ok for r in vr):
            # Stop early — can't load config for further checks
            return 1

    # Load config for server/benchmark checks
    try:
        pfcfg = PreflightConfig.from_yaml(config_path)
    except Exception as e:
        all_results.append(_check("config.preflight_load", False, str(e)))
        print_results(all_results[-1:])
        return 1

    # --server
    if server:
        print("-- server --")
        sr: list[CheckResult] = []
        for srv in pfcfg.servers:
            sr.extend(check_server(srv))
        all_results.extend(sr)
        print_results(sr)
        print()

    # --benchmark
    if benchmark:
        print("-- benchmark --")
        br: list[CheckResult] = []
        for sim_cfg in pfcfg.sims:
            br.extend(check_benchmark(sim_cfg))
        all_results.extend(br)
        print_results(br)
        print()

    # --all: run one end-to-end smoke episode
    if all_checks:
        print("-- all (1-episode dry run) --")
        dry_results: list[CheckResult] = []
        try:
            cfg = EvalConfig.from_yaml(config_path)
            cfg.episodes_per_task = 1
            cfg.max_tasks = 1
            dry_dir = Path(results_dir or "/tmp/roboeval_preflight_dry")
            orch = Orchestrator(config=cfg, results_dir=dry_dir)
            result = orch.run()
            tasks = result.get("tasks", [])
            success_count = sum(
                1
                for t in tasks
                for ep in t.get("episodes", [])
                if ep.get("metrics", {}).get("success")
            )
            total_count = sum(len(t.get("episodes", [])) for t in tasks)
            ok = total_count > 0
            msg = f"{success_count}/{total_count} episodes succeeded"
            dry_results.append(_check("dry_run.1ep", ok, msg))
        except Exception as e:
            dry_results.append(_check("dry_run.1ep", False, str(e)))
        all_results.extend(dry_results)
        print_results(dry_results)
        print()

    # Final summary
    failed_count = sum(1 for r in all_results if not r.ok)
    total_count = len(all_results)
    if failed_count == 0:
        print(f"{_GREEN}All {total_count} checks passed.{_RESET}")
    else:
        print(f"{_RED}{failed_count}/{total_count} checks failed.{_RESET}")

    return 1 if failed_count > 0 else 0
