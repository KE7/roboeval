"""roboeval CLI — 5-subcommand interface for the roboeval platform.

Subcommands:
    setup  — install backend environments via scripts/setup.sh
    run    — run sharded evaluation via the Orchestrator
    serve  — launch VLA + sim_worker servers
    merge  — merge shard result JSON files
    test   — preflight checks (validate / server / benchmark / all)

Philosophy:
    - No ``--runtime docker``, no ``ROBOEVAL_DOCKER_*`` env vars.
    - No ``--proxy-port``, no ``--replicas``.
    - All config via flat YAML + optional CLI overrides.
    - VLA_URL is forwarded via environment to subprocesses.

Usage examples:
    roboeval run --config configs/libero_spatial_pi05_smoke.yaml
    roboeval run --config configs/libero_spatial_pi05_smoke.yaml --shard-id 0 --num-shards 4
    roboeval serve --vla pi05 --sim libero
    roboeval merge --pattern 'results/*_shard*.json' -o final.json
    roboeval test --validate --config configs/libero_spatial_pi05_smoke.yaml
    roboeval test --all --config configs/libero_spatial_pi05_smoke.yaml
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="roboeval",
    help=(
        "roboeval: host-process VLA evaluation harness.\n\n"
        "Use 'roboeval <COMMAND> --help' for subcommand options."
    ),
    add_completion=False,
    no_args_is_help=True,
)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


SETUP_COMPONENTS = (
    "pi05 openvla smolvla groot internvla act diffusion_policy tdmpc2 "
    "vqbet libero libero_pro libero_infinity robocasa robotwin aloha_gym "
    "gym_pusht metaworld vlm"
)


def _find_setup_script() -> Path:
    package_dir = Path(__file__).resolve().parents[1]
    candidates = (
        package_dir.parent / "scripts" / "setup.sh",
        package_dir / "scripts" / "setup.sh",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("scripts/setup.sh")


def _resolve_serve_ports(
    *,
    vla: str,
    sim: Optional[str],
    vla_port: Optional[int],
    sim_port: Optional[int],
) -> tuple[int | None, int | None]:
    """Resolve serve ports, auto-selecting free defaults when omitted."""
    from roboeval.config import (
        find_available_port,
        find_available_port_block,
        is_port_available,
        validate_port,
    )
    from roboeval.server_runner import _SIM_DEFAULT_PORTS, _VLA_DEFAULT_PORTS

    vla_default = _VLA_DEFAULT_PORTS.get(vla, 5100)
    sim_default = _SIM_DEFAULT_PORTS.get(sim, 5300) if sim else None

    if vla_port is not None:
        vla_port = validate_port(vla_port, "--vla-port")
        if not is_port_available(vla_port):
            raise RuntimeError(f"--vla-port {vla_port} is already in use.")

    if sim_port is not None:
        sim_port = validate_port(sim_port, "--sim-port")
        if not is_port_available(sim_port):
            raise RuntimeError(f"--sim-port {sim_port} is already in use.")

    if vla_port is None and sim_port is None and sim_default == vla_default:
        base = find_available_port_block(2, preferred_start=vla_default)
        return base, base + 1

    if vla_port is None:
        vla_port = find_available_port(vla_default)
    if sim and sim_port is None:
        sim_port = find_available_port(sim_default)

    return vla_port, sim_port


# ---------------------------------------------------------------------------
# roboeval setup
# ---------------------------------------------------------------------------


@app.command(
    "setup",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def cmd_setup(
    ctx: typer.Context,
    components: Optional[list[str]] = typer.Argument(
        None,
        metavar="[COMPONENT]...",
        help=f"Components to set up. Supported: {SETUP_COMPONENTS}",
    ),
) -> None:
    """Install backend environments via scripts/setup.sh.

    Supported components:
    pi05 openvla smolvla groot internvla act diffusion_policy tdmpc2 vqbet
    libero libero_pro libero_infinity robocasa robotwin aloha_gym gym_pusht
    metaworld vlm
    """
    args = [*(components or []), *ctx.args]
    try:
        setup_script = _find_setup_script()
    except FileNotFoundError:
        typer.echo(
            "Error: bundled setup script not found: scripts/setup.sh",
            err=True,
        )
        raise typer.Exit(1)

    result = subprocess.run(["bash", str(setup_script), *args], check=False)
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# roboeval run
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    config: str = typer.Option(
        ...,
        "--config", "-c",
        help="Path to eval YAML config file.",
    ),
    shard_id: Optional[int] = typer.Option(
        None, "--shard-id",
        help="Zero-based shard index (used with --num-shards).",
    ),
    num_shards: Optional[int] = typer.Option(
        None, "--num-shards",
        help="Total number of shards.",
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="Override output directory from config.",
    ),
    vla_url: Optional[str] = typer.Option(
        None, "--vla-url",
        help="Override VLA server URL (also reads $VLA_URL).",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run sharded evaluation via the Orchestrator.

    Loads the config YAML, builds the (task, episode) work list, shards it
    (if --shard-id / --num-shards set), and launches run_sim_eval.py per
    episode as a subprocess.

    VLA_URL is forwarded to every subprocess.  Set it in the environment or
    pass --vla-url.

    Examples::

        roboeval run --config configs/libero_spatial_pi05_smoke.yaml
        roboeval run -c configs/libero_spatial_pi05_smoke.yaml --shard-id 0 --num-shards 4
    """
    _setup_logging(verbose)
    logger = logging.getLogger("roboeval.cli.run")

    # Validate shard args
    if (shard_id is None) != (num_shards is None):
        typer.echo(
            "Error: --shard-id and --num-shards must be specified together.",
            err=True,
        )
        raise typer.Exit(1)

    if shard_id is not None and shard_id >= num_shards:  # type: ignore[operator]
        typer.echo(
            f"Error: --shard-id {shard_id} must be < --num-shards {num_shards}.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        from roboeval.orchestrator import EvalConfig, Orchestrator
    except ImportError as e:
        typer.echo(f"Error: failed to import orchestrator: {e}", err=True)
        raise typer.Exit(1)

    try:
        cfg = EvalConfig.from_yaml(config)
    except FileNotFoundError:
        typer.echo(f"Error: config file not found: {config}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: failed to load config {config}: {e}", err=True)
        raise typer.Exit(1)

    # CLI overrides
    if output_dir:
        cfg.output_dir = output_dir

    # VLA URL override: CLI > env > config
    resolved_vla_url = vla_url or os.environ.get("VLA_URL") or cfg.vla_url
    extra_env: dict[str, str] = {}
    if resolved_vla_url:
        extra_env["VLA_URL"] = resolved_vla_url
        cfg.vla_url = resolved_vla_url

    typer.echo(f"roboeval run: config={config} shard={shard_id}/{num_shards}")
    typer.echo(f"  output_dir={cfg.output_dir}  vla_url={cfg.vla_url}")

    orch = Orchestrator(
        config=cfg,
        shard_id=shard_id,
        num_shards=num_shards,
        results_dir=output_dir,
        extra_env=extra_env,
    )

    try:
        result = orch.run()
    except Exception as e:
        logger.exception("Orchestrator failed: %s", e)
        raise typer.Exit(1)

    mean_success = result.get("mean_success")
    if mean_success is not None:
        typer.echo(f"\nOverall success rate: {mean_success:.1%}")

    # Print result file location
    output_path = Path(cfg.output_dir)
    typer.echo(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# roboeval serve
# ---------------------------------------------------------------------------


@app.command("serve")
def cmd_serve(
    vla: str = typer.Option(
        ...,
        "--vla",
        help="VLA to serve (pi05, pi0, smolvla, openvla, cosmos, groot, internvla).",
    ),
    sim: Optional[str] = typer.Option(
        None, "--sim",
        help="Sim backend to also launch (libero, libero_pro, robocasa, robotwin).",
    ),
    vla_port: Optional[int] = typer.Option(
        None, "--vla-port",
        help="Port for the VLA server (default: per-VLA canonical port).",
    ),
    sim_port: Optional[int] = typer.Option(
        None, "--sim-port",
        help="Port for the sim_worker (default: per-sim canonical port).",
    ),
    vla_venv: Optional[str] = typer.Option(
        None, "--vla-venv",
        help="Path to venv for the VLA server (default: .venvs/vla).",
    ),
    sim_venv: Optional[str] = typer.Option(
        None, "--sim-venv",
        help="Path to venv for the sim_worker.",
    ),
    model_id: Optional[str] = typer.Option(
        None, "--model-id",
        help="Model ID to pass to the VLA server.",
    ),
    headless: bool = typer.Option(
        True, "--headless/--no-headless",
        help="Run sim_worker in headless mode (default: True).",
    ),
    health_timeout: float = typer.Option(
        120.0, "--health-timeout",
        help="Seconds to wait for servers to become ready.",
    ),
    logs_dir: str = typer.Option(
        "logs", "--logs-dir",
        help="Directory for subprocess logs.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Launch VLA policy server and optionally a sim_worker.

    Starts processes in the background, polls /health, then waits
    for Ctrl+C.  Logs go to logs/<name>.log.

    Examples::

        roboeval serve --vla pi05
        roboeval serve --vla pi05 --sim libero --headless
        roboeval serve --vla smolvla --sim libero_object --vla-port 5101
    """
    _setup_logging(verbose)

    from roboeval.server_runner import install_signal_handlers, start_vla, start_sim, wait_for_exit

    try:
        resolved_vla_port, resolved_sim_port = _resolve_serve_ports(
            vla=vla,
            sim=sim,
            vla_port=vla_port,
            sim_port=sim_port,
        )
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Install signal handlers here (CLI entry point owns the process lifecycle).
    install_signal_handlers()

    procs = []

    # Start VLA server
    typer.echo(f"Starting VLA server: {vla}")
    try:
        vla_proc = start_vla(
            vla_name=vla,
            port=resolved_vla_port,
            venv_path=vla_venv,
            model_id=model_id,
            logs_dir=logs_dir,
            health_timeout=health_timeout,
        )
        procs.append(vla_proc)
        typer.echo(f"  VLA {vla!r} ready.")
    except (ValueError, RuntimeError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    # Start sim_worker when enabled
    if sim:
        typer.echo(f"Starting sim_worker: {sim}")
        try:
            sim_proc = start_sim(
                backend=sim,
                port=resolved_sim_port,
                venv_path=sim_venv,
                headless=headless,
                logs_dir=logs_dir,
                health_timeout=health_timeout,
            )
            procs.append(sim_proc)
            typer.echo(f"  sim_worker {sim!r} ready.")
        except RuntimeError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1)

    typer.echo("\nAll servers running.  Press Ctrl+C to stop.")
    wait_for_exit(procs)


# ---------------------------------------------------------------------------
# roboeval merge
# ---------------------------------------------------------------------------


@app.command("merge")
def cmd_merge(
    pattern: str = typer.Option(
        ...,
        "--pattern", "-p",
        help="Glob pattern for shard JSON files (e.g. 'results/*_shard*.json').",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help="Path for the merged output JSON.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Merge shard result JSON files into a single benchmark result.

    Reads all files matching PATTERN, merges episodes across shards (last-write-
    wins on duplicate episode_id), and writes to OUTPUT.

    Missing shards are allowed — result is marked partial=true.

    Examples::

        roboeval merge --pattern 'results/*_shard*.json' -o final.json
        roboeval merge -p 'results/run_*shard*.json' -o results/merged.json
    """
    _setup_logging(verbose)

    from roboeval.results.merge import (
        find_shard_files,
        load_shard_files,
        merge_shards,
        print_merge_report,
    )

    try:
        paths = find_shard_files(pattern)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(paths)} shard files.")
    for p in paths:
        typer.echo(f"  {p}")

    try:
        shards = load_shard_files(paths)
        merged = merge_shards(shards)
    except (ValueError, OSError) as e:
        typer.echo(f"Error merging shards: {e}", err=True)
        raise typer.Exit(1)

    import json

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, default=str))
    typer.echo(f"\nMerged result written to: {output_path}")

    try:
        print_merge_report(merged)
    except Exception:
        # Rich not available — just print plain summary
        rate = merged.get("mean_success", 0.0)
        total = merged.get("merge_info", {}).get("total_episodes", "?")
        typer.echo(f"Overall success: {rate:.1%} ({total} episodes)")

    # Exit non-zero if partial
    if merged.get("partial"):
        typer.echo("Warning: result is PARTIAL (some shards missing).", err=True)
        raise typer.Exit(2)


# ---------------------------------------------------------------------------
# roboeval test
# ---------------------------------------------------------------------------


@app.command("test")
def cmd_test(
    config: Optional[str] = typer.Option(
        None,
        "--config", "-c",
        help="Path to eval YAML config (required for --server, --benchmark, --all).",
    ),
    validate: bool = typer.Option(
        False, "--validate",
        help="Run fast validate-only checks (YAML, registry, ActionObsSpec).",
    ),
    server: bool = typer.Option(
        False, "--server",
        help="Probe each declared VLA server's /health + /info.",
    ),
    benchmark: bool = typer.Option(
        False, "--benchmark",
        help="Probe each declared sim's /health + /info, send /reset + zero /step.",
    ),
    all_checks: bool = typer.Option(
        False, "--all",
        help="Run all checks plus a 1-episode dry run.",
    ),
    results_dir: Optional[str] = typer.Option(
        None, "--results-dir",
        help="Directory for dry-run results (--all only).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run preflight checks to validate configuration and server health.

    Modes (combinable):
        --validate   YAML parses, registry resolves, ActionObsSpec round-trips (no network).
        --server     Probe each declared VLA's /health + /info.
        --benchmark  Probe each declared sim's /health + /info; /reset + zero /step.
        --all        All of the above + 1-episode orchestrator dry run.

    If no mode flag is given, --validate is used by default.

    Examples::

        roboeval test --validate --config configs/libero_spatial_pi05_smoke.yaml
        roboeval test --server --config configs/libero_spatial_pi05_smoke.yaml
        roboeval test --all --config configs/libero_spatial_pi05_smoke.yaml
    """
    _setup_logging(verbose)

    # Default: --validate
    if not any([validate, server, benchmark, all_checks]):
        validate = True

    # Config is required unless --validate-only
    if (server or benchmark or all_checks) and not config:
        typer.echo(
            "Error: --config is required for --server, --benchmark, and --all.",
            err=True,
        )
        raise typer.Exit(1)

    # Use a dummy config for --validate only when none supplied
    if validate and not config:
        # No config needed for pure ActionObsSpec / registry test
        _run_validate_no_config()
        return

    if config is None:
        typer.echo("Error: --config is required.", err=True)
        raise typer.Exit(1)

    from roboeval.preflight import run_preflight

    exit_code = run_preflight(
        config_path=config,
        validate=validate,
        server=server,
        benchmark=benchmark,
        all_checks=all_checks,
        results_dir=results_dir,
    )

    if exit_code != 0:
        raise typer.Exit(exit_code)


def _run_validate_no_config() -> None:
    """Run basic ActionObsSpec + registry smoke tests without a config file."""
    from roboeval.preflight import _check, print_results
    results = []

    # ActionObsSpec round-trip
    try:
        from roboeval.specs import POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS, ActionObsSpec
        for spec in [POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS]:
            d2 = spec.to_dict()
            spec2 = ActionObsSpec.from_dict(d2)
            assert spec2 == spec
        results.append(_check("specs.dimspec_roundtrip", True))
    except Exception as e:
        results.append(_check("specs.dimspec_roundtrip", False, str(e)))

    # Registry import
    try:
        from roboeval.registry import resolve_import_string
        cls = resolve_import_string("roboeval.specs:ActionObsSpec")
        assert cls.__name__ == "ActionObsSpec"
        results.append(_check("registry.import", True))
    except Exception as e:
        results.append(_check("registry.import", False, str(e)))

    # Orchestrator importable
    try:
        from roboeval.orchestrator import Orchestrator, EvalConfig
        results.append(_check("orchestrator.import", True))
    except Exception as e:
        results.append(_check("orchestrator.import", False, str(e)))

    # Results modules importable
    try:
        from roboeval.results import ResultCollector, merge_shards
        results.append(_check("results.import", True))
    except Exception as e:
        results.append(_check("results.import", False, str(e)))

    print("-- validate (no config) --")
    rc = print_results(results)
    failed = sum(1 for r in results if not r.ok)
    if failed == 0:
        print(f"All {len(results)} checks passed.")
    else:
        print(f"{failed}/{len(results)} checks failed.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``roboeval`` console script."""
    app()


if __name__ == "__main__":
    main()
