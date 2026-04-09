"""
robo-eval CLI: Unified benchmark evaluation tool.

Manages the ENTIRE stack from a single command:
  VLA servers -> round-robin proxy -> sim workers -> eval processes

Usage examples:
    # Fully self-contained: starts VLA, proxy, sims, and evals
    robo-eval run --benchmark libero --vla smolvla --suites spatial --episodes 50 --vla-replicas 4

    # Multi-GPU: each replica gets its own GPU
    robo-eval run --benchmark libero --vla pi05 --episodes 10 --vla-replicas 4 --gpus 0,1,2,3

    # Direct VLA evaluation (no VLM planner)
    robo-eval run --benchmark libero --vla pi05 --episodes 50 --mode direct

    # Full planner pipeline with VLM
    robo-eval run --benchmark libero --vla smolvla --suites spatial,object --episodes 10 --mode planner

    # Dry run: show what would happen
    robo-eval run --benchmark libero --vla smolvla --episodes 50 --vla-replicas 4 --dry-run

    # Check progress
    robo-eval status --results-dir results/my_run/

    # Server management
    robo-eval servers list
    robo-eval servers start smolvla
    robo-eval servers stop smolvla
"""

from __future__ import annotations

from contextlib import ExitStack
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer

from .config import (
    BENCHMARK_SUITES,
    DEFAULT_PROXY_PORT,
    DEFAULT_VLM_PORT,
    DEFAULT_VLM_MODEL,
    LIBERO_PRO_SUITES,
    LIBERO_SUITES,
    MODE_DESCRIPTIONS,
    SIM_CONFIGS,
    SUITE_PRESETS,
    TASKS_PER_SUITE,
    VLA_CONFIGS,
    estimate_ram_usage,
    find_available_port,
    find_available_port_block,
    get_sim_for_suite,
    get_suites_for_benchmark,
    is_port_available,
    qualify_suite,
    resolve_mode,
    validate_port,
)
from .port_allocator import PortAllocator

# ---------------------------------------------------------------------------
# App & sub-commands
# ---------------------------------------------------------------------------


def _version_callback(value: bool):
    if value:
        from . import __version__

        typer.echo(f"robo-eval {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="robo-eval",
    help="Unified CLI for VLA benchmark evaluation. Manages the full stack.",
    no_args_is_help=True,
)


@app.callback()
def _main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """Unified CLI for VLA benchmark evaluation. Manages the full stack."""


servers_app = typer.Typer(help="Manage VLA/VLM servers.")
app.add_typer(servers_app, name="servers")



def _get_suite_size(suite: str) -> int:
    sim_type = get_sim_for_suite(suite)
    if sim_type == "robocasa":
        from sims.sim_worker import RoboCasaBackend
        return len(RoboCasaBackend._TASK_INDEX)
    elif "libero" in sim_type or "libero_pro" in sim_type or "libero_infinity" in sim_type:
        return 10
    elif sim_type == "robotwin":
        return 10
    return 10


def _required_sim_workers(
    *,
    parallel: bool,
    tasks_parallel: int,
    num_tasks: int,
    suites_parallel: Optional[int],
    num_suites: int,
    debug_window: bool,
) -> int:
    """Compute the minimum sim workers needed for the requested concurrency."""
    if debug_window or not parallel:
        return 1

    workers_per_suite = max(1, min(tasks_parallel, num_tasks))
    concurrent_suites = 1
    if suites_parallel and suites_parallel > 1 and num_suites > 1:
        concurrent_suites = min(suites_parallel, num_suites)
    return workers_per_suite * concurrent_suites


def _resolve_run_ports(
    *,
    manage_stack: bool,
    use_vlm: bool,
    vla_cfg,
    vla_replicas: int,
    proxy_port: Optional[int],
    vla_base_port: Optional[int],
    sim_base_port: Optional[int],
    vlm_endpoint: Optional[str],
    actual_sim_workers: int,
) -> tuple[Optional[int], Optional[int], int, Optional[str]]:
    """Resolve concrete ports for a run, auto-picking free ones by default."""
    resolved_vla_base_port: Optional[int] = None
    resolved_proxy_port: Optional[int] = None

    if manage_stack:
        if vla_base_port is not None:
            resolved_vla_base_port = vla_base_port
        else:
            resolved_vla_base_port = find_available_port_block(
                vla_replicas,
                preferred_start=vla_cfg.port,
                search_start=vla_cfg.port,
            )

        if proxy_port is not None:
            resolved_proxy_port = proxy_port
        else:
            resolved_proxy_port = find_available_port(
                DEFAULT_PROXY_PORT,
                search_start=DEFAULT_PROXY_PORT,
            )

    if sim_base_port is not None:
        resolved_sim_base_port = sim_base_port
    else:
        resolved_sim_base_port = find_available_port_block(
            actual_sim_workers,
            preferred_start=5300,
            search_start=5300,
        )

    resolved_vlm_endpoint = vlm_endpoint
    if use_vlm and not vlm_endpoint:
        resolved_vlm_port = find_available_port(
            DEFAULT_VLM_PORT,
            search_start=DEFAULT_VLM_PORT,
        )
        resolved_vlm_endpoint = f"localhost:{resolved_vlm_port}"

    return (
        resolved_proxy_port,
        resolved_vla_base_port,
        resolved_sim_base_port,
        resolved_vlm_endpoint,
    )


def _ensure_requested_ports_available(
    *,
    manage_stack: bool,
    proxy_port: Optional[int],
    vla_base_port: Optional[int],
    vla_replicas: int,
    sim_base_port: int,
    actual_sim_workers: int,
) -> None:
    """Fail fast if explicitly chosen ports are already occupied."""
    occupied: List[str] = []

    if manage_stack and proxy_port is not None and not is_port_available(proxy_port):
        occupied.append(f"proxy port {proxy_port}")

    if manage_stack and vla_base_port is not None:
        for port in range(vla_base_port, vla_base_port + vla_replicas):
            if not is_port_available(port):
                occupied.append(f"VLA port {port}")

    for port in range(sim_base_port, sim_base_port + actual_sim_workers):
        if not is_port_available(port):
            occupied.append(f"sim worker port {port}")

    if occupied:
        ports_str = ", ".join(occupied)
        raise typer.BadParameter(
            f"Requested ports are already in use: {ports_str}. "
            "Omit the port override to let robo-eval auto-select free ports."
        )


def _reserve_run_ports(
    *,
    allocator: PortAllocator,
    manage_stack: bool,
    use_vlm: bool,
    vla_cfg,
    vla_replicas: int,
    proxy_port: Optional[int],
    vla_base_port: Optional[int],
    sim_base_port: Optional[int],
    vlm_endpoint: Optional[str],
    actual_sim_workers: int,
) -> tuple[Optional[int], Optional[int], int, Optional[str], list]:
    """Reserve concrete ports for an actual run and return active leases."""
    reservations = []
    resolved_vla_base_port: Optional[int] = None
    resolved_proxy_port: Optional[int] = None

    if manage_stack:
        vla_res = allocator.reserve_block(
            kind="vla",
            count=vla_replicas,
            preferred_start=vla_base_port or vla_cfg.port,
            search_start=vla_cfg.port,
            exact=vla_base_port is not None,
        )
        reservations.append(vla_res)
        resolved_vla_base_port = vla_res.ports[0]

        proxy_res = allocator.reserve_port(
            kind="proxy",
            preferred_port=proxy_port or DEFAULT_PROXY_PORT,
            search_start=DEFAULT_PROXY_PORT,
            exact=proxy_port is not None,
        )
        reservations.append(proxy_res)
        resolved_proxy_port = proxy_res.ports[0]

    sim_res = allocator.reserve_block(
        kind="sim",
        count=actual_sim_workers,
        preferred_start=sim_base_port or 5300,
        search_start=5300,
        exact=sim_base_port is not None,
    )
    reservations.append(sim_res)
    resolved_sim_base_port = sim_res.ports[0]

    resolved_vlm_endpoint = vlm_endpoint
    if use_vlm and not vlm_endpoint:
        vlm_res = allocator.reserve_port(
            kind="vlm",
            preferred_port=DEFAULT_VLM_PORT,
            search_start=DEFAULT_VLM_PORT,
        )
        reservations.append(vlm_res)
        resolved_vlm_endpoint = f"localhost:{vlm_res.ports[0]}"

    return (
        resolved_proxy_port,
        resolved_vla_base_port,
        resolved_sim_base_port,
        resolved_vlm_endpoint,
        reservations,
    )


# ---------------------------------------------------------------------------
# robo-eval run  (full-stack orchestration)
# ---------------------------------------------------------------------------
@app.command("run")
def run_cmd(
    # ── Model & benchmark ──
    benchmark: str = typer.Option(
        ..., "--benchmark", "-b",
        help=(
            "Benchmark to evaluate. "
            f"Valid values: {', '.join(sorted(BENCHMARK_SUITES))}."
        ),
    ),
    vla: str = typer.Option(
        ..., "--vla", "-v",
        help="VLA model to evaluate: pi05, smolvla, openvla",
    ),
    suites: Optional[str] = typer.Option(
        None, "--suites", "-s",
        help=(
            "Comma-separated SHORT suite names within the benchmark. "
            "If omitted, all suites for the benchmark are run. "
            "Example: --benchmark libero --suites spatial,goal"
        ),
    ),
    episodes: int = typer.Option(
        10, "--episodes", "-e",
        help="Number of episodes per task.",
    ),
    mode: str = typer.Option(
        "direct", "--mode", "-m",
        help=(
            "Evaluation mode: "
            "direct (VLA only, no VLM), "
            "planner (full planner pipeline + VLM), "
            "native (lerobot-eval baseline). "
            "Aliases: no-vlm=direct, vlm=planner."
        ),
    ),

    # ── Scaling ──
    vla_replicas: int = typer.Option(
        1, "--vla-replicas",
        help="Number of VLA server replicas. Traffic always routes through the proxy.",
    ),
    gpus: Optional[str] = typer.Option(
        None, "--gpus",
        help=(
            "Comma-separated GPU IDs for VLA replicas (CUDA_VISIBLE_DEVICES). "
            "Example: --gpus 0,1,2,3. Wraps if fewer GPUs than replicas."
        ),
    ),
    # ── Output ──
    results_dir: Optional[str] = typer.Option(
        None, "--results-dir", "-o",
        help="Output directory. Auto-generated if not specified.",
    ),

    # ── Task selection ──
    max_tasks: Optional[int] = typer.Option(
        None, "--max-tasks",
        help="Max number of tasks per suite (default: all 10). Useful for quick tests.",
    ),

    # ── Parallelization ──
    parallel: bool = typer.Option(
        True, "--parallel/--sequential",
        help="Run tasks in parallel (default) or sequentially.",
    ),
    tasks_parallel: int = typer.Option(
        10, "--tasks-parallel",
        help="Number of tasks to run in parallel per suite.",
    ),
    suites_parallel: Optional[int] = typer.Option(
        None, "--suites-parallel",
        help="Number of suites to run in parallel. Default: all of them.",
    ),

    # ── Ports ──
    proxy_port: Optional[int] = typer.Option(
        None, "--proxy-port",
        help="Override port for the VLA round-robin proxy. Default: auto-select a free port.",
    ),
    vla_base_port: Optional[int] = typer.Option(
        None, "--vla-base-port",
        help="Override base port for VLA replicas. Default: auto-select a free block near the model's usual port.",
    ),
    sim_base_port: Optional[int] = typer.Option(
        None, "--sim-base-port",
        help="Override base port for sim workers. Default: auto-select a free block.",
    ),

    # ── VLM options ──
    vlm_model: str = typer.Option(
        DEFAULT_VLM_MODEL, "--vlm-model",
        help="VLM model for planner mode evaluation.",
    ),
    vlm_endpoint: Optional[str] = typer.Option(
        None, "--vlm-endpoint",
        help="VLM proxy endpoint (host:port). Default: auto-select a free localhost port when auto-starting.",
    ),

    # ── Flags ──
    no_think: bool = typer.Option(
        False, "--no-think",
        help="Disable VLM thinking tokens.",
    ),
    delta_actions: bool = typer.Option(
        True, "--delta/--no-delta",
        help="Use delta actions (default: enabled).",
    ),
    resume: bool = typer.Option(
        False, "--resume",
        help="Resume: skip tasks with completed logs.",
    ),
    debug_window: bool = typer.Option(
        False, "--debug-window",
        help=(
            "Open a live GLFW MuJoCo window for interactive debugging. "
            "Supported only for one suite, sequential execution, and one sim worker."
        ),
    ),
    manage_stack: bool = typer.Option(
        True, "--manage-stack/--no-manage-stack",
        help=(
            "Auto-manage VLA servers and proxy (default). "
            "Use --no-manage-stack to use externally managed servers."
        ),
    ),
    vla_url: Optional[str] = typer.Option(
        None, "--vla-url",
        help=(
            "Override VLA/proxy URL (bypasses auto-management). "
            "Implies --no-manage-stack for VLA servers."
        ),
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed",
        help="Random seed for reproducibility. Auto-generated if not specified.",
    ),
    record_video: bool = typer.Option(
        False, "--record-video",
        help="Record episode videos from simulator.",
    ),
    record_video_n: int = typer.Option(
        3, "--record-video-n",
        help="Max episodes per task to record (default 3).",
    ),
    sim_config: Optional[str] = typer.Option(
        None, "--sim-config",
        help=(
            "Path to a YAML file with simulator-specific config, forwarded to the eval backend. "
            "Mutually exclusive with --sim-args."
        ),
    ),
    sim_args: Optional[str] = typer.Option(
        None, "--sim-args",
        help=(
            "JSON of simulator-specific args forwarded opaquely to the backend. "
            "Serialized to a YAML file in results_dir and passed as --sim-config. "
            "Mutually exclusive with --sim-config. "
            'Example: --sim-args \'{"perturbation": "combined", "max_distractors": 3}\''
        ),
    ),
    task_timeout: int = typer.Option(
        21600, "--task-timeout",
        help="Per-task timeout in seconds (default: 21600 = 6 hours).",
    ),
    runtime: str = typer.Option(
        "venv", "--runtime",
        help=(
            "Runtime backend: 'venv' (local virtualenvs, default), 'docker' (containers), "
            "'auto' (docker if available, else venv)."
        ),
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would be done without running.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose",
        help="Enable verbose logging.",
    ),
):
    """Run benchmark evaluation.

    Manages the ENTIRE stack: VLA servers, round-robin proxy, sim workers,
    and eval processes. On completion or Ctrl+C, tears down everything in
    reverse order.

    The proxy is ALWAYS used, even with 1 VLA replica. This ensures
    consistent behavior and makes scaling trivial.
    """
    # Configure logging
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(message)s",
            datefmt="%H:%M:%S",
        )

    # ── Validation ──
    if vla not in VLA_CONFIGS:
        typer.echo(f"Error: Unknown VLA '{vla}'. Available: {list(VLA_CONFIGS.keys())}")
        raise typer.Exit(1)

    # Validate benchmark
    valid_benchmarks = sorted(BENCHMARK_SUITES.keys())
    if benchmark not in BENCHMARK_SUITES:
        typer.echo(f"Error: Unknown benchmark '{benchmark}'. Valid: {', '.join(valid_benchmarks)}")
        raise typer.Exit(1)

    # Validate --sim-config / --sim-args mutual exclusion
    if sim_config and sim_args:
        typer.echo("Error: --sim-config and --sim-args are mutually exclusive.")
        raise typer.Exit(1)

    if sim_config and not Path(sim_config).exists():
        typer.echo(f"Error: --sim-config file not found: {sim_config}")
        raise typer.Exit(1)

    if sim_args:
        import json as _json
        try:
            _json.loads(sim_args)
        except _json.JSONDecodeError as e:
            typer.echo(f"Error: --sim-args is not valid JSON: {e}")
            raise typer.Exit(1)

    try:
        mode = resolve_mode(mode)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)

    # ── Runtime resolution ──
    # Explicit match with all 5 cases (FM verified: runtime selection is total).
    from .docker_backend import is_docker_available
    if runtime == "auto" and is_docker_available():
        resolved_runtime = "docker"
        typer.echo("[runtime] Docker available — using container backend.")
    elif runtime == "auto" and not is_docker_available():
        resolved_runtime = "venv"
        typer.echo("[runtime] Docker not available — using venv backend.")
    elif runtime == "docker" and is_docker_available():
        resolved_runtime = "docker"
        typer.echo("[runtime] Using Docker container backend.")
    elif runtime == "docker" and not is_docker_available():
        typer.echo(
            "Error: --runtime docker requested but Docker is not available. "
            "Install Docker and ensure the daemon is running."
        )
        raise typer.Exit(1)
    elif runtime == "venv":
        resolved_runtime = "venv"
        typer.echo("[runtime] Using local venv backend.")
    else:
        typer.echo(f"Error: Unknown --runtime value '{runtime}'. Valid: auto, docker, venv.")
        raise typer.Exit(1)

    # Validate ports
    try:
        if proxy_port is not None:
            validate_port(proxy_port, "--proxy-port")
        if sim_base_port is not None:
            validate_port(sim_base_port, "--sim-base-port")
        if vla_base_port is not None:
            validate_port(vla_base_port, "--vla-base-port")
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)

    # Resolve suites from benchmark + short names
    if suites:
        short_suites = [s.strip() for s in suites.split(",") if s.strip()]
        valid_short = get_suites_for_benchmark(benchmark)
        for s in short_suites:
            if s not in valid_short:
                typer.echo(
                    f"Error: Unknown suite '{s}' for benchmark '{benchmark}'. "
                    f"Valid: {', '.join(valid_short)}"
                )
                raise typer.Exit(1)
        suite_list = [qualify_suite(benchmark, s) for s in short_suites]
    else:
        suite_list = [qualify_suite(benchmark, s) for s in get_suites_for_benchmark(benchmark)]

    if not suite_list:
        typer.echo(f"Error: No suites resolved for benchmark '{benchmark}'.")
        raise typer.Exit(1)

    # ── Settings ──
    no_vlm = mode == "direct"
    use_vlm = mode == "planner"
    is_native = mode == "native"
    vla_cfg = VLA_CONFIGS[vla]

    # If --vla-url is given, user is managing VLA servers externally
    if vla_url:
        manage_stack = False

    # Determine sim type (all suites in a run should use the same sim)
    sim_type = get_sim_for_suite(suite_list[0])

    # Resolve max tasks per suite
    actual_max_tasks = min(max_tasks, TASKS_PER_SUITE) if max_tasks is not None else TASKS_PER_SUITE
    
    # We use actual_max_tasks to estimate workers, but the actual tasks run will be 
    # clamped per-suite in the runner. We take the max size across suites to be safe for worker estimation.
    max_suite_size = max(_get_suite_size(s) for s in suite_list)
    resolved_max_tasks = min(actual_max_tasks, max_suite_size)

    required_sim_workers = _required_sim_workers(
        parallel=parallel,
        tasks_parallel=tasks_parallel,
        num_tasks=resolved_max_tasks,
        suites_parallel=suites_parallel,
        num_suites=len(suite_list),
        debug_window=debug_window,
    )
    actual_sim_workers = required_sim_workers
    headless = not debug_window

    if debug_window:
        if len(suite_list) != 1:
            typer.echo(
                "Error: --debug-window supports exactly one suite. "
                "Use headless mode for multi-suite runs."
            )
            raise typer.Exit(1)
        if parallel:
            typer.echo(
                "Error: --debug-window requires --sequential. "
                "Use headless mode for parallel runs."
            )
            raise typer.Exit(1)
        if tasks_parallel != 1:
            typer.echo(
                "Error: --debug-window requires --tasks-parallel 1. "
                "Only one active rollout window is supported."
            )
            raise typer.Exit(1)
        if suites_parallel not in (None, 1):
            typer.echo(
                "Error: --debug-window does not support suite-level parallelism. "
                "Use headless mode for multi-suite concurrency."
            )
            raise typer.Exit(1)

    (
        resolved_proxy_port,
        resolved_vla_base_port,
        resolved_sim_base_port,
        resolved_vlm_endpoint,
    ) = _resolve_run_ports(
        manage_stack=manage_stack,
        use_vlm=use_vlm,
        vla_cfg=vla_cfg,
        vla_replicas=vla_replicas,
        proxy_port=proxy_port,
        vla_base_port=vla_base_port,
        sim_base_port=sim_base_port,
        vlm_endpoint=vlm_endpoint,
        actual_sim_workers=actual_sim_workers,
    )

    try:
        _ensure_requested_ports_available(
            manage_stack=manage_stack,
            proxy_port=proxy_port,
            vla_base_port=vla_base_port,
            vla_replicas=vla_replicas,
            sim_base_port=resolved_sim_base_port,
            actual_sim_workers=actual_sim_workers,
        )
    except typer.BadParameter as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)

    # Auto-generate results dir if not specified
    if not results_dir:
        suite_label = (suites or "all").replace(",", "_")
        results_dir = f"results/{vla}_{episodes}eps_{mode}_{benchmark}_{suite_label}"
    results_path = Path(results_dir)

    # Resolve sim_config_path from --sim-config or --sim-args
    sim_config_path: Optional[Path] = None
    if sim_config:
        sim_config_path = Path(sim_config)
    elif sim_args:
        import json as _json_2
        import yaml as _yaml
        args_dict = _json_2.loads(sim_args)
        sim_config_path = results_path / "sim_config.yaml"  # path always computed
        if not dry_run:
            results_path.mkdir(parents=True, exist_ok=True)
            sim_config_path.write_text(_yaml.safe_dump(args_dict, default_flow_style=False))

    # ── Dry Run ──
    if dry_run:
        _print_dry_run(
            vla=vla,
            vla_cfg=vla_cfg,
            mode=mode,
            suite_list=suite_list,
            episodes=episodes,
            vla_replicas=vla_replicas,
            gpus=gpus,
            actual_sim_workers=actual_sim_workers,
            proxy_port=resolved_proxy_port,
            sim_base_port=resolved_sim_base_port,
            vla_base_port=resolved_vla_base_port,
            results_dir=results_dir,
            manage_stack=manage_stack,
            vla_url=vla_url,
            parallel=parallel,
            tasks_parallel=tasks_parallel,
            suites_parallel=suites_parallel,
            use_vlm=use_vlm,
            vlm_model=vlm_model,
            vlm_endpoint=resolved_vlm_endpoint,
            delta_actions=delta_actions,
            debug_window=debug_window,
            sim_config_path=sim_config_path,
            runtime=resolved_runtime,
        )
        return

    allocator = PortAllocator()
    lease_stack = ExitStack()
    try:
        (
            resolved_proxy_port,
            resolved_vla_base_port,
            resolved_sim_base_port,
            resolved_vlm_endpoint,
            reservations,
        ) = _reserve_run_ports(
            allocator=allocator,
            manage_stack=manage_stack,
            use_vlm=use_vlm,
            vla_cfg=vla_cfg,
            vla_replicas=vla_replicas,
            proxy_port=proxy_port,
            vla_base_port=vla_base_port,
            sim_base_port=sim_base_port,
            vlm_endpoint=vlm_endpoint,
            actual_sim_workers=actual_sim_workers,
        )
        for reservation in reservations:
            lease_stack.callback(reservation.release)
    except RuntimeError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(1)

    # ── Native mode: no stack management needed ──
    try:
        if is_native:
            from .config import NATIVE_EVAL_CONFIGS
            if vla not in NATIVE_EVAL_CONFIGS:
                typer.echo(f"Error: No internal baseline config for '{vla}'.")
                raise typer.Exit(1)

            pro_suites = [s for s in suite_list if s in LIBERO_PRO_SUITES]
            if pro_suites:
                typer.echo(f"Warning: Native eval does not support PRO suites: {pro_suites}")
                suite_list = [s for s in suite_list if s not in LIBERO_PRO_SUITES]
                if not suite_list:
                    typer.echo("Error: No valid suites remain after filtering PRO suites.")
                    raise typer.Exit(1)

            from .runner import run_native_eval
            run_native_eval(vla, suite_list, results_path, episodes, task_timeout=task_timeout)
            return

        # ── Full-stack orchestration (direct / planner mode) ──

        if manage_stack:
            _run_with_managed_stack(
                vla=vla,
                vla_cfg=vla_cfg,
                vla_replicas=vla_replicas,
                vla_base_port=resolved_vla_base_port,
                gpus=gpus,
                proxy_port=resolved_proxy_port,
                sim_type=sim_type,
                sim_base_port=resolved_sim_base_port,
                actual_sim_workers=actual_sim_workers,
                suite_list=suite_list,
                results_path=results_path,
                episodes=episodes,
                no_vlm=no_vlm,
                no_think=no_think,
                use_vlm=use_vlm,
                vlm_model=vlm_model,
                vlm_endpoint=resolved_vlm_endpoint,
                delta_actions=delta_actions,
                resume=resume,
                parallel=parallel,
                tasks_parallel=tasks_parallel,
                suites_parallel=suites_parallel,
                seed=seed,
                record_video=record_video,
                record_video_n=record_video_n,
                num_tasks=resolved_max_tasks,
                headless=headless,
                sim_config_path=sim_config_path,
                task_timeout=task_timeout,
                runtime=resolved_runtime,
            )
        else:
            _run_with_external_servers(
                vla=vla,
                vla_cfg=vla_cfg,
                vla_url=vla_url,
                proxy_port=resolved_proxy_port,
                sim_base_port=resolved_sim_base_port,
                suite_list=suite_list,
                results_path=results_path,
                episodes=episodes,
                no_vlm=no_vlm,
                no_think=no_think,
                use_vlm=use_vlm,
                vlm_model=vlm_model,
                vlm_endpoint=resolved_vlm_endpoint,
                delta_actions=delta_actions,
                resume=resume,
                parallel=parallel,
                tasks_parallel=tasks_parallel,
                suites_parallel=suites_parallel,
                seed=seed,
                record_video=record_video,
                record_video_n=record_video_n,
                num_tasks=resolved_max_tasks,
                headless=headless,
                sim_config_path=sim_config_path,
                task_timeout=task_timeout,
                runtime=resolved_runtime,
            )
    finally:
        lease_stack.close()


def _run_with_managed_stack(
    vla: str,
    vla_cfg,
    vla_replicas: int,
    vla_base_port: Optional[int],
    gpus: Optional[str],
    proxy_port: int,
    sim_type: str,
    sim_base_port: int,
    actual_sim_workers: int,
    suite_list: List[str],
    results_path: Path,
    episodes: int,
    no_vlm: bool,
    no_think: bool,
    use_vlm: bool,
    vlm_model: str,
    vlm_endpoint: Optional[str],
    delta_actions: bool,
    resume: bool,
    parallel: bool,
    tasks_parallel: int,
    suites_parallel: Optional[int],
    seed: Optional[int] = None,
    record_video: bool = False,
    record_video_n: int = 3,
    num_tasks: int = 10,
    headless: bool = True,
    sim_config_path: Optional[Path] = None,
    task_timeout: int = 21600,
    runtime: str = "venv",
):
    """Run evaluation with full stack management.

    The StackManager handles:
    1. Starting VLA replicas + health checks
    2. Starting the round-robin proxy
    3. Starting sim workers
    4. Graceful teardown on completion or Ctrl+C
    """
    from .stack import StackManager

    logs_dir = results_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Clean up stale Docker containers from crashed runs before startup
    if runtime == "docker":
        from .docker_backend import docker_cleanup_stale
        docker_cleanup_stale()

    stack = StackManager(
        vla_name=vla,
        vla_replicas=vla_replicas,
        vla_base_port=vla_base_port,
        gpus=gpus,
        proxy_port=proxy_port,
        sim_type=sim_type,
        sim_base_port=sim_base_port,
        num_sim_workers=actual_sim_workers,
        sim_headless=headless,
        logs_dir=logs_dir,
        runtime=runtime,
    )

    try:
        # Start the full stack: VLA -> proxy -> sim workers
        stack.start()

        # Auto-start VLM proxy for planner mode (tracked for cleanup)
        if use_vlm:
            from .servers import check_health as _check_health
            vlm_port = 4000
            if vlm_endpoint:
                try:
                    vlm_port = int(vlm_endpoint.split(":")[-1])
                except ValueError:
                    pass
            vlm_healthy, _ = _check_health(f"http://localhost:{vlm_port}")
            if not vlm_healthy:
                typer.echo("VLM proxy not running. Starting...")
                stack.start_vlm_proxy(vlm_port, vlm_model)

        # The proxy URL is the single entry point for all eval processes
        proxy_url = stack.proxy_url

        typer.echo(f"\n[run] All infrastructure ready. Starting evaluation...")
        typer.echo(f"[run] Proxy URL: {proxy_url}")
        typer.echo(f"[run] Results: {results_path}")
        typer.echo()

        # Run evaluation through the proxy
        from .runner import run_eval_parallel, run_eval_sequential

        kwargs = dict(
            suites=suite_list,
            vla_url=proxy_url,
            vla_urls=[proxy_url],  # Single proxy URL for all processes
            results_dir=results_path,
            max_episodes=episodes,
            no_vlm=no_vlm,
            no_think=no_think or no_vlm,
            vla_name=vla,  # For per-suite model reloading (OpenVLA)
            vlm_model=vlm_model if use_vlm else None,
            vlm_endpoint=vlm_endpoint if use_vlm else None,
            delta_actions=delta_actions,
            resume=resume,
            suites_parallel=suites_parallel,
            managed_sim_pool=stack.sim_pool,  # Pass managed pool to runner
            seed=seed,
            record_video=record_video,
            record_video_n=record_video_n,
            num_tasks=num_tasks,
            max_workers=tasks_parallel,
            headless=headless,
            sim_config_path=sim_config_path,
            task_timeout=task_timeout,
        )

        if parallel:
            kwargs["sim_base_port"] = sim_base_port
            run_eval_parallel(**kwargs)
        else:
            kwargs["sim_port"] = sim_base_port
            run_eval_sequential(**kwargs)

    finally:
        # Graceful teardown: evals done -> sim workers -> proxy -> VLA servers
        typer.echo("\n[stack] Shutting down...")
        stack.stop()
        typer.echo("[stack] All services stopped.")


def _run_with_external_servers(
    vla: str,
    vla_cfg,
    vla_url: Optional[str],
    proxy_port: int,
    sim_base_port: int,
    suite_list: List[str],
    results_path: Path,
    episodes: int,
    no_vlm: bool,
    no_think: bool,
    use_vlm: bool,
    vlm_model: str,
    vlm_endpoint: Optional[str],
    delta_actions: bool,
    resume: bool,
    parallel: bool,
    tasks_parallel: int,
    suites_parallel: Optional[int],
    seed: Optional[int] = None,
    record_video: bool = False,
    record_video_n: int = 3,
    num_tasks: int = 10,
    headless: bool = True,
    sim_config_path: Optional[Path] = None,
    task_timeout: int = 21600,
    runtime: str = "venv",
):
    """Run evaluation with externally managed VLA servers.

    Used when --vla-url is specified or --no-manage-stack is set.
    Sim workers are still managed by the runner.
    """
    from .servers import check_health, start_vlm_proxy, stop_vlm_proxy

    actual_vla_url = vla_url or vla_cfg.url

    # Verify VLA is reachable
    healthy, _ = check_health(actual_vla_url)
    if not healthy:
        typer.echo(f"Error: VLA not reachable at {actual_vla_url}")
        typer.echo("Start it manually or use --manage-stack (default).")
        raise typer.Exit(1)

    # Track whether we started the VLM proxy so we can clean it up in the finally block.
    vlm_proxy_started = False
    vlm_port = 4000

    # Auto-start VLM proxy for planner mode
    if use_vlm:
        if vlm_endpoint:
            try:
                vlm_port = int(vlm_endpoint.split(":")[-1])
            except ValueError:
                pass
        vlm_healthy, _ = check_health(f"http://localhost:{vlm_port}")
        if not vlm_healthy:
            typer.echo("VLM proxy not running. Starting...")
            start_vlm_proxy(vlm_port, vlm_model)
            vlm_proxy_started = True

    from .runner import run_eval_parallel, run_eval_sequential

    kwargs = dict(
        suites=suite_list,
        vla_url=actual_vla_url,
        vla_urls=[actual_vla_url],
        results_dir=results_path,
        max_episodes=episodes,
        no_vlm=no_vlm,
        no_think=no_think or no_vlm,
        vla_name=vla,  # For per-suite model reloading (OpenVLA)
        vlm_model=vlm_model if use_vlm else None,
        vlm_endpoint=vlm_endpoint if use_vlm else None,
        delta_actions=delta_actions,
        resume=resume,
        suites_parallel=suites_parallel,
        seed=seed,
        record_video=record_video,
        record_video_n=record_video_n,
        num_tasks=num_tasks,
        max_workers=tasks_parallel,
        headless=headless,
        sim_config_path=sim_config_path,
        task_timeout=task_timeout,
    )

    try:
        if parallel:
            kwargs["sim_base_port"] = sim_base_port
            run_eval_parallel(**kwargs)
        else:
            kwargs["sim_port"] = sim_base_port
            run_eval_sequential(**kwargs)
    finally:
        # Clean up VLM proxy if we started it.
        if vlm_proxy_started:
            typer.echo("[cleanup] Stopping VLM proxy...")
            stop_vlm_proxy(vlm_port)


def _print_dry_run(
    vla: str,
    vla_cfg,
    mode: str,
    suite_list: List[str],
    episodes: int,
    vla_replicas: int,
    gpus: Optional[str],
    actual_sim_workers: int,
    proxy_port: Optional[int],
    sim_base_port: int,
    vla_base_port: Optional[int],
    results_dir: str,
    manage_stack: bool,
    vla_url: Optional[str],
    parallel: bool,
    tasks_parallel: int,
    suites_parallel: Optional[int],
    use_vlm: bool,
    vlm_model: str,
    vlm_endpoint: Optional[str],
    delta_actions: bool,
    debug_window: bool,
    sim_config_path: Optional[Path] = None,
    runtime: str = "venv",
):
    """Print dry-run summary showing what would happen."""
    actual_suites_parallel = (
        min(suites_parallel, len(suite_list))
        if suites_parallel and suites_parallel > 1 and parallel
        else 1
    )

    typer.echo("=== DRY RUN ===")
    typer.echo()

    # Runtime
    typer.echo(f"Runtime:         {runtime}")
    typer.echo()

    # Stack management
    typer.echo("--- Stack ---")
    if manage_stack:
        base_port = vla_base_port or vla_cfg.port
        assert proxy_port is not None
        typer.echo(f"VLA replicas:    {vla_replicas}x {vla}")
        typer.echo(f"  Ports:         {base_port}-{base_port + vla_replicas - 1}")
        if gpus:
            typer.echo(f"  GPUs:          {gpus}")
        typer.echo(f"Proxy:           port {proxy_port} -> {vla_replicas} backend(s)")
        typer.echo(f"  ALL traffic routes through proxy (architectural invariant)")
    else:
        typer.echo(f"VLA URL:         {vla_url or vla_cfg.url} (external)")
        typer.echo(f"Stack:           NOT managed (--no-manage-stack)")

    typer.echo(f"Sim workers:     {actual_sim_workers} on ports {sim_base_port}-{sim_base_port + actual_sim_workers - 1}")
    typer.echo()

    # Evaluation
    typer.echo("--- Evaluation ---")
    typer.echo(f"VLA:             {vla}")
    typer.echo(f"Mode:            {mode} — {MODE_DESCRIPTIONS[mode]}")
    typer.echo(f"Suites:          {', '.join(suite_list)}")
    typer.echo(f"Episodes/task:   {episodes}")
    typer.echo(f"Parallel:        {parallel}")
    typer.echo(f"Tasks/suite:     {tasks_parallel}")
    typer.echo(f"Results dir:     {results_dir}")
    typer.echo(f"Rendering:       {'windowed GLFW debug' if debug_window else 'headless EGL'}")
    typer.echo(f"Delta actions:   {delta_actions}")
    if sim_config_path is not None:
        typer.echo(f"Sim config:      {sim_config_path}")
    if use_vlm:
        typer.echo(f"VLM model:       {vlm_model}")
        typer.echo(f"VLM endpoint:    {vlm_endpoint or 'localhost:4000'}")
    typer.echo()

    # Startup sequence
    typer.echo("--- Startup Sequence ---")
    if manage_stack:
        typer.echo(f"1. Start {vla_replicas}x {vla} servers (wait for health)")
        typer.echo(f"2. Start proxy on port {proxy_port}")
        typer.echo(f"3. Start {actual_sim_workers} sim workers")
        typer.echo(f"4. Launch eval processes")
        typer.echo()
        typer.echo("--- Shutdown Sequence (on completion or Ctrl+C) ---")
        typer.echo(f"1. Wait for evals to finish")
        typer.echo(f"2. Kill sim workers")
        typer.echo(f"3. Kill proxy")
        typer.echo(f"4. Kill VLA servers")
    else:
        typer.echo(f"1. Verify VLA at {vla_url or vla_cfg.url}")
        typer.echo(f"2. Start sim workers")
        typer.echo(f"3. Launch eval processes")
    typer.echo()

    # Resource estimate
    ram = estimate_ram_usage(
        vla_name=vla,
        num_vla_servers=vla_replicas if manage_stack else 0,
        num_sim_workers=actual_sim_workers,
        use_vlm=use_vlm,
    )
    typer.echo("--- Resource Estimate ---")
    if manage_stack:
        typer.echo(f"VLA servers:     {ram['vla_servers']:.1f} GB  ({vla_replicas}x {vla})")
    typer.echo(f"Sim workers:     {ram['sim_workers']:.1f} GB  (peak concurrent)")
    if use_vlm:
        typer.echo(f"VLM proxy:       {ram['vlm_proxy']:.1f} GB")
    typer.echo(f"Eval processes:  {ram['eval_processes']:.1f} GB")
    typer.echo(f"Total estimated: {ram['total']:.1f} GB")


# ---------------------------------------------------------------------------
# robo-eval status
# ---------------------------------------------------------------------------
@app.command("status")
def status_cmd(
    results_dir: str = typer.Option(
        ..., "--results-dir", "-o",
        help="Path to results directory to check.",
    ),
):
    """Show progress of a benchmark run."""
    from .results import print_status

    results_path = Path(results_dir)
    if not results_path.exists():
        typer.echo(f"Error: Results directory not found: {results_dir}")
        raise typer.Exit(1)

    typer.echo(f"Status for: {results_dir}")
    print_status(results_path)


# ---------------------------------------------------------------------------
# robo-eval servers list
# ---------------------------------------------------------------------------
@servers_app.command("list")
def servers_list():
    """List all known servers and their status."""
    from .servers import list_servers

    servers = list_servers()

    typer.echo("Server Status:")
    typer.echo(f"{'Type':<6} {'Name':<16} {'Port':<6} {'Status':<10} {'Model'}")
    typer.echo("-" * 70)
    for s in servers:
        status = "UP" if s["healthy"] else "DOWN"
        status_padded = f"{status:<10}"
        status_color = typer.style(status_padded, fg=typer.colors.GREEN if s["healthy"] else typer.colors.RED)
        typer.echo(f"{s['type']:<6} {s['name']:<16} {s['port']:<6} {status_color} {s['model']}")


# ---------------------------------------------------------------------------
# robo-eval servers start
# ---------------------------------------------------------------------------
@servers_app.command("start")
def servers_start(
    name: str = typer.Argument(
        ...,
        help=(
            "Server to start.  Valid values: "
            "vlm, sim, vla, pi05, smolvla, openvla, proxy.  "
            "Use 'vla <name>' as an alias for direct VLA names.  "
            "Use 'sim' together with --sim to specify the backend."
        ),
    ),
    target: Optional[str] = typer.Argument(
        None,
        help=(
            "Secondary target — only used with 'vla': the VLA name "
            "(pi05, openvla, smolvla).  "
            "Example: robo-eval servers start vla pi05"
        ),
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p",
        help=(
            "Override port number.  "
            "Env-var defaults: LITELLM_PORT (vlm), VLA_PORT (vla), SIM_PORT (sim)."
        ),
    ),
    model: Optional[str] = typer.Option(
        None, "--model",
        help=(
            "Model override.  "
            "For vlm: LiteLLM model string (default: DEFAULT_VLM_MODEL).  "
            "For vla/pi05/openvla/smolvla: HuggingFace model ID."
        ),
    ),
    sim_backend: Optional[str] = typer.Option(
        None, "--sim",
        help=(
            "Simulator backend — required when name='sim'.  "
            "Valid: libero, libero_pro, robocasa, robotwin."
        ),
    ),
    host: str = typer.Option(
        "0.0.0.0", "--host",
        help="Interface to bind to (sim only, default: 0.0.0.0).",
    ),
    headless: bool = typer.Option(
        True, "--headless/--no-headless",
        help="Use EGL headless rendering (sim only, default: headless).",
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Don't wait for server to become healthy.",
    ),
    backends: Optional[List[str]] = typer.Option(
        None, "--backends",
        help=(
            "Backend URLs for proxy (required when starting proxy). "
            "Example: --backends http://localhost:5100 --backends http://localhost:5101"
        ),
    ),
):
    """Start a VLA policy server, VLM proxy, sim worker, or round-robin proxy.

    Examples:

    \\b
        # VLM proxy (LiteLLM + Vertex AI Gemini)
        robo-eval servers start vlm
        robo-eval servers start vlm --port 4001
        robo-eval servers start vlm --model vertex_ai/gemini-3-flash-preview

    \\b
        # VLA policy servers
        robo-eval servers start pi05
        robo-eval servers start vla pi05        # alias form
        robo-eval servers start openvla --port 5101
        robo-eval servers start smolvla

    \\b
        # Sim workers
        robo-eval servers start sim --sim libero
        robo-eval servers start sim --sim libero_pro --port 5010 --headless
        robo-eval servers start sim --sim robocasa --port 5002 --headless

    \\b
        # Round-robin proxy (requires --backends)
        robo-eval servers start proxy --backends http://localhost:5100 --backends http://localhost:5101
    """
    import os as _os

    from .servers import start_sim_server, start_vla_server, start_vlm_proxy

    # Handle 'vla <name>' two-word form: redirect name to the VLA name
    if name == "vla":
        if not target:
            typer.echo(
                "Error: 'vla' requires a VLA name argument.  "
                f"Available: {list(VLA_CONFIGS.keys())}"
            )
            typer.echo("Example: robo-eval servers start vla pi05")
            raise typer.Exit(1)
        name = target

    if name == "vlm":
        actual_port = port or int(_os.environ.get("LITELLM_PORT", str(DEFAULT_VLM_PORT)))
        actual_model = model or _os.environ.get("VLM_MODEL", DEFAULT_VLM_MODEL)
        typer.echo(f"[servers] Starting VLM proxy ({actual_model}) on port {actual_port}...")
        start_vlm_proxy(port=actual_port, model=actual_model, wait=not no_wait)
        typer.echo(f"[servers] VLM proxy started on port {actual_port}.")

    elif name == "sim":
        backend = sim_backend
        if not backend:
            typer.echo(
                "Error: --sim is required when starting a sim worker.  "
                f"Valid backends: {list(SIM_CONFIGS.keys())}"
            )
            raise typer.Exit(1)
        if backend not in SIM_CONFIGS:
            typer.echo(
                f"Error: Unknown sim backend '{backend}'.  "
                f"Valid: {list(SIM_CONFIGS.keys())}"
            )
            raise typer.Exit(1)
        actual_port = port or int(_os.environ.get("SIM_PORT", "5001"))
        typer.echo(
            f"[servers] Starting {backend} sim worker on {host}:{actual_port}"
            f" ({'headless' if headless else 'windowed'})..."
        )
        pid = start_sim_server(
            backend,
            port=actual_port,
            host=host,
            headless=headless,
            wait=not no_wait,
        )
        if pid:
            typer.echo(f"[servers] {backend} sim worker started (PID {pid}) on port {actual_port}.")

    elif name == "proxy":
        from .servers import start_proxy
        if not backends:
            typer.echo(
                "Error: Starting the proxy requires --backends.\n"
                "Example: robo-eval servers start proxy "
                "--backends http://localhost:5100 --backends http://localhost:5101\n"
                "Or use 'robo-eval run' which manages the proxy automatically."
            )
            raise typer.Exit(1)
        actual_port = port or DEFAULT_PROXY_PORT
        typer.echo(
            f"[servers] Starting VLA proxy on port {actual_port} -> {len(backends)} backend(s)..."
        )
        pid = start_proxy(
            backend_urls=backends,
            port=actual_port,
            wait=not no_wait,
        )
        if pid:
            typer.echo(f"[servers] VLA proxy started (PID {pid}) on port {actual_port}.")
        else:
            typer.echo(f"[servers] VLA proxy already running on port {actual_port}.")

    elif name in VLA_CONFIGS:
        cfg = VLA_CONFIGS[name]
        actual_port = port or int(_os.environ.get("VLA_PORT", str(cfg.port)))
        typer.echo(f"[servers] Starting {name} policy server on port {actual_port}...")
        pid = start_vla_server(name, port=actual_port, wait=not no_wait, model_id=model)
        if pid:
            typer.echo(f"[servers] {name} policy server started (PID {pid}) on port {actual_port}.")

    else:
        all_valid = list(VLA_CONFIGS.keys()) + ["vlm", "sim", "vla", "proxy"]
        typer.echo(f"Error: Unknown server '{name}'. Available: {all_valid}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# robo-eval servers stop
# ---------------------------------------------------------------------------
@servers_app.command("stop")
def servers_stop(
    name: str = typer.Argument(
        ...,
        help=(
            "Server to stop.  Valid values: "
            "vlm, sim, vla, pi05, smolvla, openvla, proxy."
        ),
    ),
    target: Optional[str] = typer.Argument(
        None,
        help="VLA name when using 'vla' prefix.  Example: robo-eval servers stop vla pi05",
    ),
    port: Optional[int] = typer.Option(
        None, "--port", "-p",
        help="Override port number.",
    ),
    sim_backend: Optional[str] = typer.Option(
        None, "--sim",
        help="Simulator backend (required when name='sim').",
    ),
):
    """Stop a VLA policy server, VLM proxy, sim worker, or round-robin proxy."""
    from .servers import stop_proxy, stop_sim_server, stop_vla_server, stop_vlm_proxy

    # Handle 'vla <name>' alias
    if name == "vla":
        if not target:
            typer.echo(
                "Error: 'vla' requires a VLA name argument.  "
                f"Available: {list(VLA_CONFIGS.keys())}"
            )
            raise typer.Exit(1)
        name = target

    if name == "vlm":
        stop_vlm_proxy(port=port or DEFAULT_VLM_PORT)
    elif name == "proxy":
        stop_proxy(port=port or DEFAULT_PROXY_PORT)
    elif name == "sim":
        backend = sim_backend
        if not backend:
            typer.echo(
                "Error: --sim is required when stopping a sim worker.  "
                f"Valid backends: {list(SIM_CONFIGS.keys())}"
            )
            raise typer.Exit(1)
        actual_port = port or 5001
        stop_sim_server(backend, port=actual_port)
    elif name in VLA_CONFIGS:
        stop_vla_server(name, port=port)
    else:
        all_valid = list(VLA_CONFIGS.keys()) + ["vlm", "sim", "vla", "proxy"]
        typer.echo(f"Error: Unknown server '{name}'. Available: {all_valid}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# robo-eval compare
# ---------------------------------------------------------------------------
@app.command("compare")
def compare_cmd(
    dirs: List[str] = typer.Argument(
        ...,
        help="Result directories to compare (each must contain scores.json).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Output path for comparison markdown. Default: comparison.md in first dir.",
    ),
):
    """Compare results across multiple evaluation runs.

    Reads scores.json from each directory and generates a markdown
    comparison table with suite × run matrix showing success rates
    and 95% Wilson confidence intervals.
    """
    import json as _json

    from .results import wilson_ci

    if len(dirs) < 2:
        typer.echo("Error: Need at least 2 result directories to compare.")
        raise typer.Exit(1)

    # Load scores.json from each directory
    run_data = {}  # name -> scores dict
    run_names = []
    for d in dirs:
        p = Path(d)
        scores_path = p / "scores.json"
        if not scores_path.exists():
            typer.echo(f"Warning: No scores.json in {d}, skipping.")
            continue
        try:
            with open(scores_path) as f:
                data = _json.load(f)
        except _json.JSONDecodeError:
            typer.echo(f"Warning: Invalid JSON in {scores_path}, skipping.")
            continue
        name = p.name
        run_data[name] = data
        run_names.append(name)

    if len(run_data) < 2:
        typer.echo("Error: Need at least 2 valid result directories with scores.json.")
        raise typer.Exit(1)

    # Collect all suites across all runs (preserving order)
    all_suites = []
    seen_suites = set()
    for name in run_names:
        data = run_data[name]
        # Get suites from the suites dict
        for suite_name in data.get("suites", {}):
            if suite_name not in seen_suites:
                all_suites.append(suite_name)
                seen_suites.add(suite_name)

    # Also aggregate from tasks if suite-level data is missing/wrong
    def _aggregate_suite(data, suite_name):
        """Aggregate task-level data for a suite, returns (success, total)."""
        success = 0
        total = 0
        for task in data.get("tasks", []):
            if task["suite"] == suite_name:
                success += task["n_success"]
                total += task["n_episodes"]
        return success, total

    # Build the table
    lines = []
    lines.append("# VLA Comparison\n")

    # Header row
    header = "| Suite |"
    separator = "|-------|"
    for name in run_names:
        header += f" {name} |"
        separator += f" {'-' * max(len(name), 10)} |"
    lines.append(header)
    lines.append(separator)

    # Suite rows
    for suite in all_suites:
        row = f"| {suite} |"
        for name in run_names:
            data = run_data[name]
            suite_data = data.get("suites", {}).get(suite)

            # Try to get data from tasks if suite aggregation is available
            success, total = _aggregate_suite(data, suite)

            if total > 0:
                rate = success / total * 100
                ci_lo, ci_hi = wilson_ci(success, total)
                cell = f"{success}/{total} ({rate:.1f}%, CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])"
            elif suite_data and suite_data.get("total", 0) > 0:
                s = suite_data["success"]
                t = suite_data["total"]
                rate = s / t * 100
                ci_lo, ci_hi = wilson_ci(s, t)
                cell = f"{s}/{t} ({rate:.1f}%, CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])"
            else:
                cell = "-"
            row += f" {cell} |"
        lines.append(row)

    # Overall row
    row = "| **Overall** |"
    for name in run_names:
        data = run_data[name]
        overall = data.get("overall", {})
        # Recompute from tasks for accuracy
        total_success = sum(t["n_success"] for t in data.get("tasks", []))
        total_total = sum(t["n_episodes"] for t in data.get("tasks", []))
        if total_total > 0:
            rate = total_success / total_total * 100
            ci_lo, ci_hi = wilson_ci(total_success, total_total)
            cell = f"{total_success}/{total_total} ({rate:.1f}%, CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])"
        elif overall.get("total", 0) > 0:
            s = overall["success"]
            t = overall["total"]
            rate = s / t * 100
            ci_lo, ci_hi = wilson_ci(s, t)
            cell = f"{s}/{t} ({rate:.1f}%, CI: [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])"
        else:
            cell = "-"
        row += f" {cell} |"
    lines.append(row)

    table_text = "\n".join(lines) + "\n"

    # Print to stdout
    typer.echo(table_text)

    # Save to file
    if output:
        output_path = Path(output)
    else:
        output_path = Path(dirs[0]) / "comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text)
    typer.echo(f"Comparison saved to {output_path}")


# ---------------------------------------------------------------------------
# robo-eval suites
# ---------------------------------------------------------------------------
@app.command("suites")
def suites_cmd(
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b",
        help=(
            "Benchmark to list suites for. "
            f"Valid: {', '.join(sorted(BENCHMARK_SUITES))}. "
            "If omitted, lists all benchmarks."
        ),
    ),
):
    """List available short suite names for a benchmark.

    Short suite names are used with --benchmark in robo-eval run.
    Example: robo-eval run --benchmark libero --suites spatial,goal
    """
    if benchmark:
        if benchmark not in BENCHMARK_SUITES:
            typer.echo(f"Error: Unknown benchmark '{benchmark}'. Valid: {', '.join(sorted(BENCHMARK_SUITES))}")
            raise typer.Exit(1)
        short_names = get_suites_for_benchmark(benchmark)
        typer.echo(f"Short suite names for --benchmark {benchmark}:")
        for name in short_names:
            typer.echo(f"  {name}")
        typer.echo()
        typer.echo(f"Usage: robo-eval run --benchmark {benchmark} --suites {short_names[0]}")
        typer.echo(f"       robo-eval run --benchmark {benchmark}  (runs all: {', '.join(short_names)})")
    else:
        typer.echo("Available benchmarks and their short suite names:")
        typer.echo()
        for bm in sorted(BENCHMARK_SUITES):
            short_names = get_suites_for_benchmark(bm)
            typer.echo(f"  {bm:<16}: {', '.join(short_names)}")
        typer.echo()
        typer.echo("Use --benchmark to filter. Example: robo-eval suites --benchmark libero")
        typer.echo("Usage: robo-eval run --benchmark <benchmark> [--suites <short_name,...>]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the robo-eval CLI."""
    app()


if __name__ == "__main__":
    main()
