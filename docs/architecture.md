# Architecture

roboeval is a host-process evaluation harness that runs VLA policy servers,
simulator workers, and an optional VLM proxy as independent HTTP services. The
orchestrator reads a YAML config, validates component contracts before episode
execution, runs the selected task and episode loop, and writes JSON results with
the config snapshot, per-task aggregates, per-episode records, and optional shard
metadata.

## Reader Map

- Use this page for a five-minute system overview.
- Use [extending.md](extending.md) for implementation steps.
- Use [vla_policy_architecture.md](vla_policy_architecture.md) for deeper policy-server details.
- Use [sharded_runs.md](sharded_runs.md) for shard mechanics.
- Use [liten.md](liten.md) for hierarchical evaluation details.

## Components

| Component | Responsibility |
|---|---|
| Orchestrator | Loads YAML configs, builds the task/episode work list, validates VLA and simulator contracts, launches per-episode evaluation subprocesses, tracks progress, and writes results. |
| VLA policy server | Hosts a model behind `/health`, `/info`, `/reset`, and `/predict`; declares action outputs and required observations through `ActionObsSpec`. |
| Simulator worker | Hosts a simulator backend behind `/init`, `/reset`, `/step`, `/obs`, `/success`, `/info`, and `/close`; declares consumed actions and produced observations. |
| VLM proxy | Optional planner endpoint used by LITEN-style hierarchical mode; the direct VLA/simulator interface is unchanged. |
| Results collector | Aggregates episode records into task-level and benchmark-level JSON, including metric aggregates and shard metadata when sharding is enabled. |

## Process Boundary

- Components communicate over HTTP/JSON.
- Components do not import each other at runtime.
- Each component can use its own virtual environment.
- Each component can use a different Python version.
- Each component can use different CUDA or simulator dependencies.
- The orchestrator process owns run configuration and result writing.
- The VLA server process owns model loading and inference state.
- The simulator worker process owns simulator imports, rendering, and physics state.
- The VLM endpoint is only required when `no_vlm: false`.

## Component Interfaces

### Orchestrator

- CLI entry point: `roboeval run -c <config>.yaml`.
- Config type: flat YAML loaded into `EvalConfig`.
- Work list: selected tasks multiplied by `episodes_per_task`.
- Sharding: round-robin by work-item index.
- Execution unit: one subprocess per episode.
- Output: one benchmark JSON file per run or one shard JSON file per shard.

### VLA Policy Server

- `GET /health`: reports readiness.
- `GET /info`: returns model metadata and interface declarations.
- `POST /reset`: clears per-episode state.
- `POST /predict`: returns one or more simulator-space action vectors.
- `get_action_spec()`: declares action components produced by the policy.
- `get_observation_spec()`: declares observations required by the policy.

### Simulator Worker

- `POST /init`: initializes the selected task.
- `POST /reset`: starts an episode.
- `POST /step`: applies one action.
- `GET /obs`: returns the current observation.
- `GET /success`: reports task success when the backend provides it.
- `GET /info`: returns action/observation spaces and typed specs.
- `POST /close`: releases simulator resources.

### VLM Proxy

- Used for hierarchical evaluation only.
- Exposes a LiteLLM-compatible or OpenAI-compatible endpoint.
- Receives task context from the planner.
- Emits planner text that calls `world.act("<subtask>")`.
- The low-level VLA server still receives ordinary natural-language instructions.

### Results Collector

- Records episode dictionaries under task names such as `task_0`.
- Keeps benchmark-defined metrics under `episode["metrics"]`.
- Aggregates configured metrics at task and benchmark levels.
- Promotes `params.seed` to top-level `seed` when present.
- Adds `metric_keys` when aggregates are configured.
- Writes optional shard metadata in sharded runs.

## ActionObsSpec Contract

`ActionObsSpec` is the typed compatibility contract between VLA servers and
simulator workers. It describes one action or observation component in enough
detail for the orchestrator to reject incompatible pairings before episode 1.

### Declared Fields

| Field | Meaning |
|---|---|
| `name` | Human-readable component name such as `position`, `rotation`, `gripper`, `image`, `state`, or `language`. |
| `dims` | Number of numeric dimensions; `0` is used for non-array values such as images and strings. |
| `format` | Convention string such as `delta_xyz`, `axis_angle`, `rgb_hwc_uint8`, or `binary_close_positive`. |
| `range` | Optional numeric value range, serialized as `[min, max]`. |
| `accepts` | Optional set of producer formats the consumer can convert. |
| `description` | Optional notes for edge cases or conventions. |

### Action Direction

- The VLA policy server produces actions.
- The simulator worker consumes actions.
- The orchestrator compares policy `action_spec` against simulator `action_spec`.
- Required simulator action components must be present in the policy output.
- Component dimensions must match unless one side declares `dims: 0`.
- Component formats must match unless the consumer explicitly lists the producer format in `accepts`.

### Observation Direction

- The simulator worker produces observations.
- The VLA policy server consumes observations.
- The orchestrator compares simulator `observation_spec` against policy `observation_spec`.
- Required policy observation components must be present in the simulator output.
- Camera roles, image format, state dimensionality, and state convention are declared explicitly.
- Language inputs are represented as typed observation components.

### Enforced Mismatch Types

| Mismatch | Severity |
|---|---|
| Missing required action component | Hard failure |
| Missing required observation component | Hard failure |
| No overlapping action keys when both sides declare action specs | Hard failure |
| Action dimension mismatch | Hard failure |
| Action format mismatch without consumer conversion | Hard failure |
| Observation format mismatch without consumer conversion | Hard failure |
| Image-transform declaration disagreement | Hard failure in the environment wrapper compatibility checks |
| Action range mismatch | Warning |
| Legacy components with no specs | Warning |
| Optional metadata mismatch | Ignored |

### Gate Timing

- The gate runs after both component `/info` calls are available.
- The gate runs before episode 1.
- The gate runs before expensive rollout work starts.
- Hard failures stop the run.
- Warnings are logged but do not stop execution.
- The same contract applies to direct and LITEN-style runs.

## Config To Run Flow

```text
YAML config
    |
    v
roboeval run
    |
    v
EvalConfig parser
    |
    v
orchestrator builds task x episode work list
    |
    v
VLA /info  +  simulator /info
    |
    v
ActionObsSpec compatibility gate
    |
    v
episode subprocess loop
    |
    v
ResultCollector
    |
    v
result JSON
```

## Run Flow Detail

1. The user starts component services with `roboeval serve`.
2. The user runs `roboeval run -c configs/<name>.yaml`.
3. The CLI loads YAML into `EvalConfig`.
4. The orchestrator builds a flat work list from tasks and episodes.
5. If sharding is enabled, each shard keeps work items where `index % num_shards == shard_id`.
6. The orchestrator claims a result path and progress path.
7. The collector is initialized with the benchmark name and metric keys.
8. For each work item, the orchestrator launches `python -m roboeval.run_sim_eval eval`.
9. The subprocess receives `VLA_URL` and simulator URL from the config.
10. The episode runner initializes the simulator task.
11. The episode runner checks component compatibility.
12. The episode runner executes the policy/simulator step loop.
13. The episode runner writes per-episode detail JSON.
14. The orchestrator reads the episode JSON.
15. The collector records the episode under a task name.
16. The orchestrator updates a `.progress` file.
17. The collector computes task aggregates.
18. The collector computes benchmark aggregates.
19. The orchestrator writes the final JSON.
20. Sharded output includes shard metadata.

## YAML Config Shape

```yaml
name: libero_spatial_pi05_smoke
vla: pi05
vla_url: http://localhost:5100
sim: libero
sim_url: http://localhost:5300
suite: libero_spatial
episodes_per_task: 10
max_tasks: 1
no_vlm: true
delta_actions: true
eval_python: ""
output_dir: results/libero_spatial_pi05_smoke
params: {}
```

## Important Config Fields

| Field | Meaning |
|---|---|
| `name` | Benchmark/run name used in result filenames and top-level JSON. |
| `vla` | Human-readable VLA launch name. |
| `vla_url` | HTTP URL for the policy server. |
| `sim` | Simulator backend name. |
| `sim_url` | HTTP URL for the simulator worker. |
| `suite` | Benchmark suite or task namespace. |
| `task` | Optional single task selector. |
| `tasks` | Optional list of task selectors. |
| `max_tasks` | Optional cap applied after task selection. |
| `episodes_per_task` | Number of episodes per selected task. |
| `episode_timeout_seconds` | Wall-clock timeout per episode subprocess. |
| `no_vlm` | `true` for direct evaluation, `false` for hierarchical mode. |
| `vlm_model` | Planner model name when hierarchical mode is enabled. |
| `vlm_endpoint` | Planner endpoint host and port. |
| `delta_actions` | Whether the episode runner passes delta-action mode to the simulator wrapper. |
| `eval_python` | Optional Python executable for episode subprocesses. |
| `output_dir` | Directory for result files. |
| `params` | Extra CLI parameters forwarded to the episode runner. |

## Result JSON Schema

The collector writes benchmark-level JSON. The shape below is a complete example
of the structure produced by `roboeval/results/collector.py`, including the
shard object added by the orchestrator for sharded runs.

```json
{
  "benchmark": "libero_spatial_pi05_smoke",
  "mode": "sync",
  "harness_version": "0.1.0",
  "created_at": "2026-04-29T12:00:00+00:00",
  "tasks": [
    {
      "task": "task_0",
      "episodes": [
        {
          "episode_id": 0,
          "metrics": {
            "success": true
          },
          "steps": 78,
          "elapsed_sec": 42.3
        }
      ],
      "num_episodes": 1,
      "avg_steps": 78.0,
      "mean_success": 1.0
    }
  ],
  "config": {
    "name": "libero_spatial_pi05_smoke",
    "vla_url": "http://localhost:5100",
    "sim_url": "http://localhost:5300",
    "sim": "libero",
    "suite": "libero_spatial",
    "task": null,
    "tasks": [],
    "max_tasks": 1,
    "episodes_per_task": 10,
    "episode_timeout_seconds": 1800,
    "no_vlm": true,
    "vlm_model": null,
    "vlm_endpoint": "localhost:4000",
    "delta_actions": true,
    "eval_python": "",
    "output_dir": "results/libero_spatial_pi05_smoke",
    "params": {
      "seed": 0
    }
  },
  "seed": 0,
  "metric_keys": {
    "success": "mean"
  },
  "mean_success": 1.0,
  "shard": {
    "shard_id": 0,
    "num_shards": 1
  }
}
```

## Episode Failure Records

Episode records may include failure fields when a subprocess times out, exits
nonzero, or raises an exception:

```json
{
  "episode_id": 3,
  "metrics": {
    "success": false
  },
  "steps": 0,
  "elapsed_sec": 1800.0,
  "failure_reason": "timeout",
  "failure_detail": "subprocess exceeded 1800s timeout"
}
```

## Sharded Result Files

- Non-sharded runs include a timestamp, process ID, and random suffix in the filename.
- Sharded runs use deterministic names: `<name>_shard<id>of<num>.json`.
- Each shard writes a separate JSON file.
- Each shard writes a `.progress` file while running.
- `roboeval merge` recomputes aggregates from shard episodes.
- Partial merges are marked with `"partial": true`.

## Extension Points

- Add a VLA: implement a policy server and register its module, venv, port, and setup path; see [extending.md](extending.md#add-a-vla).
- Add a benchmark: implement a simulator backend and register its worker defaults; see [extending.md](extending.md#add-a-benchmark).
- Add a planner: keep the `world.act(subtask_instruction)` boundary and integrate under the VLM planner path; see [extending.md](extending.md#liten-compatible-extensions).

### Extension cost

As of v0.1.0, integrating a new VLA averages ~200 SLOC across the 11 VLAs in the release (range: 141–361; N=11; excludes blank lines, comments, and docstrings). Integrating a new simulator backend averages ~230 SLOC (range: 40–469; N=9).

## Operational Defaults

| Service family | Typical ports |
|---|---|
| VLA policy servers | `5100` and nearby ports |
| Simulator workers | `5300` and nearby ports |
| LiteLLM VLM proxy | `4000` |
| Local vLLM endpoint | commonly `8000` |

## Design Implications

- A supported pair is a contract-compatible VLA and simulator combination.
- Compatibility is checked through declared metadata, not only through successful imports.
- A new model can be added without importing simulator code.
- A new simulator can be added without importing model code.
- Hierarchical evaluation reuses the same VLA and simulator services.
- Result aggregation is independent of the model and simulator implementations.

## File-Level Pointers

| Area | File |
|---|---|
| CLI | `roboeval/cli/main.py` |
| Orchestrator | `roboeval/orchestrator.py` |
| Spec contract | `roboeval/specs.py` |
| Server launch defaults | `roboeval/server_runner.py` |
| Result collector | `roboeval/results/collector.py` |
| Shard merge | `roboeval/results/merge.py` |
| Simulator worker | `sims/sim_worker.py` |
| VLA servers | `sims/vla_policies/` |
| LITEN docs | `docs/liten.md` |
| Extension docs | `docs/extending.md` |
