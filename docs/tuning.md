# Tuning Guide

This guide gives practical methodology for choosing shard counts, port
allocation, GPU allocation, and replica counts. It does not publish benchmark
throughput claims; measure on your own hardware and keep the config/result pair
with the run.

## Demand Versus Supply

Think of each run as demand placed on three resources:

- VLA inference.
- Simulator stepping and rendering.
- Optional VLM planning.

Supply comes from:

- Available GPU memory.
- Available CPU cores.
- Available simulator licenses/assets/render contexts.
- VLA server batching behavior.
- Number of simulator worker processes.
- Number of independent shards.

The slowest resource becomes the bottleneck.

If VLA inference dominates wall-clock time, more simulator workers may not help.

If simulator stepping dominates wall-clock time, VLA replicas may not help.

If the planner endpoint dominates hierarchical mode, direct-mode tuning numbers
will not transfer.

## First Measurement

Start with one VLA server, one simulator worker, and one non-sharded run.

Use a short config with one task and a small episode count.

Record:

- GPU memory at model load.
- GPU memory during action prediction.
- CPU utilization during simulator stepping.
- Mean episode wall-clock time.
- Whether episodes fail from timeouts or component errors.
- Whether ports remain clean after interruption.

Do not scale before this baseline is healthy.

## Shard Sizing

Use sharding when you need parallelism, restartability, or distribution across
machines.

`roboeval run` supports:

```bash
roboeval run -c configs/libero_spatial_pi05_smoke.yaml \
  --shard-id 0 \
  --num-shards 4
```

Both flags must be set together.

`--shard-id` is zero-based.

`--num-shards` is the total number of shards.

The orchestrator assigns work items round-robin:

```text
item_index % num_shards == shard_id
```

### Example: One Task, Ten Episodes, Two Shards

| Episode | Work-item index | Shard |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 1 | 1 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 0 |
| 5 | 5 | 1 |
| 6 | 6 | 0 |
| 7 | 7 | 1 |
| 8 | 8 | 0 |
| 9 | 9 | 1 |

Shard 0 runs episodes `0, 2, 4, 6, 8`.

Shard 1 runs episodes `1, 3, 5, 7, 9`.

### Example: Two Tasks, Four Episodes, Two Shards

The work list is:

```text
(task_0, ep0), (task_0, ep1), (task_0, ep2), (task_0, ep3),
(task_1, ep0), (task_1, ep1), (task_1, ep2), (task_1, ep3)
```

Shard 0 receives:

```text
(task_0, ep0), (task_0, ep2), (task_1, ep0), (task_1, ep2)
```

Shard 1 receives:

```text
(task_0, ep1), (task_0, ep3), (task_1, ep1), (task_1, ep3)
```

This keeps each shard represented across tasks.

### Choosing A Shard Count

Start with one shard per GPU when each shard owns its own VLA server.

Start with two shards per VLA server when the VLA server batches or handles
concurrent requests well.

Use more shards than GPUs when restartability matters more than immediate
parallelism.

Avoid tiny shards when setup and model-loading overhead dominate.

Avoid more concurrent shards than simulator workers unless shards share a worker
intentionally.

## Port Allocation

Keep port assignments predictable.

Default policy-server ports are in the `5100` range.

Default simulator-worker ports are in the `5300` range.

The LiteLLM proxy default is `4000`.

Local vLLM examples commonly use `8000`.

### Default Port Table

| Component | Default port |
|---|---:|
| Pi 0.5 policy server | `5100` |
| OpenVLA policy server | `5101` |
| SmolVLA policy server | `5102` |
| Cosmos policy server | `5103` |
| GR00T policy server | `5105` |
| InternVLA policy server | `5105` in shipped configs, `5104` in older config registry entries |
| VQ-BeT policy server | `5108` |
| TDMPC2 policy server | `5109` |
| LIBERO simulator worker | `5300` |
| LIBERO-Pro simulator worker | `5301` in launcher defaults; some configs reuse `5300` |
| RoboCasa simulator worker | `5302` |
| RoboTwin simulator worker | `5303` in launcher defaults; shipped configs use `5302` |
| ALOHA Gym simulator worker | `5304` in launcher defaults; ACT configs use `5001` |
| gym-pusht simulator worker | `5305` |
| Meta-World simulator worker | `5307` in setup examples; shipped configs use `5305` |
| LIBERO-Infinity simulator worker | `5308` |
| LiteLLM VLM proxy | `4000` |
| Local vLLM endpoint | `8000` |

When in doubt, use the URLs already present in the config file.

### Overriding Ports

Override server-launch ports with:

```bash
roboeval serve --vla pi05 --sim libero --vla-port 5110 --sim-port 5310 --headless
```

Then update the YAML config:

```yaml
vla_url: http://localhost:5110
sim_url: http://localhost:5310
```

For direct module launches, pass the module's `--port` argument.

For planner endpoints, update:

```yaml
vlm_endpoint: localhost:8000
```

Use `lsof -i :<port>` when a run reports that a port is already in use.

## GPU Allocation

Use `CUDA_VISIBLE_DEVICES` to pin component processes.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 roboeval serve --vla pi05 --sim libero --headless
```

For multi-GPU hosts, prefer one VLA server per GPU until you have measured
server-side batching.

For one large model across multiple GPUs, use the serving stack's tensor-parallel
configuration when supported by that model.

If a model is served through SGLang, tune tensor parallelism in the SGLang launch
command and point roboeval's `vla_url` at that server.

If a planner model is served through vLLM, tune vLLM independently and point
`vlm_endpoint` at it.

roboeval treats those servers as HTTP endpoints.

## VLA Replicas

Use multiple VLA servers when one policy process cannot keep up with simulator
demand or when you want one server per GPU.

The config layer includes `num_vla_servers` in resource-estimation helpers, but
the public run path is URL-based: each orchestrator shard should point at the
VLA server it is meant to use.

Common pattern:

```text
GPU 0 -> VLA server on :5100 -> shards 0, 1
GPU 1 -> VLA server on :5110 -> shards 2, 3
```

Create one config copy per server URL or use environment-specific config
generation in your run scripts.

Keep simulator URLs distinct if each shard owns a simulator worker.

## Simulator Workers

Use more simulator workers when simulation or rendering is the bottleneck.

The config layer includes `num_sim_workers` in resource-estimation helpers, but
each public run still points at a specific `sim_url`.

Common pattern:

```text
sim worker A -> :5300
sim worker B -> :5310
sim worker C -> :5320
```

Then assign shards to those URLs.

Do not assume every simulator backend is safe to share across many concurrent
episode loops.

When a backend has global process state, prefer one worker per active shard.

## Memory Budget

Use measured local memory, not fixed public claims, for final capacity planning.

The codebase includes rough resource estimates in `roboeval/config.py`.

Those estimates are meant for sizing discussions, not benchmark claims.

Current rough model-side estimates include:

| Component | Rough memory planning value |
|---|---:|
| Pi 0.5 | about 15 GB in `RAM_COSTS_GB` |
| OpenVLA | about 15 GB in `RAM_COSTS_GB` |
| SmolVLA | about 3 GB in `RAM_COSTS_GB` |
| GR00T | about 5 GB in `RAM_COSTS_GB` |
| InternVLA | about 7 GB in `RAM_COSTS_GB` |
| VQ-BeT | about 1 GB in `RAM_COSTS_GB` |
| TDMPC2 | about 2 GB in `RAM_COSTS_GB` |
| Simulator worker | about 2 GB CPU RAM in `RAM_COSTS_GB` |
| Evaluation process | about 0.5 GB in `RAM_COSTS_GB` |
| VLM proxy | about 0.5 GB in `RAM_COSTS_GB` |

Real GPU memory depends on dtype, attention backend, tensor parallelism, model
variant, batch behavior, and upstream package versions.

Measure with `nvidia-smi` or the profiler appropriate to your hardware.

## Practical Recipes

### Single GPU, One Direct Run

```bash
roboeval serve --vla pi05 --sim libero --headless
roboeval run -c configs/libero_spatial_pi05_smoke.yaml
```

Use this before sharding.

### Single GPU, Restartable Run

Run multiple shards sequentially or with conservative concurrency:

```bash
roboeval run -c configs/libero_spatial_pi05_smoke.yaml --shard-id 0 --num-shards 4
roboeval run -c configs/libero_spatial_pi05_smoke.yaml --shard-id 1 --num-shards 4
roboeval run -c configs/libero_spatial_pi05_smoke.yaml --shard-id 2 --num-shards 4
roboeval run -c configs/libero_spatial_pi05_smoke.yaml --shard-id 3 --num-shards 4
```

Merge after all shards finish:

```bash
roboeval merge --pattern 'results/run/*shard*.json' --output results/run/final.json
```

### Multi-GPU, One VLA Per GPU

Use distinct VLA ports and configs:

```bash
CUDA_VISIBLE_DEVICES=0 roboeval serve --vla pi05 --sim libero --vla-port 5100 --sim-port 5300 --headless
CUDA_VISIBLE_DEVICES=1 roboeval serve --vla pi05 --sim libero --vla-port 5110 --sim-port 5310 --headless
```

Assign shards so each config points to the intended URLs.

### Hierarchical Mode

Tune the VLM endpoint separately.

The planner can dominate wall-clock time even when VLA inference is fast.

Keep direct and hierarchical configs separate so `no_vlm`, `vlm_model`, and
`vlm_endpoint` are explicit.

## Related Docs

- [sharded_runs.md](sharded_runs.md)
- [failure_modes.md](failure_modes.md)
- [liten.md](liten.md)
- [extending.md](extending.md)
