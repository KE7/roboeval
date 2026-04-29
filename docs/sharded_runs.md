# Sharded Evaluation Runs in roboeval

---

## 1. When to Use Sharded Runs

| Scenario | Recommended approach |
|---|---|
| **Multi-GPU host** | 1 shard per GPU; set `CUDA_VISIBLE_DEVICES` per shard process |
| **Multi-process on a single GPU** | 2–4 shards; VLA server starts once and serves all shards concurrently; shard processes run sequentially or with server's built-in batching |
| **Distributed across multiple machines** | 1 shard per machine; shards write to shared NFS / object store; merge at the end |
| **Long full-suite run that must be restartable** | N shards, each writing a separate JSON; failed shards can be re-run without re-doing completed ones; `roboeval merge` tolerates missing shards (marks result `partial`) |
| **Single-process, short smoke test** | Don't shard — overhead not worth it for < 20 episodes |

---

## 2. CLI Commands

### 2.1 Single-process (non-sharded)

```bash
# Start VLA server once (from the roboeval project root)
cd /path/to/roboeval
.venvs/vla/bin/python \
    -m sims.vla_policies.pi05_policy --port 5100

# Start sim worker
.venvs/libero/bin/python \
    -m sims.sim_worker --backend libero --port 5300 --headless

# Run all 10 episodes on task 0
.venvs/roboeval/bin/python \
    -m roboeval run \
    --config configs/libero_spatial_pi05_smoke.yaml \
    --output-dir results/sharded_smoke
```

Output: `results/sharded_smoke/libero_spatial_pi05_smoke_<timestamp>.json`

### 2.2 Sharded run (N=2 shards)

```bash
# Start VLA server and sim worker as above (once, shared)

# Shard 0: items at even indices → episodes 0, 2, 4, 6, 8
.venvs/roboeval/bin/python \
    -m roboeval run \
    --config configs/libero_spatial_pi05_smoke.yaml \
    --shard-id 0 --num-shards 2 \
    --output-dir results/sharded_smoke

# Shard 1: items at odd indices → episodes 1, 3, 5, 7, 9
.venvs/roboeval/bin/python \
    -m roboeval run \
    --config configs/libero_spatial_pi05_smoke.yaml \
    --shard-id 1 --num-shards 2 \
    --output-dir results/sharded_smoke
```

Outputs (deterministic names — no timestamp when sharded):
- `results/sharded_smoke/libero_spatial_pi05_smoke_shard0of2.json`
- `results/sharded_smoke/libero_spatial_pi05_smoke_shard1of2.json`

### 2.3 Parallel launch via script (preferred for multi-GPU)

```bash
cd /path/to/roboeval
bash scripts/run_sharded_v2.sh \
    --config configs/libero_spatial_pi05_smoke.yaml \
    --num-shards 2 \
    --output-dir results/sharded_smoke
```

This backgrounds both shards simultaneously, waits for both, then merges automatically.

### 2.4 Merging shard results

```bash
.venvs/roboeval/bin/python \
    -m roboeval merge \
    --pattern 'results/sharded_smoke/*shard*.json' \
    --output results/sharded_smoke/final.json
```

Exit codes:
- `0` — all shards present, merge complete
- `2` — partial merge (some shards missing), result file still written with `"partial": true`

---

## 3. How Init-State IDs Are Partitioned Across Shards

Sharding is **round-robin by work-item index**, implemented in `roboeval/orchestrator.py`:

```python
# Build flat (task_id, episode) work list
work_items: list[tuple[int, int]] = []
for task_id in tasks:
    for ep in range(cfg.episodes_per_task):
        work_items.append((task_id, ep))

# Round-robin shard assignment
if self.num_shards is not None and self.shard_id is not None:
    work_items = [
        w for i, w in enumerate(work_items)
        if i % self.num_shards == self.shard_id
    ]
```

**Example — 1 task, 10 episodes, 2 shards:**

| Work-item index | (task_id, episode) | Assigned shard |
|---:|---|---:|
| 0 | (task_0, ep0) | **Shard 0** |
| 1 | (task_0, ep1) | Shard 1 |
| 2 | (task_0, ep2) | **Shard 0** |
| 3 | (task_0, ep3) | Shard 1 |
| 4 | (task_0, ep4) | **Shard 0** |
| 5 | (task_0, ep5) | Shard 1 |
| 6 | (task_0, ep6) | **Shard 0** |
| 7 | (task_0, ep7) | Shard 1 |
| 8 | (task_0, ep8) | **Shard 0** |
| 9 | (task_0, ep9) | Shard 1 |

Shard 0 runs episodes: 0, 2, 4, 6, 8  
Shard 1 runs episodes: 1, 3, 5, 7, 9  

**Multi-task example — 2 tasks, 4 episodes each, 2 shards:**

Work items: `(t0,ep0),(t0,ep1),(t0,ep2),(t0,ep3),(t1,ep0),(t1,ep1),(t1,ep2),(t1,ep3)`

Shard 0 (indices 0,2,4,6): `(t0,ep0),(t0,ep2),(t1,ep0),(t1,ep2)`  
Shard 1 (indices 1,3,5,7): `(t0,ep1),(t0,ep3),(t1,ep1),(t1,ep3)`

Each shard covers all tasks — a key benefit vs. task-splitting.

**Episode IDs and LIBERO determinism.** LIBERO seeds each episode's initial state from its episode number. Episode 0 always starts at the same init state, episode 1 at a different fixed init state, etc. Because shards use the same episode IDs as a non-sharded run, a sharded + merged result is **mathematically equivalent** to a single non-sharded run (same episodes, same init states, same outcomes).

---

## 4. Merge Behavior

The `merge_shards()` function (`roboeval/results/merge.py`):

1. **Validates** that all shard files share the same `benchmark` name and `shard.total`.
2. **Detects** missing / duplicate shard IDs and raises on duplicates.
3. **Merges** episodes per task using a `{task → {episode_id → episode}}` dict; **last-file-wins** on duplicate `episode_id` (rare in practice, logged as warning).
4. **Recomputes** all metric aggregates from the merged episode set (e.g. `mean_success`).
5. **Marks** the result `"partial": true` if any expected shard is absent.

The file-naming convention for shard files is:
```
<safe_name>_shard{shard_id}of{num_shards}.json
```
e.g. `libero_spatial_pi05_smoke_shard0of2.json`

---

## 5. Aggregate Behavior

`roboeval merge` recomputes aggregates from episode records rather than trusting shard-level summary fields. The merged output should match the episode set selected by the shard files, with deterministic policies producing identical aggregate rates to an equivalent non-sharded run over the same episodes.

For stochastic policies, two runs over the same simulator episode IDs can produce different action trajectories and therefore different outcomes. This is expected model behavior rather than a sharding issue.

---

## 6. Hardware Notes and GPU Coordination

On a single-GPU host, true multi-GPU sharding is not possible. Expected behavior:

- **Single-GPU, sequential shards**: useful for restartability, not speed.
- **Parallel shards sharing one VLA server**: policy requests may serialize inside the server.
- **Multi-GPU host**: set `CUDA_VISIBLE_DEVICES=N` per shard process and run shards in parallel with separate policy servers.

**GPU coordination rule:** Check `nvidia-smi` before starting any VLA server or eval. If GPU utilization > 50%, back off and retry.

---

## 7. Extending to True Multi-GPU

On a host with multiple GPUs:

```bash
# GPU 0 runs shard 0
CUDA_VISIBLE_DEVICES=0 \
  .venvs/vla/bin/python \
      -m sims.vla_policies.pi05_policy --port 5100 &

# GPU 1 runs shard 1
CUDA_VISIBLE_DEVICES=1 \
  .venvs/vla/bin/python \
      -m sims.vla_policies.pi05_policy --port 5102 &

# Wait for both VLA servers to be ready, then:
VLA_URL=http://localhost:5100 \
  .venvs/roboeval/bin/python \
      -m roboeval run \
      --config configs/libero_spatial_pi05_smoke.yaml \
      --shard-id 0 --num-shards 2 --output-dir results/sharded_run &

VLA_URL=http://localhost:5102 \
  .venvs/roboeval/bin/python \
      -m roboeval run \
      --config configs/libero_spatial_pi05_smoke.yaml \
      --shard-id 1 --num-shards 2 --output-dir results/sharded_run &

wait

.venvs/roboeval/bin/python \
    -m roboeval merge \
    --pattern 'results/sharded_run/*shard*.json' \
    --output results/sharded_run/final.json
```

Or use the convenience script (which handles backgrounding and merging):
```bash
bash scripts/run_sharded_v2.sh \
    --config configs/libero_spatial_pi05_smoke.yaml \
    --num-shards 4 \
    --output-dir results/sharded_run
```

**Note**: `run_sharded_v2.sh` does not currently handle per-shard `CUDA_VISIBLE_DEVICES` or separate VLA ports. For multi-GPU, set those env vars manually or extend the script.

---

## 8. Known Limitations

1. **VLA server concurrency**: If N shards share one VLA server, requests serialize inside FastAPI. High N can cause timeout if per-request latency × queue depth > server timeout. Use separate per-GPU VLA servers for true parallelism.

2. **Episode init states must be deterministic**: Sharding relies on LIBERO's fixed episode seeding. Environments that randomize init states per-run (not per-episode-ID) would produce non-reproducible splits and the sharded+merged rate would not match the non-sharded rate exactly.

3. **No automatic retry on partial failure**: A shard that crashes partway through produces a partial shard file. `roboeval merge` will include completed episodes but mark the result `partial`. There is no built-in retry — re-run the failed shard manually with the same `--shard-id`.

4. **Lock-file collision**: Two processes with the same `--shard-id --num-shards` pointing to the same `--output-dir` will contend on a `.lock` file. One will fail with `FileExistsError`. Resolve by using unique output directories or removing the stale `.lock` file.

5. **Single-GPU wall time**: On a single GPU, sequential shards offer no wall-time benefit vs. non-sharded execution. The value is fault-tolerance and restartability, not speed.

6. **`run_sharded_v2.sh` uses `roboeval` from PATH**: If `roboeval` is not on PATH, the script falls back to `python -m roboeval`. Ensure the correct venv is activated before running.

7. **Non-sharded run filename contains timestamp + pid + random hex**: `libero_spatial_pi05_smoke_1777158282_284872_637cf5.json`. Sharded runs use deterministic names: `libero_spatial_pi05_smoke_shard0of2.json`. Use the deterministic shard-file names when scripting merge.

8. **Diffusion VLA stochasticity**: Pi05 and similar diffusion-based policies sample a fresh random action trajectory each run. Even with identical LIBERO init states (same episode ID), outcomes can differ between baseline and sharded runs. This is a model property, not a sharding bug. Use 25+ episodes per task for stable rate estimates.
