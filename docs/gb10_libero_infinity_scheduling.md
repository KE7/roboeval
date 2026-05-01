# GB10 LIBERO-Infinity Scheduling Policy

This note captures the current scheduling policy for NeurIPS LIBERO-Infinity
runs on the GB10 host.

## Measured Host Facts

- `nvidia-smi` reports the device as `NVIDIA GB10`.
- `nvidia-smi` reports GPU utilization, but `memory.total` and `memory.used`
  are `N/A`, so it cannot be used as the memory gate.
- `/proc/meminfo` showed baseline `MemAvailable` around 63 GiB on May 1, 2026.
- A vLLM process is active on port `8000` for `Qwen/Qwen3.6-27B-FP8`.
- No active `roboeval`, VLA policy server, or `sim_worker` process was observed.
- The current filesystem has roughly 1.8 TiB free under the repo/results path.

## Memory Observation

Use system unified-memory pressure and process accounting:

```bash
python3 scripts/gb10_mem_probe.py
python3 scripts/gb10_mem_probe.py --interval 5 --count 120 --jsonl \
  > logs/gb10_mem_probe.jsonl
```

The probe records:

- `/proc/meminfo` `MemAvailable`, `MemTotal`, and `SwapFree`.
- Process RSS from `/proc/<pid>/status`.
- Process PSS and swap from `/proc/<pid>/smaps_rollup` when readable.
- `nvidia-smi` GPU utilization, with memory fields preserved as null when GB10
  returns `N/A`.

## Conservative Policy

- Default parallelism is one LIBERO-Infinity cell at a time.
- Full or concurrent VLA servers are blocked until pilot peak memory is measured.
- Every pilot and full run must record structured episode video.
- Full or concurrent runs are blocked until pilot video artifacts are verified
  playable.
- Keep at least 32 GiB `MemAvailable` after projected concurrent load.
- Treat process PSS as the preferred per-cell estimate. If PSS is unavailable,
  use RSS.
- Do not count `nvidia-smi` memory values on GB10 until they stop reporting
  `N/A`.
- Keep vLLM on `:8000` in the baseline unless the PM explicitly approves
  shutting it down.

## Pilot Harness

The queue harness generates per-cell configs with isolated ports and output
directories, starts a VLA/sim pair only for the selected pilot/run, samples
memory while it runs, then terminates the servers.

No servers are started by this documentation step. Use only after explicit PM
approval:

```bash
python3 scripts/run_libero_infinity_gb10_queue.py \
  --manifest configs/libero_infinity_gb10_queue.example.yaml \
  --mode pilot \
  --output-root results/libero_infinity_gb10_queue \
  --max-parallel 1
```

The harness writes:

- Generated configs under `results/libero_infinity_gb10_queue/_generated_configs/`.
- Server and run logs under `results/libero_infinity_gb10_queue/_logs/`.
- Per-cell `pilot_metrics.json` with peak PSS/RSS and `MemAvailable` delta.
- Local branch/commit provenance for roboeval and, when provided, the private
  LIBERO-Infinity worktree.

## Local Provenance Schema

Queue manifests may include a top-level `provenance` block:

```yaml
provenance:
  privacy: local-only-no-push-no-tag-no-publication
  roboeval_owner: harness/config/orchestration/video/result plumbing
  libero_infinity_owner: generator/compiler/Scenic/runtime perturbation fixes
  libero_infinity_worktree: /path/to/local/private/libero-infinity
```

The harness records:

- roboeval worktree, branch, commit, and dirty state;
- optional local private LIBERO-Infinity worktree, branch, commit, and dirty
  state;
- the ownership boundary above in generated configs and metrics JSON.

Roboeval owns fixes to harness/config/orchestration/video/result plumbing. Local
private LIBERO-Infinity branches own generator, compiler, Scenic, and runtime
perturbation fixes. Both remain local-only: no push, tag, publication, or video
artifact sharing.

## Slot Guidance

For Exp1/Exp3 articulation and combined/full pilots:

- Reserve one pilot slot only.
- Use ports starting at VLA `5510` and sim `5710` unless the PM assigns a
  different block.
- Run 1 episode and 1 task in `--mode pilot`.
- Do not start a second cell until the first pilot metrics are reviewed.

For Exp2 gated pilots:

- Reserve a separate later pilot slot, not concurrent with Exp1/Exp3 unless
  peak memory data supports it.
- Use ports starting at VLA `5520` and sim `5720` if Exp1/Exp3 is not active,
  or request a fresh block from the PM.
- Limit to 1-2 episodes after the plan-adapter mapping is confirmed.

## Promotion Rule

After pilots complete, compute:

```text
projected_available =
  baseline_MemAvailable - sum(candidate_cell_peak_available_deltas)
```

Approve concurrent cells only if:

- `projected_available >= 32 GiB`;
- no pilot used swap materially;
- no pilot had load-time OOM or CUDA allocation failures;
- every pilot produced at least one playable MP4 under the run output
  `videos/` directory;
- concurrent video writing is included in the pilot peak CPU, IO, disk, and
  memory observation;
- no long-running vLLM or other shared service has increased its baseline
  memory use.

Until those checks pass, full 200-scene cells should run sequentially.

## Video Capture Requirement

Scheduled runs must use `run_sim_eval --record-video`, not the older
`--save-videos` helper. The structured path is:

```text
<output_dir>/videos/<suite>_task<task>_ep<episode>_<instruction>.mp4
```

Videos are encoded with OpenCV `VideoWriter` using the `mp4v` codec in
`roboeval/run_sim_eval.py`. Frames are collected in memory during the episode
from `wrapper.subtask_frame_tuples`, then encoded after the episode completes.

Scheduling implications:

- video capture increases per-episode Python memory because frames are retained
  until encode time;
- MP4 encoding adds CPU load at episode end;
- concurrent cells can align encode phases and create CPU/IO spikes;
- disk impact depends on horizon and image resolution, so pilots must measure
  actual MP4 sizes before full 200-scene cells.

The GB10 queue harness forces:

```yaml
params:
  record_video: true
  record_video_n: <episodes_per_task>
```

and rejects a run when no playable MP4 is found. Playability is checked with
OpenCV by opening the MP4, reading metadata, and decoding the first frame.

## Privacy / Publication Boundary

LIBERO-Infinity fixes and generated artifacts for this scheduling effort stay
local/private. Do not push, tag, publish, or expose pilot/full videos. If a run
discovers a code fix, coordinate repo placement locally with the PM before
committing or sharing it.
