# Quick-start guide

This page gets you from a fresh clone to a CLI-launched server pair and a
passing smoke evaluation.

## 1. Prerequisites

- **Python 3.11+** and **Git** on your PATH
- **NVIDIA GPU** with CUDA 11.8 or newer (for the Pi 0.5 policy server)
- [`uv`](https://docs.astral.sh/uv/) — `roboeval setup` installs it automatically if missing

## 2. Clone and install

```bash
git clone https://github.com/KE7/roboeval.git
cd roboeval
roboeval setup pi05 libero
```

This creates `.venvs/pi05` (Python 3.11, Pi 0.5 policy server) and
`.venvs/libero` (Python 3.8, LIBERO simulator). First run downloads the Pi 0.5
checkpoint (~75 s on a fast connection). The script is idempotent — safe to
re-run.

## 3. Check the config

```bash
.venvs/roboeval/bin/roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
```

Checks the config, registry, and ActionObsSpec contracts before you spend GPU
time. Expected output: `All checks passed`.

## 4. Start servers

```bash
.venvs/roboeval/bin/roboeval serve --vla pi05 --sim libero --headless
```

`serve` is flag-driven: swap `--vla` and `--sim` to launch another supported
pair without editing YAML.

## 5. First eval

```bash
.venvs/roboeval/bin/roboeval run \
  -c configs/libero_spatial_pi05_smoke.yaml
```

Runs Pi 0.5 on `libero_spatial` — 10 episodes × 1 task. The YAML is the
reproducible invocation of `roboeval run`: it captures the full supported pair
spec, including action format, embodiment tag, port wiring, output directory,
and optional LITEN endpoint. Runtime depends on GPU, model load time, simulator
startup, and network speed for first-time checkpoint downloads.

## 6. Inspect results

```bash
find results -name '*.json' | sort | tail -1 | xargs python3 -m json.tool
```

Look for `"harness_version": "0.1.0"`, per-task episode records, and any
`failure_reason` fields. If you observe systematic failures, see
[Troubleshooting](failure_modes.md).

---

**Want to see the install + first eval in 5 minutes?  Run `bash examples/demo_recording.sh`.**

---

## Next steps

| Resource | Link |
|----------|------|
| Full install options (all VLAs, sims, GPU notes) | [`docs/install.md`](install.md) |
| Add a new VLA or simulator | [`docs/extending.md`](extending.md) |
| Run sharded sweeps across GPUs | [`docs/sharded_runs.md`](sharded_runs.md) |
