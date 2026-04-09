# LIBERO-Infinity Integration

[LIBERO-Infinity](https://github.com/KE7/libero-infinity) uses [Scenic 3](https://scenic-lang.readthedocs.io/) probabilistic programs to generate **infinite test distributions** from the standard LIBERO benchmark tasks. Instead of evaluating on fixed scenes, each episode samples a unique perturbed configuration — repositioned objects, additional distractors, combined perturbations — giving you open-ended robustness evaluation with statistical power.

robo-eval treats LIBERO-Infinity as a first-class benchmark. No CLI changes were needed to add it — just a new `LiberoInfinityBackend` class in `sims/sim_worker.py` and a `SimConfig` entry in `robo_eval/config.py`. The same `robo-eval run` command works out of the box.

---

## Prerequisites

1. **Install `libero-infinity`** into the LIBERO venv using the setup script:

   ```bash
   bash scripts/setup_envs.sh --only libero_infinity
   ```

   This creates a dedicated `.venvs/libero_infinity/` virtualenv (Python 3.11+) and installs `libero-infinity` with all its dependencies (including LIBERO and Scenic 3).

   Or install manually:

   ```bash
   python3.11 -m venv .venvs/libero_infinity
   .venvs/libero_infinity/bin/pip install "libero-infinity @ git+https://github.com/KE7/libero-infinity.git"
   .venvs/libero_infinity/bin/pip install "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
   ```

   For development (editable install from a local clone):

   ```bash
   DEV=1 bash scripts/setup_envs.sh --only libero_infinity
   ```

   > **Note:** Python 3.11+ is required (Scenic 3 dependency). Set `PYTHON311=/path/to/python3.11` if the interpreter is not on PATH.
   >
   > If you already have the `libero-infinity` repo cloned with its own `.venv/`, you can point robo-eval at it instead of creating a new venv:
   > ```bash
   > export ROBO_EVAL_LIBERO_INFINITY_VENV=/path/to/libero-infinity/.venv
   > ```

2. **GPU with EGL** — headless MuJoCo rendering, same as standard LIBERO.

3. **(Optional) `LIBERO_INFINITY_ROOT`** — only needed if you want to override package discovery. The config auto-detects the installed package via importlib; the env var is a manual override for non-standard layouts.

---

## Quick Start

```bash
# Evaluate Pi0.5 on all LIBERO-Infinity suites (50 episodes per task)
robo-eval run --benchmark libero_infinity --vla pi05 --episodes 50 --mode direct
```

That's it. robo-eval starts the Pi0.5 policy server, launches a sim worker using the `libero-infinity` virtualenv, runs evaluation with Scenic-sampled perturbations, and collects structured JSON results.

---

## Available Suites

LIBERO-Infinity mirrors the four standard LIBERO suites, each with 10 tasks:

| Suite | Qualified Name | Tasks | Max Steps | Description |
|---|---|---|---|---|
| spatial | `libero_infinity_spatial` | 10 | 300 | Spatial reasoning tasks with perturbed layouts |
| object | `libero_infinity_object` | 10 | 300 | Object manipulation with perturbed placements |
| goal | `libero_infinity_goal` | 10 | 300 | Goal-conditioned tasks with perturbed scenes |
| 10 | `libero_infinity_10` | 10 | 520 | Long-horizon tasks with perturbed configurations |

Run a specific suite:

```bash
robo-eval run --benchmark libero_infinity --vla pi05 --suites spatial --episodes 20
```

Run multiple suites:

```bash
robo-eval run --benchmark libero_infinity --vla pi05 --suites spatial,object,goal --episodes 10
```

---

## Perturbation Configuration

LIBERO-Infinity supports several perturbation types controlled via `--sim-args`. The JSON object is forwarded directly to the `LiberoInfinityBackend.init()` method as `sim_config`.

### Perturbation Types

| `perturbation` value | Description |
|---|---|
| `"position"` (default) | Randomize object positions within task-feasible regions |
| `"combined"` | Apply multiple perturbation axes simultaneously (position + distractors) |

### Sim-Args Reference

| Key | Type | Default | Description |
|---|---|---|---|
| `perturbation` | `str` | `"position"` | Perturbation type (see above) |
| `seed` | `int` | `42` | Base RNG seed for reproducible episode sampling |
| `max_distractors` | `int` | — | Maximum number of distractor objects to add |
| `min_distractors` | `int` | — | Minimum number of distractor objects |
| `max_steps` | `int` | `300` | Rollout horizon (overrides suite default) |
| `max_reset_attempts` | `int` | `5` | Max Scenic resample attempts per episode if physics settling fails |
| `post_reset_settle_steps` | `int` | `80` | Zero-action steps to let physics settle after scene sampling |

### Examples

**Default perturbation (position only):**

```bash
robo-eval run --benchmark libero_infinity --vla pi05 --episodes 50
```

**Combined perturbations with distractors:**

```bash
robo-eval run --benchmark libero_infinity --vla pi05 --episodes 50 \
  --sim-args '{"perturbation": "combined", "max_distractors": 3}'
```

**Reproducible evaluation with a specific seed:**

```bash
robo-eval run --benchmark libero_infinity --vla pi05 --episodes 50 \
  --sim-args '{"seed": 12345}'
```

**Limit to 2 suites and 3 tasks for a quick smoke test:**

```bash
robo-eval run --benchmark libero_infinity --vla smolvla \
  --suites spatial,object --max-tasks 3 --episodes 2 --mode direct \
  --sim-args '{"perturbation": "position", "seed": 42}'
```

---

## How It Works

### Architecture

LIBERO-Infinity plugs into robo-eval's simulator backend system. The `LiberoInfinityBackend` class in `sims/sim_worker.py` is registered via the `libero_infinity` `SimConfig` in `robo_eval/config.py`. The sim worker runs inside the shared LIBERO virtualenv (`.venvs/libero/`), since libero-infinity depends on libero + robosuite.

```
robo-eval run --benchmark libero_infinity
  │
  ├── VLA policy server (:5100)           # same as any benchmark
  │
  └── sim_worker.py --sim libero_infinity
        └── LiberoInfinityBackend
              ├── libero_infinity.task_config    → parse BDDL task file
              ├── libero_infinity.scenic_generator → generate .scenic program
              ├── scenic.scenarioFromFile()       → compile Scenic scenario
              ├── scenario.generate()             → sample perturbed scene (per episode)
              ├── libero_infinity.bddl_preprocessor → rewrite BDDL for sampled scene
              └── libero_infinity.simulator        → MuJoCo sim (LIBEROSimulation)
```

### Backend Lifecycle

Each evaluation task goes through this lifecycle:

1. **`init(task_name, suite, sim_config)`** — Resolves the BDDL file for the given (suite, task) pair, parses task metadata via `TaskConfig.from_bddl()`, generates a Scenic program via `generate_scenic_file()` with the requested perturbation type and distractor settings, and compiles the Scenic scenario once (expensive). Also performs a priming reset so the backend is ready for an immediate `/obs` call.

2. **`reset(episode_index)`** — Samples a new scene from the compiled Scenic scenario. Seeds the RNG deterministically as `sha256(run_seed:episode_index:attempt)` so each (run, episode) pair produces a unique but reproducible scene. Opens a BDDL context manager (which rewrites the BDDL with sampled object positions), creates a `LIBEROSimulation`, runs physics settling steps (zero-action control to let objects stabilize), and validates that no objects have toppled. If settling fails, retries up to `max_reset_attempts` times with different seeds.

3. **`step(action)`** — Forwards the 7-dim EEF delta action to the underlying `LIBEROSimulation.step_with_action()`. Returns (image, wrist_image, reward, done, info).

4. **`close()`** — Destroys the simulation, exits the BDDL context manager, and deletes the generated `.scenic` file.

### Deterministic Seeding

Each episode gets a deterministic seed derived from `sha256(run_seed:episode_index:attempt)`, ensuring:

- **Reproducibility**: the same `--sim-args '{"seed": 42}'` with the same episode indices produces identical scenes.
- **Uniqueness**: every episode samples a different perturbation from the Scenic distribution.
- **Retry safety**: if a sampled scene fails physics validation, the retry uses a different seed (via the `attempt` counter).

### Relationship to Standard LIBERO

LIBERO-Infinity uses the **same 40 base tasks** (4 suites x 10 tasks) as standard LIBERO. The difference is that standard LIBERO always resets to the same fixed initial scene, while LIBERO-Infinity samples a new scene from a Scenic distribution each episode. This means:

- Same task descriptions and success conditions
- Same robot (Panda), action space (7-dim EEF delta), and camera setup
- Different initial object positions, orientations, and (optionally) distractor objects per episode
- Results are directly comparable: standard LIBERO gives a point estimate on fixed scenes, LIBERO-Infinity gives a distributional estimate across perturbations

---

## Configuration Reference

### SimConfig (robo_eval/config.py)

```python
"libero_infinity": SimConfig(
    name="libero_infinity",
    venv=os.environ.get("ROBO_EVAL_LIBERO_INFINITY_VENV", ".venvs/libero_infinity"),
    env_vars={"MUJOCO_GL": "egl"},
)
```

The sim worker uses a dedicated Python 3.11+ venv (`.venvs/libero_infinity/`) since `libero-infinity` depends on Scenic 3 which requires Python 3.11+. `LIBERO_INFINITY_ROOT` is resolved automatically via importlib when the package is installed; no env var is required.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ROBO_EVAL_LIBERO_INFINITY_VENV` | `.venvs/libero_infinity` | Path to the LIBERO-Infinity virtualenv (Python 3.11+, separate from the base LIBERO venv) |
| `LIBERO_INFINITY_ROOT` | *(auto-detected)* | Override: path to a libero-infinity checkout. Auto-resolved via importlib → sibling dir fallback. |
| `MUJOCO_GL` | `egl` | Set automatically by the SimConfig; use `egl` for headless GPU rendering |

### Suite Registration (robo_eval/config.py)

```python
BENCHMARK_SUITES = {
    "libero_infinity": ["spatial", "object", "goal", "10"],
    ...
}
```

Suite names are qualified automatically: `qualify_suite("libero_infinity", "spatial")` returns `"libero_infinity_spatial"`.

---

## Extending

LIBERO-Infinity demonstrates robo-eval's benchmark extensibility. Adding it required:

1. **A backend class** (`LiberoInfinityBackend` in `sims/sim_worker.py`) — ~450 lines implementing the standard `init/reset/step/close` contract
2. **A SimConfig entry** in `robo_eval/config.py` — 4 lines pointing to the shared libero venv
3. **Suite registration** in `BENCHMARK_SUITES` — 1 line listing the suite names
4. **A setup function** in `scripts/setup_envs.sh` — `pip install` into the libero venv

**Zero CLI changes.** The `robo-eval run --benchmark libero_infinity` command worked immediately because the CLI dispatches based on the benchmark name in the config registry.

**Install path:** `libero-infinity` is distributed as a pip package from GitHub, installed into its own Python 3.11+ venv (`.venvs/libero_infinity/`). The `setup_envs.sh --only libero_infinity` target handles venv creation and installation automatically.

For details on adding your own benchmark, see [`docs/adding_a_benchmark.md`](adding_a_benchmark.md).
