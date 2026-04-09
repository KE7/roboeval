<div align="center">

# robo-eval

### Unified CLI for Vision-Language-Action Model Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

**One command to evaluate any VLA on any robotics benchmark. robo-eval manages the full stack — VLA servers, round-robin proxy, simulators, and parallel evaluation — so you focus on your model, not your infrastructure.**

[Quick Start](#quick-start) | [Documentation](docs/) | [Benchmark Results](#benchmark-results)

</div>

---

## Highlights

- **One CLI, Full Stack** — `robo-eval run` launches VLA servers, a round-robin load-balancing proxy, simulator workers, and evaluation processes from a single command. Stop juggling terminals.
- **Any VLA × Any Benchmark × Any Mode** — Evaluate across three configurable axes with no per-model or per-benchmark scripts:
  - **Benchmarks**: LIBERO, LIBERO-PRO, LIBERO-Infinity, RoboCasa, RoboTwin
  - **VLA policies**: Pi0.5, SmolVLA, OpenVLA — or [add your own](docs/adding_a_vla.md) in ~100 lines
  - **Modes**: Direct (VLA only) or planner-augmented (VLM + VLA)
- **Process Isolation via HTTP** — Each component (VLA, VLM, simulator) runs in its own virtualenv and process, communicating over JSON/HTTP. Different Python versions, different GPU allocations, zero import conflicts.
- **Parallel by Default** — Multi-replica VLA serving with automatic GPU assignment, concurrent task evaluation with auto port management, and structured JSON results with optional video recording.
- **Real Robot Ready** — Subclass `BaseWorldStub` to connect physical hardware; the evaluation loop works unchanged.
- **Extensible** — New VLAs and benchmarks plug in through documented interfaces ([adding a VLA](docs/adding_a_vla.md) · [adding a benchmark](docs/adding_a_benchmark.md)). [LIBERO-Infinity](docs/libero_infinity.md) (Scenic-based infinite test distributions) was added with zero CLI changes — just a backend class and a config entry.

---

## Quick Start

### Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.10+ (simulators may use older versions; `uv` manages them) |
| `uv` | Package manager — `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| GPU | NVIDIA with EGL support for headless rendering |
| Disk space | ~30 GB for model weights + simulator assets |

### Installation

```bash
# 1. Clone
git clone <this-repo> && cd robo-eval

# 2. Install the robo-eval CLI
uv pip install -e .

# 3. Set up a simulator (e.g., LIBERO)
bash scripts/setup_envs.sh --only libero

# 4. Set up a VLA (e.g., Pi0.5)
bash scripts/setup_vla.sh
```

### Run Your First Evaluation

```bash
# Evaluate Pi0.5 on LIBERO-spatial — robo-eval starts everything for you
robo-eval run --benchmark libero --vla pi05 --suites spatial --episodes 10 --mode direct
```

That's it. robo-eval starts the Pi0.5 policy server, launches simulator workers, runs the evaluation, collects structured JSON results, and tears everything down when finished.

### More Examples

```bash
# Full LIBERO benchmark (all 4 suites, 50 episodes per task)
robo-eval run --benchmark libero --vla pi05 --episodes 50 --mode direct

# Multi-GPU: 4 VLA replicas, one per GPU, with parallel task execution
robo-eval run --benchmark libero --vla smolvla --episodes 50 \
  --vla-replicas 4 --gpus 0,1,2,3 --parallel

# Planner-augmented mode (VLM decomposes task, VLA executes subtasks)
robo-eval run --benchmark libero --vla pi05 --episodes 10 --mode planner

# LIBERO-PRO OOD evaluation
robo-eval run --benchmark libero_pro --vla pi05 --episodes 10 --mode direct

# LIBERO-Infinity with Scenic perturbation distributions
robo-eval run --benchmark libero_infinity --vla smolvla \
  --suites spatial,object,goal --episodes 2 --max-tasks 2 --mode direct \
  --sim-args '{"perturbation":"position","seed":42}'

# Record video rollouts
robo-eval run --benchmark libero --vla pi05 --suites spatial \
  --episodes 5 --mode direct --record-video --record-video-n 3

# Dry run: see what would happen without executing
robo-eval run --benchmark libero --vla smolvla --episodes 50 --vla-replicas 4 --dry-run

# Check progress of a running evaluation
robo-eval status --results-dir results/my_run/

# Compare results across runs (generates markdown table with Wilson CIs)
robo-eval compare results/run_A/ results/run_B/

# List available benchmarks and suites
robo-eval suites
robo-eval suites --benchmark libero

# Server management
robo-eval servers list
robo-eval servers start smolvla
robo-eval servers stop smolvla
```

> See `robo-eval --help` for full CLI documentation.

---

## Benchmark Results

Cross-model evaluation on LIBERO and LIBERO-PRO benchmarks.

- **Pi0.5 Direct** — Pi0.5 executes the full task instruction directly with no VLM, using the built-in subtask-decomposition scaffold
- **Pi0.5 Reference Baseline** — parity run against standard [lerobot-eval](https://github.com/huggingface/lerobot) (no planner, no decomposition)
- **Pi0.5 + VLM Planner** — Planner-augmented mode: VLM (`gemini-3-flash-preview`) decomposes the task into subtasks, Pi0.5 executes each one
- **SmolVLA Direct** — SmolVLA direct (no planner, no VLM)
- **OpenVLA Baseline** — OpenVLA native eval using per-suite finetuned checkpoints

### LIBERO

All Pi0.5 and SmolVLA columns: **50 eps/task** (500 eps/suite). OpenVLA: evaluation incomplete — see notes below.

| Suite | Pi0.5 Direct | Pi0.5 Reference Baseline | Pi0.5 + VLM Planner | SmolVLA Direct | OpenVLA Baseline |
|-------|:---:|:---:|:---:|:---:|:---:|
| libero\_spatial | **90.2%** (451/500) | 87.8% (439/500) | 82.6% (413/500) | 78.6% (393/500) | INCOMPLETE<sup>†‡</sup>: 392/491 (79.8%) |
| libero\_object  | 90.4% (452/500) | **92.2%** (461/500) | 73.0% (365/500) | 91.8% (459/500) | INCOMPLETE<sup>†‡</sup>: 0/500 (0%) |
| libero\_goal    | **96.8%** (484/500) | 94.4% (472/500) | 81.2% (406/500) | 78.4% (392/500) | INCOMPLETE<sup>†‡</sup>: 0/500 (0%) |
| libero\_10      | 82.4% (412/500) | **85.2%** (426/500) | 72.8% (364/500) | 40.6% (203/500) | INCOMPLETE<sup>†‡</sup>: 0/500 (0%) |
| **Overall**     | **90.0%** (1799/2000) | 89.9% (1798/2000) | 77.4% (1548/2000) | 72.4% (1447/2000) | INCOMPLETE<sup>†‡</sup> |

Sources: Pi0.5 Direct → `results/pi05_50eps_col1/`; Pi0.5 Reference Baseline → `outputs/eval/2026-03-04/23-59-22_libero_pi05/` (spatial) + `outputs/eval/2026-03-05/` (object, goal, 10); Pi0.5 + VLM Planner → `results/pi05_50eps_col3_g3f/`; SmolVLA → `results/smolvla_50eps_col1_libero/`; OpenVLA → `scripts/run_openvla_native_eval.py`.

<sup>†</sup> **OpenVLA eval incomplete (15h timeout):** The OpenVLA libero evaluation hit the 15-hour timeout limit. `libero_spatial` ran 491/500 episodes with 392/491 successful (79.8%) before timeout; `libero_object`, `libero_goal`, and `libero_10` did not run due to timeout exhaustion. Additionally, a gripper parity fix was applied mid-run for `libero_spatial`, making those results potentially unreliable. A full re-run is planned with per-suite timeout allocations and all parity fixes applied from the start.

<sup>‡</sup> Results shown as partial completions; do not compare directly with other columns (all at 50 eps/task except where noted).

### LIBERO-PRO

Out-of-distribution generalization evaluation. Reference baseline (lerobot-eval) is N/A — lerobot-eval does not support LIBERO-PRO suites. Pi0.5: **50 eps/task**.

| Suite | Pi0.5 Direct | Pi0.5 + VLM Planner | SmolVLA Direct | OpenVLA Direct |
|-------|:---:|:---:|:---:|:---:|
| spatial\_object  | **93.8%** (469/500) | — | — | — |
| goal\_swap (OOD) | 11.8% (59/500) | — | — | — |
| with\_mug        | **84.8%** (424/500) | — | — | — |
| **Overall**      | **63.5%** (952/1500) | — | — | — |

Pi0.5 + VLM Planner LIBERO-PRO was obtained with the wrong VLM model and is excluded from this release. SmolVLA and OpenVLA LIBERO-PRO entries are marked `—` (not evaluated in this release).<sup>†</sup>

<sup>†</sup> `—` indicates the configuration was not evaluated in this release.

`goal_swap` is OOD-hard — Pi0.5 scores ~12% raw, consistent with LIBERO-PRO paper findings that VLAs collapse under semantic perturbations.

Sources: Pi0.5 Direct → `results/pi05_50eps_pro_col1/`.

> Full per-task breakdowns, per-episode logs, and methodology notes: [`docs/benchmark_results.md`](docs/benchmark_results.md).
>
> _Last updated: 2026-03-15. All results at 50 eps/task unless noted. OpenVLA at 20 eps/task (spatial) or 10 eps/task (libero\_10); no per-suite finetuned checkpoints available for object/goal._

---

## Architecture

robo-eval uses a modular client-server architecture. Each component runs as an independent HTTP service, enabling mix-and-match evaluation across VLAs, simulators, and planners without touching any internals.

```
robo-eval run  (unified CLI entry point)
 │
 ├── VLA policy servers (:5100/5101/5102)
 │     ├── pi05_policy.py       → Pi0.5 (action chunks)
 │     ├── openvla_policy.py    → OpenVLA (per-step)
 │     └── smolvla_policy.py    → SmolVLA (per-step)
 │
 ├── Round-robin proxy (:auto)
 │     └── load-balances across N VLA replicas
 │
 ├── VLM proxy (litellm :4000)
 │     └── Ollama / Vertex AI / OpenAI
 │
 └── Sim workers (sim_worker.py :5001+)
       ├── LiberoBackend         (.venvs/libero/)
       ├── LiberoProBackend      (.venvs/libero_pro/)
       ├── LiberoInfinityBackend (external .venv)
       ├── RoboCasaBackend       (.venvs/robocasa/)
       └── RoboTwinBackend       (.venvs/robotwin/)
```

### What `robo-eval run` Does

1. **Starts VLA servers** — launches N replicas across GPUs (configurable via `--vla-replicas` and `--gpus`)
2. **Starts round-robin proxy** — load-balances `/predict` requests across replicas for throughput
3. **Starts simulator workers** — one per concurrent task, each in its benchmark-specific virtualenv
4. **Runs evaluation** — parallel task execution with auto port management, structured JSON output, optional video recording
5. **Tears down** — stops all managed processes on completion or Ctrl-C

### Key Design Decisions

- **Process isolation via HTTP**: Each component runs in its own Python virtualenv (3.8 for LIBERO, 3.10 for RoboTwin, 3.11 for VLA/VLM). No shared memory, no import conflicts.
- **Standalone policy servers**: Each VLA runs as an independent FastAPI service with a uniform `/predict` interface. The orchestrator doesn't need to know model internals.
- **Auto port management**: Managed runs coordinate through `/tmp` lease files to avoid port races. Concurrent `robo-eval run` invocations coexist safely.

### Evaluation Modes

**Direct** — The VLA receives the full task instruction and executes it end-to-end. No planner, no VLM. The standard VLA evaluation approach.

```bash
robo-eval run --benchmark libero --vla pi05 --episodes 50 --mode direct
```

**Planner-augmented** — A VLM decomposes the task into subtask instructions executed sequentially by the VLA. Currently ships with a VLM-based planner (based on [LITEN](https://arxiv.org/abs/2510.19752)), which optionally feeds past episode experience back to improve planning over successive episodes.

```bash
robo-eval run --benchmark libero --vla pi05 --episodes 10 --mode planner
```

> VLA policy server contracts: [`docs/vla_policy_architecture.md`](docs/vla_policy_architecture.md)

---

## Supported VLAs and Benchmarks

### VLA Policies

| Model | Parameters | Checkpoint | Action Chunks | Suites |
|---|---|---|---|---|
| **Pi0.5** | 4B | `lerobot/pi05_libero_finetuned` | 50 steps | All LIBERO + PRO |
| **OpenVLA** | 7B | `openvla/openvla-7b-finetuned-libero-*` | 1 step | Per-suite checkpoints |
| **SmolVLA** | 0.45B | `HuggingFaceVLA/smolvla_libero` | 1 step | All LIBERO |

Adding a new VLA requires implementing a single FastAPI server with a `/predict` endpoint — see [`docs/adding_a_vla.md`](docs/adding_a_vla.md).

### Simulators

| Simulator | Tasks | Action Dim | Python | Description |
|---|---|---|---|---|
| **LIBERO** | 40 (4 suites × 10) | 7 | 3.8 | Standard manipulation benchmark |
| **LIBERO-PRO** | 30+ (OOD perturbations) | 7 | 3.8 | Robustness evaluation with position/task/semantic perturbations |
| **LIBERO-Infinity** | 40 base tasks × Scenic distributions | 7 | external `.venv` | Open-ended robustness evaluation with composable perturbation axes |
| **RoboCasa** | Kitchen tasks | 7 | 3.11 | Mobile-base kitchen manipulation |
| **RoboTwin** | Dual-arm tasks | 14 | 3.10 | Bimanual tabletop manipulation |

Adding a new benchmark requires implementing a simulator backend and registering it — see [`docs/adding_a_benchmark.md`](docs/adding_a_benchmark.md).

---

## Setup

### VLM Setup (Planner-Augmented Mode Only)

Required only when using `--mode planner`.

**Option A: Vertex AI (Gemini)**

```bash
export VERTEXAI_PROJECT=your-project
export VERTEXAI_LOCATION=us-central1
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
VLM_MODEL=vertex_ai/gemini-3-flash-preview bash scripts/start_vlm.sh
```

**Option B: Local Ollama + LiteLLM proxy**

```bash
ollama pull qwen3-vl:latest
uv venv .venvs/litellm --python 3.11
.venvs/litellm/bin/pip install litellm pillow
bash scripts/start_vlm.sh
```

**Option C: OpenAI directly**

```bash
echo "sk-..." > utils/openaikey.txt
```

### VLA Setup

```bash
# Install VLA dependencies (Pi0.5 + OpenVLA)
bash scripts/setup_vla.sh

# Start Pi0.5 policy server (port 5100, ~75s model load)
bash scripts/start_pi05_policy.sh
```

The Pi0.5 model (`lerobot/pi05_libero_finetuned`, ~7.4 GB) downloads from HuggingFace on first run.

Other policy servers:
```bash
bash scripts/start_openvla_policy.sh   # port 5101
bash scripts/start_smolvla_policy.sh   # port 5102
```

> **Note:** When using `robo-eval run`, VLA servers are started and stopped automatically — manual server management is only needed for debugging or custom setups.

### Simulator Setup

```bash
# Set up all simulators:
bash scripts/setup_envs.sh

# Or one at a time:
bash scripts/setup_envs.sh --only libero       # Python 3.8
bash scripts/setup_envs.sh --only libero_pro   # Python 3.8
bash scripts/setup_envs.sh --only robocasa     # Python 3.11
bash scripts/setup_envs.sh --only robotwin     # Python 3.10
```

For LIBERO-Infinity (creates a dedicated Python 3.11+ venv):

```bash
bash scripts/setup_envs.sh --only libero_infinity
# Or manually:
# python3.11 -m venv .venvs/libero_infinity
# .venvs/libero_infinity/bin/pip install "libero-infinity @ git+https://github.com/KE7/libero-infinity.git"
```

### Virtualenv Layout

```
.venvs/
  robo-eval/   Python 3.11  -- Core CLI + orchestrator
  litellm/     Python 3.11  -- LiteLLM proxy
  vla/         Python 3.11  -- Pi0.5 + OpenVLA policy servers
  smolvla/     Python 3.11  -- SmolVLA policy server (separate due to dep conflicts)
  libero/      Python 3.8   -- LIBERO simulator
  libero_pro/  Python 3.8   -- LIBERO-PRO simulator
  robocasa/    Python 3.11  -- RoboCasa kitchen tasks
  robotwin/    Python 3.10  -- RoboTwin dual-arm tasks
```

---

## Project Structure

```
.
├── robo_eval/                  # Core CLI package
│   ├── cli.py                  # Typer CLI (robo-eval run, servers, suites, compare, status)
│   ├── runner.py               # Evaluation orchestrator
│   ├── proxy.py                # Round-robin VLA proxy
│   ├── servers.py              # Server lifecycle management
│   ├── stack.py                # Full-stack start/stop coordination
│   ├── results.py              # Structured JSON result collection
│   └── config.py               # Benchmark/VLA/port configuration
│
├── run_sim_eval.py             # Single-task eval (used internally by robo-eval run)
├── run.py                      # Physical robot CLI entrypoint
├── world_stubs.py              # BaseWorldStub interface (act, reset, VLM queries)
├── run_utils.py                # Shared utilities (ICA saving, video)
│
├── vlm_hl/                     # VLM high-level reasoning
│   ├── vlm_methods.py          # Plan generation, assessment
│   └── prompts/                # Prompt templates
│
├── sims/                       # Simulator and VLA infrastructure
│   ├── env_wrapper.py          # SimWrapper (BaseWorldStub subclass for sim eval)
│   ├── sim_worker.py           # FastAPI sim server (all backends)
│   └── vla_policies/           # Standalone VLA policy servers
│       ├── pi05_policy.py      # Pi0.5 (port 5100)
│       ├── openvla_policy.py   # OpenVLA (port 5101)
│       └── smolvla_policy.py   # SmolVLA (port 5102)
│
├── ica/                        # In-Context Adaptation
│   └── reasoning_ica.py        # Experience data structures
│
├── scripts/                    # Setup and launch scripts
│   ├── setup_envs.sh           # Create all simulator virtualenvs
│   ├── setup_vla.sh            # Create VLA virtualenv
│   ├── start_vlm.sh            # Start LiteLLM VLM proxy
│   ├── start_sim.sh            # Start simulator worker
│   └── start_*_policy.sh       # Start individual VLA policy servers
│
├── docs/                       # Documentation
├── tests/                      # Unit and integration tests
└── .venvs/                     # Isolated virtualenvs (not in git)
```

---

## Advanced Usage

### Physical Robot (No Simulator)

Subclass `BaseWorldStub` and implement `act()` + `physical_reset()`:

```python
from world_stubs import BaseWorldStub

class MyRobotWorld(BaseWorldStub):
    def act(self, command: str):
        # Send command to VLA, execute actions on robot
        # Update self.current_image and self.subtask_frame_tuples
        ...

    def physical_reset(self):
        # Move robot to home, update self.current_image
        ...
```

Then run:
```bash
python run.py planner my_experiment
```

### Single-Task Mode (`run_sim_eval.py`)

For custom debugging setups or manual server management, run the lower-level orchestrator directly:

```bash
# Terminal 1 -- VLM proxy (only needed for planner-augmented mode)
bash scripts/start_vlm.sh

# Terminal 2 -- VLA policy server
bash scripts/start_pi05_policy.sh

# Terminal 3 -- Simulator (headless)
bash scripts/start_sim.sh --sim libero --port 5001 --headless

# Terminal 4 -- Run single-task evaluation
.venvs/litellm/bin/python run_sim_eval.py eval \
    --sim libero --task 0 --suite libero_spatial \
    --sim-url http://localhost:5001 --headless
```

<details>
<summary><strong>run_sim_eval.py CLI arguments</strong></summary>

| Argument | Default | Description |
|---|---|---|
| `--sim` | required | `libero`, `libero_pro`, `robocasa`, `robotwin` |
| `--task` | required | Task name (substring match) or numeric index |
| `--sim-url` | required | Simulator worker URL |
| `--suite` | `None` | LIBERO suite: `libero_spatial`, `libero_goal`, `libero_object`, `libero_10` |
| `--vlm-endpoint` | `localhost:4000` | LiteLLM proxy host:port |
| `--vlm-model` | `None` | VLM model name override |
| `--max-episodes` | `1` | Episodes to run |
| `--start-episode` | `0` | Episode index to start from (for resuming runs) |
| `--max-steps` | sim default | Max steps per subtask |
| `--camera-resolution` | `256` | Camera image resolution (square, pixels) |
| `--experience-dir` | `sim_experience` | ICA experience storage |
| `--save-videos/--no-save-videos` | `True` | Save rollout videos to `demo_videos/` (use `--no-save-videos` to disable) |
| `--record-video` | flag | Record structured episode videos to `--results-dir/videos/` |
| `--record-video-n` | `3` | Max episodes per task to record |
| `--results-dir` | `None` | Results directory for structured video and JSON output |
| `--seed` | `None` | Random seed for reproducibility (auto-generated if omitted) |
| `--sim-config` | `None` | Path to YAML file with extra sim configuration |
| `--headless` | `False` | EGL offscreen rendering (must match sim worker) |
| `--no-vlm` | flag | Skip VLM reasoning (task decomposition only) |
| `--no-think` | flag | Disable VLM thinking tokens |
| `--delta-actions` | flag | Use delta EEF control (required for LIBERO) |

</details>

### Windowed Debug Mode

```bash
robo-eval run \
  --benchmark libero --vla pi05 --suites spatial \
  --episodes 1 --mode direct --sequential \
  --tasks-parallel 1 --debug-window
```

---

## Service Port Map

| Service | Default Port | Start Script |
|---|---|---|
| LiteLLM VLM proxy | 4000 | `scripts/start_vlm.sh` |
| Pi0.5 policy server | 5100 | `scripts/start_pi05_policy.sh` |
| OpenVLA policy server | 5101 | `scripts/start_openvla_policy.sh` |
| SmolVLA policy server | 5102 | `scripts/start_smolvla_policy.sh` |
| Simulator worker | 5001+ | `scripts/start_sim.sh` |

> When using `robo-eval run`, ports are allocated automatically. The defaults above apply to manual server management only.

### Environment Variables

| Variable | Default | Used By |
|---|---|---|
| `LITELLM_PORT` | `4000` | `start_vlm.sh` |
| `VLM_MODEL` | `vertex_ai/gemini-3-flash-preview` | `start_vlm.sh` |
| `VLA_URL` | `http://localhost:5100` | benchmark scripts |
| `MUJOCO_GL` | `egl` | sim/VLA start scripts |
| `NO_THINK` | unset | Set to `1` to disable VLM thinking tokens |
| `LIBERO_INFINITY_ROOT` | *(auto-detected)* | Override path to `libero-infinity` (auto-resolved via importlib) |
| `ROBO_EVAL_LITELLM_VENV` | `.venvs/litellm` | Path to LiteLLM venv directory |
| `ROBO_EVAL_VLA_VENV` | `.venvs/vla` | Path to pi05/openvla venv directory |
| `ROBO_EVAL_SMOLVLA_VENV` | `.venvs/smolvla` | Path to smolvla venv directory |
| `ROBO_EVAL_LIBERO_VENV` | `.venvs/libero` | Path to LIBERO simulator venv |
| `ROBO_EVAL_LIBERO_PRO_VENV` | `.venvs/libero_pro` | Path to LIBERO-PRO simulator venv |
| `ROBO_EVAL_LIBERO_INFINITY_VENV` | `.venvs/libero_infinity` | Path to LIBERO-Infinity venv (Python 3.11+) |
| `ROBO_EVAL_ROBOCASA_VENV` | `.venvs/robocasa` | Path to RoboCasa simulator venv |
| `ROBO_EVAL_ROBOTWIN_VENV` | `.venvs/robotwin` | Path to RoboTwin simulator venv |
| `ROBO_EVAL_OPENVLA_NATIVE_VENV` | `.venvs/openvla_native` | Path to OpenVLA native eval venv |
| `ROBO_EVAL_DOCKER_GPU` | *(auto-detected)* | Docker GPU mode: `dri` (GB10/Jetson), `cdi`, or `gpus` |

---

## Docker Support

robo-eval supports running all components in Docker containers for reproducible evaluation environments:

```bash
# Build images
docker build -f docker/sim-libero.Dockerfile -t robo-eval/sim-libero:latest .
docker build -f docker/vla-lerobot.Dockerfile -t robo-eval/vla-lerobot:latest .

# Run with Docker backend
robo-eval run --benchmark libero --vla pi05 --episodes 10 --runtime docker
```

GPU passthrough is auto-detected: `--device=/dev/dri` on GB10/Jetson (unified memory ARM64), CDI on discrete GPUs with Container Toolkit, or classic `--gpus all` as fallback. Override with `ROBO_EVAL_DOCKER_GPU=dri|cdi|gpus`.

> Full Docker guide: [`docs/docker.md`](docs/docker.md)

---

## Documentation

| Document | Description |
|---|---|
| [`docs/adding_a_vla.md`](docs/adding_a_vla.md) | How to add a new VLA policy |
| [`docs/adding_a_benchmark.md`](docs/adding_a_benchmark.md) | How to add a new benchmark/simulator |
| [`docs/libero_infinity.md`](docs/libero_infinity.md) | LIBERO-Infinity integration guide (Scenic-based perturbation testing) |
| [`docs/docker.md`](docs/docker.md) | Docker runtime guide (GPU modes, GB10 support, standalone containers) |
| [`docs/benchmark_results.md`](docs/benchmark_results.md) | Complete benchmark tables with per-task breakdowns |
| [`docs/vla_policy_architecture.md`](docs/vla_policy_architecture.md) | VLA policy server interface contracts |

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT

---

<div align="center">
  <sub>Built for the robotics research community</sub>
</div>
