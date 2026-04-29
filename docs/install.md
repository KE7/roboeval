# Install Guide

This guide covers every path from a fresh clone to a working roboeval
evaluation — from the one-command quick path to a fully manual per-venv
setup.

---

## Table of Contents

1. [Quick path](#quick-path)
2. [Per-VLA notes](#per-vla-notes)
3. [Per-simulator notes](#per-simulator-notes)
4. [Platform notes](#platform-notes)
5. [Troubleshooting](#troubleshooting)

---

## Quick path

For most users starting with Pi 0.5 + LIBERO:

```bash
git clone https://github.com/KE7/roboeval.git
cd roboeval
roboeval setup pi05 libero
roboeval serve --vla pi05 --sim libero --headless
roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
```

## Installation prerequisites

### Required

Some VLAs use micromamba envs instead of uv venvs because their CUDA-compiled deps ship as binaries on conda-forge but only as sdists on PyPI. `roboeval setup` handles both transparently. See README.md prerequisites for installing micromamba.

| Requirement | Notes |
|---|---|
| **Python 3.13** | For the orchestrator venv (auto-provisioned by `roboeval setup`). VLA venvs use Python 3.11/3.12 per upstream constraints. LIBERO/LIBERO-Pro require Python 3.8; RoboTwin requires Python 3.10. The orchestrator package itself supports Python ≥3.11 for manual installs. |
| **[uv](https://docs.astral.sh/uv/)** | Astral's package manager. `roboeval setup` installs it if missing; or: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **[micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)** | Lightweight conda-forge manager needed for VLAs that require binary CUDA packages (flash-attn, libcudss). Auto-installed by `roboeval setup` when needed; or manually: `"${SHELL}" <(curl -L micro.mamba.pm/install.sh)` |
| **Git** | Needed to clone non-PyPI packages (LIBERO, robosuite, RoboTwin, etc.). |

### GPU / CUDA

All VLA backends require a CUDA-capable NVIDIA GPU.

| VLA | Minimum VRAM | Framework |
|---|---|---|
| Pi 0.5 | 8 GB | PyTorch |
| Pi 0 | 8 GB | PyTorch |
| OpenVLA 7B | 14 GB | PyTorch |
| SmolVLA | 6 GB | PyTorch |
| GR00T-N1.6-3B | 8 GB | PyTorch |
| InternVLA-A1 | 16 GB | PyTorch |

Verify CUDA:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

### EGL (headless rendering)

LIBERO and RoboCasa use MuJoCo, which requires EGL for headless rendering on
servers without a display.

```bash
sudo apt-get install libegl1-mesa-dev libgl1-mesa-dev libosmesa6-dev
```

Verify after installing a sim venv:

```bash
MUJOCO_GL=egl .venvs/libero/bin/python -c "import mujoco; print('MuJoCo OK')"
```

If MuJoCo falls back to software rendering, simulations will be slow and
may time out.

### Disk space

| Component | Download size (approx.) |
|---|---|
| Pi 0.5 model | ~7.4 GB |
| Pi 0.5 base model | ~8 GB |
| Pi 0 (LIBERO finetuned) | ~7.4 GB |
| OpenVLA 7B | ~14 GB |
| SmolVLA | ~2–4 GB |
| GR00T-N1.6-3B | ~6 GB |
| InternVLA-A1 | ~16 GB |
| LIBERO datasets | ~5 GB per suite |
| RoboCasa assets | ~10 GB |

Plan for **≥55 GB** free for a full install.

---

## Quick path details

```bash
git clone https://github.com/KE7/roboeval.git
cd roboeval
roboeval setup pi05 libero   # installs only Pi 0.5 + LIBERO
```

The script creates isolated venvs under `.venvs/`:

```
.venvs/
  roboeval/    ← orchestrator CLI (always created)
  pi05/        ← Pi 0.5 VLA server
  libero/      ← LIBERO simulator server
```

After setup:

```bash
# Terminal 1 — orchestrator
source .venvs/roboeval/bin/activate
roboeval --help

# Terminal 2 — VLA policy server + simulator worker
roboeval serve --vla pi05 --sim libero --headless

# Terminal 1 — validate and run the reproducible smoke invocation
roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
roboeval run -c configs/libero_spatial_pi05_smoke.yaml
```

`serve` is flag-driven for interactive launch. `run` is config-driven so the
YAML captures the full supported pair spec: action format, embodiment tag, port
wiring, output directory, and optional LITEN endpoint.

### Component list

```
VLAs (Python 3.11):
  pi05        Pi 0.5 via LeRobot
  pi0         Pi 0 via LeRobot (predecessor to pi05; pairs for pi0-vs-pi05 eval)
  openvla     OpenVLA 7B via HuggingFace Transformers
  smolvla     SmolVLA via LeRobot
  groot       GR00T-N1.6 (NVIDIA) — see Per-VLA notes
  internvla   InternVLA-A1 (InternRobotics) — setup auto-clones fork

Sims:
  libero      LIBERO benchmark (Python 3.8, MuJoCo)
  libero_pro  LIBERO-Pro extended suites (Python 3.8)
  robocasa    RoboCasa kitchen envs (Python 3.11, MuJoCo)
  robotwin    RoboTwin 2.0 bimanual (Python 3.10, SAPIEN) — see Per-simulator notes
  aloha_gym   gym-aloha bimanual ALOHA (Python 3.10, MuJoCo) — pure uv, aarch64-compatible

Mode extras:
  vlm         LiteLLM VLM proxy for hierarchical-planner mode

  all         Everything (installs all of the above)
```

Multiple components can be specified together:

```bash
roboeval setup smolvla openvla libero robocasa
roboeval setup all
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ROBOEVAL_VENDORS_DIR` | `~/.local/share/roboeval/vendors` | Where non-PyPI repos are cloned |
| `LIBERO_PRO_DIR` | `$VENDORS_DIR/LIBERO-PRO` | Override LIBERO-PRO clone path |
| `SKIP_SYSTEM_DEPS` | `0` | Set to `1` to skip `apt-get` installs |
| `DRY_RUN` | `0` | Set to `1` to print the plan without executing |

---

## Manual per-venv setup

For advanced users who want full control.

### 3.1 Orchestrator (always needed)

```bash
uv venv .venvs/roboeval --python 3.13
uv pip install --python .venvs/roboeval/bin/python -e .
source .venvs/roboeval/bin/activate
roboeval --help
```

### 3.2 VLA venvs (Python 3.11)

Replace `<vla>` with one of: `pi05`, `pi0`, `openvla`, `smolvla`, `groot`,
`internvla`.

```bash
uv venv .venvs/<vla> --python 3.11
uv pip install --python .venvs/<vla>/bin/python -e ".[<vla>]"
```

### 3.3 LIBERO / LIBERO-Pro venvs (Python 3.8)

`requires-python = ">=3.11"` in `pyproject.toml` means `-e .[libero]` won't
work in a Python 3.8 venv. Install deps directly:

```bash
uv venv .venvs/libero --python 3.8
uv pip install --python .venvs/libero/bin/python \
    "numpy>=1.24" "pillow>=9.0" "fastapi>=0.100" "uvicorn[standard]>=0.22" \
    "h5py>=3.8" "bddl>=3.0.0" "mujoco>=2.3.7" \
    "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@master" \
    "libero @ git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
```

### 3.4 RoboCasa venv (Python 3.11)

```bash
uv venv .venvs/robocasa --python 3.11
uv pip install --python .venvs/robocasa/bin/python -e ".[robocasa]"
```

### 3.5 RoboTwin venv (Python 3.10)

See [RoboTwin notes](#robotwin) for the full multi-step install.

---

## Per-VLA notes

### pi05

**Python 3.12 required** — `roboeval setup` creates the pi05 venv with Python 3.12.
`lerobot[pi]` 0.4.5 (the supported version) uses `Python>=3.10`, but
`lerobot>=0.5.0` on PyPI requires `Python>=3.12`. To avoid ambiguity,
`roboeval setup` always uses 3.12 for pi05.

**Model IDs**

| Model | Use case | VRAM |
|---|---|---|
| `lerobot/pi05_libero_finetuned` | LIBERO tasks | 8 GB |
| `lerobot/pi05_base` | DROID / general tasks | 8 GB |

Models are downloaded automatically to `~/.cache/huggingface` on first use.

**lerobot version pinning**

`roboeval setup` installs `lerobot[pi]` from the `v0.4.5` git tag. This version
uses the `fix/lerobot_openpi` branch of transformers (not PyPI transformers).
Do not upgrade lerobot to ≥0.5.0 in the pi05 venv: it pins transformers≥5.3
which breaks openvla and smolvla.

<details>
<summary>PyTorch dynamo workaround</summary>

On PyTorch 2.10, `torch.where(int64_mask, ...)` fails inside `torch.compile`.
The Pi 0.5 policy server automatically unwraps `compiled_sample_actions` at
load time. If you see a dynamo error, check that you're using the server in
`sims/vla_policies/pi05_policy.py` (not a raw lerobot eval script).

</details>

**Start command**

```bash
roboeval serve --vla pi05 --model-id lerobot/pi05_libero_finetuned
```

---

### openvla

**CRITICAL: `transformers==4.40.1` (exact pin)**

Two distinct failure modes justify the exact pin:

1. **Import error** — `AutoModelForVision2Seq` was removed in transformers 5.0.
2. **Image-conditioning incompatibility** — transformers 4.41–4.57+ can break
   `predict_action` image conditioning without raising an error. The root cause is
   that `modeling_prismatic.py` in the openvla checkpoint was written against the
   4.40.1 API.

Never upgrade transformers in the openvla venv. If you see `pip` or `uv pip`
pulling a newer version, force-reinstall:

```bash
.venvs/openvla/bin/pip install "transformers==4.40.1" --force-reinstall
```

<details>
<summary>Attention implementation compatibility note</summary>

**Do not** pass `attn_implementation="eager"` when loading the model.
That flag is only needed for transformers≥4.50 to work around an SDPA compat
issue; on 4.40.1 it triggers a causal-mask shape mismatch
(`RuntimeError: tensor a (291) vs b (290) at dim 3`).

</details>

**Model**

`openvla/openvla-7b` (~14 GB). Downloaded on first use.

**Start command**

```bash
roboeval serve --vla openvla --model-id openvla/openvla-7b
```

**unnorm-key**

The unnorm key selects the action-space normalisation statistics baked into the
checkpoint. Common values: `libero_spatial`, `libero_object`, `libero_goal`,
`libero_10`. Use the key that matches your task suite.

---

### smolvla

SmolVLA uses lerobot's `make_pre_post_processors` for byte-for-byte compatible
preprocessing with the lerobot native eval pipeline.

**lerobot version pinning**

The `smolvla` extra pins `lerobot[smolvla]==0.4.4` — the last PyPI release with
`Python>=3.10` support and `transformers<5.0`. Do not upgrade to `lerobot>=0.5`
in this venv.

**Start command**

```bash
roboeval serve --vla smolvla
```

---

### groot

`gr00t` (NVIDIA Isaac-GR00T) is **not on PyPI**. `roboeval setup` auto-clones
[NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) into
`~/.local/share/roboeval/vendors/Isaac-GR00T`, **checks out the pinned
N1.6 revision**, and installs it editable
into `.venvs/groot`.

> **micromamba env** — unlike every other VLA venv, `.venvs/groot` is created
> by micromamba (conda-forge), not uv. See
> [§ Why uv for most / micromamba for groot](#why-two-tools) below.

```bash
roboeval setup groot           # clones + checks out e29d8fc + installs gr00t
```

If you already have a local clone (or work behind a firewall), pass it directly:

```bash
roboeval setup groot --with-isaac-groot=/path/to/Isaac-GR00T
# or equivalently:
ISAAC_GROOT_PATH=/path/to/Isaac-GR00T roboeval setup groot
```

The clone must contain the pinned N1.6 revision; `roboeval setup` will refuse to
install from an incompatible shallow clone.

**Disk requirement:** the Isaac-GR00T clone is approximately **~5 GB** (model
checkpoints are downloaded separately on first use and are **not** included in
the clone).

**What gets installed automatically**

- The `gr00t` Python package, editable, with `--no-deps` (so the venv's
  existing CUDA-enabled torch is preserved).
- N1.6's Python-only inference deps pinned to match gr00t's own `pyproject.toml`:
  **`transformers==4.51.3`** (the critical pin — eliminates all
  N1.7-era vendor patches), `diffusers==0.35.1`, `peft==0.17.1`,
  `albumentations==1.4.18`, `numpy==1.26.4`, etc.
- Roboeval's own `[groot]` extras (FastAPI server + image utils + aarch64
  lib provisioners `nvpl==25.11` and `nvidia-cudss-cu13`).

After `roboeval setup groot`, check the install with:

```bash
.venvs/groot/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy; from gr00t.data.embodiment_tags import EmbodimentTag; print('gr00t N1.6 OK')"
```

**Python version**: gr00t's own `pyproject.toml` pins
`requires-python = "==3.10.*"` and `torch==2.7.1`.  Neither version is available
as a CUDA-capable wheel on PyPI for aarch64, so `roboeval setup` creates `.venvs/groot`
as a **micromamba env** (Python 3.12) and installs
`pytorch=2.10.0` + `flash-attn=2.8.3` from conda-forge.
gr00t itself is installed with `--no-deps --ignore-requires-python` so the
conda-forge torch is preserved; transformers, diffusers, and peft are pinned to
N1.6's exact versions so the model code runs unmodified.

<a name="why-two-tools"></a>

#### Why uv for most VLAs / micromamba for groot

roboeval uses **two package-management tools** intentionally:

- **uv** (default, 14+ venvs): 10× faster than pip/conda for pure-PyPI closures,
  requires no solver overhead, and handles every VLA except groot correctly.
- **micromamba** (groot only): PyPI does **not** ship aarch64 binary wheels for
  `flash-attn` — only an sdist whose source build is slow on aarch64 and
  frequently fails on gcc ≥ 13.  conda-forge ships pre-built `flash-attn=2.8.3`
  and `pytorch=2.10.0` for aarch64+CUDA13 as ordinary binary packages.  micromamba
  is the lightest path to conda-forge (~3 MB single binary, no base env, no GUI)
  without bringing full conda/mamba install overhead.  The resulting env exposes a
  standard `bin/python`, so all subsequent `uv pip install` calls into it work
  without modification.

**Per-VLA environment type:**

| VLA | Env type | Notes |
|---|---|---|
| pi05 | uv | Pure PyPI; lerobot[pi] + torch from PyTorch index |
| openvla | uv | Pure PyPI; transformers==4.40.1 exact pin |
| smolvla | uv | Pure PyPI; lerobot==0.4.4 |
| **groot** | **micromamba (conda-forge)** | flash-attn=2.8.3 + pytorch=2.10.0 from conda-forge |
| internvla | uv | SDPA default; no flash-attn needed for correct inference |
| diffusion_policy | uv | Pure PyPI |
| vqbet | uv | Pure PyPI |
| act | uv | Pure PyPI |
| tdmpc2 | uv | Pure PyPI |

**Why N1.6.** N1.6 (`gr00t/model/gr00t_n1d6/`, Eagle-Block2A-2B-v2 backbone)
is available from the public upstream repository, loads cleanly under
transformers 4.51.3 with **zero vendor patches**, and matches the closure that
the public community LIBERO fine-tune (below) was trained against.

**Default checkpoints**

| Checkpoint | Embodiment tag | Paired sim | Action space | Notes |
|---|---|---|---|---|
| `0xAnkitSingh/GR00T-N1.6-LIBERO` | `LIBERO_PANDA` | libero | 7-dim EEF delta | Community fine-tune; reportedly **94.9 % LIBERO avg** (Spatial 96.6 / Object 98.4 / Goal 96.8 / Long 87.8) per the upstream README. |
| `nvidia/GR00T-N1.6-3B` | `ROBOCASA_PANDA_OMRON` | robocasa | 12-dim mobile-base EEF delta | Foundation model; no upstream RoboCasa fine-tune was published for this pairing. |

The `ROBOCASA_PANDA_OMRON` enum entry is **native to N1.6**
(`gr00t/data/embodiment_tags.py`); no enum patch is required, unlike under N1.7.

The `nvidia/Eagle-Block2A-2B-v2` backbone is **shipped inside the
Isaac-GR00T clone** (Python source + JSON/tokenizer data files in
`gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/`).  Both checkpoints above
use this backbone — no license-gated downloads and no `huggingface-cli login`
required.

**Default model** is `0xAnkitSingh/GR00T-N1.6-LIBERO` with embodiment tag
`LIBERO_PANDA`.  Override via `GROOT_MODEL_ID` and `GROOT_EMBODIMENT_TAG`.

To run GR00T-N1.6 against LIBERO:

```bash
GROOT_MODEL_ID=0xAnkitSingh/GR00T-N1.6-LIBERO \
GROOT_EMBODIMENT_TAG=LIBERO_PANDA \
  roboeval serve --vla groot --sim libero --headless

roboeval run -c configs/libero_spatial_groot_smoke.yaml
```

The smoke YAML is the reproducible `roboeval run` invocation for this pair,
including the LIBERO action format, `LIBERO_PANDA` embodiment tag, server port,
and output path.

RoboCasa x GR00T remains a documented capability boundary in v0.1.0 and does
not ship as a root config.

No new venv is needed — `.venvs/groot` (Python 3.10) supports both checkpoints.

<details>
<summary>aarch64 NVIDIA Performance Libraries (libnvpl)</summary>

On an aarch64 GPU host, GR00T's scipy dependency requires
`libnvpl_lapack_lp64_gomp.so.0`.  `roboeval setup` installs `nvpl==25.11` via uv pip
on aarch64; the wheel ships `libnvpl_*.so` files inside `site-packages`
where the dynamic linker finds them automatically.  On x86_64 the wheel is a
no-op stub (~8 kB).

No manual steps required — handled automatically inside the `=== groot_libnvpl ===`
block of `roboeval setup groot`.

> **Note on libcudss:** `nvidia-cudss-cu13` is no longer installed separately.
> conda-forge's `pytorch=2.10.0` pulls `libcudss` and other CUDA runtime deps as
> conda package dependencies automatically (vs. the old Jetson SBSA torch build,
> which needed a manual `nvidia-cudss-cu13` PyPI wheel to satisfy a baked-in RPATH).

</details>

**Start command**

```bash
GROOT_MODEL_ID=nvidia/GR00T-N1.6-3B \
GROOT_EMBODIMENT_TAG=robocasa_panda_omron \
  roboeval serve --vla groot
```

---

### internvla

InternVLA-A1 uses a patched lerobot fork that is **not on PyPI**. `roboeval setup
internvla` clones that fork into `$ROBOEVAL_VENDORS_DIR/InternVLA-A1`, installs
it with the dependency constraints needed by the policy server, and patches the
known DynamicCache API mismatch idempotently. No manual clone step is required.

**Memory requirements**

InternVLA-A1-3B is hardcoded to **float32** (~12 GB weights). The flow-matching
denoise loop in the action head requires fp32 — bfloat16 introduces rounding
error in the intermediate ODE trajectory. fp16 and bf16 paths are intentionally
not exposed.


**Start command**

```bash
roboeval serve --vla internvla
```

---

### act

ACT (Action Chunking Transformer) ships via the `lerobot` package — no manual
install step required.  The checkpoint downloads automatically from HuggingFace
on first use (~300 MB for the aloha checkpoints).

**Model IDs**

| Model | Task | VRAM |
|---|---|---|
| `lerobot/act_aloha_sim_transfer_cube_human` | AlohaTransferCube-v0 (default) | ~2 GB |
| `lerobot/act_aloha_sim_insertion_human` | AlohaInsertion-v0 | ~2 GB |

**Action space**

ACT predicts **absolute joint positions** (14-dim: 7 joints × 2 arms).  This is
a `joint_pos_abs` contract — distinct from the `eef_delta` contract used by
LIBERO-trained VLAs (Pi0.5, SmolVLA, OpenVLA).  Always pair ACT with the
`aloha_gym` sim; do not mix with LIBERO sims.

**Chunk size**

ACT generates 100 actions per forward pass and replays them step-by-step.  The
`reset()` call (triggered by the orchestrator at each episode boundary) flushes
the internal chunk buffer.

**Setup**

```bash
roboeval setup act
```

This creates `.venvs/act` (Python 3.11) and installs `lerobot>=0.4.4` plus
`torch`.

**Start command**

```bash
roboeval serve --vla act --model-id lerobot/act_aloha_sim_transfer_cube_human
```

**Pair with gym-aloha**

ACT is designed for the bimanual ALOHA sim.  Start the gym-aloha worker
alongside the ACT policy server:

```bash
# Terminal 1 — ACT VLA server + gym-aloha sim worker
roboeval serve --vla act --sim aloha_gym --headless

# Terminal 2 — orchestrator
roboeval run -c configs/aloha_gym_act_smoke.yaml
```

---

<!-- === START tdmpc2 === -->
### tdmpc2

<a name="tdmpc2"></a>

TDMPC2 (**Temporal-Difference Model-Predictive Control v2**, Hansen et al. 2024)
is a **model-based RL policy** in the shipped VLA roster — every other
shipped VLA is pure imitation / behaviour-cloning.  TDMPC2 learns a latent
world model jointly with a Q-function and an action prior, then plans
short-horizon trajectories with **MPPI** at every step (falling back to the
learned Q-value beyond the planning horizon).

**Default checkpoint:** `nicklashansen/tdmpc2` (HuggingFace, **MIT-licensed**) —
the official MT80 metaworld release from the upstream authors.  Trained on
80 metaworld manipulation tasks; emits **4-dim Sawyer eef-delta**
(`[dx, dy, dz, gripper]`) — an exact match for the metaworld sim backend.

| Checkpoint | Tasks | Size | Action |
|---|---|---|---|
| `nicklashansen/tdmpc2` | MT80 (80 metaworld tasks) | ~150 MB | 4-dim eef-delta |

**Why this pairing matters.** The metaworld backend expects 4-dim actions, while
most shipped VLAs emit 7-dim actions. TDMPC2 emits the compatible
`eef_delta_xyz_gripper` × 4-dim contract, so the orchestrator's `ActionObsSpec`
gate accepts this pairing.

**Install:**

```bash
roboeval setup tdmpc2 metaworld
```

This creates `.venvs/tdmpc2` (Python 3.11) and installs `lerobot>=0.4.4` plus
torch+cuda.  No TF dependency (unlike octo's `dlimp` cascade) — pure Python +
torch keeps the install path clean on aarch64.

**Run smoke:**

```bash
# Terminal 1 — TDMPC2 policy server + metaworld sim worker
roboeval serve --vla tdmpc2 --sim metaworld --headless

# Terminal 2 — orchestrator
roboeval run -c configs/metaworld_tdmpc2_smoke.yaml
```

**Loading strategy.**  The server tries lerobot's TDMPC2 wrapper at
`lerobot.policies.tdmpc2.modeling_tdmpc2.TDMPC2Policy` first, then the
older `lerobot.common.policies.tdmpc2.modeling_tdmpc2` path, and finally
falls back to the upstream `tdmpc2` package (Nicklas Hansen) — see
`load_model()` in `sims/vla_policies/tdmpc2_policy.py`.

**Paradigm note.**  TDMPC2 is task-conditioned via a task-ID embedding in
its policy prior, not free-text language.  The server accepts the
`instruction` field but ignores it; the task to perform is selected by the
metaworld backend's `task` config field (e.g. `button-press-v2`).

<!-- === END tdmpc2 === -->

### vqbet

<a name="vqbet"></a>

VQ-BeT (**Vector-Quantized Behavior Transformer**) is a behavior-cloning policy
that vector-quantizes actions via a VQ-VAE codebook and then predicts action
codes with a transformer.  The original BeT paper used **PushT** as its
canonical evaluation — the same benchmark Diffusion Policy targets — making
VQ-BeT vs Diffusion Policy a textbook architectural comparison on identical
gym_pusht observations and actions.

**Why VQ-BeT (replacing octo)**

The `[octo]` extra was removed in v0.1 because octo's transitive dependency
`dlimp==0.0.1` upstream-pins TF==2.15.0, which on aarch64 installs
`tensorflow_cpu_aws` — an empty stub with no `tf.image` module — breaking
`from octo.model.octo_model import OctoModel` at import time.  VQ-BeT is
shipped in its place: a vector-quantized action family that is **architecturally
distinct** from both Diffusion Policy (DDPM denoising) and pi05/openvla
(autoregressive transformer over continuous actions).  Octo restoration is
tracked for v0.1.1 if dlimp upstream relaxes its TF pin.

**Python 3.11 required.**  VQ-BeT lives in its own `.venvs/vqbet` (override via
`ROBOEVAL_VQBET_VENV`).  No special extras beyond the bundled `lerobot==0.4.4`.

**Model IDs**

| Model | Use case | VRAM |
|---|---|---|
| `lerobot/vqbet_pusht` | gym_pusht 2-D PushT pushing task | <1 GB |

**Action space**

2-dim absolute (x, y) end-effector position in PushT coordinates — identical
to Diffusion Policy on PushT.  This deliberately matches Diffusion Policy so
the two policies are drop-in interchangeable for head-to-head comparison.

**Setup**

```bash
roboeval setup vqbet
```

**Start command**

```bash
roboeval serve --vla vqbet
```

**Smoke run**

```bash
# Terminal 1 — VQ-BeT VLA server + gym_pusht sim worker
roboeval serve --vla vqbet --sim gym_pusht --headless

# Terminal 2 — orchestrator
roboeval run -c configs/gym_pusht_vqbet_smoke.yaml
```

**VQ-BeT vs Diffusion Policy head-to-head**

Run the two configs back-to-back on the identical gym_pusht task:

```bash
roboeval run -c configs/gym_pusht_diffusion_policy_smoke.yaml
roboeval run -c configs/gym_pusht_vqbet_smoke.yaml
```

Each config is a reproducible `roboeval run` invocation for comparing
vector-quantized codes vs DDPM denoising on the same canonical PushT task.

---

## Per-simulator notes

### libero

**Python 3.8 is required.** LIBERO's MuJoCo bindings do not support 3.9+.

**Datasets**

LIBERO datasets (libero_spatial, libero_object, libero_goal, libero_10) are
not downloaded automatically. Obtain them from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
and symlink them under `~/.libero/`:

```bash
mkdir -p ~/.libero
ln -s /path/to/libero_spatial ~/.libero/libero_spatial
# repeat for each suite
```

**Headless rendering**

```bash
roboeval serve --vla pi05 --sim libero --headless
```

If you see `Failed to initialize OpenGL` or a blank observation, check that
`libegl1-mesa-dev` is installed and your NVIDIA driver is loaded.

---

### libero_pro

Same Python 3.8 requirement as LIBERO.

**LIBERO-PRO clone**

`roboeval setup` clones `https://github.com/Zxy-MLlab/LIBERO-PRO.git` into
`$ROBOEVAL_VENDORS_DIR/LIBERO-PRO`. If you already have a clone, set:

```bash
export LIBERO_PRO_DIR=/path/to/your/LIBERO-PRO
roboeval setup libero_pro
```

<details>
<summary>HDF5 native libraries on aarch64</summary>

On ARM64 machines, h5py may fail to build because pip's bundled HDF5 does not
include the native BLAS/HDF5 libraries. Fix:

```bash
micromamba install -n libero_libs -c conda-forge hdf5 -y
export LD_LIBRARY_PATH=~/.micromamba/envs/libero_libs/lib:$LD_LIBRARY_PATH
```

Add the `LD_LIBRARY_PATH` export to the script or shell that starts the
libero_pro sim worker.

</details>

---

<!-- === START libero_infinity === -->
### libero_infinity

LIBERO-Infinity extends LIBERO with Scenic-based perturbation testing — tasks
are generated on-the-fly by sampling random object placements, distractors, and
lighting conditions from a Scenic scene description.

**Python version**

`libero-infinity` (PyPI) requires **Python ≥ 3.11**.  `roboeval setup` creates a
dedicated `.venvs/libero_infinity` venv at Python 3.11 that contains both the
LIBERO upstream simulator and the `libero-infinity` package side by side.

```bash
roboeval setup libero_infinity
```

**LIBERO clone**

`roboeval setup` clones `https://github.com/Lifelong-Robot-Learning/LIBERO.git` into
`$ROBOEVAL_VENDORS_DIR/LIBERO` (shared with the base `libero` venv; the clone
is reused if it already exists).  To point to an existing clone:

```bash
export LIBERO_DIR=/path/to/your/LIBERO
roboeval setup libero_infinity
```

**Datasets**

Same dataset layout as the base LIBERO sim: symlink task datasets under
`~/.libero/` (see [LIBERO docs](https://github.com/Lifelong-Robot-Learning/LIBERO)).

**Starting the sim worker**

```bash
roboeval serve --vla pi05 --sim libero_infinity --headless
```

**Scenic dependency**

`libero-infinity` depends on [Scenic](https://scenic-lang.org/) for procedural
scene generation.  Scenic is installed automatically from PyPI (`scenic>=3.0.0`)
during `roboeval setup libero_infinity`.

**Platform notes**

- Supported on x86_64 Linux with CUDA 12 and aarch64 GPU hosts.
- HDF5 native libs: if `h5py` fails to build on aarch64, apply the same
  micromamba fix as for `libero_pro` (see [§libero_pro notes](#libero_pro)).

<!-- === END libero_infinity === -->

---

### robocasa

**Assets (~10 GB)**

RoboCasa downloads kitchen assets (models, textures, objects) into
`~/.robocasa/` (or `$ROBOCASA_ASSET_DIR`) on first run. The download happens
automatically when you first call `/init` on the sim server.

**Robot**

RoboCasa tasks use `PandaOmron` (mobile-base Panda). The robot is defined in
robosuite 1.5; ensure you installed robosuite from git master (the PyPI package
lags behind).

**Headless rendering**

```bash
roboeval serve --vla cosmos --sim robocasa --headless
```

---

### robotwin

RoboTwin 2.0 is a SAPIEN-based bimanual simulator. It has several non-standard
installation requirements.

**Python 3.10 is required.**

**Step 1 — Install SAPIEN**

SAPIEN is distributed through normal Python package channels for supported
platforms. `roboeval setup` first uses the standard online install path:

```bash
.venvs/robotwin/bin/pip install sapien --pre \
    --extra-index-url https://storage.googleapis.com/sapien-nightly/
```

On Linux/aarch64 with Python 3.10, PyPI/nightly index resolution may not expose
the required wheel. For that platform, `roboeval setup robotwin` downloads the
official SAPIEN GitHub release wheel and caches it under `$ROBOEVAL_VENDORS_DIR`:

```text
https://github.com/haosulab/SAPIEN/releases/download/3.0.3/sapien-3.0.3-cp310-cp310-linux_aarch64.whl
```

The download is SHA256-verified and installed without dependency resolution so
RoboTwin's `numpy<2` planner stack is preserved. If you need a custom or offline
wheel, set `ROBOEVAL_SAPIEN_WHL=/path/to/sapien*.whl`; if the upstream asset
moves, set `ROBOEVAL_SAPIEN_AARCH64_URL` and `ROBOEVAL_SAPIEN_AARCH64_SHA256`.

**Step 2 — Motion-planning C++ stack (curobo + mplib)**

RoboTwin's motion planners (`CuroboPlanner`, `MplibPlanner`) require two
compiled C++ extension packages. `roboeval setup` handles this automatically based
on your CPU architecture; no manual steps are needed.

*x86_64 users:* `roboeval setup robotwin` installs `nvidia-curobo` and
`mplib` from PyPI wheels automatically. No further action needed.

*aarch64 users:* see [§ aarch64 notes below](#robotwin-aarch64).

**Step 3 — RoboTwin repo**

`roboeval setup` clones [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)
into `$ROBOEVAL_VENDORS_DIR/RoboTwin` and adds it to the venv's `sys.path`
via a `.pth` file. The server's `os.chdir(robotwin_dir)` call in `sim_worker.py`
handles asset loading (RoboTwin uses relative paths).

> **Asset download (~3.7 GB):** `roboeval setup` automatically runs `python _download.py`
> inside `$ROBOTWIN_DIR/assets/` after cloning. This downloads `embodiments.zip`
> (~220 MB) and `objects.zip` (~3.5 GB) from HuggingFace (`TianxingChen/RoboTwin2.0`).
> Allow several minutes on a typical connection. If the download fails you can re-run it
> manually: `cd $ROBOEVAL_VENDORS_DIR/RoboTwin/assets && python _download.py`

**CRITICAL: import order**

`import torch` **must** come before `import sapien` in any Python process that
uses both. A reverse import order causes a CUDA segfault. `sim_worker.py`
enforces this.

**Start command**

```bash
ROBOTWIN_DIR=$HOME/.local/share/roboeval/vendors/RoboTwin \
  roboeval serve --vla internvla --sim robotwin --headless
```

---

<details>
<summary><a name="robotwin-aarch64"></a>aarch64 automated source build for RoboTwin planners</summary>

On aarch64, RoboTwin's motion-planning C++ stack (curobo + mplib) is built from
source via a script-managed conda-forge environment. The setup is
script-managed; no user action is needed beyond the standard `roboeval setup
robotwin` invocation.

| | Detail |
|---|---|
| Build time | Varies by CPU, storage, and compiler cache state |
| Additional disk | ~3.9 GB (`mplib_libs` conda env 3.1 GB + curobo 671 MB + mplib 104 MB) |
| micromamba location | `~/.micromamba/` (auto-installed under `~/.local/bin/micromamba`, no sudo) |
| CUDA prerequisite | `cuda-toolkit-13-0` (or matching version) must already be at `/usr/local/cuda` |

**Version pins (exact; do not loosen):**

- `curobo v0.7.7` — RoboTwin's `planner.py` uses the v0.7 module layout
  (`curobo.types.base`, `curobo.wrap.reacher`). curobo v0.8 reorganised these
  modules and breaks the import.
- `pinocchio 2.6.21` — mplib's source uses `UrdfVisitorBaseTpl` and other
  pinocchio-2.x APIs that were **removed** in pinocchio 3.x. The conda-forge
  package is pinned to exactly 2.6.21 in the `mplib_libs` env.
- `ompl 1.6.0` — matches mplib's official Dockerfile; newer ompl versions
  trigger conda-forge solver conflicts with `libccd-double`.
- `fcl < 0.7.1` — FCL 0.7.1+ pulls a conflicting `libccd` variant.

**Upstream patches applied by `roboeval setup` automatically:**

1. **Relax `-Werror` in `CMakeLists.txt`** — gcc 13 raises
   `-Wmaybe-uninitialized` on Eigen template instantiations from pinocchio
   2.6.21. The patch strips `-Werror` and adds `-Wno-uninitialized
   -Wno-maybe-uninitialized`. Safe to apply; these are spurious template
   warnings, not real bugs.

2. **`dev/mkdoc.sh` python3 variable** — cmake spawns `mkdoc.sh` using the
   bare `python3` name, which resolves to the system Python (lacking `libclang`).
   The patch changes it to `"${PYTHON3:-python3}"` so the script honours the
   `PYTHON3` env var set to the venv interpreter.

3. **Linker `-L` flag** — mplib's `target_link_libraries` uses bare library
   names (e.g. `urdfdom_model`) without adding the conda-forge prefix to the
   linker search path. The patch passes
   `-DCMAKE_{EXE,SHARED,MODULE}_LINKER_FLAGS=-L$MMENV/lib` via `CMAKE_ARGS`
   and sets `LDFLAGS` accordingly so `ld` finds the libraries.

</details>

---

### aloha_gym

`gym-aloha` is HuggingFace's pure-Python ALOHA bimanual simulator and is the
**uv-based / aarch64-compatible / no-conda** counterpart to RoboTwin.  It complements
(does not replace) RoboTwin: identical 14-dim absolute joint-position action
contract, so InternVLA drives both backends natively, but `gym-aloha` installs
purely from PyPI wheels — no SAPIEN nightly index, no curobo source build, no
mplib, no micromamba.

**Python 3.10+ required.**  The `roboeval setup` helper creates `.venvs/aloha_gym`
with Python 3.10 (the lowest version that satisfies `gymnasium>=0.29` *and*
upstream `gym-aloha`).

**Install**

```bash
roboeval setup internvla aloha_gym
```

This is fully unattended: the `setup_aloha_gym()` function installs
`gym-aloha`, `gymnasium`, `mujoco`, `dm-control`, plus the FastAPI/uvicorn
shim used by the sim worker.  No external assets need to be downloaded —
gym-aloha bundles its MuJoCo XML scenes and meshes inside the wheel.

**Tasks**

| Task id | Description |
|---|---|
| `AlohaTransferCube-v0` | Right arm picks up the red cube and transfers it to the left gripper. |
| `AlohaInsertion-v0` | Both arms pick up a socket and a peg respectively, then insert mid-air. |

The backend also accepts numeric task indices (`"0"`, `"1"`) and a few short
forms (`transfer_cube`, `insertion`).

**Action / observation contract**

| Field | Format |
|---|---|
| Action | 14-dim absolute joint targets (6 joint × 2 arms + 1 gripper × 2). |
| Observation | `pixels.top` (primary) + `pixels.angle` (secondary), 14-dim `agent_pos` state. |
| Image transform | `applied_in_sim` — both frames are flipped 180° (`[::-1, ::-1]`) inside `AlohaGymBackend._extract_image`. Consumers MUST NOT reapply. |
| Reward | 0–4 staircase; success = reward ≥ 4. |

**Compatible shipped VLAs**

| VLA | Compatible? | Notes |
|---|---|---|
| InternVLA-A1 | ✅ native | 14-dim joint_pos head matches exactly (same contract as RoboTwin). |
| GR00T | ⚠️ multi-embodiment | Works in degraded mode if the bimanual head is configured for ALOHA-14. |
| Pi 0.5 / SmolVLA / OpenVLA | ❌ | These are 7-dim eef-delta single-arm; the orchestrator's `ActionObsSpec` gate rejects them. |

**Start command**

```bash
roboeval serve --vla act --sim aloha_gym --headless
```

**Smoke config:** `configs/aloha_gym_act_smoke.yaml` (supported ACT pairing,
TransferCube).

> **Why we ship both gym-aloha and RoboTwin:** they cover different value
> props.  RoboTwin offers the larger task suite and SAPIEN's articulated
> object physics, while gym-aloha gives users the same ALOHA bimanual action
> contract through a smaller uv-based install path.

---

### diffusion_policy

<!-- === START diffusion_policy === -->
Diffusion Policy is a DDPM-based visuomotor policy from Chi et al. (2023),
implemented via lerobot's `DiffusionPolicy` class.  The canonical checkpoint
shipped with roboeval is **`lerobot/diffusion_pusht`** — trained on the
PushT 2-D block-pushing task (~50 MB).

> **Domain note (v0.1):** `lerobot/diffusion_pusht` was trained on PushT, not
> LIBERO or ALOHA. Use the PushT config for the supported Diffusion Policy
> pairing. A LIBERO-finetuned Diffusion Policy checkpoint and matching sim
> integration are planned for a later release.

**Python 3.11 required.** Uses the same `lerobot==0.4.4` base as SmolVLA —
the last PyPI release with Python 3.11 support prior to lerobot's transformers≥5.0
requirement in v0.5.0.

**Install**

```bash
roboeval setup diffusion_policy
```

This creates `.venvs/diffusion_policy` (Python 3.11) and installs
`lerobot==0.4.4` plus CUDA-enabled `torch`.

**Manual per-venv install**

```bash
uv venv .venvs/diffusion_policy --python 3.11
uv pip install --python .venvs/diffusion_policy/bin/python -e ".[diffusion_policy]"
```

**Start server**

```bash
roboeval serve --vla diffusion_policy --model-id lerobot/diffusion_pusht
```

The model (~50 MB) is downloaded from HuggingFace on first use.
To use the ALOHA sim insertion checkpoint instead:

```bash
roboeval serve --vla diffusion_policy --model-id lerobot/diffusion_aloha_sim_insertion_human
```

**Action / observation contract (lerobot/diffusion_pusht)**

| Field | Format |
|---|---|
| Action | 2-dim absolute (x, y) end-effector position (PushT coordinates). |
| Observation | `observation.image`: top-down RGB (96 × 96). `observation.state`: 2-dim (x,y) robot keypoint. |
| Language | Not used — Diffusion Policy is not language-conditioned. |
| Image transform | `none` — no domain-specific flip needed for PushT checkpoint. |

**Compatible sims (v0.1)**

| Sim | Compatible? | Notes |
|---|---|---|
| PushT (`gym-pusht`) | ✅ native | Direct match — checkpoint trained on PushT. |
| LIBERO | ⚠️ unsupported pairing | Action space mismatch (2-dim vs 7-dim); use a LIBERO-finetuned checkpoint before enabling this path. |
| gym-aloha | ⚠️ with `--model-id lerobot/diffusion_aloha_sim_insertion_human` | ALOHA checkpoint is 14-dim joint_pos; wire up to gym-aloha sim. |

**VRAM requirement:** ~2 GB (DDPM denoising loop; no LLM backbone).

**Smoke config:** `configs/gym_pusht_diffusion_policy_smoke.yaml`
<!-- === END diffusion_policy === -->

---

<!-- === START gym_pusht === -->
### gym_pusht

gym-pusht is the **canonical companion simulator for Diffusion Policy** (Chi
et al., 2023). A T-shaped block must be pushed into a target zone drawn on a
2-D workspace using a disk-shaped end-effector.

- **Package:** `gym-pusht` (PyPI)
- **Python:** 3.11+
- **Action space:** 2-dim continuous (x, y) absolute end-effector position
- **Observation:** top-down RGB frame (96 × 96 by default), 2-dim agent state
- **Success criterion:** T-block coverage of target zone ≥ 90 % (`reward ≥ 0.9`)
- **No GPU required.** No MuJoCo, no conda, no system libs. aarch64-compatible.

**Disk space:** < 1 MB (gym-pusht + pymunk + pygame wheels).

**Install**

```bash
roboeval setup gym_pusht
```

This creates `.venvs/gym_pusht` (Python 3.11) and installs `gym-pusht`,
`gymnasium`, and the FastAPI/uvicorn shim. No external assets are downloaded.

**Manual per-venv install**

```bash
uv venv .venvs/gym_pusht --python 3.11
uv pip install --python .venvs/gym_pusht/bin/python -e ".[gym_pusht]"
```

**Start command**

```bash
roboeval serve --vla diffusion_policy --sim gym_pusht --headless
```

No `MUJOCO_GL` is needed — gym-pusht uses pymunk (2-D physics), not MuJoCo.

**Tasks**

| Task id | Description |
|---|---|
| `gym_pusht/PushT-v0` | Push the T-shaped block into the target zone. |

The backend also accepts `"0"` (numeric index), `"pusht"`, and `"push_t"` as
aliases for `gym_pusht/PushT-v0`.

**Action / observation contract**

| Field | Format |
|---|---|
| Action | 2-dim absolute (x, y) end-effector target, workspace-normalised. |
| Observation | `pixels` dict → 96 × 96 RGB top-down view; `agent_pos` 2-dim state. |
| Language | Not used — DP is not language-conditioned. |
| Image transform | `none` — no flip needed. |
| Reward | Continuous coverage in [0, 1]; success = reward ≥ 0.9. |

**Compatible shipped VLAs**

| VLA | Compatible? | Notes |
|---|---|---|
| Diffusion Policy (`lerobot/diffusion_pusht`) | ✅ native | Canonical pairing; the DP checkpoint was trained on this environment. |
| All other VLAs | ❌ | Action spaces are incompatible (7-dim EEF delta or 14-dim joint_pos vs 2-dim XY); `ActionObsSpec` gate rejects them. |

**Smoke config:** `configs/gym_pusht_diffusion_policy_smoke.yaml`

> **Why gym-pusht?** PushT is the simplest supported example for the full
> DP ↔ sim integration: it is the environment the checkpoint was trained on,
> has the smallest download footprint (<1 MB), and requires zero GPU, zero
> MuJoCo, and zero conda. If you observe systematic failures, see
> [Troubleshooting](failure_modes.md).
<!-- === END gym_pusht === -->

---

<!-- === START maniskill2 === -->
### <a name="maniskill2"></a>maniskill2

ManiSkill2 is Hao Su lab's SAPIEN-based manipulation benchmark with a rich set
of tabletop tasks (PickCube, StackCube, PegInsertionSide, and ~20 more).

<details>
<summary>aarch64 platform boundary for ManiSkill2</summary>

`mani_skill2==0.5.3` requires `sapien==2.2.2`, which ships only
`manylinux2014_x86_64` wheels on PyPI.  `sapien 3.x` has aarch64 wheels as
of 3.0.3, but ManiSkill2 0.5.3 is not compatible with that API line; using it
would require a ManiSkill3 backend migration.  On aarch64 the sim worker
starts but `/init` raises `RuntimeError` with a pointer to the workaround.
`get_info()` returns a valid spec regardless so orchestrator tooling can
introspect the contract.

Workarounds:

1. Run on x86_64 and use `roboeval setup maniskill2`.
2. Provide a compatible `sapien==2.2.2` aarch64 wheel, or migrate this backend
   to ManiSkill3/SAPIEN 3.
3. Use `aloha_gym` or `robotwin` for aarch64-compatible benchmarks.

</details>

**Python 3.10 required** — compatible with `sapien==2.2.2`'s `cp310` wheel.

**Install (x86_64)**

```bash
roboeval setup maniskill2
```

This installs `sapien==2.2.2` then `mani_skill2==0.5.3` into `.venvs/maniskill2`.

**Install (aarch64 — degraded mode)**

`roboeval setup maniskill2` still completes on aarch64 but prints a blocker warning
and skips the SAPIEN install.  The venv is usable for introspection (e.g.
`/info`) but `/init` raises `RuntimeError`.

**Tasks**

| Task id | Description |
|---|---|
| `PickCube-v0` | Pick up the red cube and lift it to a target height. |
| `StackCube-v0` | Stack the green cube on top of the red cube. |
| `PegInsertionSide-v0` | Insert a peg into a side-hole box. |

The backend also accepts numeric task indices (`"0"`, `"1"`, `"2"`).

**Action / observation contract**

| Field | Format |
|---|---|
| Action | 7-dim `pd_ee_delta_pose` (Δx, Δy, Δz, Δax, Δay, Δaz, gripper). Same as LIBERO. |
| Observation | `base_camera` 256×256 RGB (primary). No wrist camera in default config. |
| State | 9-dim `qpos(7)+qvel(2)`. |
| delta_actions | `true` (controller outputs deltas). |

**Compatible VLAs**

| VLA | Compatible? | Notes |
|---|---|---|
| Pi 0.5 | ✅ native | 7-dim eef_delta matches exactly. |
| SmolVLA | ✅ native | Same format. |
| OpenVLA | ✅ native | Same format. |
| InternVLA-A1 | ❌ | 14-dim bimanual; spec gate rejects. |

**Start command (x86_64)**

```bash
roboeval serve --vla pi05 --sim maniskill2 --headless
```

No root smoke config ships for ManiSkill2 in v0.1.0; the backend remains an
x86_64-only platform boundary until ManiSkill3/SAPIEN 3 support lands.
<!-- === END maniskill2 === -->

---

<!-- === START metaworld === -->
### metaworld

Meta-World is a well-known benchmark of ~50 Sawyer single-arm manipulation
tasks (push-button, pick-place, door-open, drawer-close, hammer, reach, …)
built on top of MuJoCo.  It is a **pure PyPI / uv-based / aarch64-compatible**
simulator — MuJoCo ships official aarch64 wheels and `metaworld` itself has
no compiled extensions beyond what MuJoCo provides.

**Python 3.11+** (metaworld 2.0.0 requires Python ≥ 3.9; roboeval setup uses 3.11).

**Install**

```bash
roboeval setup metaworld
```

Fully unattended.  `setup_metaworld()` creates `.venvs/metaworld` (Python 3.11),
installs `metaworld==2.0.0`, `mujoco`, `gymnasium`, FastAPI/uvicorn shim.
No external assets need to be downloaded — Meta-World bundles its MuJoCo XML
scenes inside the wheel.

**Tasks (~50)**

All 50 MT50 task names are accepted.  A few representative ones:

| Task name | Description |
|---|---|
| `button-press-v2` | Press the red button. |
| `pick-place-v2` | Pick up the puck and place it at the goal. |
| `door-open-v2` | Open the door. |
| `reach-v2` | Move the robot hand to the red goal sphere. |
| `drawer-open-v2` | Open the drawer. |
| `window-open-v2` | Open the window. |

The backend also accepts numeric task indices (`"0"`–`"49"`) and short forms
with underscores (`button_press_v2`, `pick_place_v2`).

**Action / observation contract**

| Field | Format |
|---|---|
| Action | 4-dim EEF delta `[dx, dy, dz, gripper]`, range [−1, 1] |
| Observation | `corner` camera (primary) + `behindGripper` camera (wrist), 39-dim proprioceptive state |
| Image transform | `applied_in_sim` — both frames are flipped 180° (`[::-1, ::-1]`) inside `MetaWorldBackend._render_camera`. Consumers MUST NOT reapply. |
| Success | `info['success']` from the Meta-World environment |

**Compatible VLAs**

| VLA | Compatible? | Notes |
|---|---|---|
| *(none in v0.1)* | — | No shipped VLA emits 4-dim actions natively. |
| Pi 0.5 / SmolVLA / OpenVLA | ❌ | These emit 7-dim (6-DoF EEF delta + gripper). The `ActionObsSpec` gate blocks the pairing — **expected in v0.1; proves the gate works**. |

> **Action-dim mismatch — v0.2 plan:** A 7→4 adapter strip (keep xyz + gripper,
> drop orientation) or a 4-dim-native VLA checkpoint is tracked for v0.2.
> `MetaWorldBackend.step()` already implements the 7→4 truncation defensively
> for isolated tests; only the spec-gate check is the production blocker.

**Start command**

```bash
roboeval serve --vla tdmpc2 --sim metaworld --headless
```

**Smoke config:** `configs/metaworld_tdmpc2_smoke.yaml` (10 episodes,
`button-press-v2`). SmolVLA x Meta-World is intentionally not shipped because
the ActionObsSpec gate catches the 7-vs-4 action-dimension mismatch.

> **Why we include Meta-World:** (1) TDMPC2 gives v0.1.0 a supported
> 4-dim control pairing. (2) Meta-World is the most widely used single-arm
> benchmark in the VLA/RL literature. (3) It is aarch64-compatible, with no extra install
> risk.
<!-- === END metaworld === -->

---

## Platform notes

Most users can start from the quick path and only return to these notes when a
component mentions their architecture or native library stack. The detailed
platform guidance is kept in-place with the component it affects:

- GR00T aarch64 BLAS/LAPACK runtime details are folded under
  [aarch64 NVIDIA Performance Libraries](#groot).
- LIBERO-Pro HDF5 native-library setup is folded under
  [LIBERO-Pro](#libero_pro).
- RoboTwin aarch64 planner source-build notes are folded under
  [RoboTwin](#robotwin).
- ManiSkill2 aarch64 install boundaries are folded under
  [ManiSkill2](#maniskill2).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'libero'`

The LIBERO Python package is installed from git, not PyPI. Make sure you ran
`roboeval setup libero` (or the manual install steps above) and are using the
`.venvs/libero/bin/python` interpreter — not your system Python.

---

### `Failed to initialize OpenGL` / blank observations

MuJoCo cannot find EGL. Fixes (in order of likelihood):

1. `sudo apt-get install libegl1-mesa-dev`
2. Set `MUJOCO_GL=egl` in the environment that starts the sim worker.
3. Verify the NVIDIA driver is loaded: `nvidia-smi`.
4. On some headless servers, `MUJOCO_GL=osmesa` works as a fallback (software
   rendering, slower).

---

### `AutoModelForVision2Seq` removed / transformers 5.x error

You have `transformers>=5.0` in the openvla venv. The openvla extra pins the
exact version `transformers==4.40.1` — re-install:

```bash
.venvs/openvla/bin/pip install "transformers==4.40.1" --force-reinstall
```

### OpenVLA returns 200 OK but rollouts fail unexpectedly

Your openvla venv may have `transformers` 4.41–4.57+. Those versions can break
image conditioning in `predict_action` without raising an error. The fix is:

```bash
.venvs/openvla/bin/pip install "transformers==4.40.1" --force-reinstall
```

---

### `torch.where` dynamo error (Pi 0.5, PyTorch 2.10)

This is a known PyTorch 2.10 bug. The Pi 0.5 policy server already applies a
workaround at load time. If you're seeing it in a custom script, add:

```python
if hasattr(policy, "compiled_sample_actions"):
    policy.sample_actions = policy.compiled_sample_actions.__wrapped__
```

---

### h5py fails to build on aarch64

See [§ LIBERO-Pro notes](#libero_pro) for the micromamba HDF5 fix.

---

### `PermissionError` on `~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/` (groot)

GR00T uses `trust_remote_code=True` when loading the Eagle backbone, which caches dynamic
module files in `~/.cache/huggingface/modules/`.  If a previous groot run was executed as
root (or via sudo), that directory may be root-owned and unwritable by your user.

**Fix (preferred):** restore ownership:

```bash
sudo chown -R $USER ~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/
```

**Workaround (no sudo):** redirect the modules cache to a user-writable location:

```bash
mkdir -p "$HOME/.cache/roboeval/hf_modules"
HF_MODULES_CACHE="$HOME/.cache/roboeval/hf_modules" \
  roboeval serve --vla groot --sim libero --headless
```

---

### `setup_groot` fails with `micromamba: command not found`

micromamba is required for the groot venv (conda-forge flash-attn/pytorch). Install it:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PATH="$HOME/.local/bin:$PATH"
# then re-run: roboeval setup groot
```

See [README.md Prerequisites](../README.md) for the full list of prerequisites.

---

### `uv` not found after install

The uv installer adds `~/.local/bin` to `PATH` in your shell's rc file, but
not the current shell session. Fix:

```bash
export PATH="$HOME/.local/bin:$PATH"
# then re-run roboeval setup
```

---

### RoboCasa asset download fails / times out

The kitchen asset download (~10 GB) happens on first `/init` call. If it
times out, trigger it manually:

```bash
.venvs/robocasa/bin/python -c "import robocasa; robocasa.download_assets()"
```

Set `ROBOCASA_ASSET_DIR` to a path with enough free space if `~/.robocasa/`
is on a small partition.

---

### RoboTwin planner stubs are active on aarch64

If the aarch64 curobo/mplib source build fails during `roboeval setup robotwin`,
the script falls back to lightweight Python stubs for both libraries so that
the venv starts cleanly. In stub mode `CuroboPlanner` and `MplibPlanner` are
no-ops, which means motion-planning calls return without moving the robot arm.

**How to detect:**

```bash
.venvs/robotwin/bin/python -c "
import mplib
if getattr(mplib, '_IS_STUB', False):
    print('WARNING: mplib stub active — planners disabled')
else:
    print('mplib OK:', mplib.__file__)
"
```

**Fix:** re-run `roboeval setup robotwin` after ensuring the CUDA
toolkit is installed (`nvcc --version` should succeed) and that
`~/.local/share/roboeval/vendors/curobo` and
`~/.local/share/roboeval/vendors/MPlib` are writable. The build is
idempotent and skips already-completed steps.

See [§ aarch64 notes](#robotwin-aarch64) for version pin requirements.

---

### SAPIEN wheel not found (RoboTwin)

On most platforms, re-run `roboeval setup robotwin`; setup uses the normal
SAPIEN package indexes. On Linux/aarch64, setup uses the official GitHub release
wheel URL documented in the RoboTwin section above. If GitHub changes the asset
name or you need an offline build, provide an explicit wheel:

```bash
ROBOEVAL_SAPIEN_WHL=/path/to/sapien-...whl roboeval setup robotwin
```

For a moved upstream release asset, override both the URL and checksum:

```bash
ROBOEVAL_SAPIEN_AARCH64_URL=https://github.com/haosulab/SAPIEN/releases/download/.../sapien-...linux_aarch64.whl \
ROBOEVAL_SAPIEN_AARCH64_SHA256=<sha256> \
roboeval setup robotwin
```

---

For quickstart instructions (Pi 0.5 + LIBERO from setup to first local run),
see [docs/quickstart.md](quickstart.md).
