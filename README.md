# roboeval

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/github/actions/workflow/status/KE7/roboeval/ci.yml?branch=main&label=tests)](https://github.com/KE7/roboeval/actions/workflows/ci.yml)

roboeval is a CLI-driven evaluation harness for running VLAs against simulator backends through isolated HTTP services. It provides an `ActionObsSpec` compatibility gate before episode execution, per-component virtual environments for dependency isolation, sharded result collection, and built-in support for LITEN-style hierarchical evaluation in which a VLM planner issues subtask instructions to a low-level VLA.

## Method / Contracts

roboeval treats each VLA and simulator as an independently launched component. The orchestrator communicates with a VLA policy server and a simulator worker over HTTP/JSON, validates their declared contracts, and records episode-level results from a reproducible YAML run config.

The main contract surfaces are:

| Surface | Role |
|---|---|
| `ActionObsSpec` gate | VLA and simulator components declare action format, dimensionality, value range, camera roles, image format, state layout, and language inputs. Under the default strict mode, incompatible declarations stop the run before episode 1. |
| Host-process isolation | VLA servers, simulator workers, and optional VLM proxy processes run in separate `.venvs/` environments. This allows different Python and CUDA dependency stacks to coexist without a monolithic runtime. |
| Dependency isolation | Each VLA and simulator keeps its upstream package pins, Python version, CUDA assumptions, and optional micromamba/uv environment separate. This is a design choice: adding a new backend should not force the orchestrator or other backends onto the same dependency closure. |
| LITEN-style hierarchical evaluation | The hierarchical mode integrates the VLM-planner method introduced by Shah et al. ([Learning Affordances at Inference-Time for Vision-Language-Action Models](https://arxiv.org/abs/2510.19752)). The planner emits subtask calls that are executed by the same VLA server interface used for direct evaluation. roboeval is, to our knowledge, the first public VLA evaluation harness to ship a working LITEN integration. |
| Result records | `roboeval run` writes JSON with harness version, config snapshot, per-episode metadata, success flags, and optional shard metadata. |

## Documentation map

For a compact system overview, design rationale, supported-pair notes, tuning guidance, related systems, and decision records, see [architecture](docs/architecture.md), [design](docs/design.md), [supported pairs](docs/supported_pairs.md), [tuning](docs/tuning.md), [related work](docs/related_work.md), and the [RFC index](docs/rfcs/).

## Installation

For full prerequisites, platform notes, and per-component dependency details, see [docs/install.md](docs/install.md).

```bash
git clone https://github.com/KE7/roboeval.git
cd roboeval
roboeval setup pi05 libero
```

The setup script provisions the orchestrator plus the requested VLA and simulator environments under `.venvs/`.

## Quickstart

```bash
roboeval setup pi05 libero
roboeval serve --vla pi05 --sim libero --headless
roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
roboeval run -c configs/libero_spatial_pi05_smoke.yaml
```

`serve` launches the selected VLA and simulator workers. `run` executes the YAML configuration, including the declared VLA/simulator pair, task suite, episode count, server URLs, output directory, and optional LITEN endpoint. Additional examples are in [docs/quickstart.md](docs/quickstart.md).

## Supported VLAs and Simulators

The table describes shipped coverage. It is a support matrix, not a benchmark table; supported pairs are tested end-to-end.

| VLA | Simulator | Coverage | Example config |
|---|---|---|---|
| Pi0.5 | LIBERO | direct, LITEN | `configs/libero_spatial_pi05_smoke.yaml`, `configs/libero_spatial_pi05_liten_smoke.yaml` |
| Pi0.5 | LIBERO-Pro | direct, LITEN | `configs/libero_pro_pi05_smoke.yaml`, `configs/libero_pro_pi05_liten_smoke.yaml` |
| Pi0.5 | LIBERO-Infinity | direct, LITEN | `configs/libero_infinity_pi05_smoke.yaml`, `configs/libero_infinity_pi05_liten_smoke.yaml` |
| SmolVLA | LIBERO | direct, LITEN | `configs/libero_object_smolvla_smoke.yaml`, `configs/libero_object_smolvla_liten_smoke.yaml` |
| OpenVLA | LIBERO | direct, LITEN | `configs/libero_spatial_openvla_smoke.yaml`, `configs/libero_spatial_openvla_liten_smoke.yaml` |
| GR00T | LIBERO | direct, LITEN | `configs/libero_spatial_groot_smoke.yaml`, `configs/libero_spatial_groot_liten_smoke.yaml` |
| InternVLA | RoboTwin | direct, LITEN | `configs/robotwin_internvla_smoke.yaml`, `configs/robotwin_internvla_liten_smoke.yaml` |
| ACT | ALOHA Gym | direct, LITEN | `configs/aloha_gym_act_smoke.yaml`, `configs/aloha_gym_act_liten_smoke.yaml` |
| Diffusion Policy | gym-pusht | direct | `configs/gym_pusht_diffusion_policy_smoke.yaml` |
| VQ-BeT | gym-pusht | direct | `configs/gym_pusht_vqbet_smoke.yaml` |
| TDMPC2 | Meta-World | direct | `configs/metaworld_tdmpc2_smoke.yaml` |
| InternVLA | ALOHA Gym | CI smoke | `configs/ci/aloha_gym_internvla_smoke.yaml` |
| ManiSkill2 | ManiSkill2 backend | backend scaffold; x86_64 execution path | setup target `maniskill2` |
| RoboCasa | RoboCasa backend | simulator backend and registry support | setup target `robocasa` |

Supported VLA launch names are `pi05`, `vqbet`, `tdmpc2`, `smolvla`, `openvla`, `cosmos`, `groot`, and `internvla`. Supported simulator launch names are `libero`, `libero_pro`, `libero_infinity`, `robocasa`, `robotwin`, `aloha_gym`, `gym_pusht`, `maniskill2`, and `metaworld`.

## Current limitations

- ManiSkill2 is platform-blocked on aarch64 because the required SAPIEN 2.x wheels are x86_64-only.
- `bridge_octo` is platform-blocked on aarch64 by its current TensorFlow/dlimp dependency chain and does not ship in the v0.1.0 support matrix.
- Some technically expressible pairs remain capability boundaries and do not ship root configs, including RoboCasa x GR00T.

## Planned features

- Multi-architecture CI matrix. aarch64 is currently the primary CI path; x86_64 execution paths exist but are not in the CI matrix.
- Additional VLAs as their checkpoints become available.
- More simulators. Community contributions are welcome; see [docs/extending.md](docs/extending.md).

## Extending

**Extension cost.** Adding a new VLA averages ~200 SLOC; adding a new simulator backend averages ~230 SLOC (across the v0.1.0 release; excludes blank lines, comments, and docstrings).

- Add a VLA by implementing a policy server with `/health`, `/info`, `/reset`, and `/predict`, then registering it with `roboeval serve`.
- Add a simulator by implementing a `SimBackendBase` backend with `/init`, `/reset`, `/step`, `/obs`, `/success`, and `/info` support through the sim worker.
- Add a new compatibility path by declaring `ActionObsSpec` records on both sides and adding a smoke config under `configs/`.

See [docs/extending.md](docs/extending.md) for the extension architecture and step-by-step entry points.

## Citations

If you use roboeval in your research, please cite us.

```bibtex
@software{elmaaroufi2026roboeval,
  title   = {roboeval: A reproducible evaluation harness for Vision-Language-Action models},
  author  = {Elmaaroufi, Karim and OMAR and Seshia, Sanjit A. and Zaharia, Matei},
  version = {0.1.0},
  date    = {2026-04-29},
  url     = {https://github.com/KE7/roboeval},
  license = {BSD-3-Clause}
}
```

## License

roboeval is released under the [BSD-3-Clause License](LICENSE).
