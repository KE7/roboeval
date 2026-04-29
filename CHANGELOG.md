# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-26

### Added

- Added a tools-paper README rewrite for the v0.1.0 release, replacing results-paper and landing-page framing with method, contracts, coverage, extension, validation, citation, and license sections.
- Added prominent LITEN attribution to Shah et al. and documented roboeval's integration boundary for the planner method.
- Added groot x LIBERO support using the `nvidia/GR00T-N1.7-LIBERO` checkpoint, with smoke coverage for `configs/libero_spatial_groot_smoke.yaml`.
- Added TDMPC2 as a model-based reinforcement-learning policy backend for Meta-World, including smoke configs, setup support, server-runner registration, and tests.
- Added VQ-BeT policy-server integration for PushT, including setup support, smoke configs, tests, and documentation.
- Added a dedicated `libero_infinity` setup path with its own Python 3.11 environment, server-runner defaults, dependency extras, tests, and install documentation.
- Added smoke configs for additional validated model/simulator pairings, including `configs/libero_infinity_pi05_smoke.yaml`, `configs/libero_spatial_groot_smoke.yaml`, and `configs/robotwin_internvla_smoke.yaml`.
- Added ManiSkill2 backend scaffolding with typed action and observation specs, setup entry, tests, and install documentation. aarch64 support is documented as limited by upstream wheel availability.
- Added Meta-World backend support with a pure-uv setup path, typed specs, smoke config, tests, and install/results documentation.
- Added ACT policy-server integration for ALOHA Gym, including setup support, smoke config, tests, and documentation.
- Added `gym_pusht` backend support for PushT evaluation, including smoke config, tests, setup support, and documentation.
- Added `examples/demo_recording.sh` for a launch walkthrough.
- Added `configs/robotwin_internvla_smoke.yaml` for InternVLA-A1-3B x RoboTwin validation.

### Changed

- Trimmed public docs from 15 tracked files to 9 by merging extension guides into `docs/extending.md`, moving CI notes into `CONTRIBUTING.md`, and reframing release results material as `docs/validation.md`.
- Documented the post-reorganization layout with the top-level package under `roboeval/` and setup/maintenance entry points consolidated under `scripts/`.
- Replaced the planned Octo slot for v0.1.0 with VQ-BeT because Octo's upstream dependency stack was not compatible with the target aarch64 release environment.
- Updated Meta-World coverage from spec-gate-only to end-to-end validated through the TDMPC2 pairing.
- Updated README, install docs, results docs, and setup instructions to match the shipped v0.1.0 command surface and component registry.
- Reframed shipped configs as validated, reproducible `roboeval run` invocations for supported VLA/simulator pairs.
- Documented component-specific setup details for new policy servers and simulation backends.

### Fixed

- Fixed SmolVLA x LIBERO smoke configuration and server-runner defaults so the orchestrator connects to the SmolVLA server on its default port.
- Fixed OpenVLA readiness handling so evaluation waits for model loading to complete before launching episodes.
- Fixed groot x LIBERO integration details for camera key mapping, model subfolder configuration, ActionObsSpec declarations, and gripper state extraction.
- Fixed InternVLA memory use by loading model weights in bfloat16 by default while keeping action post-processing numerically stable.
- Fixed fresh-clone groot setup by cloning and installing Isaac-GR00T during `setup_groot()`, with an override for existing local clones.
- Fixed RoboTwin runtime setup issues needed for end-to-end smoke execution.
- Fixed OpenVLA dependency pins to restore image-conditioned action prediction.
- Fixed test-suite interaction between mocked `torch` modules and scipy rotation validation.
- Fixed several setup and documentation inconsistencies discovered during v0.1.0 validation.

### Removed

- Removed Octo from the v0.1.0 shipped backend set and removed its policy server, tests, smoke config, setup entries, dependency extras, and docs rows.
- Dropped `robosuite_twoarm` from v0.1.0 because no shipped public policy target matched its action space.

### Known Limitations

- ManiSkill2 is documented as limited on aarch64 because required upstream simulator wheels are not available for that platform.
- Some smoke configs validate wiring and action/observation compatibility; full benchmark numbers may require a longer evaluation run.

## [0.1.0] - 2026-04-25

### Added

- Added host-process execution for VLA policy servers and simulation workers using per-component Python environments and HTTP/JSON communication.
- Added ActionObsSpec contracts for typed action and observation compatibility checks between policy servers and simulation backends.
- Added hierarchical-planner support while preserving the direct evaluation path.
- Added sharded evaluation and result merge support in `roboeval/orchestrator.py`.
- Added `roboeval test` preflight checks for server reachability and spec compatibility.
- Added the `roboeval run`, `serve`, `merge`, and `test` subcommands.
- Added `roboeval/registry.py`, `roboeval/rotation.py`, `roboeval/preflight.py`, and `roboeval/server_runner.py`.
- Added example YAML configs in `configs/` for canonical smoke and evaluation workflows.
- Added RoboTwin aarch64 setup support through source builds where binary wheels are unavailable.
- Added `SECURITY.md`, `THIRD_PARTY.md`, and `docs/liten.md`.
- Added regression tests for server readiness polling and signal-handler installation.

### Changed

- Bumped the orchestrator virtual environment default to Python 3.13 while retaining project support for Python 3.11 and newer.
- Renamed `DimSpec` to `ActionObsSpec`; the deprecated alias is retained for one release.
- Lifted simulation backends to a `SimBackendBase` abstract base class.
- Added typed action and observation specs to RoboCasa, RoboTwin, and LIBERO-Infinity backend metadata.
- Kept legacy `obs_requirements` as a transition fallback while making ActionObsSpec the canonical contract.
- Added lifecycle compatibility to `RoboTwinBackend` for the shared backend interface.
- Sharpened the README and documented public design rationale for the v0.1.0 release.

### Fixed

- Fixed Python 3.8 compatibility for LIBERO subprocess imports.
- Fixed packaging of the `roboeval.results` package in fresh clones.
- Updated user-facing setup references to the unified `scripts/setup.sh <component>` interface.
- Corrected docs and examples to use the shipped `--config <yaml>` CLI flow.
- Clarified strict spec-contract behavior in docs and error messages.
- Fixed server health polling, sim worker launch flags, subprocess cleanup, action validation, and load-failure diagnostics.
- Added ActionObsSpec declarations to InternVLA.
- Added RoboTwin asset download support in setup.
- Moved signal-handler installation out of import time.
- Added scipy to development dependencies so rotation cross-validation runs in tests.
- Removed stale Docker config fields from runtime configuration.
- Updated groot virtual-environment defaults to use the project-local environment path.
- Corrected `examples/eval.yaml` to match the current config schema.
- Renamed a manual batched-server script so pytest does not collect it.
- Moved legacy development notes out of the repo root.
- Tightened one-sided spec validation under strict specs.
- Matched LIBERO-Infinity image orientation behavior to LIBERO.
- Fixed `LiberoInfinityBackend` inheritance to reuse the LIBERO-Pro backend hierarchy.

### Migrated

- Completed the migration from container orchestration to host-process execution.

### Removed

- Removed the Docker backend, proxy, port allocator, VLA manager, stack module, Docker images, Docker runtime flags, Docker environment variables, and multi-replica proxy logic.
- Removed stale Docker documentation.
- Removed obsolete tests for deleted Docker and legacy runner/result modules.

[0.1.0]: https://github.com/KE7/roboeval/releases/tag/v0.1.0
