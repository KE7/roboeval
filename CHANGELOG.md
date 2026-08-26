# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-08-26

A packaging and repository-hygiene release. No functional changes to evaluation
behaviour, result schemas, or the supported VLA/simulator matrix.

**Packaging note for anyone installing from PyPI:** the published sdist and wheel
now contain only the `roboeval` and `sims` Python packages and the bundled setup
script. Previous releases also unpacked `configs/` and `scripts/` directories into
`site-packages`. Nothing at runtime reads those bundled copies — configs are always
passed by path — but if you were relying on the installed copies, take the configs
from the repository instead. Installing from a clone is unaffected.

### Changed

- Bumped the harness version to 0.1.1 across `pyproject.toml`, `CITATION.cff`, and
  the emitted `harness_version` field, and refreshed the version shown in the
  documentation's sample result output.
- Replaced the packaging configuration with an explicit allow-list. `package-data`
  previously used `../configs/**/*.yaml`, which escaped the package directory and
  swept every file under `configs/` into both artifacts, so new files were published
  by default. Added a `MANIFEST.in` that excludes everything and re-adds only what
  ships, and set `include-package-data = false` so the sdist list can never feed the
  wheel implicitly.
- Moved `scripts/setup.sh` to `roboeval/_scripts/setup.sh` so it ships inside the
  package rather than as a top-level `scripts/` directory in `site-packages`.
  `./scripts/setup.sh` still works from a checkout — it is now a shim — and
  `roboeval setup` resolves the packaged copy. The script locates the source
  checkout by searching for `pyproject.toml` and honours `ROBOEVAL_PROJECT_ROOT`,
  and now fails with a clear message when no checkout is present instead of
  misresolving its project root.
- Hardened the release workflow: it verifies that the pushed tag matches the project
  version before building, builds from a clean `dist/`, asserts that no config,
  document, or credential file is present in the built artifacts, pins the PyPI
  publish action to a commit SHA, and runs with least-privilege permissions under a
  concurrency guard.
- Broadened `.gitignore` into category-based rules for local working documents and
  scratch experiment configs, replacing a hand-maintained list of individual file
  names that had to be edited for every new file.
- Added ignore rules for common secret and credential files (`.env` and variants,
  private keys, `.netrc`, and similar), which were previously not covered.
- Simplified inline comments and test names in the simulator compatibility layer so
  they describe what each fix does, and consolidated the corresponding regression
  tests as `tests/test_sim_compat_patches.py`. Test coverage is unchanged.
- Rewrote the skip-list config headers (`configs/libero_infinity_skip_indices_seed42.yaml`,
  `configs/libero_infinity_task8_skip_init_states.yaml`) to document the schema and
  the regeneration procedure on their own terms.
- Expanded the `wilson_difference_ci` docstring to describe the interval it computes
  and to state where it diverges from an exact Newcombe interval.
- Replaced the vulnerability-reporting channel in `SECURITY.md` with GitHub private
  vulnerability reporting.
- Resolved the vendored `libero-infinity` source location in the test suite through
  `ROBOEVAL_LIBERO_INFINITY_SRC` / `ROBOEVAL_VENDOR_DIR` / `XDG_DATA_HOME` instead of
  a hard-coded path, so the tests run against a checkout in any location.

### Removed

- Removed internal planning and status documents, and the reporting script tied to a
  local results layout, from the published tree. They documented a private workflow
  and had no bearing on using the harness.

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

[0.1.1]: https://github.com/KE7/roboeval/releases/tag/v0.1.1
[0.1.0]: https://github.com/KE7/roboeval/releases/tag/v0.1.0
