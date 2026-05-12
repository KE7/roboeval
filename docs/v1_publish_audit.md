# roboeval ↔ sibling-repo doc comparison (snapshot at 8e3c534)

## Section A: Gaps where roboeval is shorter / missing

| Topic | sibling depth | roboeval depth | Recommendation |
|---|---|---|---|
| Architecture overview | Matched-to-deep: concise public diagram, component table, protocol, config, result collection, and error handling in `docs/architecture.md`. | Partial: README method table plus very deep `docs/vla_policy_architecture.md`, but no short public system overview. | Add a compact `docs/architecture.md` or trim/promote the current developer guide into a public architecture page. |
| Design philosophy | Deep: explicit principles for freshness, convenience, abstraction, quality, reproducibility, and openness. | No: design rationale is implicit in README contracts and extension docs. | Add a short "Design principles" page or README section; this would help public readers understand why the harness uses isolated services and strict specs. |
| RFC / decision record index | Deep: RFC index plus implemented RFCs for protocol, model server hierarchy, isolation, episode execution, sharding, and batching. | No: design history is folded into implementation docs. | Before public push, add a lightweight ADR/RFC index for major contracts, even if only 3-4 pages. |
| Reproduction reports | Deep: per-model reports, common pitfalls, and running guide. | No public equivalent after validation evidence removal. | Keep quantitative validation internal, but add a non-numeric "supported pair notes" page explaining intended pairings and caveats. |
| Tuning guide | Deep: demand/supply methodology, shard and batch sizing, resource allocation, and worked examples. | Partial: `docs/sharded_runs.md` covers sharding mechanics and single-host notes, but not methodology. | Add a short tuning section for choosing shard counts, ports, and GPU allocation without publishing run-throughput claims. |
| Docker / environment isolation story | Deep: sibling documents Docker benchmark isolation as a core public concept. | Partial: roboeval documents per-component venv/micromamba isolation, but the architecture-level story is scattered. | Promote dependency-isolation design into README/architecture so users see it as a design choice, not only install detail. |
| Contribution pathways | Matched: clear "ways to contribute", tests, linting, project tree, benchmark/model instructions. | Partial: good setup and CI notes, but less inviting and less specific about contribution categories. | Add a brief "Ways to contribute" section and link concrete VLA/benchmark entry points. |
| Competitive landscape / references | Deep: proposal and reference survey compare related systems. | No: README cites only this package. | Optional for v0.1.0; useful later as a paper appendix or website page, but not required for package release. |
| Planned features / roadmap | Moderate: architecture lists planned video and reference-score comparison; RFCs expose status. | No: public docs avoid roadmap. | Add only a restrained "Current limitations" section if maintainers want expectation-setting; avoid speculative roadmap. |
| Result schema | Moderate: sibling architecture shows structured result fields and collector behavior. | Partial: README states result records include version, config, metadata, success flags, and shard metadata. | Add a small result JSON schema snippet to quickstart or architecture. |
| Testing commands | Matched: sibling documents smoke tests, lint, type checks, and CI expectations. | Partial: CONTRIBUTING covers CI and smoke gates, quickstart covers `test --validate`. | Add a single "Local checks before PR" command block in CONTRIBUTING. |
| Public installation breadth | Moderate: sibling quick install is short because Docker/uv script model hides complexity. | Verbose: roboeval install guide is much more exhaustive and platform-specific. | Keep the depth, but move rare platform/debug notes behind clearly named subsections to reduce first-read load. |

## Section B: Over-disclosures where roboeval reveals too much

| Location | Content | Recommendation |
|---|---|---|
| `docs/failure_modes.md` | The whole page reads like an internal UX audit with P0/P1 labels, code line references, "Current behaviour", "Fix", and an implementation map. | Remove from public docs or rewrite as a polished troubleshooting guide with stable symptoms, causes, and remedies. |
| `docs/vla_policy_architecture.md` | "Confirmed Bug History", "before fixing", "gold standard", "Key Lessons Learned", and parity-test narrative disclose fix history. | Rewrite as design rules and reference implementation guidance; drop historical discovery language. |
| `docs/vla_policy_architecture.md` | `GB10`, PyTorch 2.10 aarch64 failure notes, and commands like `pkill -f pi05_policy` tie guidance to a specific dev box/platform path. | Keep generic hardware constraints and move machine-specific notes to internal runbooks. |
| `docs/install.md` | Several "validated version", "validated revision", "fully end-to-end validated", and "for the first time" phrases remain after quantitative validation was removed. | Replace with "supported", "pinned", or "compatible"; avoid validation-evidence framing in public docs. |
| `docs/install.md` | "Expect 30-80 % success on a single episode" and "If the smoke produces 0 % success" are result-quality claims. | Remove success-rate expectations or move them to internal validation/reproduction notes. |
| `docs/install.md` | Platform notes repeatedly name `NVIDIA GB10 / DGX Spark`, plus timings like "~30-second pure-uv install" and "22-minute source build". | Generalize to architecture classes and prerequisites; retain timings only if they are measured release claims. |
| `docs/install.md` | Zero-shot/infrastructure-validation wording for Diffusion Policy against LIBERO exposes internal validation intent. | Present it as a compatibility/config example or omit if not a recommended public path. |
| `docs/quickstart.md` | Runtime estimate says "~8 min on a GB10/A100-class GPU." | Use a broader "runtime depends on GPU, model, and simulator" note or omit the estimate. |
| `docs/sharded_runs.md` | "This box is a single NVIDIA GB10" and observed behavior make the doc sound copied from a local validation machine. | Reframe as generic single-GPU guidance and optionally mention "for example" hardware separately. |
| `docs/sharded_runs.md` | Multi-GPU examples name 4x A100/H100 and discuss success-rate comparison protocols. | Keep multi-GPU patterns, but remove model-result language unless this page becomes a formal tuning guide. |
| `docs/vla_policy_architecture.md` | "Results should be within ~5%" and "If LITEN score is 0%" are validation thresholds. | Make this an internal compatibility checklist or state qualitatively that parity should be checked against upstream behavior. |
| `README.md` / `docs/extending.md` | "smoke config" appears frequently as the public term for release examples. | Usually acceptable, but consider "example config" or "compatibility config" where the docs are not specifically about CI smoke gates. |

## Section C: Original strengths in roboeval (keep)

- The `ActionObsSpec` compatibility gate is clearer and more concrete than the sibling's flexible payload convention; keep it prominent.
- The install guide has unusually strong per-model and per-simulator operational detail, especially for mixed uv/micromamba environments.
- LITEN support is a differentiator the sibling docs do not cover; keep it as a first-class public feature.
- `docs/extending.md` gives actionable VLA and benchmark implementation steps with enough contract detail to be useful.
- `CITATION.cff` is already present; the sibling README has a citation block but no CFF file in the requested read set.
- The support matrix is explicit that it is not a benchmark table, which is the right public posture after validation evidence removal.

## Section D: Tools-paper voice diff

roboeval is more restrained than the sibling in marketing claims: it avoids large leaderboard, throughput, and "zero setup" positioning, and its README reads closer to a methods/tool paper. The tradeoff is that several deeper docs still carry internal engineering-audit voice, especially bug-history and validation phrasing, while the sibling is more coherent as a public project narrative despite being more promotional and evidence-heavy.

## Section E: Top 3 actionable changes for v0.1.0 push

1. Remove or rewrite `docs/failure_modes.md` as public troubleshooting; it is the largest remaining internal-process leak.
2. Scrub residual validation/result language from `docs/install.md`, `docs/vla_policy_architecture.md`, `docs/quickstart.md`, and `docs/sharded_runs.md` while preserving generic compatibility/testing guidance.
3. Add one compact public architecture/design page that explains isolated services, `ActionObsSpec`, config/result flow, and extension points without bug history or validation evidence.

# Harness efficiency audit (snapshot at db0052c)

Current footprint: 3,658 SLOC across 19 files.

Files audited: `cli/__init__.py`, `cli/main.py`, `__init__.py`, `__main__.py`, `config.py`, `episode_logger.py`, `orchestrator.py`, `preflight.py`, `registry.py`, `rotation.py`, `server_runner.py`, `specs.py`, `results/__init__.py`, `results/collector.py`, `results/merge.py`, `run.py`, `run_sim_eval.py`, `run_utils.py`, `world_stubs.py`.

### Likely vestigial (delete with no functional change)

| File / function | SLOC | Evidence |
|---|---:|---|
| `run.py` physical-robot Typer commands, after preserving the three helper functions imported by `run_sim_eval.py` | ~600 | `roboeval` console entrypoint is `roboeval.cli.main:app`; current run path is `cli/main.py` -> `Orchestrator` -> `python -m roboeval.run_sim_eval eval`. `run_sim_eval.py` imports only `get_reasoning_steps`, `get_top_level_task_assessment`, and `save_video` from `run.py`. The rest is an interactive physical-robot CLI with `setup_world()` hard-coded to raise `NotImplementedError`; `python -m roboeval.run` is mentioned in `pyproject.toml` as "not registered", so verify maintainers do not treat it as a supported module entrypoint. |
| `config.py` mode/native/RAM planning block (`ModeConfig`, mode aliases, `EVAL_PYTHON`, `NATIVE_EVAL_CONFIGS`, `estimate_ram_usage`) | ~115 | Live `roboeval run/serve/test/merge/setup` does not import these symbols. Uses found were tests and docs; the current CLI has no `--mode` or native-eval command. Likely residue from a broader CLI design, but keep if those helpers are considered public API. |
| `config.py` port allocation helpers (`is_port_available`, `find_available_port`, `find_available_port_block`) | ~55 | `roboeval serve` uses `server_runner._assert_port_free()` rather than automatic allocation; references are tests only. These could be removed or moved to tests if automatic port assignment is not planned. |
| `config.py` VLA/sim launch registries (`VLA_CONFIGS`, `SIM_CONFIGS`) duplicate `server_runner.py` maps | ~100 | `server_runner.py` owns the live launch maps for modules, ports, and venvs. `VLA_CONFIGS` is referenced by backend tests; `SIM_CONFIGS` was not found in live harness callers. Removing outright could affect public imports, so verify API posture first. |
| `episode_logger.py` readback helpers (`load_episode_results`, `episode_results_for_task`) | ~55 | Live path writes episode JSON via `save_episode_result()` and then `orchestrator._read_episode_json()` reads the single expected file directly. These bulk read helpers are covered by tests but not used by the harness. |
| `results/merge.py::merge_shards_from_pattern` | ~25 | `roboeval merge` reimplements the same find/load/merge/write sequence in `cli/main.py` rather than calling this helper. Either delete the unused helper or make the CLI use it; deletion is no behavior change for the current command path. |
| `orchestrator.py::ServerConfig` | ~10 | Defined near `EvalConfig` but not used by `EvalConfig.from_dict`, `preflight`, or server launch code. |

### Duplication (could unify)

| Concern | Locations | Approx LOC saved if unified |
|---|---|---:|
| VLA and sim subprocess launch lifecycle | `server_runner.start_vla()` and `server_runner.start_sim()` share port check, project-root resolution, venv/python resolution, env merge, log opening, `Popen`, health polling, log tail, termination, and managed-process cleanup. | ~60 |
| Launcher registries | `config.py` has `VLA_CONFIGS`/`SIM_CONFIGS`; `server_runner.py` has `_VLA_MODULE_MAP`, `_VLA_DEFAULT_VENVS`, `_VLA_DEFAULT_PORTS`, `_SIM_DEFAULT_PORTS`, `_SIM_DEFAULT_VENVS`. Values also drift (`internvla`/`groot` ports differ between the two files). | ~50 |
| Result JSON write path | `orchestrator._atomic_write_json()`, `episode_logger.save_episode_result()`, `results.merge.merge_shards_from_pattern()`, and `cli/main.py` merge output all open/write JSON directly with slightly different atomicity guarantees. | ~30 |
| Result summary formatting | `results.collector.ResultCollector.print_summary()` and `results.merge.print_merge_report()` both have rich/plain branches and per-task success rendering; `print_task_table()` already proves a shared helper works. | ~20 |
| Preflight HTTP probing | `preflight.check_server()` and `preflight.check_benchmark()` both do `/health` and `/info` request/exception/result bookkeeping before diverging into spec vs reset/step checks. | ~25 |
| Config parsing defaults | `orchestrator.EvalConfig.from_dict()` and `preflight.PreflightConfig.from_dict()` duplicate core defaults for `vla_url`, `sim_url`, `sim`, `suite`, and naming. | ~20 |
| CLI result merge wrapper | `cli/main.py::cmd_merge()` manually expands, loads, merges, writes, and reports instead of using `merge_shards_from_pattern()` plus `print_merge_report()`. | ~20 |

### Quick wins (<=10 lines each)

- Remove unused imports: `sys` in `cli/main.py`, `json` in `preflight.py`, and `os`/`Optional` in `episode_logger.py`.
- Replace repeated `task=int(task) if str(task).isdigit() else 0` in `run_sim_eval.py` with one local `task_idx`.
- Replace repeated frame-flattening loops in `run_sim_eval.py` with a tiny helper or reuse `run.collect_all_frames()`.
- Use one `write_json(path, payload, atomic=False)` helper for `Path.write_text(json.dumps(...))` call sites, then make only result-file writes atomic.
- Hoist `{"success": "mean"}` to a shared results constant; it appears as the default in both collector and merge/orchestrator paths.
- Consider making `EpisodeResult.timestamp` use `field(default_factory=...)`; that removes the only `__post_init__` in the audited package.
- Narrow `except ImportError` rich fallbacks to a small formatting helper so collector/merge do not each carry their own import branch.
- If `python -m roboeval.cli.main` is not a supported entrypoint, drop the `if __name__ == "__main__"` block in `cli/main.py`; keep `roboeval/__main__.py` for `python -m roboeval`.

### Estimated total cut

- Vestigial removal: ~760 SLOC
- Duplication merge: ~125 SLOC
- Quick wins: ~30 SLOC
- **Total estimated savings: ~915 SLOC (~25%)**

### Caveats

- `run.py` is likely the largest cut, but `pyproject.toml` explicitly documents `python -m roboeval.run` as an unregistered physical-robot CLI. Delete only if maintainers agree that module entrypoint is not part of v0.1.0 support.
- Tests intentionally cover several likely-vestigial helpers (`config.py` mode/RAM/port utilities, `episode_logger.py` readers). Removing code means pruning tests too, even if live CLI behavior is unchanged.
- `world_stubs.py` is not vestigial despite living in `roboeval/`: `sims/env_wrapper.py` subclasses `BaseWorldStub`, so deleting it would affect integration code.
- `run_utils.py` is not independently reachable from the current `roboeval` console command, but it is used by both `run.py` and `run_sim_eval.py`; keep or relocate the two helpers `run_sim_eval.py` needs before trimming the physical CLI.
- `config.py` has broad public-looking helpers. Several are not used by live harness commands, but some are documented and tested; treat them as API-policy decisions, not pure dead-code deletions.
- This pass used static reachability and line inspection only. I did not run tests or execute harness commands because the request was read-only and refactor-audit scoped.
