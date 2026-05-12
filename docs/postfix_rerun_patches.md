# POSTFIX-RERUN patches

Tracking the four parked vendor/runtime fixes assigned by
`li-postfix-rerun-pm2 → patch-and-rerun-pilot` (2026-05-04).

Status legend: ✅ landed, 🟡 partial, ⏸ deferred with rationale.

## (1) ✅ libero10 OffScreenRenderEnv-None / simulator.py:848

- File: `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/simulator.py`
- Change:
  - Wrap `OffScreenRenderEnv(**env_cfg)` (~line 835) in `try/except` that
    logs `env_cfg` keys + bddl path and re-raises a `RuntimeError`.
  - Add explicit `None` guard immediately after construction.
  - Add a second `None`-guard on `self.libero_env` / `.env` right before the
    default-pose capture at line 848.
- Behaviour change: failures that previously surfaced as `'NoneType' object
  has no attribute 'env'` now propagate as a clean `RuntimeError` with the
  underlying construction exception chained via `raise … from exc`.
- Tests: `tests/test_postfix_rerun_patches.py::test_simulator_construction_failure_raises_runtimeerror`
  (text-asserts the guard is in place — full instantiation requires MuJoCo).

## (3) 🟡 task8 visibility cap + SimWrapper timeout

- File: `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/validation_errors.py`
  - `MAX_VISIBILITY_RETRIES` default raised `10 → 100` and made overridable
    via `LIBERO_VISIBILITY_RETRIES` env var. NOTE: spec said the cap was
    "25"; live source had `10`. Either way the new default (100) is the
    intended target.
- File: `roboeval/sims/env_wrapper.py`
  - `_post` and `_get` HTTP timeouts raised from hard-coded `120s → 300s`
    with `ROBOEVAL_SIM_HTTP_TIMEOUT` env override.
- Tests:
  - `test_visibility_retries_default_is_bumped`
  - `test_visibility_retries_env_override`
  - `test_visibility_retries_invalid_falls_back`
  - `test_sim_wrapper_post_uses_env_timeout`
  - `test_sim_wrapper_post_default_timeout_is_300`
- POSTFIX-RERUN PATCH (3-ext): added during smoke (2) interpretation —
  the *outer* Scenic resample loop in `sims/sim_worker.py` (around line
  1594) had its own hard-coded default of `25` (read as `sim_config.get
  ("max_reset_attempts", 25)`). The error string "after N Scenic resample
  attempts" surfaced at line 1845 uses *this* counter, not
  `MAX_VISIBILITY_RETRIES`. Bumped default to 100 via env var
  `LIBERO_MAX_RESET_ATTEMPTS`. Explicit `sim_config.max_reset_attempts`
  values still win. New test:
  `test_max_reset_attempts_env_override_text_present`.
- Deferred portion: the spec also called for an arena-XML `nconmax`
  bump from 5000 → 10000. No arena XML in the libero/libero-infinity vendor
  contains `nconmax=5000`; the only `nconmax` declarations are in
  articulated-object XMLs (`flat_stove.xml` etc., currently 100). The
  observed runtime "ncon=5000" warning therefore originates from the
  default MuJoCo allocation rather than an XML override; bumping it
  requires touching robosuite's `MjSim` construction path and is out of
  scope for a "minimal/reversible" patch. Recommend a follow-up after
  reproducing the warning under the patched retry budget.

## (B) ✅ PATCH-B-LANDED — MuJoCo nconmax XML splice

- File: `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/simulator.py`
  - New helper block (top of file, ~line 73): `_resolve_nconmax_target`,
    `_splice_nconmax`, `_scoped_nconmax_injector` (a `@contextmanager`).
  - `LIBEROSimulation.setup` now runs `OffScreenRenderEnv(**env_cfg)` inside
    `with _ncon_ctx:` (immediately adjacent to the patch (1) try/except),
    monkey-patching `mujoco.MjModel.from_xml_string` / `from_xml_path` for
    the duration of construction so every loaded XML gets
    `<size nconmax="N"/>` spliced in.
- Why XML splice (option b) and not post-construction set (option a):
  empirically verified that `mujoco.MjModel.nconmax` is read-only
  (`AttributeError: property of 'MjModel' object has no setter`) under the
  mujoco bindings shipped in `.venvs/libero_infinity` — option (a) is a
  no-op. The XML splice was probed and confirmed: `nconmax` goes from -1
  (auto) → 10000 in a minimal scene after splice.
- Activation: env var `LIBERO_NCONMAX` (default `10000`, parses as int,
  fall back to 10000 on garbage; set to `0` to disable). Same pattern as
  patch (3) `LIBERO_VISIBILITY_RETRIES`.
- Blast radius: the monkey-patch is scoped via context manager; original
  `from_xml_string` / `from_xml_path` are restored on exit even on
  exception. Concurrent OffScreenRenderEnv constructions in the same
  process would race, but the LIBERO-Infinity backend builds envs
  serially per worker — acceptable.
- Tests (`tests/test_postfix_rerun_patches.py`):
  - `test_patch_b_text_present` — guards the wiring + env-var name.
  - `test_resolve_nconmax_target_default_and_override` — default 10000,
    env override, garbage fallback, `0`-disables.
  - `test_splice_nconmax_inserts_when_absent` — covers the three splice
    paths (no `<size>`, `<size>` without nconmax, `<size>` with nconmax).
- Smoke coverage: smoke (3) (`exp4_task8`) and the queued exp2_task8
  smoke will exercise patch B as a side effect of running task8 +
  combined perturbation under the scoped XML splice. Full final report
  will record per-smoke whether patch B was active.

## (5) ✅ PATCH-5-LANDED — libero10 first-policy-step done-clear guard

- File: `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/simulator.py`
- Root-cause analysis (post-mortem of smoke (1) failure):
  - The 2026-05-03 fix at `setup()` already set `env.done = False` and
    `env.timestep = 0` at the end of setup, AND `step_with_action` had a
    defensive guard at the top that cleared `env.done` if both
    `timestep == 0` AND `done == True`.
  - Yet smoke (1) still surfaced
    `ValueError: executing action in terminated episode` at the FIRST
    `/step` for libero10 task3+task4 combined-perturbation cells, after
    `/init` and `/reset` succeeded. This means *something* between
    setup() and step_with_action either (a) flipped env.done back to
    True without resetting timestep, or (b) advanced timestep past 0
    while leaving done=True — slipping the gate.
- Bounded fix: replace the fragile `(timestep == 0 AND done)` gate with
  an explicit per-episode marker `self._policy_step_taken` (initialized
  False in `__init__`, re-armed False at the end of setup() right next
  to the existing 2026-05-03 done=False reset). On the first call to
  `step_with_action` after setup, we unconditionally clear
  `env.done`/`env.timestep`/`env.cur_time` regardless of their values
  and flip the marker True. Subsequent calls fall through to the
  legacy `(timestep == 0 AND done)` defensive clear, so legitimate
  episode terminations after a real step still propagate.
- Why this is safe:
  - Real episode termination only ever occurs *after* at least one
    policy step has been taken (robosuite sets `done` only inside
    `_post_action`, which runs only inside `step()`).
  - Therefore at the very first policy step, `env.done == True` is
    *always* spurious and clearing it cannot mask any real failure.
- Tests (`tests/test_postfix_rerun_patches.py`):
  - `test_patch_5_first_policy_step_guard_text_present` — guards the
    marker name, the constructor init, and the new step_with_action
    branch.
  - `test_patch_5_first_step_clears_done_with_mock_env` — behavioural:
    a mock env with `done=True, timestep=0` survives the first
    step_with_action call, and a synthetic second-step termination
    *does* propagate.
- Smoke validation: NOT EXECUTED in this cycle (sibling pilots
  pi05-t9-rerun-pilot/pi05-full-rerun-pilot held the GPU/policy
  servers; running additional 5570/5770 smokes risked port/memory
  collision with the active reruns). Recommended smoke before any
  full libero10 launch:

      ports 5570/5770, manifest results/neurips2026/postfix_rerun/_smokes/libero10/manifest.yaml,
      episodes_per_task=5, run --mode pilot. Goal: ep0 reaches step
      loop on at least one task; if all 5 produce real ≥1-step
      rollouts, declare VALIDATED.

  The patch is purely additive to the existing 2026-05-03 fix and is
  reversible by deleting the marker block (one constructor line + one
  setup() line + one step_with_action branch).

## (6) ✅ PATCH-6-LANDED — /init timeout knob + per-task init-state skip-list

Two-pronged fix for exp4 task8 (combined perturbation, Scenic compile
> 5 min on the long tail).

### (6a) Separated `/init` timeout knob

- File: `roboeval/sims/env_wrapper.py::_post`
- Change: `/init` now reads `ROBOEVAL_SIM_INIT_TIMEOUT` (falling back to
  `ROBOEVAL_SIM_HTTP_TIMEOUT`, default `900`s). All other paths (incl.
  `/step`, `/reset`, `/obs`, `/health`) keep the existing
  `ROBOEVAL_SIM_HTTP_TIMEOUT` knob with default `300`s.
- Why separated: a 900s `/step` timeout would mask runaway policy
  behaviour for 15 minutes. `/init` is the only path that legitimately
  takes >5 minutes (Scenic compile + nconmax-spliced MjModel
  construction).
- Tests:
  - `test_patch_6_init_timeout_knob_text_present`
  - `test_patch_6_init_path_uses_init_timeout` — covers default 900s
    on `/init`, override via `ROBOEVAL_SIM_INIT_TIMEOUT`, and that
    `/step` still uses `ROBOEVAL_SIM_HTTP_TIMEOUT`.

### (6b) Per-task init-state skip-list

- New env var: `LIBERO_TASK8_SKIP_INIT_STATES` (parallel to
  `LIBERO_SCENIC_SKIP_INDICES`).
- New file: `configs/libero_infinity_task8_skip_init_states.yaml` —
  ships with empty `indices` and `per_task['8']` so enabling the env
  var causes zero behaviour change until an operator populates the
  list from observed failures.
- Runner wiring: `sims/sim_worker.py`
  - new module-level `_load_init_state_skip_indices(task_name)` helper
    + `_INIT_STATE_SKIP_CACHE` (mirrors PATCH-4's loader),
  - `LIBEROInfinity.init` now stores
    `self._init_state_skip_indices` alongside the existing
    `self._scenic_skip_indices`,
  - `LIBEROInfinity.reset` raises a clean `RuntimeError` prefixed with
    `[INIT-STATE-SKIP-LIST]` *before* invoking
    `self._scenario.generate(...)` — same short-circuit shape as
    PATCH-4, treated as a clean skip by sim_worker.
- Tests:
  - `test_patch_6_init_state_skip_loader_unset_returns_empty`
  - `test_patch_6_init_state_skip_loader_honors_yaml`
  - `test_patch_6_init_state_skip_loader_bad_path_is_defensive`
  - `test_patch_6_init_state_skip_yaml_shipped_loads`
  - `test_patch_6_reset_short_circuits_on_init_state_skip` — guards
    the runtime wiring text in sim_worker.py.
- Activation for the exp4 task8 rerun:

      export ROBOEVAL_SIM_INIT_TIMEOUT=900
      export LIBERO_TASK8_SKIP_INIT_STATES=configs/libero_infinity_task8_skip_init_states.yaml

  (the timeout alone is sufficient to recover the long tail; the
  skip-list is only needed if specific init states refuse to compile
  even at 900s).
- Smoke validation: NOT EXECUTED (same GPU/port-budget reason as
  PATCH-5). Recommended smoke before any full exp4 task8 rerun:

      ports 5602/5802, manifest results/neurips2026/postfix_rerun/_smokes/exp4_task8/manifest.yaml,
      episodes_per_task=5, ROBOEVAL_SIM_INIT_TIMEOUT=900,
      LIBERO_TASK8_SKIP_INIT_STATES=configs/libero_infinity_task8_skip_init_states.yaml.
      Goal: at least 1/5 episodes produces real steps>0; if 3/5 do,
      declare VALIDATED.

## (2) ✅ PATCH-2-LANDED — Scenic veneer activate/deactivate symmetry (verified)

- Investigation outcome: the upstream Scenic call sites for
  `activate(...)` / `deactivate()` are exactly two:
  1. `scenic/syntax/translator.py:277-323` — the primary compile path.
     Already correctly wrapped: `veneer.activate(...)` is followed by
     `try:` ... `finally: veneer.deactivate()`. A failed compile thus
     cannot leak the global `activity` counter.
  2. `scenic/core/scenarios.py::_Activator/_Deactivator` — the pickle
     path. Activate and deactivate live on *different* helper objects
     placed in the pickle stream by `_ScenarioPickleMixin.__getstate__`.
     Pickle protocol calls them in sequence, so they are not a single
     caller pairing — there is no place to wrap them in a single
     try/finally without restructuring the pickle contract (which
     would violate the "minimal/reversible" constraint).
- Decision: no monkey-patch landed. The existing translator wrap is
  sufficient for every compile path exercised by the LIBERO-Infinity
  runner. The pickle path was not observed firing in any postfix_rerun
  smoke and remains mitigated by the runner-level fresh-process retry.
- Tests added:
  - `test_scenic_translator_compile_wraps_in_try_finally` — text-asserts
    the structural wrap at translator.py:277-323. Guards against an
    upstream regression (e.g., a Scenic upgrade) that drops the wrap.
  - `test_scenic_compile_failure_does_not_leak_activity` — functional:
    runs `scenarioFromString("invalid !!!")` against the live Scenic
    install, asserts `veneer.activity == 0` after the raise. Skips if
    Scenic is unavailable in the test venv.
- Smoke validation: not required (no source change made; tests
  already cover the property of interest).

## RESIDUAL DEFECT — out of scope for this cycle (HISTORICAL)

### `executing action in terminated episode` on libero10 task3/task4 ep0

Observed on the smoke (1) pilot run (`results/neurips2026/postfix_rerun/_smokes/libero10/_logs/libero10_task{3,4}_smoke/run_pilot.log`).
Both ep0s succeeded at `/init` (patch (1) verified — zero `NoneType.env`
hits) but failed at the first `/step`, hitting:

```
ValueError: executing action in terminated episode
  at robosuite/environments/base.py:376                   (raise site)
  via libero/libero/envs/bddl_base_domain.py:806
  via libero/libero/envs/env_wrapper.py:88
  via vendors/libero-infinity/src/libero_infinity/simulator.py:1239 (step_with_action)
  via roboeval/sims/sim_worker.py:1859 (LIBEROInfinity.step)
```

Hypothesis: the `done` flag is set inside the env on/before the first
action, likely because `OffScreenRenderEnv.reset()` returns an already
goal-satisfied observation under combined-perturbation seed=42 on these
LIBERO-10 cells, or an internal step-count guard fires on step 0. The
defect is upstream of every patch in this cycle — neither patch (1),
(3), (4), nor (B) interacts with the robosuite step-loop or LIBERO
done-flag logic.

Recommendation for follow-up ticket:
- Repro: launch any LIBERO-10 combined-perturbation cell, watch the
  first observation's `done` value before stepping.
- Likely fix layer: LIBERO `bddl_base_domain.step` reset-state check, or
  robosuite `Environment.step`'s terminated guard.
- Not blocking the NeurIPS reruns for exp1/exp2/exp3/exp4 (those did
  not exhibit this signature in any of the postfix_rerun logs).
- **UPDATE (residual-defects-pilot 2026-05-04):** addressed by
  PATCH-5-LANDED above. The first-policy-step done-clear guard is
  unconditional, regardless of how the gate was previously slipping.

## (2) ⏸ Scenic veneer state-leak (deferred — superseded by PATCH-2-LANDED above)

- File: vendored Scenic `scenic/syntax/veneer.py:527` (also 343, 641, 651
  carry the same `assert activity == 0` invariant).
- The `activity` counter is incremented in `activate()` and decremented in
  `deactivate()`; an asymmetric raise between the two leaves the global at
  `>0` and poisons every subsequent compile in the same process. Locating
  the asymmetric path requires reproducing the failure under instrumented
  Scenic, which the runner already mitigates via fresh-process retry.
- Recommendation: keep the runner-level fresh-process retry as the
  workaround; open a tracked Scenic upstream bug rather than monkey-patch
  the veneer globals (high regression surface — would affect every Scenic
  consumer in the project).

## (4) ✅ PATCH-4-LANDED — Scenic Subtype-A seed-deterministic skip-list

- Source data: extracted from `results/neurips2026/postfix_rerun/exp3_parallel/_logs/{combined_task0,combined_task1}/run.log`
  by grepping `Episode task=N ep=M exited with code 1`. The 23-index
  intersection across task0 ∩ task1 is the byte-identical Subtype-A list:
  `{10,19,29,31,53,63,64,67,69,73,80,85,86,93,101,103,105,114,116,117,119,120,142}`.
  task-0 has one extra deterministic failure at index 43.
- New file: `configs/libero_infinity_skip_indices_seed42.yaml` — committed
  YAML containing `seed:`, flat `indices:`, and `per_task:` sections.
- Runner wiring: `sims/sim_worker.py`
  - new module-level `_load_scenic_skip_indices(task_name)` helper +
    cache, gated on env var `LIBERO_SCENIC_SKIP_INDICES`,
  - `LIBEROInfinity.init` now stores `self._task_name` and
    `self._scenic_skip_indices`,
  - `LIBEROInfinity.reset` raises a clean `RuntimeError` prefixed with
    `[SCENIC-SKIP-LIST]` *before* invoking `self._scenario.generate(...)`,
    avoiding the 25k-iteration burn. Existing runner skip+continue path
    treats the RuntimeError as an episode skip.
- Activation: `export LIBERO_SCENIC_SKIP_INDICES=configs/libero_infinity_skip_indices_seed42.yaml`
  before launching a run; deactivate by `unset`. Default behaviour
  (env unset) is unchanged.
- Tests (`tests/test_postfix_rerun_patches.py`):
  - `test_skip_list_yaml_contains_known_23_shared_indices`
  - `test_load_scenic_skip_indices_unset_returns_empty`
  - `test_load_scenic_skip_indices_honors_yaml`
  - `test_load_scenic_skip_indices_bad_path_is_defensive`
  - `test_shipped_yaml_loads_via_loader`

## (5b) 🟡 PATCH-5b-LANDED-SMOKE-RED — walk-to-leaf done-clear (code landed, smoke still RED)

**Verdict from re-smoke (residual-defects-v2-pilot 2026-05-04):** code is
landed and tested at the unit level, but the libero10 task3+task4 smoke
on ports 5602 still surfaces ``ValueError: executing action in
terminated episode`` at the FIRST policy /step. **Walking to the leaf
did not clear the symptom.**

### Why the leaf walk is empirically a no-op for this chain

The LIBERO env composition is exactly one ``.env`` deep:
``LIBEROSimulation.libero_env`` (= ``OffScreenRenderEnv`` →
``ControlEnv``) → ``.env`` (= a ``BDDLBaseDomain`` subclass which
inherits from ``SingleArmEnv`` → ``ManipulationEnv`` → ``RobotEnv`` →
``robosuite.environments.base.MujocoEnv``). The
``BDDLBaseDomain`` instance IS the ``MujocoEnv`` instance via
inheritance — there is no ``self.env`` deeper. So
``_resolve_leaf_env(self.libero_env)`` returns ``self.libero_env.env``,
which is exactly what patch (5) was already writing to. The
walk-to-leaf is correct as a defensive idiom but does not resolve THIS
specific failure.

### Next layer of root cause (handoff for the next pilot)

Confirmed empirically:

1. ``robosuite/environments/base.py:376`` raises ``ValueError`` when
   ``self.done`` is True at the start of ``MujocoEnv.step``. ``self`` is
   the ``BDDLBaseDomain`` instance.
2. There is no ``done`` property/setter on the inheritance chain (grep
   confirms).
3. There is no ``__setattr__`` interceptor.
4. Setup() ends with ``leaf.done = False``; ``_post_reset_settle``
   restores ``saved_done = False``; ``step_with_action`` first-step
   path again writes ``leaf.done = False`` immediately before
   ``self.libero_env.step(action)``.

Therefore *something between our last write and the
``MujocoEnv.step`` read* sets ``self.done = True``. Candidates left to
investigate (in priority order):

- (i) ``ControlEnv.step`` does ``return self.env.step(action)``. If
  ``self.env`` is reassigned between init and first /step (e.g. by a
  hidden retry path or by the orchestrator calling ``set_init_state``
  / ``regenerate_obs_from_state`` which calls ``set_state`` then
  ``check_success`` then ``_post_process``), the leaf object identity
  could change. Worth printing ``id(self.libero_env.env)`` at three
  points: end of setup, end of post_reset_settle, top of
  step_with_action.
- (ii) ``BDDLBaseDomain.step`` line 806 calls ``super().step(action)``
  but the BDDL subclasses for libero10 task3+task4 may override
  ``step`` themselves and call into a different super. Check
  ``TASK_MAPPING`` for the actual class name surfaced by these BDDLs
  and grep for ``self.done = True`` in that subclass.
- (iii) The ``hard_reset=True`` flag passed by ``OffScreenRenderEnv``
  triggers a *full* MjSim rebuild on every ``reset()`` call. If
  anything between setup() and first /step calls ``env.reset()`` (e.g.
  a wrapper-level retry on a transient render error), the rebuilt env
  loses our ``done = False`` write. Probe: instrument ``MujocoEnv.reset``
  with a stack-trace print restricted to libero10 cells.
- (iv) **Pragmatic mitigation** (recommended if (i)-(iii) take more
  than one cycle to diagnose): pass ``ignore_done=True`` to
  ``OffScreenRenderEnv`` via the ``env_kwargs`` block in
  ``LIBEROSimulator.__init__``. With ``ignore_done=True``, robosuite's
  ``_post_action`` sets ``self.done = (timestep >= horizon) and not
  ignore_done`` → ``False`` always. The terminated-episode raise at
  base.py:376 cannot fire. The user-facing ``done`` returned from
  ``BDDLBaseDomain.step`` is computed from ``_check_success`` (success
  = goal predicate true), so ``ignore_done=True`` does not affect
  evaluation semantics — only the internal horizon-termination flag.
  This is one line in ``sim_worker.LIBEROInfinity.init`` (or in the
  LIBEROSimulator default ``env_kwargs``).

### Source state

- Helper: ``_resolve_leaf_env(env)`` at ~line 86 of
  ``vendors/libero-infinity/src/libero_infinity/simulator.py``.
- Setup() end-of-method clear: ~line 1216.
- ``step_with_action`` first-step clear: ~line 1397.
- Tests: ``test_patch_5b_text_present``,
  ``test_patch_5b_resolve_leaf_env_walks_chain``.

Recommendation: keep the walk-to-leaf change committed (it's a
correctness improvement against any future deeper-wrapping). DO NOT
authorize a libero10 full rerun with combined-perturbation cells until
the next-layer fix lands.

### Original landing notes (preserved for trail)

- File: `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/simulator.py`
- Hypothesis from `final-smokes-pilot` (2026-05-04): patch (5)'s
  single-level write `self.libero_env.env.done = False` did not reach
  the robosuite ``MujocoEnv`` instance whose ``self.done`` is checked
  at ``base.py:376``. The leaf is reached by walking ``.env`` until no
  ``.env`` attribute remains. **(Verdict above: hypothesis incorrect
  for this wrapper composition; leaf is at depth 1.)**
- Fix:
  - New module-level helper `_resolve_leaf_env(env)` (top of file, ~line
    74) walks the wrapper chain defensively (`seen` guard against cycles).
  - `setup()` end-of-method clear and `step_with_action()` first-policy-
    step clear both now call `_resolve_leaf_env(self.libero_env)` and
    write `done`, `_done`, `timestep`, `cur_time` only when each
    attribute exists on the leaf.
- Tests:
  - `test_patch_5b_text_present` — guards the marker + helper name.
  - `test_patch_5b_resolve_leaf_env_walks_chain` — builds a 3-deep
    wrapper chain (`Wrap(Wrap(Wrap(Leaf)))`) and asserts the resolver
    returns the leaf and that the patch-5b write pattern actually
    clears `leaf.done`/`leaf._done`/`leaf.timestep`/`leaf.cur_time`.

## (6b) ✅ PATCH-6b-LANDED — visibility-validator cap + skip-list suite scoping

Two-part follow-up after `final-smokes-pilot` proved (a) 100 outer-loop
retries are not enough for some libero_infinity_spatial cells under
combined perturbation, and (b) the flat-`indices:` block of the seed=42
skip-list authored for `libero_infinity_goal` was bleeding into
`libero_infinity_spatial` task8 (init state 10).

### (6b-a) `LIBERO_VISIBILITY_VALIDATOR_RETRIES` env var

- File: `sims/sim_worker.py::LIBEROInfinity.init`
- New env var (default `1000`) is read alongside `LIBERO_MAX_RESET_ATTEMPTS`
  (default `100`) and the effective default is `max(...)` of the two —
  so the new knob can only RAISE the cap. Explicit
  `sim_config.max_reset_attempts` still wins. Garbage falls back to
  1000.
- Activation for the exp4 task8 rerun: env var is set in the smoke
  scripts; the flat-config rerun harness inherits it from the operator
  shell (or the queue runner script).
- Tests:
  - `test_patch_6b_visibility_validator_retries_text_present`
  - `test_patch_6b_visibility_validator_retries_env_override` — exercises
    default 1000, env override, garbage fallback, max() interaction with
    LIBERO_MAX_RESET_ATTEMPTS, and explicit sim_config still wins.

### (6b-b) Suite-scoped skip-list schema

- New schema in `configs/libero_infinity_skip_indices_seed42.yaml` and
  `configs/libero_infinity_task8_skip_init_states.yaml`:
      seed: <int>
      per_suite_task:
        "<suite>":
          "<task>": [int, ...]
      legacy_per_task_suite: <suite>   # optional gate for the legacy
                                       # `indices:` and `per_task:`
                                       # lists; only honored when the
                                       # caller's suite matches.
- Loaders `_load_scenic_skip_indices` and `_load_init_state_skip_indices`
  in `sims/sim_worker.py` now take a `suite=` kwarg (passed by
  `LIBEROInfinity.init` from its incoming `suite` argument). Resolution
  is scoped: `per_suite_task[suite][task]` always applies; the legacy
  flat lists ONLY apply when `legacy_per_task_suite` matches.
- The shipped seed=42 yaml has been migrated: the 23-entry shared list
  now lives under `per_suite_task.libero_infinity_goal["0"]` (with
  task-0's extra `43`) and `per_suite_task.libero_infinity_goal["1"]`.
  No indices apply to any other suite.
- Tests:
  - `test_skip_list_yaml_contains_known_23_shared_indices` — updated to
    assert the new schema location.
  - `test_load_scenic_skip_indices_honors_yaml` — rewritten for new
    schema.
  - `test_load_scenic_skip_indices_no_cross_suite_leak` (NEW) — proves
    that a list authored under `libero_infinity_goal` does NOT bleed
    into `libero_infinity_spatial` task8, even when both
    `per_suite_task` and the legacy flat blocks are present.
  - `test_patch_6_init_state_skip_loader_honors_yaml` — rewritten.
  - `test_patch_6_init_state_skip_yaml_shipped_loads` — updated.

## Smoke pilots

NOT executed. Live smoke pilots require:
  - GPU/MuJoCo capable runtime + per-policy checkpoints,
  - confirmed-free port pairs (5570/5770, 5582/5782, etc.) — sibling
    `pi05-tail-completion` may still hold 5530/5730/5571/5771,
  - the exact `scripts/run_libero_infinity_gb10_queue.py` invocation per
    cell.

Returning to parent for go/no-go on launching them with the patched
vendor tree.
