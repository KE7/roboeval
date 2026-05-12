# Recurring Issues — Root Cause Analysis

Author: `re-recurring-rca` (Workstream 4 of 4 toward making libero-infinity publishable)
Date: 2026-05-11
Scope: ~10 Claude/Codex roboeval transcripts (~17,075 LI mentions) under
`~/.claude/projects/-home-batman-Documents-research-roboeval/`.

Shared context: `~/.omar/ea/4/libero_infinity_validation_plan.md` — every
recurring bug below is a publication blocker for the 66,560-condition
LIBERO-Infinity validation matrix.

This is **RCA, not fixes**. Only one trivially-safe P0 fix is proposed for a
follow-up PR; items 1–4 and 6 ship as proposed fixes for review first.

---

## Severity Ledger

| #  | Issue                                                          | Severity | Status                                    |
|----|----------------------------------------------------------------|----------|-------------------------------------------|
| 1  | VisibilityError retry bypass                                   | **P0**   | Proposed fix (no implementation yet)       |
| 2  | RejectionException not caught at right layer                   | **P0**   | Proposed fix (no implementation yet)       |
| 3  | `sim_worker.py` dying mid-run with no recovery                 | **P1**   | Proposed fix (no implementation yet)       |
| 4  | Port reuse races on parallel sim_worker launches               | **P1**   | Proposed fix (no implementation yet)       |
| 5  | Robosuite terminated-episode signature drift                   | **P0**   | **Deferred — owned by WS3 (`postfix-rerun-patches`, commit `c606b7c`)** |
| 6  | Gate-script shell bug                                          | **P1**   | Proposed fix (no implementation yet)       |

P0 = blocks validation. P1 = makes validation flake. P2 = cosmetic.

---

## 1. VisibilityError retry bypass — **P0**

### Symptom (transcripts)
- `Reset/startup failure excluded from eval episode task=8 ep=0 scene_ep=0 attempt=1/20; retrying` immediately followed by `Episode task=8 ep=5 exited with code 1` with **no further attempt logs** — the retry loop never iterated.
- "Exhausted N retries after VisibilityError" surfacing on what should be the first sample, with `n_visibility=0` in logs.
- Numerous "VisibilityError" → `InfeasibleScenarioError` chains in stderr that bypass the outer `sim_worker` resample loop.

### Code path
- Outer resample loop: `sims/sim_worker.py:1902–1971` (`LiberoInfinityBackend.reset_episode`).
- Inner per-attempt visibility retries: `libero-infinity` vendor at
  `~/.local/share/roboeval/vendors/libero-infinity/src/libero_infinity/simulator.py:2454–2549`
  (`run_with_validation_loop`), driven by `MAX_VISIBILITY_RETRIES`.
- Env overrides:
  - `LIBERO_VISIBILITY_VALIDATOR_RETRIES` (inner)
  - `LIBERO_MAX_RESET_ATTEMPTS` (outer; capped to ≥ inner — `sim_worker.py:1732`)

### Root cause (not symptom)
The outer loop at `sim_worker.py:1947–1952` is gated on a brittle **string
substring match**:

```python
except RuntimeError as exc:
    if not ("Invalid Scenic sample after MuJoCo settling" in str(exc)
            or "Invalid Scenic sample after visibility check" in str(exc)):
        raise
```

Three structural failure modes bypass this gate, each empirically observed
in transcripts:

1. **`VisibilityError` / `InfeasibleScenarioError`** raised by
   `run_with_validation_loop` are *not* `RuntimeError` subclasses and *do not*
   contain those magic substrings — they propagate up untouched and the
   outer retry never fires.
2. **`scenic.core.distributions.RejectionException`** raised when
   `scenario.generate(maxIterations=…)` exhausts its budget (see issue #2)
   is not a `RuntimeError` either.
3. Any vendor wrapper that reformats the error message (e.g. translating
   "visibility check" to "validator failed" — observed in newer LI vendor
   patches) silently drops out of the substring filter.

The reason this **keeps coming back** is that the gate is *type-agnostic* and
*string-coupled*: every refactor in the vendor that touches error text breaks
it again. Patch (3-ext) in `docs/postfix_rerun_patches.md` raised the *budget*
but did not address the *bypass*.

### Proposed fix
- Replace the substring-match in `sim_worker.py:1947–1952` with an explicit
  list of recovery-eligible exception types imported from
  `libero_infinity.errors` (`VisibilityError`, `InfeasibleScenarioError`)
  plus `scenic.core.distributions.RejectionException`. Fall back to the
  current substring match only when the optional imports are unavailable.
- Log at WARNING with `attempt`, `max_reset_attempts`, exception type, and
  `seed_i` on every retry, so the bypass cannot regress silently.
- Add a unit test that injects a `VisibilityError`-raising stub scenario and
  asserts `_max_reset_attempts` consecutive retries occur before raise.

### Coordination
None — outer loop only.

---

## 2. `RejectionException` (from Scenic) not caught at right layer — **P0**

### Symptom (transcripts)
- `scenic.core.distributions.RejectionException: failed to generate scenario
  in 25000 iterations` on Exp4 task2 / task3 → 100% of reset attempts fail
  (`li-exp4-task23-scenic-rca` report).
- Hits the orchestrator as `nonzero_exit_1` with no recovery path — every
  episode is logged as a hard failure even though the underlying issue is
  scene-infeasibility, not environment death.

### Code path
- Raised at: `scenario.generate(maxIterations=…)` inside `sim_worker.py:1912`.
- Inner LI loop at `simulator.py:2502` *also* calls `scenario.generate(…)`
  but wraps only `CollisionError` and `VisibilityError`, not
  `RejectionException`.
- Orchestrator: `roboeval/orchestrator.py:708–724` records the subprocess
  exit code; there is no per-error-class branch.

### Root cause
`RejectionException` is a Scenic-internal control-flow signal indicating
"the sampler couldn't satisfy `require` constraints in maxIterations". It
maps semantically to "this scene is infeasible — try another seed", which is
exactly what `_max_reset_attempts` is designed for. But:

1. No layer along the stack `(Scenic → LI run_with_validation_loop →
   sim_worker.reset_episode → orchestrator)` catches it. It propagates as
   an unhandled exception, the subprocess exits non-zero, and the episode
   is counted as a hard failure — eating the per-episode budget without
   re-sampling.
2. The root-cause failure mode it indicates (unsatisfiable
   `require (distance from bowl to fixture) > 0.27 m` against a bowl placed
   at ≤0.05 m offset; see `li-exp4-task23-scenic-rca` write-up) is an
   **infeasible feasible-set** — increasing `maxIterations` is futile. The
   actual fix is in the LI renderer (skip fixture-clearance constraints
   when the object is `on` the fixture), but until that lands the eval
   harness should at minimum stop reporting these as crashes.

It keeps coming back because each new perturbation axis (articulation,
distractor occlusion) introduces new constraint geometries that can become
mutually unsatisfiable, and nothing in the harness flags this distinct from
a genuine crash.

### Proposed fix
- Catch `scenic.core.distributions.RejectionException` at
  `sim_worker.py:1947` alongside the typed exceptions from issue #1 and
  treat it as a resample-eligible failure (re-seed and retry).
- After `_max_reset_attempts` exhausted RejectionExceptions, raise a *new*
  typed exception `SceneInfeasibleError` so the orchestrator can label this
  distinctly from `nonzero_exit_*` in
  `roboeval/orchestrator.py:716–724`. This unblocks gate G3 (Scenic
  sampling within maxIterations=2000) accounting for the validation matrix.
- Long-term (out of scope for this PR): the LI renderer fix described in
  `li-exp4-task23-scenic-rca` should land in `libero-infinity` separately.

### Coordination
None for the harness-side catch. The renderer-side fix is an LI-repo
follow-up.

---

## 3. `sim_worker.py` dying mid-run with no recovery — **P1**

### Symptom (transcripts)
- `WARNING roboeval.orchestrator: Episode task=8 ep=5 exited with code 1`
  with no traceback in stdout (the subprocess died before flushing).
- Long-running server-mode sim workers (started via `roboeval serve`) become
  unresponsive after a `OffScreenRenderEnv` returns `None` (patch (1)
  scenario) or after a MuJoCo `ncon` overflow; subsequent `/step` calls hit
  `Connection refused`.
- Per-episode subprocess mode is robust (orchestrator launches a fresh
  process per episode) but the **server-mode** path used by
  `scripts/run_libero_infinity_gb10_queue.py` does not restart a dead
  worker — every subsequent episode in the cell fails until the harness
  itself is killed and relaunched.

### Code path
- Per-episode subprocess: `roboeval/orchestrator.py:687–724`.
- Long-lived server: `roboeval/server_runner.py` (sim and vla servers),
  launched by `roboeval serve` from `roboeval/cli/main.py:75–110+`.
- Sim HTTP client: `sims/env_wrapper.py` (`_post`, `_get`) — timeout raised
  from 120 → 300 s by postfix patch (3) but no liveness probe / restart.

### Root cause
There are two distinct sub-issues conflated under "sim_worker dying":

1. **Subprocess mode** (orchestrator): exit code is correctly captured but
   the failure is attributed as a generic episode failure even when the root
   cause is a transient `MuJoCo nconmax` exceedance or `OffScreenRenderEnv`
   returning `None`. WS3 patches (1) and (B) address the **construction**
   side; the orchestrator side has no classifier to distinguish
   transient-and-retryable from terminal.
2. **Server mode** (`roboeval serve`): the HTTP client in
   `sims/env_wrapper.py` retries on TimeoutError but **does not health-check
   or respawn the server**. When the underlying sim_worker process dies,
   every subsequent request gets `ConnectionRefusedError`, the harness keeps
   trying for the configured timeout × episode-count, and the cell is
   effectively wedged.

The reason this keeps coming back: every postfix patch addresses a specific
failure mode (None env, nconmax, terminated-episode) but the absence of a
*supervisor* means the next unknown failure produces the same wedged state.

### Proposed fix
- Subprocess mode: Add an exit-code classifier in
  `roboeval/orchestrator.py` that distinguishes
  `nonzero_exit_{construction,transient,terminal}` based on stderr
  signatures, and add a single bounded retry for `construction` and
  `transient` classes only.
- Server mode: Wrap the sim HTTP client in `sims/env_wrapper.py` with a
  liveness probe (`GET /healthz`, already exists for vla — check sim side)
  and on `ConnectionRefusedError`, fire a single respawn via the same
  launch path used at startup. Cap respawns per cell (default: 3) to
  prevent infinite respawn loops on a deterministic crash.
- Persist the most recent 50 KB of sim_worker stderr to a sidecar log file
  on respawn so post-hoc debugging stops requiring transcript scraping.

### Coordination
Touches the same files (`sims/sim_worker.py`, `sims/env_wrapper.py`) that
WS1's scenic-hardening and WS3's postfix patches modify — sequence after
both land.

---

## 4. Port reuse races on parallel `sim_worker` launches — **P1**

### Symptom (transcripts)
- `OSError: [Errno 98] Address already in use` after `roboeval/cli` reported
  the port as free.
- Two sim workers launched seconds apart in `run_exp1_exp3_gated.sh`
  occasionally claim the same port; the second worker exits with port-bind
  error and the cell records every episode in its lane as a hard failure.

### Code path
- TOCTOU port check: `roboeval/config.py:96–127` (`is_port_available`,
  `find_available_port`).
- Block allocation: `roboeval/config.py:130–156` (`find_available_port_block`).
- Callers: `roboeval/cli/main.py:75–110+` (CLI `serve` flow) and
  `scripts/run_libero_infinity_gb10_queue.py` (parallel lane harness).

### Root cause
Classic TOCTOU. `is_port_available` `bind`s + closes the socket, returns
True, the caller then `Popen`s a server process that re-binds the same
port. Between the close and the re-bind, **any** other process — including
a sibling sim_worker in the same parallel harness — can grab it.

Why it keeps coming back: every patch that touches port allocation
(`find_available_port_block`, env_wrapper retries) treats this as a
flakiness symptom and adds retries. The races persist because the
allocator never *holds* the port across the launch boundary, and the
parallel harness has no cross-process port-allocation lock.

### Proposed fix
- Either: (a) keep the listening socket open in `find_available_port[*]`
  and pass the file descriptor to the child via `SO_REUSEADDR` + `fork`
  / `subprocess.Popen(pass_fds=…)`; or (b) introduce a small flock-backed
  port reservation table at `~/.cache/roboeval/port_reservations.json`
  (TTL 60 s) so concurrent harness lanes coordinate. Option (b) is the
  smaller blast radius and is the recommended path.
- Until the holder fix lands, add a bounded retry around the launch
  itself: if the child exits with EADDRINUSE within 2 s, free the
  reservation and re-allocate.
- Regression test: parametrized pytest with `multiprocessing.Pool` calling
  `find_available_port` concurrently and asserting all returned ports are
  unique under contention.

### Coordination
None — touches `roboeval/config.py` and the two launch sites only.

---

## 5. Robosuite terminated-episode signature drift — **P0 — DEFERRED**

### Symptom (transcripts)
- `ValueError: executing action in terminated episode` on first `/step`
  after `setup()` + `reset()`, libero10 task3/task4 ep0 combined
  perturbation cells.
- Survives the 2026-05-03 `env.done=False` + `env.timestep=0` reset and the
  defensive `(timestep == 0 AND done)` clear in `step_with_action`.

### Status
**Owned by Workstream 3** — branch `postfix-rerun-patches`, commit
`c606b7c "fix(libero-infinity): postfix-rerun runtime patches (G5/G6
gates)"`. The fix introduces an explicit `_policy_step_taken` per-episode
marker (rationale in `docs/postfix_rerun_patches.md:95–125`) replacing the
fragile `(timestep == 0 AND done)` gate.

No action from this RCA per task spec item 1f. PR review should confirm
that:
1. `_policy_step_taken` is re-armed `False` at the end of every
   `setup()` call (not just `__init__`).
2. The unconditional clear in `step_with_action` runs before the legacy
   defensive gate.
3. `ignore_done=False` is preserved at the env-construction call site.

### Coordination
This RCA defers entirely to WS3. If WS3's PR does not land before
validation, escalate; do not duplicate the fix.

---

## 6. Gate-script shell bug — **P1**

### Symptom (transcripts)
- File named `run_remaining_after_gate_bug.sh` in `results/neurips2026/postfix_rerun/`,
  with the commit comment "Replacement launcher after gate-locked harness
  was killed (option B)" — direct evidence of a prior gate-script bug.
- Cells whose `run_one_cell` invocation failed silently incremented
  EXP1_FAILURES=0 in the harness log despite the underlying harness
  crashing.

### Code path
- `results/neurips2026/postfix_rerun/run_exp1_exp3_gated.sh:12` (`set -e`)
- `run_exp1_exp3_gated.sh:56–93` (`run_one_cell` function).
- Loop sites: `run_exp1_exp3_gated.sh:114–117` and `134–137`:
  `run_one_cell ... || EXP1_FAILURES=$((EXP1_FAILURES + 1))`.

### Root cause
`set -e` interacts pathologically with the `run_one_cell` function:

```bash
run_one_cell() {
  ...
  $PYTHON $HARNESS ...
  local rc=$?
  echo "... rc=$rc"
  return $rc
}
```

Under `set -e`, when `$PYTHON $HARNESS` exits non-zero, the function exits
*immediately* at that line — `local rc=$?` is never executed, the trailing
`return $rc` is never reached, and the function terminates with a non-zero
status. *Normally* `set -e` would then kill the whole script — but because
the caller has `... || EXP1_FAILURES=...`, `set -e` is suppressed inside
the function call as a "checked context". Bash's documented behavior is
that `set -e` is **disabled inside the entire called function** when its
return value is being tested (`||`/`&&`/`if`).

The compounding effect: any failure *inside* `run_one_cell` other than the
final harness call (e.g. the `$PYTHON -c "..."` single-cell manifest
extraction at lines 66–78) silently exits the function with non-zero
*before* reaching the harness, and the cell is recorded as a single
counted failure — but no harness was actually launched, so the user thinks
"one harness failed" when in fact *zero* harnesses ran. This is what the
filename "after gate bug" alludes to: the gate function returned early on
a manifest-extraction error and the for-loop kept iterating.

It keeps coming back because the pattern `set -e` + functions + `|| counter`
is widely copy-pasted across `results/neurips2026/**/*.sh`.

### Proposed fix
- In every `run_one_cell`-style function under
  `results/neurips2026/**/*.sh`:
  - Remove `local rc=$?` and instead invoke the harness with
    `$PYTHON $HARNESS ... ; rc=$?` (or wrap in `if ! $PYTHON $HARNESS …;
    then rc=$?; …; fi`).
  - Explicitly `set +e` at function entry and `set -e` at exit (or drop
    `set -e` from the script and use `pipefail`-style checks per-call).
  - Distinguish manifest-extraction failures from harness failures via
    distinct exit codes (`exit 2` / `exit 3`) so the for-loop log records
    *what* failed.
- Add a `shellcheck` CI lint over `results/**/*.sh` and
  `scripts/**/*.sh`. The current state contains multiple SC2155
  violations (`local rc=$(cmd)`) which are textbook instances of this
  bug class.
- Regression: a small bats test that invokes a stubbed `run_one_cell`
  with both a manifest failure and a harness failure, asserting both
  increment the counter and produce distinguishable log lines.

### Coordination
None — touches `results/**/*.sh` only. Safe to land in parallel with any
other workstream.

---

## Summary

- 4 of 6 are P0/P1 harness bugs with clear, bounded, type-safe fixes.
- Item 5 (terminated-episode) is fully owned by WS3 — do not duplicate.
- Item 1 (VisibilityError bypass) is the highest-leverage single fix:
  switching from substring-match to typed-exception recovery in
  `sim_worker.reset_episode` simultaneously resolves issues #1 and #2 and
  prevents future vendor refactors from silently breaking the retry
  contract.
- Item 4 (port races) is the most pernicious "flake symptom" — recommend
  the flock-reservation path over more retry layers.

## Trivially-safe P0 fix included in follow-up

None. Issues #1 and #2 are P0 but both touch `sim_worker.py:1947–1952`,
which is also modified on WS1 (`scenic-hardening`) and WS3
(`postfix-rerun-patches`). Landing either fix from this RCA before those
two branches merge risks a non-trivial conflict at exactly the line range
both other workstreams edit. Recommendation: open the RCA PR for review
*now*, wait for WS1 + WS3 to merge, then open a follow-up PR implementing
issue #1 + #2 together, since the typed-exception list naturally subsumes
both.

Items #3, #4, #6 are P1 and have no overlap with WS1/WS3 — they can ship
as independent follow-up PRs after this RCA is reviewed.
