# 0002: ActionObsSpec Contract

## Status

Accepted.

## Context

VLA/simulator integration failures often look like normal rollout failures.

A policy can emit the wrong number of action dimensions.

A simulator can expect a different gripper sign convention.

A policy can expect axis-angle state while a simulator provides quaternion state.

A camera transform can be applied in the wrong place.

A bimanual policy can be paired with a single-arm simulator.

A planar PushT policy can be paired with a 7-dimensional manipulation backend.

If these mistakes are discovered only after trajectories are collected, the run
can waste GPU time and produce misleading results.

roboeval needs a public contract that is explicit enough for extension authors
and strict enough for run-time protection.

## Decision

roboeval uses `ActionObsSpec` as the typed action/observation contract.

VLA policy servers declare:

- Actions they produce.
- Observations they require.

Simulator workers declare:

- Actions they consume.
- Observations they provide.

Each spec includes:

- `name`
- `dims`
- `format`
- optional `range`
- optional `accepts`
- optional `description`

The orchestrator and environment wrapper compare these declarations before
episode execution.

Hard mismatches stop the run before episode 1.

Warnings are reserved for non-fatal metadata differences such as range mismatch.

Consumers can declare accepted producer formats through `accepts` when conversion
is intentionally supported.

## Consequences

Positive:

- Action dimensionality errors are caught early.
- State representation errors are caught early.
- Gripper convention mismatches are made explicit.
- Image format and transform ownership become documented interface details.
- Extension authors have a concrete checklist.
- Compatibility becomes inspectable without running full rollouts.
- Direct and hierarchical modes share the same low-level gate.

Negative:

- New integrations must write more metadata.
- Incorrectly declared specs can block otherwise runnable experiments.
- Some legacy components only provide partial metadata and may need adapters.
- Flexible simulators must decide which conversions they officially accept.

## Alternatives Considered

### Advisory Warnings Only

The harness could log compatibility warnings and continue.

This was rejected because common mismatches corrupt the meaning of the rollout.

For example, action dimension or gripper sign mistakes should not be treated as a
normal evaluation failure.

### Runtime-Only Checks

The harness could wait for the first invalid action or missing observation at
runtime.

This was rejected because runtime failures happen after expensive setup and can
be harder to diagnose from simulator errors.

Pre-episode checks provide clearer failure messages.

### Per-Pair Hardcoded Assertions

Each config could contain bespoke compatibility checks.

This was rejected because it does not scale to new VLAs and simulators.

The contract belongs to components, not to every pair-specific config.

### Untyped Free-Form Metadata

Components could publish arbitrary dictionaries without a structured schema.

This was rejected because extension authors need stable fields and the
orchestrator needs deterministic mismatch rules.
