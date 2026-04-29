# Design Principles

This page documents the main design decisions behind roboeval. The goal is to
make the harness easy to inspect, easy to reproduce, and strict enough to catch
interface mistakes before long evaluation runs.

## Process Isolation

roboeval treats each VLA, simulator, and optional planner endpoint as a separate
host process.

The default installation model uses per-component virtual environments under
`.venvs/`.

That choice is deliberate.

It lets a VLA server use one Python version while a simulator uses another.

It lets a model stack use one CUDA dependency set while a physics backend uses a
different one.

It lets old simulator packages coexist with newer model packages.

It keeps heavyweight optional dependencies out of the orchestrator environment.

It makes import failures local to the component that owns the dependency.

It avoids requiring a single monolithic environment that satisfies every VLA and
every simulator at once.

It also makes launch behavior explicit: a VLA policy server is a service, a
simulator worker is a service, and the orchestrator coordinates them over HTTP.

The tradeoff is operational: users need to know which component owns which
dependency and which port each process binds.

The payoff is that mixed Python versions, mixed CUDA stacks, and incompatible
robotics packages remain usable in one evaluation project.

## Strict Specs

roboeval uses `ActionObsSpec` as a hard compatibility contract.

A policy server declares what actions it produces and what observations it
requires.

A simulator worker declares what actions it consumes and what observations it
provides.

The orchestrator compares both sides before episode execution.

Hard failures are intentional.

Action dimensionality mismatches should not become silent rollout failures.

Image format mismatches should not become confusing model behavior.

State layout mismatches should not be hidden in adapter code.

Gripper sign conventions should be declared instead of guessed.

Rotation representations should be declared instead of inferred from vector
length alone.

The contract catches common mistakes:

- 7-dimensional end-effector actions sent to a 14-dimensional joint-position simulator.
- 2-dimensional planar actions sent to a 7-dimensional manipulation backend.
- Axis-angle state expected by a policy but quaternion state provided by a simulator.
- Binary gripper close-positive and close-negative conventions mixed accidentally.
- Image-transform ownership split incorrectly between simulator and policy code.

Warnings remain useful for non-fatal differences such as range metadata.

The default posture is strict because the cost of a false start is lower than the
cost of collecting invalid trajectories.

## YAML-Driven Runs

roboeval uses `roboeval run -c <config>.yaml` as the primary invocation form.

The YAML file captures the run contract:

- VLA identity.
- VLA URL.
- Simulator identity.
- Simulator URL.
- Suite.
- Task selection.
- Episode count.
- Output directory.
- Planner endpoint.
- Extra parameters.

That makes a run portable.

It also makes it reviewable in code review.

A command line with many flags is easy to lose in shell history.

A config file can be committed, diffed, copied, and re-run.

The CLI still supports operational flags for sharding and output overrides.

The durable description of an evaluation remains the config.

The result JSON stores a config snapshot so later readers can see the evaluated
pair and run settings.

## CLI As The API

roboeval exposes user workflows through the CLI:

- `roboeval setup`
- `roboeval serve`
- `roboeval test --validate`
- `roboeval run`
- `roboeval merge`

Scripts and policy modules are implementation details.

The CLI gives users a stable surface while component internals remain free to
change.

`roboeval setup` owns environment provisioning.

`roboeval serve` owns component launch.

`roboeval test --validate` owns fast compatibility checks.

`roboeval run` owns evaluation execution.

`roboeval merge` owns shard aggregation.

This structure also gives documentation a small set of commands to teach.

Advanced users can still launch component modules directly when debugging.

The public path remains the CLI.

## Hierarchical Evaluation

roboeval treats hierarchical evaluation as a first-class mode.

The LITEN-style planner integration is architectural, not a one-off script.

The planner is above the same VLA policy server used for direct evaluation.

The simulator worker is the same worker used for direct evaluation.

The low-level call boundary is `world.act(subtask_instruction)`.

The high-level planner can decompose a task into subtask instructions.

The low-level VLA continues to receive ordinary language-conditioned prediction
requests.

The result path remains the same collector and JSON format.

The compatibility gate remains the same `ActionObsSpec` gate.

This keeps direct and hierarchical evaluations comparable at the systems level.

It also makes new VLA and simulator integrations automatically eligible for
planner experiments when their direct contracts are valid.

The tradeoff is that planner safety and endpoint trust must be documented
explicitly.

Planner-generated programs are not a sandbox boundary.

Trusted endpoint assumptions are documented in [liten.md](liten.md) and
[../SECURITY.md](../SECURITY.md).

## Reproducibility Over Convenience

roboeval favors explicit settings over hidden defaults when the setting affects a
run.

Ports are visible.

URLs are visible.

Suites are visible.

Episode counts are visible.

Shard IDs are visible.

The config snapshot is written into the result.

The harness version is written into the result.

This does not make every upstream simulator deterministic by itself.

It does give each run an inspectable record.

## Compatibility Before Throughput

The first priority is to run the intended VLA/simulator pair correctly.

Throughput features such as sharding and multiple workers are built on top of
the same contract checks.

Scaling a wrong contract only creates more bad data.

The harness therefore checks compatibility before episode 1 and then lets users
scale runs with shards, replica processes, and explicit GPU allocation.

## Small Public Surface

The public extension path is intentionally narrow.

New VLAs implement the policy-server interface.

New simulators implement the simulator-worker backend interface.

New planners preserve the `world.act(...)` boundary.

The internal module layout can evolve while those contracts remain stable.

See [extending.md](extending.md) for the concrete entry points.
