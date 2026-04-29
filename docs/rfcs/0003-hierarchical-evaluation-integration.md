# 0003: Hierarchical Evaluation Integration

## Status

Accepted.

## Context

LITEN, introduced by Shah et al. in "Learning Affordances at
Inference-Time for Vision-Language-Action Models," uses a VLM planner to produce
subtask-level calls for a lower-level VLA policy.

roboeval already has a direct VLA/simulator evaluation path.

The project needs hierarchical evaluation without duplicating every VLA and
simulator integration.

A separate script could run planner experiments for one model and one simulator,
but that would bypass the normal contract gate, result collector, sharding path,
and extension surface.

The architectural question is whether LITEN-style evaluation should be integrated
as a first-class mode or left as an external experiment script.

## Decision

roboeval integrates hierarchical evaluation at the architecture level.

The VLM planner is an optional service.

The low-level VLA policy server is the same server used for direct evaluation.

The simulator worker is the same worker used for direct evaluation.

The planner boundary is `world.act(subtask_instruction: str)`.

Each `world.act(...)` call invokes the normal VLA/simulator interaction loop.

The same `ActionObsSpec` gate validates the low-level pair.

The same result collection path records the run.

YAML configs enable the mode with `no_vlm: false`, `vlm_model`, and
`vlm_endpoint`.

## Consequences

Positive:

- Direct and hierarchical runs share component interfaces.
- New VLA integrations can participate in hierarchical mode without a second
  policy implementation.
- New simulator integrations can participate in hierarchical mode without a
  second backend implementation.
- Planner experiments use the same result collector and sharding machinery.
- The LITEN integration boundary is documented and inspectable.

Negative:

- Planner endpoint trust becomes part of the run model.
- Planner latency can dominate run time.
- Debugging hierarchical failures requires inspecting planner output as well as
  VLA and simulator logs.
- Some low-level policies may not respond well to planner-generated subtasks
  even when their direct contract is valid.

## Alternatives Considered

### Separate LITEN Script

A standalone script would be simpler for a single demonstration.

It was rejected because it would duplicate setup, bypass the contract gate, and
create a second result path.

### Planner Inside The VLA Server

Embedding the planner inside each VLA server would hide one service boundary.

It was rejected because planner choice should be independent of the low-level
policy implementation.

### Planner Inside The Simulator Worker

Embedding the planner inside the simulator worker would couple task planning to
simulator internals.

It was rejected because the planner operates above both VLA and simulator
services.

### No Hierarchical Mode

Leaving hierarchical evaluation out would keep the harness smaller.

It was rejected because VLM-guided decomposition is a meaningful evaluation mode
for long-horizon VLA studies.
