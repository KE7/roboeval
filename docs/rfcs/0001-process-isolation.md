# 0001: Process Isolation

## Status

Accepted.

## Context

VLA evaluation combines dependencies that are difficult to place in one Python
environment.

Policy servers may require current PyTorch, Transformers, LeRobot, custom CUDA
kernels, or vendor repositories.

Simulator workers may require older Python versions, MuJoCo packages, SAPIEN,
Robosuite variants, or task-specific assets.

Planner endpoints may use LiteLLM, vLLM, SGLang, hosted APIs, or local serving
stacks.

A single monolithic environment would have to satisfy all of those constraints
at once.

That makes installation fragile.

It also makes debugging ambiguous because a failed import might come from a
component unrelated to the run.

roboeval needs one orchestrator that can evaluate many VLA/simulator pairs
without forcing every optional dependency to be installed together.

The public extension path also needs to let contributors add a VLA or simulator
without destabilizing unrelated components.

## Decision

roboeval runs major components as independent host processes.

The orchestrator is one process.

Each VLA policy server is a separate process.

Each simulator worker is a separate process.

The optional VLM planner endpoint is a separate service.

Components communicate over HTTP/JSON.

The default setup creates per-component virtual environments under `.venvs/`.

Launch defaults map component names to modules, virtual environments, and ports.

The CLI exposes this through `roboeval setup`, `roboeval serve`, and
`roboeval run`.

The orchestrator does not import model or simulator internals during normal
evaluation.

## Consequences

Positive:

- Components can use different Python versions.
- Components can use different CUDA and simulator dependency stacks.
- Optional dependencies stay local to the component that needs them.
- Import failures are easier to attribute.
- New VLA and simulator integrations have clearer ownership boundaries.
- Long-running services can be health-checked independently.
- The same VLA server can serve direct and hierarchical runs.

Negative:

- Users must manage ports.
- Users may need multiple terminals or process supervisors.
- Error messages cross process boundaries.
- Local debugging sometimes requires checking several logs.
- Shared filesystem paths and assets must be configured consistently across
  component environments.

## Alternatives Considered

### Monolithic Environment

One environment would simplify activation and direct imports.

It was rejected because supported model and simulator stacks have incompatible
Python and package requirements.

It would also force users to install dependencies for components they are not
using.

### Conda-Only Environments

A conda-only approach would help with native dependencies.

It was rejected as the only mechanism because several components install cleanly
with `uv` and do not need conda.

Forcing conda everywhere would make lightweight paths heavier.

### Micromamba-Only Environments

Micromamba is useful for some binary dependency cases.

It was rejected as the universal mechanism for the same reason as conda-only:
not every component needs it, and a single tool should not be required where a
standard virtual environment is sufficient.

### In-Process Plugin Registry

An in-process plugin registry would make Python APIs convenient.

It was rejected for evaluation execution because it would reintroduce dependency
coupling between policies, simulators, and the orchestrator.
