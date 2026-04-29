# Related Work

roboeval sits in a landscape of model-evaluation tools, robotics simulators, and
task-specific VLA evaluation scripts. The table below is factual and
non-exhaustive; it describes where each tool is commonly used rather than
ranking them.

| Tool | Scope | Sim coverage | VLA coverage | Hierarchical mode | Hard-fail spec gate | Process isolation |
|---|---|---|---|---|---|---|
| roboeval | VLA evaluation harness across multiple simulator backends | LIBERO-family, RoboTwin, gym-aloha, gym-pusht, Meta-World, and registered backend scaffolds | Multiple HTTP policy servers with typed action/observation contracts | Yes, LITEN-style planner mode through the same VLA/sim interface | Yes, via `ActionObsSpec` before episode 1 | Yes, per-component environments and HTTP services |
| LeRobot eval | Evaluation utilities for policies in the LeRobot ecosystem | Strongest for LeRobot datasets and gym-style environments | Strongest for LeRobot policy classes and checkpoints | Not a primary architectural mode | Policy/environment compatibility is mostly handled by task-specific code paths | Usually one Python environment for the eval process |
| OpenVLA-style harnesses | Model-specific evaluation scripts for OpenVLA checkpoints | Commonly LIBERO-focused | OpenVLA-family checkpoints | Not a primary architectural mode | Usually task-specific assertions or runtime failures | Usually model-script environment based |
| lm-eval-harness | General LLM evaluation harness | Text and multimodal benchmark tasks, not robotics simulators | Language models rather than robot policies | Not robotics-hierarchical | Task schema validation, not robot action/observation spec gating | Single eval environment with model backends configured per run |
| HuggingFace evaluate | General metric and evaluation library | Dataset/metric oriented, not simulator orchestration | Model-agnostic metric computation | No robotics planner mode | Metric input validation, not robot interface contract gating | Library-level integration in the caller's environment |
| Simulator-native eval scripts | Scripts shipped by individual simulators or model repositories | Usually one simulator family | Usually one model family or policy API | Rarely first-class | Usually local checks inside the script | Usually one environment per script or upstream project |
| vla-eval | Unified evaluation harness for VLA models via WebSocket+msgpack protocol | 14 simulation benchmarks (e.g., LIBERO, Meta-World) | 6 model servers with single predict() interface | Not a primary architectural mode | Compatibility handled by protocol and Docker isolation | Yes, Docker-based environment isolation |

roboeval differs from these in two main ways: it treats VLA policy servers,
simulator workers, and optional VLM planners as separately launched HTTP
services, and it makes the action/observation compatibility gate a first-class
contract before any rollout begins. This makes it useful when a study needs to
mix robotics packages with incompatible dependency stacks, compare several
VLA/simulator pairings through one result format, or evaluate direct and
hierarchical modes through the same low-level policy interface.

## What roboeval Is Not

- It is not a simulator.
- It is not a model-training framework.
- It is not a metric library for arbitrary datasets.
- It is not a leaderboard.
- It is not a replacement for upstream model-specific validation scripts.

## Where It Fits

- Use roboeval when the evaluation subject is a VLA policy interacting with a
  simulator.
- Use roboeval when the VLA and simulator dependencies do not fit comfortably in
  one Python environment.
- Use roboeval when action, observation, camera, state, and gripper conventions
  need to be declared explicitly.
- Use roboeval when direct and planner-assisted runs should share the same
  component interfaces.
- Use simulator-native scripts when studying one upstream model in its original
  evaluation stack.
- Use LeRobot tools when the policy and environment both live cleanly inside the
  LeRobot ecosystem.
- Use general LLM evaluation harnesses when the task is text, tool-use, or
  multimodal model evaluation rather than robot control.

## Comparison Notes

### LeRobot Eval

LeRobot is closest when the policy, processor, dataset, and environment all fit
inside the LeRobot ecosystem.

roboeval can host LeRobot-backed policies, but it treats them as HTTP services.

That boundary matters when the paired simulator has incompatible dependencies.

It also matters when several policy families need to share one result format.

### OpenVLA-Style Harnesses

OpenVLA-style scripts are useful for reproducing model-specific upstream
behavior.

They often encode the assumptions of one model family and one target benchmark.

roboeval keeps model-specific preprocessing inside the policy server and exposes
the resulting policy through the same HTTP contract as other VLAs.

### lm-eval-harness

lm-eval-harness is a mature reference point for general language-model
evaluation.

It standardizes tasks, model adapters, and result reporting for language tasks.

roboeval addresses a different layer: closed-loop robot policy execution against
stateful simulators.

The main shared idea is that a harness should make evaluation inputs and outputs
explicit.

### HuggingFace Evaluate

HuggingFace evaluate is useful when the central object is a metric computation.

roboeval's central object is an episode rollout.

Metrics are recorded after simulator interaction rather than computed over a
static dataset alone.

### Simulator-Native Scripts

Simulator-native scripts remain valuable for debugging upstream environment
behavior.

They are often the most direct way to inspect a simulator's original API.

roboeval is useful once the goal is to compare multiple VLA servers or reuse the
same policy interface across simulator backends.

### vla-eval

vla-eval (Allen AI) is concurrent work that emerged after roboeval's initial development.

Both target the O(N×M) integration burden, but vla-eval uses WebSocket+msgpack with Docker containers for benchmarks and models, while roboeval uses HTTP with process environments.

vla-eval's model integration is a single predict() method with the framework handling chunking and batching automatically.

roboeval requires implementing a full HTTP server, making the contract explicit at the boundary.

vla-eval discovers compatibility issues through systematic comparison against reference implementations, where undocumented parameters can cause 55 percentage point swings.

roboeval makes compatibility a first-class gate through `ActionObsSpec` before any episode begins, rejecting mismatches at startup.

vla-eval prioritizes cross-evaluation matrices and a leaderboard with 657 results; roboeval prioritizes hierarchical evaluation modes and explicit action/observation contracts.

## Dimensions That Matter

- **Closed-loop control:** Robot policies must act repeatedly in a mutable
  environment.
- **Action semantics:** Equal vector length is not enough; conventions must
  match.
- **Observation semantics:** Camera role, image format, state layout, and
  language fields must be explicit.
- **Dependency isolation:** Robotics packages often have stronger version
  constraints than ordinary metric libraries.
- **Planner integration:** Hierarchical evaluation should reuse the same
  low-level policy and simulator contracts.
- **Result shape:** Episode records, task aggregates, and shard metadata should
  be machine-readable.

## Practical Selection

Choose the narrowest tool that answers the research question.

Use an upstream model script for upstream parity checks.

Use LeRobot tooling for LeRobot-native policy and environment studies.

Use general LLM evaluation harnesses for text and multimodal benchmarks.

Use roboeval when the study is about VLA behavior in simulator episodes and the
VLA/simulator boundary itself needs to be explicit.
