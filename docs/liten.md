# LITEN Integration

The hierarchical evaluation mode in roboeval integrates LITEN, a VLM-based task-planning method introduced by **Shah et al.** in [Learning Affordances at Inference-Time for Vision-Language-Action Models](https://arxiv.org/abs/2510.19752).

roboeval is, to our knowledge, the first public VLA evaluation harness to ship a working LITEN integration. In roboeval, this mode appears as hierarchical evaluation: a high-level VLM planner emits subtask instructions, and the existing low-level VLA policy server executes those instructions through the same `/predict` interface used for direct evaluation.

```bibtex
@article{shah2025liten,
  title   = {Learning Affordances at Inference-Time for Vision-Language-Action Models},
  author  = {Shah, Ameesh and Chen, William and Godbole, Adwait and Mora, Federico and Seshia, Sanjit A. and Levine, Sergey},
  journal = {arXiv preprint arXiv:2510.19752},
  year    = {2025}
}
```

## Planner Boundary

The boundary between the planner and the VLA is `world.act(subtask_instruction: str)`.

1. The VLM receives the task context and produces a small Python program.
2. The program calls `world.act("...")` for each subtask.
3. Each `world.act(...)` call runs the normal VLA/simulator inner loop until success, timeout, or step-budget exhaustion.
4. The VLA server interface is unchanged between direct and hierarchical mode.

Example planner output:

```python
world.act("grasp the black bowl on the stove")
world.act("place the black bowl on the plate")
```

The program is executed with a restricted global namespace containing the `world` object. This is not a security sandbox. Use this mode only with trusted VLM endpoints; see [../SECURITY.md](../SECURITY.md) for the trust model.

## Running Hierarchical Evaluation

Start the VLA and simulator pair:

```bash
roboeval serve --vla pi05 --sim libero --headless
```

Start a LiteLLM-compatible VLM endpoint:

```bash
bash scripts/start_vlm.sh
```

Run a config with `no_vlm: false` and a configured VLM endpoint:

```bash
roboeval run -c configs/libero_spatial_pi05_liten_smoke.yaml
```

Relevant YAML fields:

```yaml
no_vlm: false
vlm_model: vertex_ai/gemini-3-flash-preview
vlm_endpoint: localhost:4000
```

## Mode Comparison

| Property | Direct | LITEN-style hierarchical |
|---|---|---|
| VLM required | No | Yes |
| VLA server | `/predict` policy server | Same server |
| Simulator worker | Same worker | Same worker |
| Instruction passed to VLA | Full task instruction | Planner-generated subtask instruction |
| Main added dependency | None beyond VLA/sim pair | LiteLLM-compatible VLM endpoint |
| Primary use | Native VLA evaluation | Planner/VLA interaction studies and long-horizon decomposition experiments |

## Extending the Planner

The planner entry point is under `vlm_hl/`. Alternative planners should preserve the `world.act(...)` output contract so they remain compatible with the existing orchestrator and VLA policy servers.
