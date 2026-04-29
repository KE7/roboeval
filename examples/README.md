# roboeval examples

This directory contains copy-paste-able contributor templates and example
configuration files.

## Templates

| File | Purpose |
|------|---------|
| [`new_vla_template.py`](new_vla_template.py) | Minimal but complete template for a new VLA policy server. Copy, fill in 4 marked spots, have a working HTTP server. |
| [`new_sim_template.py`](new_sim_template.py) | Template for a new simulator backend. Copy, implement 6 required methods, register in `BACKENDS`. |
| [`eval.yaml`](eval.yaml) | Example evaluation config for `robo-eval run`. |

## Quick start

**Add a new VLA model:**

```bash
cp examples/new_vla_template.py sims/vla_policies/my_model_policy.py
# Fill in load_model(), predict(), get_info(), and optionally reset()
```

**Add a new simulator:**

```bash
cp examples/new_sim_template.py sims/my_sim_backend.py
# Implement init(), reset(), step(), get_obs(), check_success(), close(), get_info()
# Register in sims/sim_worker.py BACKENDS dict
```

## Demo script

A script for validating the install, smoke-test, and first-evaluation flow.

```bash
# Dry-run (prints all planned commands, exits 0 — no GPU needed)
bash examples/demo_recording.sh --simulate

# Full run (requires NVIDIA GPU + CUDA)
bash examples/demo_recording.sh

# Skip re-install if .venvs already exist
bash examples/demo_recording.sh --quick
```

**Options**

| Flag | Effect |
|------|--------|
| `--quick` | Skip `setup.sh`; reuse existing `.venvs/` for fast re-runs. |
| `--no-color` | Disable ANSI color codes (plain text output). |
| `--simulate` | Print every planned command without executing. Useful for validation. |
| `--help` | Show usage summary. |

## Full documentation

See [`docs/extending.md`](../docs/extending.md) for the narrative walkthrough,
including:
- How the VLA server and simulator communicate (HTTP + ActionObsSpec contract)
- Step-by-step guide for adding a new VLA or sim
- Image-flip convention
- Common pitfalls: gripper sign convention, axis-angle vs. quaternion state,
  action chunking, delta vs. absolute actions
