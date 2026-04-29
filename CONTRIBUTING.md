# Contributing to roboeval

This document summarizes the public contribution workflow for roboeval.

## Ways to Contribute

roboeval accepts small fixes and larger compatibility additions. Good entry
points include:

- **Add a VLA** — implement the policy-server contract and registration path.
  See [`docs/extending.md#add-a-vla`](docs/extending.md#add-a-vla).
- **Add a simulator** — implement the simulator-backend contract and server
  wiring. See [`docs/extending.md#add-a-simulator`](docs/extending.md#add-a-simulator).
- **Improve docs** — file an issue or open a PR. Small corrections, clearer
  setup notes, and reproducible examples are welcome.
- **Report a bug** — use GitHub issues. Include your platform, Python version,
  relevant package versions, reproduction steps, and the output of
  `roboeval test --validate -c <config>.yaml`.
- **Propose a design change** — use an issue or PR and link the relevant
  decision record. See the [RFC index](docs/rfcs/index.md).

## Development Setup

```bash
# Clone the repo
git clone https://github.com/KE7/roboeval.git && cd roboeval

# Install the CLI in development mode
uv pip install -e ".[dev]"

# Set up a simulator (optional; only needed for running evaluations)
roboeval setup libero
```

## Code Style

- **Formatter**: [Black](https://github.com/psf/black) (default settings)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type hints**: Required on all public function signatures
- **Docstrings**: Required on all public classes and functions (Google style)

Run checks before submitting:

```bash
# 1. Lint and format
ruff check .
ruff format --check .

# 2. Unit tests
.venvs/roboeval/bin/python -m pytest -q tests/

# 3. Spec-contract validation against an example config
roboeval test --validate -c configs/libero_spatial_pi05_smoke.yaml
```

## Project Structure

- `roboeval/` — Core CLI package (Python 3.10+)
- `sims/` — Simulator backends and VLA policy servers
- `vlm_hl/` — VLM reasoning and plan generation
- `scripts/` — Setup and launch scripts
- `tests/` — Unit and integration tests
- `docs/` — Documentation

## Adding a New VLA

See [`docs/extending.md`](docs/extending.md#add-a-vla) for the policy-server contract. The template at `sims/vla_policies/template_policy.py` provides a starting point.

## Adding a New Benchmark

See [`docs/extending.md`](docs/extending.md#add-a-benchmark) for the simulator-backend contract.

## Continuous Integration

`.github/workflows/ci.yml` runs three jobs on every pull request and push to `main`:

| Job | What it checks |
|---|---|
| `lint` | `ruff check` + `ruff format --check` across Python 3.11–3.13 |
| `test` | `pytest tests/` across Python 3.11–3.13 |
| `fresh-install-smoke` | `setup.sh libero` from a clean clone, then boots the sim worker and calls `/init` |

These jobs run on standard GitHub-hosted runners and require no GPU.

## Eval Results in Pull Requests

GPU eval does not run in CI. If your PR adds or modifies a VLA policy, simulator backend, or evaluation config, please run the relevant smoke configs locally and paste the results as a comment on the PR before requesting review.

Use `roboeval serve` + `roboeval run` against the configs under `configs/`:

```bash
# Example: new VLA on libero_spatial
roboeval serve --vla <your_vla> --sim libero --headless &
roboeval run -c configs/libero_spatial_<your_vla>_smoke.yaml
```

A minimal result comment looks like:

```
**Eval results** (local, 10 ep)
| Pair | Score | Baseline | Delta |
|---|---|---|---|
| libero_spatial × <your_vla> | 8/10 | — | — |
```

Including a score helps reviewers gauge whether the change is a regression or improvement. PRs that affect evaluation behaviour without results may be held for author follow-up.

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all existing tests pass (`ruff check .`, `ruff format --check .`, `pytest tests/`)
5. Update documentation if needed
6. If your change affects eval behaviour, run the relevant smoke configs locally and paste results as a PR comment (see [Eval Results in Pull Requests](#eval-results-in-pull-requests))
7. Submit a PR with a clear description of what and why

## Reporting Issues

When reporting bugs, please include:
- Your hardware and OS (especially GPU model)
- Python version and relevant package versions
- Steps to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.
