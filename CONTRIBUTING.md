# Contributing to robo-eval

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/KE7/robo-eval.git && cd robo-eval

# Install the CLI in development mode
uv pip install -e ".[dev]"

# Set up simulators (optional — only needed for running evaluations)
bash scripts/setup_envs.sh --only libero
```

## Code Style

- **Formatter**: [Black](https://github.com/psf/black) (default settings)
- **Linter**: [Ruff](https://github.com/astral-sh/ruff)
- **Type hints**: Required on all public function signatures
- **Docstrings**: Required on all public classes and functions (Google style)

Run checks before submitting:

```bash
ruff check robo_eval/ sims/ vlm_hl/
black --check robo_eval/ sims/ vlm_hl/
pytest tests/
```

## Project Structure

- `robo_eval/` — Core CLI package (Python 3.10+)
- `sims/` — Simulator backends and VLA policy servers (may run in older Python venvs)
- `vlm_hl/` — VLM reasoning and plan generation
- `scripts/` — Setup and launch scripts
- `tests/` — Unit and integration tests
- `docs/` — Documentation

## Adding a New VLA

See [`docs/adding_a_vla.md`](docs/adding_a_vla.md) for a step-by-step guide. The template at `sims/vla_policies/template_policy.py` provides a starting point.

## Adding a New Benchmark

See [`docs/adding_a_benchmark.md`](docs/adding_a_benchmark.md) for how to integrate a new simulator.

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Ensure all existing tests pass
5. Update documentation if needed
6. Submit a PR with a clear description of what and why

## Reporting Issues

When reporting bugs, please include:
- Your hardware and OS (especially GPU model)
- Python version and relevant package versions
- Steps to reproduce the issue
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
