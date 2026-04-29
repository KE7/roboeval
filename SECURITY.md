# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Instead, email the maintainers at **k.e@berkeley.edu** with:

- A description of the vulnerability and its potential impact.
- Steps to reproduce or a proof-of-concept (if applicable).
- Any suggested mitigations you are aware of.

You will receive an acknowledgement within **48 hours** and a resolution timeline within **7 days**.

## Security Model and Known Assumptions

### LITEN Hierarchical Planner — `exec()` of VLM Output

roboeval includes a hierarchical planning mode (LITEN) in which a high-level Vision-Language
Model (VLM) generates a Python program that is then executed with `exec()` in the evaluation
process.

**This feature is designed for use with trusted VLM endpoints only.**

- Do **not** point roboeval at an untrusted or adversarially-controlled VLM endpoint.
- A malicious or prompt-injected VLM response could cause the generated program to execute
  arbitrary code in the orchestration process.
- The `exec()` sandbox does not restrict filesystem access, network access, or subprocess
  spawning. It only scopes the global namespace to `{"world": wrapper}` — it is not a
  security boundary.

Typical trusted endpoints: Vertex AI (Gemini), OpenAI API, a locally-hosted Ollama instance
on a private network. Do **not** expose a roboeval instance that feeds untrusted user
prompts to the planner.

### VLA Policy Servers

VLA policy servers communicate over HTTP on `localhost`. They are not designed to be exposed
on public network interfaces. Running `roboeval serve` binds to `localhost` only by default.

### Credentials

The LiteLLM proxy can read an API key from `utils/openaikey.txt` as a fallback. This file
is listed in `.gitignore`. Never commit API keys to the repository.
