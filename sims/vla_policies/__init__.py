"""VLA policy servers exposing /predict HTTP endpoints.

Each server is a standalone FastAPI process that loads a specific VLA model
and serves action predictions over HTTP. They share a common API contract:

  GET  /health  → readiness check
  GET  /info    → model metadata (action_space, state_dim, chunk_size)
  POST /reset   → reset per-episode state (no-op for stateless models)
  POST /predict → {obs: {image, instruction, state?, image2?}} → {actions}

Available servers:
  - ``pi05_policy``: Pi 0.5 (lerobot, port 5100)
  - ``openvla_policy``: OpenVLA (transformers, port 5101)
  - ``smolvla_policy``: SmolVLA (lerobot, port 5102)
  - ``openvla_batched_server``: OpenVLA with dynamic request batching (port 5103)
"""
