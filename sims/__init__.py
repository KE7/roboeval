"""robo-eval simulator backends and environment wrappers.

This package contains:
  - ``sim_worker``: FastAPI HTTP server exposing simulator environments.
  - ``env_wrapper``: ``SimWrapper`` class that wraps simulators as ``BaseWorldStub``.
  - ``litellm_vlm``: Helper to configure the litellm VLM proxy connection.
  - ``vla_policies/``: Standalone VLA policy servers (Pi0.5, OpenVLA, SmolVLA).
"""
