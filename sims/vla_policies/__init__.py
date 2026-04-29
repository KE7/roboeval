"""VLA policy servers exposing /predict HTTP endpoints.

Each server is a standalone FastAPI process that loads a specific VLA model
and serves action predictions over HTTP.  All servers share a common API
contract defined in :mod:`sims.vla_policies.base`:

  GET  /health  → readiness check
  GET  /info    → model metadata (action_space, state_dim, chunk_size)
  POST /reset   → reset per-episode state (no-op for stateless models)
  POST /predict → {obs: {images, instruction, state}} → {actions, ...}

New VLA integration
-------------------
1. Copy ``template_policy.py`` to ``your_policy.py``.
2. Subclass :class:`~sims.vla_policies.base.VLAPolicyBase`.
3. Implement ``load_model()``, ``predict()``, ``get_info()``.
4. Call :func:`~sims.vla_policies.base.make_app` in ``main()``.
5. Add a ``# /// script`` uv inline-dependency block at the top.

Available servers
-----------------
- ``pi05_policy``      : Pi 0.5 (lerobot, port 5100)
- ``pi0_policy``       : Pi 0 (lerobot, port 5106) — predecessor to pi05
- ``openvla_policy``   : OpenVLA (transformers, port 5101); adds ``/reload``
- ``smolvla_policy``   : SmolVLA (lerobot, port 5102)
- ``cosmos_policy``    : Cosmos-Policy (NVIDIA, port 5103)
- ``groot_policy``     : GR00T-N1.6 (NVIDIA, port 5105)
- ``internvla_policy`` : InternVLA-A1 (InternRobotics, port 5200)
- ``tdmpc2_policy``    : TDMPC2 model-based RL (lerobot, port 5109) — pairs with metaworld
"""
