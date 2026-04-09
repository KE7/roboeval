"""Unified observation schema for VLA policy servers.

This module defines the canonical observation and prediction request schemas
shared by all VLA policy servers (pi05, openvla, smolvla, groot, internvla).
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel

class VLAObservation(BaseModel):
    """Observation payload sent to VLA policy ``/predict`` endpoints.

    Attributes:
        instruction: Natural-language task instruction.
        images: Camera images keyed by **role name**, each value a
            base64-encoded JPEG/PNG string.  Standard role keys:

            * ``"primary"`` -- main third-person / agentview camera.
            * ``"wrist"``   -- wrist / eye-in-hand camera (optional).
            * ``"secondary"`` -- additional side camera (optional).

            Policy servers consume whichever roles they need and
            silently ignore the rest.
        state: Robot proprioceptive state, keyed by format.
            ``{"flat": [float, ...]}`` for simple vectors,
            ``{"structured": {"joint_pos": [...]}}`` for named fields.
    """
    instruction: str
    images: Dict[str, str]
    state: Dict[str, Any]

class PredictRequest(BaseModel):
    obs: VLAObservation
