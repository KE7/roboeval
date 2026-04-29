"""Interface convention specifications for action and observation formats.

Every ``ModelServer`` and ``StepBenchmark`` must declare what it produces
and what it consumes via ``get_action_spec()`` and ``get_observation_spec()``.
The orchestrator compares these at episode start and logs warnings on
mismatches — catching convention bugs (wrong rotation format, inverted
gripper, delta-vs-absolute, missing state) before they waste GPU hours.

Usage::

    from robo_eval.specs import ActionObsSpec, POSITION_DELTA, ROTATION_AA, GRIPPER_CLOSE_POS

    class MyModelServer(PredictModelServer):
        def get_action_spec(self) -> dict[str, ActionObsSpec]:
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_AA,
                "gripper": GRIPPER_CLOSE_POS,
            }

Validation policy
-----------------
``check_specs()`` returns a list of ``(severity, message)`` tuples where
severity is one of ``"HARD"``, ``"WARN"``, or ``"IGNORE"``.  The caller
(orchestrator / env_wrapper) should abort on any HARD entry before starting
an episode, log WARN entries loudly, and silently drop IGNORE entries.

Severity rules:

==========================================================  =========
Mismatch                                                    Severity
==========================================================  =========
Missing required component (server expects key, sim lacks)  HARD
image_transform declaration disagrees                       HARD
Action dims mismatch                                        HARD
Action format mismatch, consumer cannot convert             HARD
Action format mismatch, consumer ``accepts`` includes it    (pass)
Action range differs                                        WARN
No ``action_spec`` / ``observation_spec`` (legacy server)   WARN
Optional metadata mismatch                                  IGNORE
==========================================================  =========
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Severity constants
HARD = "HARD"
WARN = "WARN"
IGNORE = "IGNORE"


@dataclass(frozen=True)
class ActionObsSpec:
    """Specification for one component of the action/observation interface.

    Attributes:
        name: Human-readable component name (e.g. ``"position"``, ``"gripper"``).
        dims: Number of dimensions in the array.  Use 0 for non-array data
            (images, language strings).
        format: Convention string (e.g. ``"delta_xyz"``, ``"binary_close_positive"``).
            Use predefined constants where possible to avoid typos.
        range: Expected ``(min, max)`` value range, or ``None`` if unconstrained.
        accepts: Set of format strings this consumer can handle.  When set,
            ``is_compatible()`` checks membership instead of exact equality.
            Use on the *consumer* side (benchmark action spec, model server
            observation spec) to declare which formats can be converted.
        description: Free-text notes for edge cases.
    """

    name: str
    dims: int
    format: str
    range: tuple[float, float] | None = None
    accepts: frozenset[str] | None = None
    description: str = ""

    def validate(self, value: np.ndarray) -> list[str]:
        """Validate a value against this spec.  Returns a list of errors (empty = valid)."""
        errors: list[str] = []
        flat = np.asarray(value, dtype=np.float64).flatten()
        if self.dims > 0 and len(flat) < self.dims:
            errors.append(f"{self.name}: expected {self.dims}D, got {len(flat)}D")
        if self.range and self.dims > 0:
            lo, hi = self.range
            chunk = flat[: self.dims]
            if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
                errors.append(f"{self.name}: contains NaN/Inf")
            elif np.any(chunk < lo - 0.01) or np.any(chunk > hi + 0.01):
                errors.append(f"{self.name}: values outside [{lo}, {hi}]")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for wire protocol / JSON)."""
        d: dict[str, Any] = {"name": self.name, "dims": self.dims, "format": self.format}
        if self.range is not None:
            d["range"] = list(self.range)
        if self.accepts is not None:
            d["accepts"] = sorted(self.accepts)
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionObsSpec":
        """Deserialize from a plain dict."""
        return cls(
            name=d["name"],
            dims=d["dims"],
            format=d["format"],
            range=tuple(d["range"]) if "range" in d else None,
            accepts=frozenset(d["accepts"]) if "accepts" in d else None,
            description=d.get("description", ""),
        )

    def is_compatible(self, other: "ActionObsSpec") -> tuple[bool, str]:
        """Check if ``self`` (producer) is consumable by ``other`` (consumer).

        When *other* has ``accepts`` set, checks format membership.
        Otherwise checks format and dims equality.
        """
        if other.accepts is not None:
            if self.format not in other.accepts:
                return False, f"{self.name}: {self.format} not in accepts {set(other.accepts)}"
            return True, ""
        if self.format != other.format:
            return False, f"{self.name}: {self.format} vs {other.format}"
        if self.dims != other.dims and self.dims > 0 and other.dims > 0:
            return False, f"{self.name}: {self.dims}D vs {other.dims}D"
        return True, ""


def check_specs(
    server_action: dict[str, ActionObsSpec],
    bench_action: dict[str, ActionObsSpec],
    server_obs: dict[str, ActionObsSpec],
    bench_obs: dict[str, ActionObsSpec],
) -> list[tuple[str, str]]:
    """Compare server and benchmark specs.

    Returns a list of ``(severity, message)`` tuples.  Severity is one of
    ``HARD``, ``WARN``, or ``IGNORE``.

    Severity rules (see module docstring for the full table):

    - Missing required component → HARD
    - Dims mismatch → HARD
    - Format mismatch (consumer cannot convert) → HARD
    - Format mismatch (consumer ``accepts`` includes producer's format) → (pass)
    - Range mismatch → WARN
    - Legacy: no specs declared → WARN

    Parameters
    ----------
    server_action:
        ActionObsSpec mapping declared by the VLA server for its action outputs.
    bench_action:
        ActionObsSpec mapping declared by the benchmark/sim for the actions it consumes.
    server_obs:
        ActionObsSpec mapping declared by the VLA server for the observations it needs.
    bench_obs:
        ActionObsSpec mapping declared by the benchmark/sim for the observations it provides.
    """
    results: list[tuple[str, str]] = []

    # ── Legacy fallback: if either side has no specs, warn once ──────────────
    if not server_action and not bench_action and not server_obs and not bench_obs:
        results.append((WARN, "no action_spec or observation_spec declared (legacy server)"))
        return results

    # ── Action: server produces → benchmark consumes ─────────────────────────
    if server_action and bench_action and not (server_action.keys() & bench_action.keys()):
        results.append(
            (HARD, "action: no overlapping keys between server and benchmark specs")
        )

    for key in bench_action:
        if key not in server_action and server_action:
            results.append(
                (HARD, f"action [{key}]: benchmark expects it but server doesn't declare it")
            )

    for key in server_action.keys() & bench_action.keys():
        prod = server_action[key]
        cons = bench_action[key]

        # Dimension mismatches are hard failures.
        if prod.dims > 0 and cons.dims > 0 and prod.dims != cons.dims:
            results.append(
                (HARD, f"action [{key}]: dims mismatch {prod.dims}D vs {cons.dims}D")
            )
            continue

        # Format compatibility
        ok, reason = prod.is_compatible(cons)
        if not ok:
            results.append((HARD, f"action [{key}]: {reason}"))
            continue

        # Range mismatches are warnings.
        if prod.range and cons.range and prod.range != cons.range:
            results.append(
                (WARN, f"action [{key}]: range mismatch {prod.range} vs {cons.range}")
            )

    # ── Observation: benchmark produces → server consumes ────────────────────
    for key in server_obs:
        if key not in bench_obs:
            results.append(
                (HARD, f"observation [{key}]: server expects it but benchmark doesn't provide it")
            )

    for key in server_obs.keys() & bench_obs.keys():
        ok, reason = bench_obs[key].is_compatible(server_obs[key])
        if not ok:
            results.append((HARD, f"observation [{key}]: {reason}"))

    return results


# ---------------------------------------------------------------------------
# Predefined constants — use these instead of raw strings
# ---------------------------------------------------------------------------

# Position
POSITION_DELTA = ActionObsSpec("position", 3, "delta_xyz", (-1, 1))
POSITION_ABS = ActionObsSpec("position", 3, "absolute_xyz")
POSITION_ABSOLUTE = POSITION_ABS  # alias

# Rotation
ROTATION_EULER = ActionObsSpec("rotation", 3, "euler_xyz", (-3.15, 3.15))
ROTATION_AA = ActionObsSpec("rotation", 3, "axis_angle", (-3.15, 3.15))
ROTATION_QUAT = ActionObsSpec("rotation", 4, "quaternion_xyzw", (-1, 1))
ROTATION_ROT6D_INTERLEAVED = ActionObsSpec("rotation", 6, "rot6d_interleaved")

# Rotation — consumer variants that accept multiple formats
ROTATION_EULER_ACCEPTS_AA = ActionObsSpec(
    "rotation",
    3,
    "euler_xyz",
    (-3.15, 3.15),
    accepts=frozenset({"euler_xyz", "axis_angle"}),
)

# Gripper
GRIPPER_CLOSE_POS = ActionObsSpec("gripper", 1, "binary_close_positive", (-1, 1))
GRIPPER_CLOSE_NEG = ActionObsSpec("gripper", 1, "binary_close_negative", (-1, 1))
GRIPPER_01 = ActionObsSpec("gripper", 1, "continuous_01", (0, 1))
GRIPPER_RAW = ActionObsSpec("gripper", 1, "raw")

# Observation — images
IMAGE_RGB = ActionObsSpec("image", 0, "rgb_hwc_uint8")

# Observation — state
STATE_EEF_POS_QUAT_GRIP = ActionObsSpec("state", 8, "eef_pos3_quat4_gripper1")
STATE_EEF_POS_AA_GRIP = ActionObsSpec("state", 8, "eef_pos3_axisangle3_gripper2")
STATE_EEF_POS_EULER_GRIP = ActionObsSpec("state", 8, "eef_pos3_euler3_gripper2")
STATE_ROT6D_PROPRIO_20D = ActionObsSpec("state", 20, "rot6d_interleaved_proprio_20d")
STATE_JOINT = ActionObsSpec("state", 0, "joint_positions")

# Observation — language
LANGUAGE = ActionObsSpec("language", 0, "language")

# Generic passthrough (no convention enforced)
RAW = ActionObsSpec("raw", 0, "raw")


# ---------------------------------------------------------------------------
# Deprecated alias retained for compatibility.
DimSpec = ActionObsSpec
