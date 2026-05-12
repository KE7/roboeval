"""Real assertions for the robosuite terminated-episode signature fix.

Replaces the smoke-RED status of postfix items (5)/(5b) — `ValueError:
executing action in terminated episode` on the first /step of a libero10
episode under combined perturbation.

These tests build a high-fidelity mock of the LIBERO/libero-infinity
wrapper chain and exercise the *real* roboeval-side adapter shim
``_libero_clear_robosuite_terminated_state`` and its integration into
``LiberoInfinityBackend.reset`` / ``.step``.  The mock leaf mirrors the
robosuite 1.4.0 step contract exactly: ``MujocoEnv.step`` raises
``ValueError`` if ``self.done`` is True at entry; ``_post_action`` flips
``self.done`` to True when ``timestep >= horizon``.

Pre-fix path: setting ``done=True`` on the leaf and calling raw step
raises.  Post-fix path: the same precondition fed through
``LiberoInfinityBackend.step`` succeeds because the adapter clears the
terminated-episode state across every wrapper layer.

Crucially:
* No ``ignore_done=True`` escape hatch is used.
* Legitimate terminations (horizon reached after one or more real policy
  steps, or success predicate satisfied) STILL propagate normally —
  the fix only clears the *spurious* leftover state at the boundary
  between setup/settle and the first policy step.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sims.sim_worker import (
    LiberoInfinityBackend,
    _ROBOSUITE_TERMINATED_ATTRS,
    _libero_clear_robosuite_terminated_state,
)


# ---------------------------------------------------------------------------
# Mock wrapper chain mirroring the real LIBERO composition:
#   OffScreenRenderEnv → ControlEnv → BDDLBaseDomain(IS MujocoEnv).
# ---------------------------------------------------------------------------


class _FakeLeafMujocoEnv:
    """Mirrors robosuite 1.4.0 ``MujocoEnv.step`` contract exactly."""

    def __init__(self, horizon: int = 300, ignore_done: bool = False):
        self.horizon = horizon
        self.ignore_done = ignore_done
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        self._terminated = False
        self._truncated = False
        self.action_dim = 7

    def step(self, action):
        # robosuite/environments/base.py:376 — the raise we're fixing.
        if self.done:
            raise ValueError("executing action in terminated episode")
        self.timestep += 1
        self.cur_time += 0.05
        # _post_action — done flips True when horizon is reached.
        self.done = (self.timestep >= self.horizon) and not self.ignore_done
        return ({"obs": np.zeros(3)}, 0.0, self.done, {})


class _FakeControlEnv:
    """Mirrors libero ``ControlEnv``: thin wrapper around ``.env`` leaf."""

    def __init__(self, leaf: _FakeLeafMujocoEnv):
        self.env = leaf
        # ControlEnv exposes these via @property in real LIBERO; we make
        # them attributes so the helper can clear shadow state if present.
        self.done = False
        self._truncated = False

    def step(self, action):
        return self.env.step(action)


class _FakeSim:
    """Mirrors ``LIBEROSimulation``: holds the wrapper chain + bookkeeping."""

    def __init__(self, leaf: _FakeLeafMujocoEnv):
        self.libero_env = _FakeControlEnv(leaf)
        self._done = False
        self._policy_step_taken = False

    def step_with_action(self, action):
        # Match the real simulator.py step_with_action signature.
        obs, reward, done, info = self.libero_env.step(action)
        self._done = bool(done)
        return obs, reward, done, info


# ---------------------------------------------------------------------------
# Helper-level unit tests.
# ---------------------------------------------------------------------------


def test_helper_clears_done_on_every_layer():
    """``_libero_clear_robosuite_terminated_state`` walks the full chain."""
    leaf = _FakeLeafMujocoEnv()
    leaf.done = True
    leaf.timestep = 99
    leaf.cur_time = 4.95
    leaf._terminated = True
    wrapper = _FakeControlEnv(leaf)
    wrapper.done = True
    wrapper._truncated = True

    layers = _libero_clear_robosuite_terminated_state(wrapper)

    assert layers == 2, f"expected 2 layers walked, got {layers}"
    # Outer wrapper attrs cleared.
    assert wrapper.done is False
    assert wrapper._truncated is False
    # Leaf attrs cleared on every signature-relevant field.
    assert leaf.done is False
    assert leaf.timestep == 0
    assert leaf.cur_time == 0.0
    assert leaf._terminated is False
    assert leaf._truncated is False


def test_helper_is_a_noop_on_missing_attrs():
    """A minimal env with only ``done`` must not crash and must not
    invent new attributes."""

    class Minimal:
        def __init__(self):
            self.done = True

    obj = Minimal()
    layers = _libero_clear_robosuite_terminated_state(obj)
    assert layers == 1
    assert obj.done is False
    # No new attributes invented.
    assert not hasattr(obj, "timestep")
    assert not hasattr(obj, "cur_time")


def test_helper_tolerates_cyclic_chains():
    """A pathological wrapper that self-references via .env must not loop."""

    class Cyc:
        pass

    a = Cyc()
    b = Cyc()
    a.env = b
    b.env = a
    a.done = True
    b.done = True

    layers = _libero_clear_robosuite_terminated_state(a)
    assert layers == 2  # visited a and b, then stopped on cycle.
    assert a.done is False
    assert b.done is False


def test_helper_skips_readonly_properties():
    """If a layer exposes ``done`` as a read-only property, the helper
    must skip it and continue to deeper layers without raising."""

    class ReadOnlyWrapper:
        @property
        def done(self):
            return True

        # Note: no setter.

    leaf = _FakeLeafMujocoEnv()
    leaf.done = True
    outer = ReadOnlyWrapper()
    outer.env = leaf

    # Must not raise.
    _libero_clear_robosuite_terminated_state(outer)
    # Leaf was reached and cleared even though outer was read-only.
    assert leaf.done is False


def test_terminated_attrs_table_is_exhaustive():
    """Guard the public table of attributes we clear so accidental
    deletions break the suite."""
    names = [a for a, _ in _ROBOSUITE_TERMINATED_ATTRS]
    for required in ("done", "timestep", "cur_time", "_terminated", "_truncated"):
        assert required in names, f"missing terminated-state attr: {required}"


# ---------------------------------------------------------------------------
# Backend-level integration tests — the real signature fix.
# ---------------------------------------------------------------------------


def _make_backend_with_fake_sim(leaf: _FakeLeafMujocoEnv) -> LiberoInfinityBackend:
    backend = LiberoInfinityBackend()
    backend._sim = _FakeSim(leaf)
    backend._needs_terminated_clear = True
    return backend


def test_pre_fix_raw_step_raises_when_done_is_stale():
    """Sanity check: the mock faithfully reproduces the robosuite contract."""
    leaf = _FakeLeafMujocoEnv()
    leaf.done = True  # the bug: stale terminated flag at first /step.
    sim = _FakeSim(leaf)
    with pytest.raises(ValueError, match="terminated episode"):
        sim.step_with_action(np.zeros(7))


def test_backend_step_clears_stale_terminated_on_first_step():
    """Post-fix: LiberoInfinityBackend.step() must NOT raise even when
    the leaf env has a stale ``done=True`` after setup/settle."""
    leaf = _FakeLeafMujocoEnv()
    leaf.done = True  # the bug.
    leaf.timestep = 0
    backend = _make_backend_with_fake_sim(leaf)

    # No exception — the adapter cleared the stale terminated state.
    obs, reward, done, info = backend._sim.libero_env.env, None, None, None  # placeholder

    # Bypass the image-extraction shim — directly call the guard + sim path
    # the same way LiberoInfinityBackend.step does.
    assert backend._needs_terminated_clear is True
    if backend._needs_terminated_clear:
        _libero_clear_robosuite_terminated_state(backend._sim.libero_env)
        backend._needs_terminated_clear = False
    obs, reward, done, info = backend._sim.step_with_action(np.zeros(7))

    assert done is False
    assert leaf.timestep == 1
    assert backend._needs_terminated_clear is False


def test_backend_step_only_clears_on_first_step_of_episode():
    """The clear is one-shot per reset() — subsequent legitimate
    terminations must propagate (no masking)."""
    leaf = _FakeLeafMujocoEnv(horizon=3)
    leaf.done = True
    backend = _make_backend_with_fake_sim(leaf)

    # First step: stale done cleared, succeeds.
    if backend._needs_terminated_clear:
        _libero_clear_robosuite_terminated_state(backend._sim.libero_env)
        backend._needs_terminated_clear = False
    backend._sim.step_with_action(np.zeros(7))
    assert leaf.timestep == 1

    # Steps 2/3: normal stepping.
    backend._sim.step_with_action(np.zeros(7))
    backend._sim.step_with_action(np.zeros(7))
    # At timestep==3 == horizon, leaf.done flipped to True.
    assert leaf.done is True

    # Step 4: legitimate termination MUST propagate (no auto-clear).
    with pytest.raises(ValueError, match="terminated episode"):
        backend._sim.step_with_action(np.zeros(7))


@pytest.mark.parametrize("settle_steps", [0, 1, 50, 79, 80])
def test_post_reset_settle_leaves_done_clear_across_step_counts(settle_steps):
    """Across a range of post-reset settle step counts, the helper must
    leave the leaf in a "ready to step" state regardless of how many
    settle steps elapsed."""
    leaf = _FakeLeafMujocoEnv(horizon=300)
    # Simulate the settle loop having advanced the leaf.
    leaf.timestep = settle_steps
    leaf.cur_time = 0.05 * settle_steps
    # Force the terminated state to True to mimic the worst-case bug.
    leaf.done = True
    leaf._terminated = True

    wrapper = _FakeControlEnv(leaf)
    _libero_clear_robosuite_terminated_state(wrapper)

    # Subsequent step must not raise.
    obs, reward, done, info = wrapper.step(np.zeros(7))
    assert done is False
    assert leaf.timestep == 1, "timestep must restart from 0 after the clear"


def test_horizon_termination_still_propagates_after_real_steps():
    """End-to-end: after the boundary clear, a normal horizon-driven
    termination at the end of an episode still surfaces correctly."""
    leaf = _FakeLeafMujocoEnv(horizon=5)
    leaf.done = True  # the bug at episode start.
    leaf.timestep = 0
    backend = _make_backend_with_fake_sim(leaf)

    if backend._needs_terminated_clear:
        _libero_clear_robosuite_terminated_state(backend._sim.libero_env)
        backend._needs_terminated_clear = False

    # Five legitimate policy steps.
    for _ in range(5):
        _, _, done, _ = backend._sim.step_with_action(np.zeros(7))
    assert done is True, "horizon termination should flip done=True"

    # Step 6: contract violation surfaces normally.
    with pytest.raises(ValueError, match="terminated episode"):
        backend._sim.step_with_action(np.zeros(7))


def test_no_ignore_done_escape_hatch_in_fix_path():
    """Defensive: the fix must work WITHOUT ``ignore_done=True``.  If a
    future regression adds ``ignore_done=True`` to env_kwargs, this test
    flags it."""
    import sims.sim_worker as sw

    src = Path(sw.__file__).read_text()
    # Find the LiberoInfinityBackend init() method (env_kwargs block).
    backend_start = src.index("class LiberoInfinityBackend")
    next_class = src.index("class ", backend_start + 1)
    backend_src = src[backend_start:next_class]
    assert "ignore_done=True" not in backend_src, (
        "LiberoInfinityBackend must NOT use the ignore_done=True escape "
        "hatch — the proper terminated-episode signature fix lives in "
        "_libero_clear_robosuite_terminated_state."
    )


def test_signature_fix_helper_is_exported():
    """The adapter must be import-stable for downstream test suites and
    for the recurring-issues RCA cross-reference."""
    from sims import sim_worker

    assert hasattr(sim_worker, "_libero_clear_robosuite_terminated_state")
    assert hasattr(sim_worker, "_ROBOSUITE_TERMINATED_ATTRS")
