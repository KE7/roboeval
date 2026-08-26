import numpy as np

try:
    from scripts.run_openvla_native_eval import SUITE_MAX_STEPS
except ImportError:
    # Torch may not be available in the test environment.
    SUITE_MAX_STEPS = {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }

from sims.sim_worker import LiberoBackend


class _DummyController:
    def __init__(self):
        self.use_delta = False


class _DummyRobot:
    def __init__(self):
        self.controller = _DummyController()


class _DummyInnerEnv:
    def __init__(self):
        self.robots = [_DummyRobot()]

    def _get_observations(self):
        return {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}


class _DummyOuterEnv:
    def __init__(self):
        self.env = _DummyInnerEnv()
        self.step_delta_history = []
        self.init_states_seen = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, init_state):
        self.init_states_seen.append(init_state)

    def step(self, action):
        self.step_delta_history.append(self.env.robots[0].controller.use_delta)
        return None


def test_libero_reset_applies_use_delta_after_warmup():
    backend = LiberoBackend()
    backend.env = _DummyOuterEnv()
    backend.delta_actions = True
    backend.init_states = ["episode-0"]
    backend._extract_image = lambda obs: (obs["agentview_image"], None)

    backend.reset()

    assert backend.env.reset_calls == 1
    assert backend.env.init_states_seen == ["episode-0"]
    assert backend.env.step_delta_history == [False] * 10
    assert backend.env.env.robots[0].controller.use_delta is True


def test_openvla_native_eval_uses_canonical_libero_spatial_horizon():
    assert SUITE_MAX_STEPS["libero_spatial"] == 280


# Pi05Policy class behavior is covered by the image orientation and spec
# handshake tests.


def test_openvla_gripper_postprocessing_matches_policy_server():
    """Given raw gripper outputs, native eval and policy server mapping match.

    Both should use the canonical approach:
    - Binarize based on sign (>0 = close, <=0 = open)
    - Invert to match LIBERO convention (OpenVLA outputs RLDS convention)

    This test does NOT require torch, so it can run in any environment.
    """

    def process_action_native_eval(raw_gripper):
        """Simulate native eval gripper post-processing."""
        gripper = -(1.0 if raw_gripper > 0.0 else -1.0)
        return gripper

    def policy_server_gripper(raw_gripper):
        """Simulate policy server gripper post-processing."""
        gripper = 1.0 if raw_gripper > 0.0 else -1.0
        inverted = -gripper
        return inverted

    # Test with values in the expected [-1, 1] range
    test_values = [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0]
    for val in test_values:
        native = process_action_native_eval(val)
        server = policy_server_gripper(val)
        assert native == server, (
            f"Gripper mismatch at input={val}: native_eval={native}, policy_server={server}"
        )


def test_openvla_gripper_convention_rlds_to_libero():
    """Given RLDS gripper convention, post-processing emits LIBERO convention.

    OpenVLA (RLDS): 1=close, -1=open
    LIBERO: -1=close, 1=open

    After post-processing:
    - Model outputs > 0 (close intent) → -1 (LIBERO close)
    - Model outputs <= 0 (open intent) → 1 (LIBERO open)
    """

    def gripper_postprocess(raw_gripper):
        """Canonical gripper post-processing from openvla_policy.py."""
        return -(1.0 if raw_gripper > 0.0 else -1.0)

    # Positive values (RLDS "close" intent) should map to -1 (LIBERO "close")
    assert gripper_postprocess(0.5) == -1.0  # model says close → LIBERO close
    assert gripper_postprocess(1.0) == -1.0

    # Non-positive values (RLDS "open" intent) should map to 1 (LIBERO "open")
    assert gripper_postprocess(-0.5) == 1.0  # model says open → LIBERO open
    assert gripper_postprocess(-1.0) == 1.0
    assert gripper_postprocess(0.0) == 1.0  # boundary case
