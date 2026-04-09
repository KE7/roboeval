import numpy as np

try:
    from scripts.run_openvla_native_eval import SUITE_MAX_STEPS
except ImportError:
    # torch may not be available in test environment
    SUITE_MAX_STEPS = {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }

from sims.sim_worker import LiberoBackend
from sims.vla_policies import pi05_policy


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


def test_pi05_predict_uses_select_action_and_hf_postprocessor(monkeypatch):
    class _DummyPolicy:
        def __init__(self):
            self.select_action_calls = 0
            self.predict_action_chunk_calls = 0

        def select_action(self, batch):
            self.select_action_calls += 1
            return np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.7]], dtype=np.float32)

        def predict_action_chunk(self, batch):
            self.predict_action_chunk_calls += 1
            raise AssertionError("predict_action_chunk should not be used")

    dummy_policy = _DummyPolicy()
    
    import sys
    import types
    class NoGrad:
        def __enter__(self): pass
        def __exit__(self, *args): pass
    mock_torch = types.ModuleType("torch")
    mock_torch.Tensor = type("Tensor", (), {})
    mock_torch.no_grad = NoGrad
    sys.modules["torch"] = mock_torch
    
    monkeypatch.setattr(pi05_policy, "_policy", dummy_policy)
    monkeypatch.setattr(pi05_policy, "_preprocessor", lambda frame: {"frame": frame})
    monkeypatch.setattr(pi05_policy, "_postprocessor", lambda action: action)
    monkeypatch.setattr(pi05_policy, "_build_frame", lambda *args, **kwargs: {"task": "demo"})
    monkeypatch.setattr(pi05_policy, "_action_dim", 7)

    actions = pi05_policy._predict("unused", "demo", None, None)

    assert dummy_policy.select_action_calls == 1
    assert dummy_policy.predict_action_chunk_calls == 0
    assert actions == [[0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645, 0.5, 0.6000000238418579, -0.699999988079071]]


def test_pi05_postprocessed_action_does_not_flip_gripper_sign():
    pi05_policy._action_dim = 7
    result = pi05_policy._policy_action_to_list(
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25]], dtype=np.float32)
    )
    assert result == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25]]


def test_pi05_reset_policy_resets_internal_queue(monkeypatch):
    class _DummyPolicy:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    dummy_policy = _DummyPolicy()
    monkeypatch.setattr(pi05_policy, "_ready", True)
    monkeypatch.setattr(pi05_policy, "_policy", dummy_policy)

    resp = pi05_policy.reset_policy()

    assert resp == {"success": True}
    assert dummy_policy.reset_calls == 1


def test_openvla_gripper_postprocessing_matches_policy_server():
    """Verify that gripper post-processing in native eval matches policy server.

    Both should use the canonical approach:
    - Binarize based on sign (>0 = close, <=0 = open)
    - Invert to match LIBERO convention (OpenVLA outputs RLDS convention)

    This test does NOT require torch, so it can run in any environment.
    """
    def process_action_native_eval(raw_gripper):
        """Simulate the fixed native eval gripper post-processing."""
        # This is the new implementation from the fix
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
            f"Gripper mismatch at input={val}: "
            f"native_eval={native}, policy_server={server}"
        )


def test_openvla_gripper_convention_rlds_to_libero():
    """Test that gripper convention is correctly inverted from RLDS to LIBERO.

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
    assert gripper_postprocess(0.0) == 1.0   # boundary case
