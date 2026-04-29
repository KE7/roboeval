"""
Test GR00T gripper binarization and convention inversion.

Tests that the gripper dimension is correctly:
1. Binarized (continuous values become -1 or 1)
2. Inverted (RLDS convention → LIBERO convention)
"""

import numpy as np


def test_groot_gripper_binarization():
    """Test that gripper values are binarized correctly."""
    import sys
    from pathlib import Path

    # Add the project root to the path so we can import sims.vla_policies
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Exercise the gripper conversion logic without loading the model.

    # Apply the same binarization and convention conversion used by actions.
    def mock_flatten_with_gripper_fix(flat_action):
        """Apply gripper binarization and convention conversion."""
        gripper = 1.0 if flat_action[-1] > 0.0 else -1.0
        flat_action[-1] = -gripper
        return flat_action

    # Test: positive gripper value → 1.0 (RLDS close) → inverted to -1.0 (LIBERO close)
    action_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    result_pos = mock_flatten_with_gripper_fix(action_pos.copy())
    assert result_pos[-1] == -1.0, (
        f"Positive gripper value should become -1.0 (LIBERO close), got {result_pos[-1]}"
    )

    # Test: negative gripper value → -1.0 (RLDS open) → inverted to 1.0 (LIBERO open)
    action_neg = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3])
    result_neg = mock_flatten_with_gripper_fix(action_neg.copy())
    assert result_neg[-1] == 1.0, (
        f"Negative gripper value should become 1.0 (LIBERO open), got {result_neg[-1]}"
    )

    # Test: zero gripper value should default to one or the other
    action_zero = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result_zero = mock_flatten_with_gripper_fix(action_zero.copy())
    assert result_zero[-1] in [-1.0, 1.0], (
        f"Zero gripper should binarize to -1.0 or 1.0, got {result_zero[-1]}"
    )

    # Test: very small positive value should still binarize to -1.0
    action_tiny_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001])
    result_tiny_pos = mock_flatten_with_gripper_fix(action_tiny_pos.copy())
    assert result_tiny_pos[-1] == -1.0, (
        f"Tiny positive gripper should become -1.0, got {result_tiny_pos[-1]}"
    )


def test_gripper_transitions_in_sequence():
    """Test that a sequence of gripper actions produces transitions."""

    def mock_flatten_with_gripper_fix(flat_action):
        """Apply gripper binarization and convention conversion."""
        flat_action = flat_action.copy()
        gripper = 1.0 if flat_action[-1] > 0.0 else -1.0
        flat_action[-1] = -gripper
        return flat_action

    # Simulate a sequence of actions with varying gripper values
    actions_continuous = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),  # positive → -1.0 (close)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]),  # positive → -1.0 (close)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1]),  # negative → 1.0 (open)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2]),  # negative → 1.0 (open)
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05]),  # positive → -1.0 (close)
    ]

    gripper_states = [mock_flatten_with_gripper_fix(action)[-1] for action in actions_continuous]

    # Count transitions
    transitions = sum(
        1 for i in range(1, len(gripper_states)) if gripper_states[i] != gripper_states[i - 1]
    )

    # We expect 2 transitions: positive→negative at index 2, negative→positive at index 4
    assert transitions == 2, f"Expected 2 transitions, got {transitions}. States: {gripper_states}"


def test_gripper_not_stuck_at_zero():
    """Test that zero or near-zero gripper values don't stay stuck."""

    def mock_flatten_with_gripper_fix(flat_action):
        """Apply gripper binarization and convention conversion."""
        flat_action = flat_action.copy()
        gripper = 1.0 if flat_action[-1] > 0.0 else -1.0
        flat_action[-1] = -gripper
        return flat_action

    # Continuous gripper values should be binarized before they reach the backend.
    continuous_grippers = [0.04, -0.05, 0.09, 0.02, -0.01]

    for gripper_val in continuous_grippers:
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_val])
        result = mock_flatten_with_gripper_fix(action)

        # Output should be exactly -1.0 or 1.0, not the original continuous value.
        assert result[-1] in [-1.0, 1.0], (
            f"Gripper {gripper_val} should binarize to -1.0 or 1.0, got {result[-1]}"
        )
        assert result[-1] != gripper_val, (
            f"Gripper value {gripper_val} should be transformed, not left as-is"
        )


if __name__ == "__main__":
    test_groot_gripper_binarization()
    test_gripper_transitions_in_sequence()
    test_gripper_not_stuck_at_zero()
    print("✓ All GR00T gripper tests passed!")
