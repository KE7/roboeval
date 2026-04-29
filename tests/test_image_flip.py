"""Image transform behavior tests.

Tests:
1. LiberoBackend._extract_image applies [::-1, ::-1] to both primary and wrist images.
2. Pi05Policy.get_info() advertises image_transform="applied_in_sim".
3. SmolVLAPolicy.get_info() advertises image_transform="applied_in_sim".
4. OpenVLAPolicy.get_info() advertises image_transform="applied_in_sim".
5. _apply_image_transform(img, "applied_in_sim") is a no-op.
6. _apply_image_transform(img, "none") is a no-op.
7. LiberoBackend.get_info() includes image_transform="applied_in_sim" in obs_space.
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Test 1–2: LiberoBackend._extract_image and get_info
# ---------------------------------------------------------------------------


def _make_libero_backend():
    """Instantiate LiberoBackend without importing robosuite/LIBERO.

    We patch the heavyweight sim import so the test runs in a plain venv.
    """
    # Provide minimal stubs so sim_worker imports cleanly without robosuite
    for mod in [
        "robosuite",
        "robosuite.wrappers",
        "libero",
        "libero.envs",
        "libero.envs.libero_envs",
        "libero.envs.bddl_base_domain",
        "libero.libero",
        "libero.libero.benchmark",
        "bddl",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # robosuite.wrappers.GymWrapper stub
    rw = sys.modules["robosuite.wrappers"]
    if not hasattr(rw, "GymWrapper"):
        rw.GymWrapper = object

    # libero.libero.benchmark stubs
    benchmark_mod = sys.modules["libero.libero.benchmark"]
    if not hasattr(benchmark_mod, "get_benchmark_dict"):
        benchmark_mod.get_benchmark_dict = lambda: {}

    # libero.envs stubs
    envs_mod = sys.modules["libero.envs"]
    if not hasattr(envs_mod, "OffScreenRenderEnv"):
        envs_mod.OffScreenRenderEnv = object

    from sims.sim_worker import LiberoBackend

    return LiberoBackend


class TestLiberoBackendExtractImage(unittest.TestCase):
    """_extract_image must return 180°-flipped ([::-1,::-1]) images."""

    def setUp(self):
        LiberoBackend = _make_libero_backend()
        self.backend = LiberoBackend.__new__(LiberoBackend)

    def _make_obs(self, primary_fill: int, wrist_fill: int, h: int = 8, w: int = 8):
        """Create a synthetic observation dict with distinct patterns."""
        primary = np.zeros((h, w, 3), dtype=np.uint8)
        # Gradient: row=0 is filled with primary_fill, last row with 0
        primary[0, :, 0] = primary_fill
        primary[-1, :, 0] = 0

        wrist = np.zeros((h, w, 3), dtype=np.uint8)
        wrist[0, :, 1] = wrist_fill
        wrist[-1, :, 1] = 0

        return {
            "agentview_image": primary,
            "robot0_eye_in_hand_image": wrist,
        }

    def test_primary_flipped(self):
        """Primary image must equal raw[::-1, ::-1]."""
        obs = self._make_obs(200, 100)
        raw_primary = np.asarray(obs["agentview_image"], dtype=np.uint8).copy()
        img, _wrist = self.backend._extract_image(obs)
        expected = raw_primary[::-1, ::-1]
        np.testing.assert_array_equal(
            img, expected, err_msg="_extract_image did not flip primary image"
        )

    def test_wrist_flipped(self):
        """Wrist image must equal raw[::-1, ::-1]."""
        obs = self._make_obs(200, 100)
        raw_wrist = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
        _primary, wrist_out = self.backend._extract_image(obs)
        expected = raw_wrist[::-1, ::-1]
        np.testing.assert_array_equal(
            wrist_out, expected, err_msg="_extract_image did not flip wrist image"
        )

    def test_flip_is_not_identity(self):
        """Sanity: a non-symmetric image should differ before/after flip."""
        obs = self._make_obs(255, 128)
        raw = np.asarray(obs["agentview_image"], dtype=np.uint8).copy()
        img, _ = self.backend._extract_image(obs)
        self.assertFalse(
            np.array_equal(img, raw),
            "Expected flipped image to differ from raw (input is not centro-symmetric)",
        )

    def test_no_wrist_returns_none(self):
        """When only agentview is present, wrist component is None."""
        obs = {"agentview_image": np.zeros((8, 8, 3), dtype=np.uint8)}
        img, wrist = self.backend._extract_image(obs)
        self.assertIsNone(wrist)
        self.assertIsNotNone(img)

    def test_missing_cameras_raises(self):
        """KeyError raised when no camera images are present."""
        with self.assertRaises(KeyError):
            self.backend._extract_image({"state": [0.0]})

    def test_get_info_obs_space_image_transform(self):
        """get_info() must declare image_transform='applied_in_sim' in obs_space."""
        info = self.backend.get_info()
        transform = info.get("obs_space", {}).get("image_transform")
        self.assertEqual(
            transform,
            "applied_in_sim",
            f"Expected obs_space.image_transform='applied_in_sim', got {transform!r}",
        )


# ---------------------------------------------------------------------------
# Test 3: Pi05Policy.get_info
# ---------------------------------------------------------------------------


class TestPi05PolicyGetInfo(unittest.TestCase):
    """Pi05Policy.get_info() must advertise image_transform='applied_in_sim'."""

    def _make_policy(self):
        from sims.vla_policies.pi05_policy import Pi05Policy

        policy = Pi05Policy.__new__(Pi05Policy)
        policy.model_id = "lerobot/pi05_libero_finetuned"
        policy._image_transform = "flip_hw"  # detected model image transform
        policy._action_dim = 7
        policy._state_dim = 8
        policy._action_chunk_size = 50
        return policy

    def test_image_transform_is_applied_in_sim(self):
        policy = self._make_policy()
        info = policy.get_info()
        transform = info["obs_requirements"]["image_transform"]
        self.assertEqual(
            transform,
            "applied_in_sim",
            f"Pi05Policy.get_info() should return 'applied_in_sim', got {transform!r}",
        )


# ---------------------------------------------------------------------------
# Test 4: SmolVLAPolicy.get_info
# ---------------------------------------------------------------------------


class TestSmolVLAPolicyGetInfo(unittest.TestCase):
    """SmolVLAPolicy.get_info() must advertise image_transform='applied_in_sim'."""

    def _make_policy(self):
        from sims.vla_policies.smolvla_policy import SmolVLAPolicy

        policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
        policy.model_id = "HuggingFaceVLA/smolvla_libero"
        policy._image_transform = "flip_hw"
        policy._action_dim = 7
        policy._state_dim = 8
        policy._action_chunk_size = 1
        return policy

    def test_image_transform_is_applied_in_sim(self):
        policy = self._make_policy()
        info = policy.get_info()
        transform = info["obs_requirements"]["image_transform"]
        self.assertEqual(
            transform,
            "applied_in_sim",
            f"SmolVLAPolicy.get_info() should return 'applied_in_sim', got {transform!r}",
        )


# ---------------------------------------------------------------------------
# Test 5: OpenVLAPolicy.get_info
# ---------------------------------------------------------------------------


class TestOpenVLAPolicyGetInfo(unittest.TestCase):
    """OpenVLAPolicy.get_info() must advertise image_transform='applied_in_sim'."""

    def _make_policy(self):
        from sims.vla_policies.openvla_policy import OpenVLAPolicy

        policy = OpenVLAPolicy.__new__(OpenVLAPolicy)
        policy.model_id = "openvla/openvla-7b-finetuned-libero-spatial"
        return policy

    def test_image_transform_is_applied_in_sim(self):
        policy = self._make_policy()
        info = policy.get_info()
        transform = info["obs_requirements"]["image_transform"]
        self.assertEqual(
            transform,
            "applied_in_sim",
            f"OpenVLAPolicy.get_info() should return 'applied_in_sim', got {transform!r}",
        )


# ---------------------------------------------------------------------------
# Test 6–8: _apply_image_transform no-op cases
# ---------------------------------------------------------------------------


class TestApplyImageTransform(unittest.TestCase):
    """_apply_image_transform must treat 'applied_in_sim' and 'none' as no-ops."""

    def setUp(self):
        from PIL import Image as PILImage

        arr = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        self.img = PILImage.fromarray(arr, mode="RGB")
        self.arr = arr.copy()

    def test_applied_in_sim_is_noop(self):
        """'applied_in_sim' must return the same image object unchanged."""
        from sims.env_wrapper import _apply_image_transform

        result = _apply_image_transform(self.img, "applied_in_sim")
        # Must be the identical object (no copy)
        self.assertIs(
            result, self.img, "_apply_image_transform('applied_in_sim') must return input unchanged"
        )

    def test_none_is_noop(self):
        """'none' must return the same image object unchanged."""
        from sims.env_wrapper import _apply_image_transform

        result = _apply_image_transform(self.img, "none")
        self.assertIs(result, self.img)

    def test_empty_string_is_noop(self):
        """Empty string must return the same image object unchanged."""
        from sims.env_wrapper import _apply_image_transform

        result = _apply_image_transform(self.img, "")
        self.assertIs(result, self.img)

    def test_flip_hw_actually_flips(self):
        """Sanity: 'flip_hw' must still produce a flipped image."""
        from sims.env_wrapper import _apply_image_transform

        result = _apply_image_transform(self.img, "flip_hw")
        result_arr = np.array(result)
        expected = self.arr[::-1, ::-1]
        np.testing.assert_array_equal(result_arr, expected)

    def test_unknown_transform_raises(self):
        """Unknown transform string must raise ValueError."""
        from sims.env_wrapper import _apply_image_transform

        with self.assertRaises(ValueError):
            _apply_image_transform(self.img, "magic_flip")


# ---------------------------------------------------------------------------
# Parity: LiberoInfinityBackend MUST flip images identically to LiberoBackend.
# This keeps camera preprocessing consistent across LIBERO backends.
# ---------------------------------------------------------------------------


class TestLiberoInfinityFlipParity(unittest.TestCase):
    """LiberoInfinityBackend._extract_images must mirror LiberoBackend._extract_image."""

    def setUp(self):
        LiberoBackend = _make_libero_backend()
        from sims.sim_worker import LiberoInfinityBackend

        self.libero = LiberoBackend.__new__(LiberoBackend)
        self.infinity = LiberoInfinityBackend.__new__(LiberoInfinityBackend)

    def _make_obs(self):
        primary = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        wrist = (255 - primary).copy()
        return {
            "agentview_image": primary,
            "robot0_eye_in_hand_image": wrist,
        }

    def test_extract_images_match_libero(self):
        obs = self._make_obs()
        libero_primary, libero_wrist = self.libero._extract_image(obs)
        inf_primary, inf_wrist = self.infinity._extract_images(obs)
        np.testing.assert_array_equal(
            inf_primary,
            libero_primary,
            err_msg="LiberoInfinity primary image flip diverges from LiberoBackend",
        )
        np.testing.assert_array_equal(
            inf_wrist,
            libero_wrist,
            err_msg="LiberoInfinity wrist image flip diverges from LiberoBackend",
        )

    def test_get_info_advertises_applied_in_sim(self):
        from sims.sim_worker import LiberoInfinityBackend

        backend = LiberoInfinityBackend.__new__(LiberoInfinityBackend)
        backend._camera_resolution = 256
        backend._max_steps = 280
        info = backend.get_info()
        self.assertEqual(info["obs_space"]["image_transform"], "applied_in_sim")
        # ActionObsSpec contracts must also be present.
        self.assertIn("action_spec", info)
        self.assertIn("observation_spec", info)
        self.assertEqual(info["observation_spec"]["state"]["dims"], 8)


if __name__ == "__main__":
    unittest.main()
