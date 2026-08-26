"""
Tests for roboeval/config.py — configuration constants and helpers.

Covers:
- get_sim_for_suite()
- get_suites_for_benchmark()
- qualify_suite()
- get_suite_max_steps()
- get_qualified_suites()
- is_port_available() (mocked)
- find_available_port() (mocked)
- find_available_port_block() (mocked)
- VLAConfig dataclass
"""

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# qualify_suite
# ---------------------------------------------------------------------------


class TestQualifySuite:
    def test_libero_spatial(self):
        from roboeval.config import qualify_suite

        assert qualify_suite("libero", "spatial") == "libero_spatial"

    def test_libero_pro_goal_swap(self):
        from roboeval.config import qualify_suite

        assert qualify_suite("libero_pro", "goal_swap") == "libero_pro_goal_swap"

    def test_libero_infinity(self):
        from roboeval.config import qualify_suite

        assert qualify_suite("libero_infinity", "spatial") == "libero_infinity_spatial"

    def test_robocasa(self):
        from roboeval.config import qualify_suite

        assert qualify_suite("robocasa", "kitchen") == "robocasa_kitchen"

    def test_robotwin(self):
        from roboeval.config import qualify_suite

        assert qualify_suite("robotwin", "aloha_agilex") == "robotwin_aloha_agilex"


# ---------------------------------------------------------------------------
# get_suites_for_benchmark
# ---------------------------------------------------------------------------


class TestGetSuitesForBenchmark:
    def test_libero(self):
        from roboeval.config import get_suites_for_benchmark

        result = get_suites_for_benchmark("libero")
        assert result == ["spatial", "object", "goal", "10"]

    def test_libero_pro(self):
        from roboeval.config import get_suites_for_benchmark

        result = get_suites_for_benchmark("libero_pro")
        assert result == ["spatial_object", "goal_swap", "spatial_with_mug"]

    def test_libero_infinity(self):
        from roboeval.config import get_suites_for_benchmark

        result = get_suites_for_benchmark("libero_infinity")
        assert result == ["spatial", "object", "goal", "10"]

    def test_robocasa(self):
        from roboeval.config import get_suites_for_benchmark

        assert get_suites_for_benchmark("robocasa") == ["kitchen"]

    def test_robotwin(self):
        from roboeval.config import get_suites_for_benchmark

        assert get_suites_for_benchmark("robotwin") == ["aloha_agilex"]

    def test_unknown_benchmark_raises(self):
        from roboeval.config import get_suites_for_benchmark

        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_suites_for_benchmark("nonexistent_benchmark")


# ---------------------------------------------------------------------------
# get_qualified_suites
# ---------------------------------------------------------------------------


class TestGetQualifiedSuites:
    def test_libero(self):
        from roboeval.config import get_qualified_suites

        result = get_qualified_suites("libero")
        assert result == ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    def test_libero_pro(self):
        from roboeval.config import get_qualified_suites

        result = get_qualified_suites("libero_pro")
        assert result == [
            "libero_pro_spatial_object",
            "libero_pro_goal_swap",
            "libero_pro_spatial_with_mug",
        ]


# ---------------------------------------------------------------------------
# get_sim_for_suite
# ---------------------------------------------------------------------------


class TestGetSimForSuite:
    def test_libero_spatial(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_spatial") == "libero"

    def test_libero_object(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_object") == "libero"

    def test_libero_goal(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_goal") == "libero"

    def test_libero_10(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_10") == "libero"

    def test_libero_pro_spatial_object(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_pro_spatial_object") == "libero_pro"

    def test_libero_pro_goal_swap(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_pro_goal_swap") == "libero_pro"

    def test_robocasa_kitchen(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("robocasa_kitchen") == "robocasa"

    def test_robotwin_aloha_agilex(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("robotwin_aloha_agilex") == "robotwin"

    def test_unknown_suite_falls_back(self):
        from roboeval.config import get_sim_for_suite

        # Unknown suites fall back to 'libero_pro'
        assert get_sim_for_suite("some_unknown_suite") == "libero_pro"

    def test_libero_infinity_spatial(self):
        from roboeval.config import get_sim_for_suite

        assert get_sim_for_suite("libero_infinity_spatial") == "libero_infinity"


# ---------------------------------------------------------------------------
# get_suite_max_steps
# ---------------------------------------------------------------------------


class TestGetSuiteMaxSteps:
    def test_known_suite(self):
        from roboeval.config import get_suite_max_steps

        assert get_suite_max_steps("libero_spatial") == 280

    def test_libero_10(self):
        from roboeval.config import get_suite_max_steps

        assert get_suite_max_steps("libero_10") == 520

    def test_libero_goal(self):
        from roboeval.config import get_suite_max_steps

        assert get_suite_max_steps("libero_goal") == 300

    def test_robocasa_kitchen(self):
        from roboeval.config import get_suite_max_steps

        assert get_suite_max_steps("robocasa_kitchen") == 500

    def test_unknown_suite_gets_default(self):
        from roboeval.config import DEFAULT_MAX_STEPS, get_suite_max_steps

        assert get_suite_max_steps("totally_unknown") == DEFAULT_MAX_STEPS
        assert DEFAULT_MAX_STEPS == 500


# ---------------------------------------------------------------------------
# is_port_available (mocked)
# ---------------------------------------------------------------------------


class TestIsPortAvailable:
    def test_port_available(self):
        from roboeval.config import is_port_available

        with patch("roboeval.config.socket.socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value
            mock_sock.bind.return_value = None
            assert is_port_available(5100) is True
            mock_sock.close.assert_called_once()

    def test_port_in_use(self):
        from roboeval.config import is_port_available

        with patch("roboeval.config.socket.socket") as mock_socket_cls:
            mock_sock = mock_socket_cls.return_value
            mock_sock.bind.side_effect = OSError("Address already in use")
            assert is_port_available(5100) is False
            mock_sock.close.assert_called_once()


# ---------------------------------------------------------------------------
# find_available_port (mocked)
# ---------------------------------------------------------------------------


class TestFindAvailablePort:
    def test_preferred_available(self):
        from roboeval.config import find_available_port

        with patch("roboeval.config.is_port_available", return_value=True):
            assert find_available_port(5100) == 5100

    def test_preferred_unavailable_finds_next(self):
        from roboeval.config import find_available_port

        available = {5101: False, 5102: True}
        with patch(
            "roboeval.config.is_port_available",
            side_effect=lambda p: available.get(p, True),
        ):
            result = find_available_port(5101, search_start=5101, search_end=5110)
            assert result == 5102

    def test_no_port_available_raises(self):
        from roboeval.config import find_available_port

        with patch("roboeval.config.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="No available TCP port"):
                find_available_port(search_start=5100, search_end=5102)

    def test_no_preferred(self):
        from roboeval.config import find_available_port

        calls = []

        def mock_available(port):
            calls.append(port)
            return port == 1024

        with patch("roboeval.config.is_port_available", side_effect=mock_available):
            assert find_available_port(search_start=1024, search_end=1030) == 1024


# ---------------------------------------------------------------------------
# find_available_port_block (mocked)
# ---------------------------------------------------------------------------


class TestFindAvailablePortBlock:
    def test_preferred_block_available(self):
        from roboeval.config import find_available_port_block

        with patch("roboeval.config.is_port_available", return_value=True):
            result = find_available_port_block(3, preferred_start=5300)
            assert result == 5300

    def test_preferred_block_partial_unavailable(self):
        from roboeval.config import find_available_port_block

        # 5300 unavailable, 5301-5303 available
        def avail(port):
            return port != 5300

        with patch("roboeval.config.is_port_available", side_effect=avail):
            result = find_available_port_block(
                3, preferred_start=5300, search_start=5300, search_end=5310
            )
            assert result == 5301

    def test_count_zero_raises(self):
        from roboeval.config import find_available_port_block

        with pytest.raises(ValueError, match="count must be >= 1"):
            find_available_port_block(0)

    def test_no_block_available_raises(self):
        from roboeval.config import find_available_port_block

        with patch("roboeval.config.is_port_available", return_value=False):
            with pytest.raises(RuntimeError, match="No available block"):
                find_available_port_block(3, search_start=5300, search_end=5305)


# ---------------------------------------------------------------------------
# VLAConfig dataclass
# ---------------------------------------------------------------------------


class TestVLAConfig:
    def test_url_property(self):
        from roboeval.config import VLA_CONFIGS

        assert VLA_CONFIGS["pi05"].url == "http://localhost:5100"

    def test_known_vla_names(self):
        from roboeval.config import VLA_CONFIGS

        assert set(VLA_CONFIGS.keys()) == {
            "pi05",
            "vqbet",
            "tdmpc2",
            "openvla",
            "smolvla",
            "cosmos",
            "internvla",
            "groot",
        }

    def test_startup_timeout(self):
        from roboeval.config import VLA_CONFIGS

        assert VLA_CONFIGS["pi05"].startup_timeout == 300
        assert VLA_CONFIGS["openvla"].startup_timeout == 600

    def test_cosmos_config(self):
        from roboeval.config import VLA_CONFIGS

        cosmos = VLA_CONFIGS["cosmos"]
        assert cosmos.port == 5103
        assert cosmos.model_id == "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B"
        assert cosmos.url == "http://localhost:5103"
        assert cosmos.startup_timeout == 600

    def test_internvla_config(self):
        from roboeval.config import VLA_CONFIGS

        internvla = VLA_CONFIGS["internvla"]
        assert internvla.port == 5104
        assert internvla.model_id == "InternRobotics/InternVLA-A1-3B-RoboTwin"
        assert internvla.url == "http://localhost:5104"


# ---------------------------------------------------------------------------
# validate_port — additional edge cases
# ---------------------------------------------------------------------------


class TestValidatePortExtended:
    def test_non_integer_raises(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError, match="must be an integer"):
            validate_port("5100")  # type: ignore

    def test_boundary_1024(self):
        from roboeval.config import validate_port

        assert validate_port(1024) == 1024

    def test_boundary_65535(self):
        from roboeval.config import validate_port

        assert validate_port(65535) == 65535

    def test_boundary_1023_rejected(self):
        from roboeval.config import validate_port

        with pytest.raises(ValueError, match="privileged"):
            validate_port(1023)
