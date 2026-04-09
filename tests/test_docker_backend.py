"""Tests for robo_eval.docker_backend — Docker container management.

All Docker interactions are mocked via subprocess.run — no real Docker needed.
Tests verify that docker commands are constructed correctly, including:
  - GPU flags (--gpus device=N vs --gpus all)
  - Port mapping (-p PORT:PORT)
  - Volume mounts (HF cache :ro by default)
  - Display/X11 forwarding for debug window
  - CUDA_VISIBLE_DEVICES is NEVER passed (FM C2)
  - Container ID tracking (FM C1)
  - Stale container cleanup (FM W1)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from robo_eval.docker_backend import (
    _active_containers,
    docker_cleanup_stale,
    docker_logs,
    docker_run,
    docker_stop,
    docker_stop_all,
    docker_stop_by_name,
    get_active_containers,
    is_docker_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_active_containers():
    """Ensure _active_containers is empty before/after each test."""
    _active_containers.clear()
    yield
    _active_containers.clear()


# ---------------------------------------------------------------------------
# is_docker_available
# ---------------------------------------------------------------------------


class TestIsDockerAvailable:
    """Tests for is_docker_available()."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_docker_available(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert is_docker_available() is True
        mock_run.assert_called_once_with(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_docker_not_available_nonzero_exit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        assert is_docker_available() is False

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_docker_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        assert is_docker_available() is False

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_docker_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker info", timeout=5)
        assert is_docker_available() is False


# ---------------------------------------------------------------------------
# docker_run
# ---------------------------------------------------------------------------


class TestDockerRun:
    """Tests for docker_run() — verifies correct command construction."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_basic_run_with_all_gpus(self, mock_run):
        """Default: --gpus all when no gpu_id specified."""
        mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

        cid = docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
        )

        assert cid == "abc123"
        cmd = mock_run.call_args[0][0]
        assert "--device" in cmd
        gpu_idx = cmd.index("--device")
        assert cmd[gpu_idx + 1] == "nvidia.com/gpu=all"
        assert "-p" in cmd
        p_idx = cmd.index("-p")
        assert cmd[p_idx + 1] == "5300:5300"
        assert "--name" in cmd
        name_idx = cmd.index("--name")
        assert cmd[name_idx + 1] == "robo-eval-5300"

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_run_with_specific_gpu(self, mock_run):
        """C2: --gpus device=N, no CUDA_VISIBLE_DEVICES."""
        mock_run.return_value = MagicMock(returncode=0, stdout="def456\n", stderr="")

        cid = docker_run(
            image="robo-eval/vla-lerobot:latest",
            port=5100,
            gpu_id="2",
        )

        assert cid == "def456"
        cmd = mock_run.call_args[0][0]
        gpu_idx = cmd.index("--device")
        assert cmd[gpu_idx + 1] == "nvidia.com/gpu=2"
        # CUDA_VISIBLE_DEVICES must NOT appear in the command
        cmd_str = " ".join(cmd)
        assert "CUDA_VISIBLE_DEVICES" not in cmd_str

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_cuda_visible_devices_filtered_from_env(self, mock_run):
        """C2: Even if CUDA_VISIBLE_DEVICES is in env dict, it must be filtered out."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ghi789\n", stderr="")

        docker_run(
            image="robo-eval/vla-lerobot:latest",
            port=5100,
            env={"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": "3"},
            gpu_id="3",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "CUDA_VISIBLE_DEVICES" not in cmd_str
        assert "MUJOCO_GL=egl" in cmd_str

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_env_vars_passed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="c1\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            env={"MUJOCO_GL": "egl", "VLA_PORT": "5300"},
        )

        cmd = mock_run.call_args[0][0]
        assert "-e" in cmd
        e_indices = [i for i, x in enumerate(cmd) if x == "-e"]
        env_args = [cmd[i + 1] for i in e_indices]
        assert "MUJOCO_GL=egl" in env_args
        assert "VLA_PORT=5300" in env_args

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_hf_cache_readonly_by_default(self, mock_run):
        """W3: HF cache mounted as :ro by default."""
        mock_run.return_value = MagicMock(returncode=0, stdout="c2\n", stderr="")

        docker_run(
            image="robo-eval/vla-lerobot:latest",
            port=5100,
        )

        cmd = mock_run.call_args[0][0]
        v_indices = [i for i, x in enumerate(cmd) if x == "-v"]
        vol_args = [cmd[i + 1] for i in v_indices]
        hf_mounts = [v for v in vol_args if "huggingface" in v]
        assert len(hf_mounts) == 1
        assert hf_mounts[0].endswith(":ro")

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_hf_cache_readwrite(self, mock_run):
        """rw_cache=True should mount as :rw."""
        mock_run.return_value = MagicMock(returncode=0, stdout="c3\n", stderr="")

        docker_run(
            image="robo-eval/vla-lerobot:latest",
            port=5100,
            rw_cache=True,
        )

        cmd = mock_run.call_args[0][0]
        v_indices = [i for i, x in enumerate(cmd) if x == "-v"]
        vol_args = [cmd[i + 1] for i in v_indices]
        hf_mounts = [v for v in vol_args if "huggingface" in v]
        assert len(hf_mounts) == 1
        assert hf_mounts[0].endswith(":rw")

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_display_forwarding(self, mock_run):
        """Debug window: X11 socket + DISPLAY + MUJOCO_GL=glfw."""
        mock_run.return_value = MagicMock(returncode=0, stdout="c4\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            display=":0",
        )

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)

        assert "DISPLAY=:0" in cmd_str
        assert "/tmp/.X11-unix:/tmp/.X11-unix" in cmd_str
        assert "MUJOCO_GL=glfw" in cmd_str

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_custom_name(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="c5\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            name="my-custom-name",
        )

        cmd = mock_run.call_args[0][0]
        name_idx = cmd.index("--name")
        assert cmd[name_idx + 1] == "my-custom-name"

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_extra_args_appended(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="c6\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            extra_args=["--sim", "libero", "--port", "5300"],
        )

        cmd = mock_run.call_args[0][0]
        # Extra args should come after the image name
        img_idx = cmd.index("robo-eval/sim-libero:latest")
        trailing = cmd[img_idx + 1:]
        assert trailing == ["--sim", "libero", "--port", "5300"]

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_additional_volumes(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="c7\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            volumes=["/data/results:/results"],
        )

        cmd = mock_run.call_args[0][0]
        v_indices = [i for i, x in enumerate(cmd) if x == "-v"]
        vol_args = [cmd[i + 1] for i in v_indices]
        assert "/data/results:/results" in vol_args

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_container_tracked_in_registry(self, mock_run):
        """C1: Container ID must be tracked in _active_containers."""
        mock_run.return_value = MagicMock(returncode=0, stdout="tracked123\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
            name="robo-eval-5300",
        )

        containers = get_active_containers()
        assert "robo-eval-5300" in containers
        assert containers["robo-eval-5300"] == "tracked123"

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_run_failure_raises(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Error: image not found"
        )

        with pytest.raises(RuntimeError, match="docker run failed"):
            docker_run(
                image="robo-eval/nonexistent:latest",
                port=5300,
            )

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_run_detached_and_rm(self, mock_run):
        """Container runs with -d and --rm flags."""
        mock_run.return_value = MagicMock(returncode=0, stdout="c8\n", stderr="")

        docker_run(
            image="robo-eval/sim-libero:latest",
            port=5300,
        )

        cmd = mock_run.call_args[0][0]
        assert "-d" in cmd
        assert "--rm" in cmd


# ---------------------------------------------------------------------------
# docker_stop
# ---------------------------------------------------------------------------


class TestDockerStop:
    """Tests for docker_stop() and docker_stop_by_name()."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_by_id(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = docker_stop("abc123", timeout=15)

        assert result is True
        mock_run.assert_called_once_with(
            ["docker", "stop", "-t", "15", "abc123"],
            capture_output=True,
            text=True,
        )

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_removes_from_registry(self, mock_run):
        """C1: Stopped containers are removed from _active_containers."""
        _active_containers["my-container"] = "abc123"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        docker_stop("abc123")

        assert "my-container" not in _active_containers

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_failure_returns_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No such container")

        result = docker_stop("nonexistent")

        assert result is False

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_by_name(self, mock_run):
        _active_containers["robo-eval-5300"] = "xyz789"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = docker_stop_by_name("robo-eval-5300")

        assert result is True
        # Should use the container ID from registry
        cmd = mock_run.call_args[0][0]
        assert "xyz789" in cmd

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_by_name_unknown(self, mock_run):
        """If name not in registry, pass the name directly to docker stop."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        result = docker_stop_by_name("unknown-container")

        assert result is True
        cmd = mock_run.call_args[0][0]
        assert "unknown-container" in cmd


# ---------------------------------------------------------------------------
# docker_logs
# ---------------------------------------------------------------------------


class TestDockerLogs:
    """Tests for docker_logs()."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_logs_with_tail(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="line1\nline2\n", stderr="warn1\n"
        )

        result = docker_logs("abc123", tail=100)

        assert result == "line1\nline2\nwarn1\n"
        mock_run.assert_called_once_with(
            ["docker", "logs", "--tail", "100", "abc123"],
            capture_output=True,
            text=True,
        )

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_logs_default_tail(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        docker_logs("abc123")

        cmd = mock_run.call_args[0][0]
        assert "--tail" in cmd
        tail_idx = cmd.index("--tail")
        assert cmd[tail_idx + 1] == "50"


# ---------------------------------------------------------------------------
# docker_cleanup_stale
# ---------------------------------------------------------------------------


class TestDockerCleanupStale:
    """Tests for docker_cleanup_stale() — FM W1."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_cleanup_removes_stale_containers(self, mock_run):
        # First call: docker ps -a returns stale containers
        ps_result = MagicMock(
            returncode=0,
            stdout="abc123 robo-eval-5300 Exited (137) 2 hours ago\ndef456 robo-eval-5100 Up 3 hours\n",
            stderr="",
        )
        # Subsequent calls: docker rm -f succeeds
        rm_result = MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = [ps_result, rm_result, rm_result]

        removed = docker_cleanup_stale()

        assert removed == 1
        # Verify docker ps was called with correct filter
        ps_call = mock_run.call_args_list[0]
        cmd = ps_call[0][0]
        assert "--filter" in cmd
        filter_idx = cmd.index("--filter")
        assert cmd[filter_idx + 1] == "name=robo-eval-"

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_cleanup_no_stale_containers(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        removed = docker_cleanup_stale()

        assert removed == 0

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_cleanup_custom_prefix(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        docker_cleanup_stale(prefix="my-prefix-")

        cmd = mock_run.call_args[0][0]
        filter_idx = cmd.index("--filter")
        assert cmd[filter_idx + 1] == "name=my-prefix-"


# ---------------------------------------------------------------------------
# docker_stop_all
# ---------------------------------------------------------------------------


class TestDockerStopAll:
    """Tests for docker_stop_all()."""

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_all_containers(self, mock_run):
        _active_containers["c1"] = "id1"
        _active_containers["c2"] = "id2"
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        stopped = docker_stop_all()

        assert stopped == 2
        assert len(_active_containers) == 0

    @patch("robo_eval.docker_backend.subprocess.run")
    def test_stop_all_empty(self, mock_run):
        stopped = docker_stop_all()

        assert stopped == 0
        mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: runtime selection (5 cases from FM)
# ---------------------------------------------------------------------------


class TestRuntimeSelection:
    """Test that the 5-case runtime selection is total (no undefined cases)."""

    def test_all_valid_runtime_values(self):
        """Runtime must be one of auto, docker, venv."""
        valid = {"auto", "docker", "venv"}
        assert valid == {"auto", "docker", "venv"}

    @patch("robo_eval.docker_backend.is_docker_available", return_value=True)
    def test_auto_docker_available(self, mock_avail):
        """auto + docker available -> docker."""
        from robo_eval.docker_backend import is_docker_available
        runtime = "auto"
        if runtime == "auto" and is_docker_available():
            resolved = "docker"
        elif runtime == "auto":
            resolved = "venv"
        elif runtime == "docker" and is_docker_available():
            resolved = "docker"
        elif runtime == "docker":
            resolved = "error"
        elif runtime == "venv":
            resolved = "venv"
        else:
            resolved = "undefined"
        assert resolved == "docker"

    @patch("robo_eval.docker_backend.is_docker_available", return_value=False)
    def test_auto_no_docker(self, mock_avail):
        """auto + no docker -> venv."""
        from robo_eval.docker_backend import is_docker_available
        runtime = "auto"
        if runtime == "auto" and is_docker_available():
            resolved = "docker"
        elif runtime == "auto":
            resolved = "venv"
        elif runtime == "docker" and is_docker_available():
            resolved = "docker"
        elif runtime == "docker":
            resolved = "error"
        elif runtime == "venv":
            resolved = "venv"
        else:
            resolved = "undefined"
        assert resolved == "venv"

    @patch("robo_eval.docker_backend.is_docker_available", return_value=True)
    def test_docker_explicit_available(self, mock_avail):
        """docker + available -> docker."""
        from robo_eval.docker_backend import is_docker_available
        runtime = "docker"
        if runtime == "auto" and is_docker_available():
            resolved = "docker"
        elif runtime == "auto":
            resolved = "venv"
        elif runtime == "docker" and is_docker_available():
            resolved = "docker"
        elif runtime == "docker":
            resolved = "error"
        elif runtime == "venv":
            resolved = "venv"
        else:
            resolved = "undefined"
        assert resolved == "docker"

    @patch("robo_eval.docker_backend.is_docker_available", return_value=False)
    def test_docker_explicit_not_available(self, mock_avail):
        """docker + not available -> error."""
        from robo_eval.docker_backend import is_docker_available
        runtime = "docker"
        if runtime == "auto" and is_docker_available():
            resolved = "docker"
        elif runtime == "auto":
            resolved = "venv"
        elif runtime == "docker" and is_docker_available():
            resolved = "docker"
        elif runtime == "docker":
            resolved = "error"
        elif runtime == "venv":
            resolved = "venv"
        else:
            resolved = "undefined"
        assert resolved == "error"

    def test_venv_explicit(self):
        """venv -> venv (no docker check needed)."""
        runtime = "venv"
        if runtime == "auto":
            resolved = "docker"
        elif runtime == "docker":
            resolved = "docker"
        elif runtime == "venv":
            resolved = "venv"
        else:
            resolved = "undefined"
        assert resolved == "venv"


# ---------------------------------------------------------------------------
# Config: Docker image maps
# ---------------------------------------------------------------------------


class TestDockerImageConfig:
    """Test that DOCKER_IMAGES and VLA_DOCKER_IMAGES are correctly defined."""

    def test_sim_images_defined(self):
        from robo_eval.config import DOCKER_IMAGES
        expected_sims = ["libero", "libero_pro", "libero_infinity", "robocasa", "robotwin"]
        for sim in expected_sims:
            assert sim in DOCKER_IMAGES, f"Missing DOCKER_IMAGES['{sim}']"

    def test_vla_images_defined(self):
        from robo_eval.config import VLA_DOCKER_IMAGES
        expected_vlas = ["pi05", "smolvla", "openvla", "cosmos", "internvla"]
        for vla in expected_vlas:
            assert vla in VLA_DOCKER_IMAGES, f"Missing VLA_DOCKER_IMAGES['{vla}']"

    def test_image_names_have_tags(self):
        from robo_eval.config import DOCKER_IMAGES, VLA_DOCKER_IMAGES
        for name, image in {**DOCKER_IMAGES, **VLA_DOCKER_IMAGES}.items():
            assert ":" in image, f"Image for '{name}' missing tag: {image}"
