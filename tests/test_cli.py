"""
Tests for roboeval CLI and version info.

Covers:
- __version__ existence and format
- _version_callback behavior
"""


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------


class TestVersionInfo:
    def test_version_exists(self):
        from roboeval import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0
        parts = __version__.split(".")
        assert len(parts) >= 2


# ---------------------------------------------------------------------------
# Bundled setup script
#
# `setup.sh` lives inside the package (roboeval/_scripts/setup.sh) so that it
# ships in the wheel. The packaging allow-list in pyproject.toml is explicit,
# so a rename or a missed package-data entry would silently break
# `roboeval setup` for everyone who installed from PyPI. These tests fail
# loudly instead.
# ---------------------------------------------------------------------------


class TestSetupScriptIsBundled:
    def test_setup_script_resolves_inside_the_package(self):
        from roboeval.cli.main import _find_setup_script

        script = _find_setup_script()
        assert script.is_file()
        # Must resolve to the copy inside the package, not the repo-root shim,
        # because only the packaged copy exists in an installed wheel.
        assert script.parent.name == "_scripts"
        assert script.parent.parent.name == "roboeval"

    def test_setup_script_is_listed_in_package_data(self):
        """The wheel only contains what package-data names explicitly."""
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        package_data = data["tool"]["setuptools"]["package-data"]["roboeval"]
        assert "_scripts/setup.sh" in package_data
        # The RoboCasa backend loads this pickle package-relative at runtime.
        assert (
            "robocasa_controller_configs.pkl" in data["tool"]["setuptools"]["package-data"]["sims"]
        )
        # A "../" escape would sweep the whole repo into the artifacts.
        assert not any(entry.startswith("../") for entry in package_data)
        assert data["tool"]["setuptools"]["include-package-data"] is False

    def test_setup_help_runs(self):
        """`roboeval setup --help` must work from an installed package."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "roboeval.cli.main", "setup", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "setup" in result.stdout.lower()
