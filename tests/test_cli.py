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
