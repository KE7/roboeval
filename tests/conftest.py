"""pytest configuration and cross-cutting fixtures for the roboeval test suite.

Some backend tests stub optional GPU dependencies with ``MagicMock`` so policy
modules can be imported in CPU-only environments.

When scipy inspects array backends, its ``_issubclass_fast`` helper can find a
stubbed ``torch`` module in ``sys.modules``, resolve ``torch.Tensor`` to another
``MagicMock``, and then call ``issubclass(numpy.ndarray, MagicMock())``, which
raises::

    TypeError: issubclass() arg 2 must be a class, a tuple of classes, or a union

The fixture below removes leaked torch-related stubs before each test, preventing
fake modules from corrupting scipy's array-API compatibility layer.
"""

from __future__ import annotations

import sys
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def _remove_torch_mock_before_test() -> Generator[None, None, None]:
    """Remove MagicMock torch stubs from sys.modules before every test.

    If a test file stubbed ``torch`` (or related packages) with a ``MagicMock``
    and did not clean up, scipy's ``is_torch_array`` helper will raise a
    ``TypeError`` when it tries to call ``issubclass(cls, MagicMock().Tensor)``.

    This fixture detects that case and removes the fake entry so scipy sees
    "torch not installed" (safe ``return False`` path) rather than a broken stub.
    The original entry is restored after the test so tests that explicitly need
    the stub (and set it up themselves) are unaffected.
    """
    from unittest.mock import MagicMock

    TORCH_KEYS = [
        "torch",
        "torchvision",
        "torchvision.transforms",
    ]

    # Snapshot any MagicMock-backed optional dependency stubs.
    stashed: dict[str, object] = {}
    for key in TORCH_KEYS:
        val = sys.modules.get(key)
        if isinstance(val, MagicMock):
            stashed[key] = val
            del sys.modules[key]

    # Clear scipy's cached backend check after cleaning sys.modules.
    try:
        from scipy._lib.array_api_compat.common._helpers import _issubclass_fast

        _issubclass_fast.cache_clear()
    except (ImportError, AttributeError):
        pass

    yield

    # Restore any stubs we removed for tests that installed them explicitly.
    sys.modules.update(stashed)
