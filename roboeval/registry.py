"""Import string resolution for benchmarks and model servers.

Resolves ``"module.path:ClassName"`` strings from config files into actual
Python objects using ``importlib``.

Example::

    cls = resolve_import_string("roboeval.specs:ActionObsSpec")
    assert cls.__name__ == "ActionObsSpec"
"""

from __future__ import annotations

import importlib
from typing import Any


def resolve_import_string(import_path: str) -> Any:
    """Resolve a ``"module.path:AttributeName"`` string to the actual object.

    The string must contain exactly one colon separating the module path from
    the attribute name.  The attribute is looked up on the imported module via
    ``getattr``, so it works for classes, functions, and module-level constants.

    Parameters
    ----------
    import_path:
        A string of the form ``"some.module.path:AttributeName"``.

    Returns
    -------
    Any
        The resolved Python object.

    Raises
    ------
    ValueError
        If ``import_path`` does not contain exactly one colon.
    ImportError
        If the module cannot be imported.
    AttributeError
        If the attribute does not exist on the module.

    Examples
    --------
    >>> cls = resolve_import_string("roboeval.specs:ActionObsSpec")
    >>> cls.__name__
    'ActionObsSpec'
    """
    if import_path.count(":") != 1:
        raise ValueError(
            f"import_path must be of the form 'module.path:AttributeName', got {import_path!r}"
        )
    module_path, attr_name = import_path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
