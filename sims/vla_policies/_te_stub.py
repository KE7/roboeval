"""
Transformer-Engine stub installer.

Import this module BEFORE any cosmos_policy / megatron imports to install
a meta-path finder that auto-stubs transformer_engine and
transformer_engine_torch. These NVIDIA packages require CUDA kernels
unavailable on some platforms; only the optimizer uses them at runtime,
so inference works fine with stubs.

Usage:
    import sims.vla_policies._te_stub  # noqa: F401
"""

import importlib.abc
import importlib.machinery
import sys
import types


class _StubMeta(type):
    """Metaclass so that class-level attribute access (e.g. _Stub.Sequential)
    also returns _Stub, not just instance-level access."""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return cls


class _Stub(metaclass=_StubMeta):
    """Universal stub that supports arbitrary attribute access and calling."""

    __version__ = "2.2.0"

    def __init__(self, *a, **kw):
        pass

    def __call__(self, *a, **kw):
        return _Stub()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Stub

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def __bool__(self):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    @classmethod
    def __class_getitem__(cls, item):
        return cls


class _TEStubModule(types.ModuleType):
    """Module that returns stub objects for any attribute access."""

    __version__ = "2.2.0"

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _Stub


class _TEStubFinder(importlib.abc.MetaPathFinder):
    _PKGS = ("transformer_engine", "transformer_engine_torch")

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            mod = _TEStubModule(spec.name)
            mod.__loader__ = self
            mod.__path__ = []
            mod.__package__ = spec.name
            mod.__spec__ = spec
            return mod

        def exec_module(self, module):
            pass

    _loader = _Loader()

    def find_spec(self, fullname, path, target=None):
        for pkg in self._PKGS:
            if fullname == pkg or fullname.startswith(pkg + "."):
                return importlib.machinery.ModuleSpec(
                    fullname, self._loader, is_package=True
                )
        return None


if not any(isinstance(f, _TEStubFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _TEStubFinder())
