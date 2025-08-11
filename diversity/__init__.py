
from __future__ import annotations

import importlib
import types
from typing import Any

__all__ = [
    "pipeline",
    "__version__",
]

__version__ = "0.1.0"


def _lazy_import_module(qualified_name: str) -> types.ModuleType:
    return importlib.import_module(qualified_name)


def __getattr__(name: str) -> Any:
    if name == "pipeline":
        mod = _lazy_import_module("diversity.pipeline")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'diversity' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)