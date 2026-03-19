"""
Helper: detect whether a real (non-stub) PyTorch is installed.

Must be imported BEFORE conftest.py installs stubs, so it is kept as a
separate module that test files import explicitly at their top level,
before any src imports that might trigger stub registration.
"""

import importlib

def _check() -> bool:
    try:
        spec = importlib.util.find_spec("torch")  # type: ignore[attr-defined]
        if spec is None:
            return False
        # Try to load a small attribute that only real torch has
        import torch
        return callable(getattr(torch, "FloatTensor", None))
    except Exception:
        return False


TORCH_AVAILABLE: bool = _check()
