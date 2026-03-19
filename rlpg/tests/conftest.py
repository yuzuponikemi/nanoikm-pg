"""
pytest configuration for rlpg tests.

Stubs out optional heavy dependencies (torch, matplotlib, tqdm) so that
the test suite can run in lean environments where only numpy is installed.
"""

import sys
import os
import types

# Ensure the rlpg root is on sys.path
_RLPG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RLPG_ROOT not in sys.path:
    sys.path.insert(0, _RLPG_ROOT)


# ---------------------------------------------------------------------------
# Minimal stub factory
# ---------------------------------------------------------------------------

class _AnyAttr:
    """Attribute access always returns self, so stub.anything.else == stub."""
    def __getattr__(self, name):
        return self

    def __call__(self, *a, **kw):
        return self

    def __iter__(self):
        return iter([])


def _make_stub(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # make it look like a package
    return m


# ---------------------------------------------------------------------------
# torch stub
# ---------------------------------------------------------------------------

def _stub_torch() -> None:
    if "torch" in sys.modules:
        return  # real torch already imported

    # Check if real torch is importable
    try:
        import importlib
        importlib.import_module("torch")
        return  # real torch available
    except ImportError:
        pass

    torch = _make_stub("torch")

    # torch.nn
    nn = _make_stub("torch.nn")
    # Classes used in type hints and isinstance checks
    nn.Module = type("Module", (), {"__init__": lambda s, *a, **kw: None})
    nn.Linear = type("Linear", (), {"__init__": lambda s, *a, **kw: None})
    nn.ReLU = type("ReLU", (), {"__init__": lambda s, *a, **kw: None})
    nn.Tanh = type("Tanh", (), {"__init__": lambda s, *a, **kw: None})
    nn.Sequential = type("Sequential", (), {"__init__": lambda s, *a, **kw: None})
    torch.nn = nn
    sys.modules["torch.nn"] = nn

    # torch.optim
    optim = _make_stub("torch.optim")
    optim.Adam = type("Adam", (), {"__init__": lambda s, *a, **kw: None})
    torch.optim = optim
    sys.modules["torch.optim"] = optim

    # torch.distributions
    dist = _make_stub("torch.distributions")
    dist.Normal = _AnyAttr()
    torch.distributions = dist
    sys.modules["torch.distributions"] = dist

    # Basic tensor / dtype stubs
    torch.FloatTensor = _AnyAttr()
    torch.tensor = _AnyAttr()
    torch.no_grad = lambda: _AnyAttr()
    torch.clamp = _AnyAttr()

    sys.modules["torch"] = torch


# ---------------------------------------------------------------------------
# matplotlib stub
# ---------------------------------------------------------------------------

def _stub_matplotlib() -> None:
    if "matplotlib" in sys.modules:
        return

    try:
        import importlib
        importlib.import_module("matplotlib")
        return
    except ImportError:
        pass

    mpl = _make_stub("matplotlib")
    mpl.use = lambda *a, **kw: None
    sys.modules["matplotlib"] = mpl

    _any = _AnyAttr()
    for sub in ("pyplot", "animation", "patches", "lines", "gridspec", "cm"):
        full = f"matplotlib.{sub}"
        m = types.ModuleType(full)
        m.__path__ = []
        # Proxy attribute access to _AnyAttr so plt.subplots() etc. work
        m.__getattr__ = lambda name, _a=_any: _a  # noqa: E731
        setattr(mpl, sub, m)
        sys.modules[full] = m


# ---------------------------------------------------------------------------
# tqdm stub
# ---------------------------------------------------------------------------

def _stub_tqdm() -> None:
    if "tqdm" in sys.modules:
        return

    try:
        import importlib
        importlib.import_module("tqdm")
        return
    except ImportError:
        pass

    stub = _make_stub("tqdm")

    class _tqdm:
        def __init__(self, iterable=None, **kwargs):
            self._it = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self._it)

        def set_postfix(self, *args, **kwargs):
            pass

    stub.tqdm = _tqdm
    sys.modules["tqdm"] = stub


# ---------------------------------------------------------------------------
# Detect real availability BEFORE stubs are installed
# ---------------------------------------------------------------------------

def _detect_torch() -> bool:
    """Return True only if a real, functional torch is importable."""
    try:
        import importlib
        importlib.import_module("torch")
        import torch  # noqa: F401
        _ = torch.FloatTensor  # sanity check a real attribute
        return True
    except (ImportError, AttributeError):
        return False


REAL_TORCH_AVAILABLE: bool = _detect_torch()

# ---------------------------------------------------------------------------
# Apply all stubs at collection time
# ---------------------------------------------------------------------------

_stub_torch()
_stub_matplotlib()
_stub_tqdm()
