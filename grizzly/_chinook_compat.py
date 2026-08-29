"""Import-time shims so chinook loads without optional GUI deps."""

from __future__ import annotations

import sys
from types import ModuleType


def ensure_chinook_headless_ok() -> None:
    """Allow ``import chinook.ARPES_lib`` when Tk / ``_tkinter`` is missing.

    chinook ``Tk_plot`` imports ``matplotlib.backends.backend_tkagg`` *before*
    its own try/except around ``tkinter``, so Homebrew Pythons without Tk
    fail at import. GrizzlyME does not need the interactive map GUI.
    """
    try:
        import _tkinter  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    name = "chinook.Tk_plot"
    if name in sys.modules:
        return

    stub = ModuleType(name)
    stub.tk_found = False

    def tk_query() -> bool:
        return False

    def plot_intensity_interface(*_args, **_kwargs):
        raise RuntimeError(
            "chinook interactive Tk plotting is unavailable "
            "(no _tkinter). Use numpy/matplotlib on Ig instead."
        )

    stub.tk_query = tk_query
    stub.plot_intensity_interface = plot_intensity_interface
    sys.modules[name] = stub
