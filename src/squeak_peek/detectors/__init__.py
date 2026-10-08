"""
Detection engines — self-registering plugins.

    PSDDetector   — Power Spectral Density detector
    BSCDDetector  — Bayesian Sequential Change Detection
    RBDDetector   — Relative Bayesian Difference
    MLDetector    — Random Forest sliding-window detector (see squeak_peek.ml
                    for training/calibration)
    PitchDetector — one event per stretch where a pitch contour holds (the
                    same tracking the Visualization tab draws)

To add a new detector: drop a module in this package defining an
AbstractDetector subclass with a unique ``id`` (see base.py). Every module
in this package is imported below for its registration side effect, so
nothing else needs editing — see base.py's docstring for details.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import AbstractDetector  # noqa: F401  (re-exported for convenience)

for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "base" and not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")

del _module_info
