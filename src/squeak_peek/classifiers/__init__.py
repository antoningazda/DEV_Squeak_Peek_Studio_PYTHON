"""
Call-type classifiers — self-registering plugins.

A classifier takes the events a detector already found (a list of Label
objects with start/end times) plus the raw audio, and returns labels with
the ``.label`` field (the call type, e.g. "d", "5", "sk") filled in or
refined. It never invents new event boundaries — that is a detector's job.

To add a new classifier: drop a module in this package defining an
AbstractClassifier subclass with a unique ``id`` (see base.py, and
duration.py for a minimal worked example). Every module in this package
is imported below for its registration side effect, so nothing else needs
editing — the Detection tab's classifier list and the Settings tab's
per-classifier form are generated from ``AbstractClassifier.all()``.
"""

from __future__ import annotations

import importlib
import pkgutil

from .base import AbstractClassifier  # noqa: F401  (re-exported for convenience)

for _module_info in pkgutil.iter_modules(__path__):
    if _module_info.name != "base" and not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")

del _module_info
