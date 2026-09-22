"""
Shared self-registration machinery for detector and classifier plugins.

Both squeak_peek.detectors.base.AbstractDetector and
squeak_peek.classifiers.base.AbstractClassifier subclass Plugin. Any class
that subclasses one of those *and* sets a non-empty ``id`` registers itself
automatically the moment its module is imported — each package's
__init__.py imports every module in the package for exactly this side
effect (see detectors/__init__.py, classifiers/__init__.py).

Nothing else — a GUI list, a Settings-tab form, a dispatch table — needs
editing to add a new detector or classifier: drop a module defining the
class in the right package and it appears everywhere automatically.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel


class NoParams(BaseModel):
    """Empty parameter set, for plugins that take no configuration."""

    model_config = {"populate_by_name": True}


class Plugin(ABC):
    """
    Self-registering base class.

    Concrete plugin families (AbstractDetector, AbstractClassifier) each
    declare their own ``_registry: ClassVar[dict] = {}`` so detectors and
    classifiers register separately. A subclass registers itself as soon
    as it defines a non-empty ``id``; intermediate abstract bases (which
    leave ``id`` at its default ``""``) are silently skipped.
    """

    id: ClassVar[str] = ""
    display_name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    Params: ClassVar[type[BaseModel]] = NoParams

    _registry: ClassVar[dict[str, type[Plugin]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.id:
            return  # an intermediate/abstract base, e.g. AbstractDetector itself

        registry = cls._registry
        existing = registry.get(cls.id)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate plugin id {cls.id!r}: {existing.__module__}.{existing.__name__} "
                f"and {cls.__module__}.{cls.__name__} both use it. Plugin ids must be unique "
                "within their family."
            )
        registry[cls.id] = cls

    @classmethod
    def get(cls, plugin_id: str) -> type[Plugin]:
        """Look up a registered subclass by its ``id``."""
        try:
            return cls._registry[plugin_id]
        except KeyError:
            raise KeyError(
                f"Unknown {cls.__name__} id {plugin_id!r}. Registered: {sorted(cls._registry)}"
            ) from None

    @classmethod
    def all(cls) -> list[type[Plugin]]:
        """All registered subclasses, sorted by id."""
        return [cls._registry[k] for k in sorted(cls._registry)]
