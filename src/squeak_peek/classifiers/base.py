"""
Abstract base class for all call-type classifiers.

A classifier is the second stage of the pipeline: a detector finds *where*
events are; a classifier decides *what* each one is (assigns/refines the
`.label` call-type string, e.g. "d", "sk", "5" — see
AppSettings.label_edit.classification_list for the configured vocabulary).

To add a classifier: create a module in this package that defines an
AbstractClassifier subclass with a unique ``id``, a ``Params`` pydantic
model describing its hyperparameters, and a ``classify()`` implementation.
Drop the file in squeak_peek/classifiers/ — it is auto-imported by this
package's __init__.py and self-registers (see squeak_peek.plugins.Plugin).
Nothing else needs editing.

See squeak_peek.classifiers.duration.DurationClassifier for a minimal
worked example.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from squeak_peek.plugins import Plugin

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class AbstractClassifier(Plugin):
    """Common interface for all call-type classification algorithms."""

    _registry: ClassVar[dict[str, type["AbstractClassifier"]]] = {}

    def __init__(self, params) -> None:
        """
        Parameters
        ----------
        params : an instance of this class's ``Params`` model.
        """
        self.params = params

    @abstractmethod
    def classify(
        self,
        labels: list["Label"],
        signal: np.ndarray,
        fs: int,
    ) -> list["Label"]:
        """
        Assign or refine the call-type label of each input event.

        Parameters
        ----------
        labels : events already found by a detector (start/end times, etc.)
        signal : the full mono audio signal the events were detected in
        fs     : sample rate in Hz

        Returns
        -------
        list[Label] with `.label` set to a call type. Does not need to
        preserve the input list's identity/order, but normally does.
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable classifier name, e.g. 'Duration threshold'."""
        return self.display_name or self.id
