"""
Abstract base class for all USV detectors.

All four detectors (PSD, BSCD, RBD, ML) implement this interface so they
are interchangeable in the GUI and CLI pipelines.

Phase 2 will provide concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class AbstractDetector(ABC):
    """Common interface for all USV detection algorithms."""

    @abstractmethod
    def detect(
        self,
        signal: np.ndarray,
        fs: int,
    ) -> list["Label"]:
        """
        Run detection on a mono audio signal.

        Parameters
        ----------
        signal : 1-D float32 array — bandpass-filtered or raw audio
        fs     : sample rate in Hz

        Returns
        -------
        List of Label objects (start/end time, label string, frequencies).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name, e.g. 'PSD'."""
        ...
