"""
Abstract base class for all USV detectors.

To add a new detector: create a module in this package that defines an
AbstractDetector subclass with a unique ``id``, a ``Params`` pydantic model
describing its hyperparameters, and a ``detect()`` implementation. Drop the
file in squeak_peek/detectors/ — it is auto-imported by this package's
__init__.py and self-registers (see squeak_peek.plugins.Plugin). Nothing
else needs editing: the Detection tab's list, tooltips and dispatch, and
the Settings tab's per-detector form, are all generated from
``AbstractDetector.all()``.

See squeak_peek.detectors.psd.PSDDetector for a complete worked example.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from squeak_peek.plugins import Plugin

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


class AbstractDetector(Plugin):
    """Common interface for all USV detection algorithms."""

    _registry: ClassVar[dict[str, type[AbstractDetector]]] = {}

    #: Whether the pipeline may denoise this detector's input (when its
    #: Params has ``denoise`` on; algorithm settings in Detection.PRE, see
    #: squeak_peek.audio.denoise). Learned detectors set False and instead
    #: apply whatever denoising their model was trained with, so training
    #: and inference always see the same kind of audio.
    uses_pipeline_preprocessing: ClassVar[bool] = True

    def __init__(self, params) -> None:
        """
        Parameters
        ----------
        params : an instance of this class's ``Params`` model.
        """
        self.params = params

    @abstractmethod
    def detect(
        self,
        signal: np.ndarray,
        fs: int,
    ) -> list[Label]:
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
    def wants_denoise(self) -> bool:
        """True when the pipeline should hand this detector denoised audio:
        its Params has ``denoise`` switched on and it is not a learned
        detector that applies its model's own denoising."""
        return self.uses_pipeline_preprocessing and bool(getattr(self.params, "denoise", False))

    @property
    def name(self) -> str:
        """Human-readable detector name, e.g. 'PSD'."""
        return self.display_name or self.id
