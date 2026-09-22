"""
CNN (Faster R-CNN) detector: inference over a model trained by
squeak_peek.cnn.train.

Object-detection alternative to MLDetector's frame classifier — predicts
a (start_time, end_time, start_frequency, end_frequency) box per call
directly, tiling the signal the same way training tiles did and merging
duplicate detections from overlapping tiles.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from squeak_peek.cnn.spectrogram_image import signal_to_image
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.model import Label

# torch and squeak_peek.cnn.train are imported inside the methods that need
# them: detectors/__init__.py imports every module in this package for its
# registration side effect, so a module-level torch import here would make
# the whole app unusable for anyone without the optional 'cnn' extra.

_INFERENCE_BATCH = 8


class CNNParams(BaseModel):
    """Runtime parameters for the CNN (Faster R-CNN) detector."""

    modelPath: str = Field(
        "",
        description="Path to a trained Faster R-CNN checkpoint (.pt) used for detection.",
        json_schema_extra={"widget": "file", "file_filter": "PyTorch checkpoints (*.pt)"},
    )
    minEventDuration: float = Field(
        0.003, ge=0.0, le=10.0,
        description="Shortest event kept after merging overlapping tile detections.",
        json_schema_extra={"unit": "s", "decimals": 4},
    )
    sensitivity: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Box-score cutoff for accepting a predicted call.",
        json_schema_extra={
            "decimals": 4,
            "caption": "Lower sensitivity finds more (and weaker) calls; higher sensitivity keeps only confident detections.",
        },
    )

    model_config = {"populate_by_name": True}


def _nms_by_time(boxes: list[tuple[float, float, float, float, float]], iou_thresh: float = 0.3):
    """Suppress duplicate detections (from overlapping tiles) by 1-D time IoU, keeping the highest score."""
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    for box in boxes:
        s0, e0 = box[0], box[1]
        overlaps = False
        for k in kept:
            s1, e1 = k[0], k[1]
            inter = max(0.0, min(e0, e1) - max(s0, s1))
            union = max(e0, e1) - min(s0, s1)
            if union > 0 and inter / union > iou_thresh:
                overlaps = True
                break
        if not overlaps:
            kept.append(box)
    return kept


class CNNDetector(AbstractDetector):
    """Faster R-CNN sliding-tile USV detector."""

    id = "CNN"
    display_name = "CNN"
    description = (
        "Deep-learning (Faster R-CNN) detector predicting a time/frequency box per call. "
        "Needs the optional 'cnn' extra (pip install squeak-peek-studio[cnn]) and a model "
        "file configured in Settings -> CNN detector."
    )
    Params = CNNParams

    def __init__(self, params: CNNParams) -> None:
        super().__init__(params)
        self._checkpoint: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._checkpoint is None:
            if not self.params.modelPath:
                raise ValueError(
                    "CNNDetector: params.modelPath is not set. Train a model "
                    "(squeak_peek.cnn.train.train_cnn) and point CNNParams.modelPath "
                    "at the saved .pt checkpoint."
                )
            try:
                from squeak_peek.cnn.train import load_checkpoint
            except ImportError as exc:
                raise RuntimeError(
                    "The CNN detector needs torch/torchvision. "
                    "Install with: pip install squeak-peek-studio[cnn]"
                ) from exc
            self._checkpoint = load_checkpoint(self.params.modelPath)
        return self._checkpoint

    def detect(self, signal: np.ndarray, fs: int) -> list[Label]:
        # _load() first: it reports a missing model or a missing 'cnn' extra
        # in plain language, where a bare `import torch` here would surface a
        # raw ImportError instead.
        checkpoint = self._load()

        import torch

        model = checkpoint["model"]
        device = checkpoint["device"]
        tp = checkpoint["tile_params"]

        window_len = round(tp["window_s"] * fs)
        if window_len <= 0:
            return []
        window_len = min(window_len, len(signal))
        if window_len <= 0:
            return []
        hop_len = max(1, window_len // 2)

        starts = list(range(0, max(len(signal) - window_len, 0) + 1, hop_len))
        if not starts:
            starts = [0]

        raw_boxes: list[tuple[float, float, float, float, float]] = []

        for batch_start in range(0, len(starts), _INFERENCE_BATCH):
            batch_starts = starts[batch_start:batch_start + _INFERENCE_BATCH]
            tensors = []
            tile_meta = []
            for start in batch_starts:
                end = start + window_len
                image, freqs, times = signal_to_image(
                    signal[start:end], fs, tp["fcut_min"], tp["fcut_max"],
                    segment_length=tp["segment_length"], overlap_factor=tp["overlap_factor"],
                )
                tensors.append(torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1).to(device))
                tile_meta.append((start / fs, freqs, times))

            with torch.no_grad():
                predictions = model(tensors)

            for pred, (t0, freqs, times) in zip(predictions, tile_meta):
                boxes = pred["boxes"].cpu().numpy()
                scores = pred["scores"].cpu().numpy()
                for (x0, y0, x1, y1), score in zip(boxes, scores):
                    if score < self.params.sensitivity:
                        continue
                    xi0, xi1 = int(np.clip(x0, 0, len(times) - 1)), int(np.clip(x1, 0, len(times) - 1))
                    yi0, yi1 = int(np.clip(y0, 0, len(freqs) - 1)), int(np.clip(y1, 0, len(freqs) - 1))
                    start_time = t0 + float(times[xi0])
                    end_time = t0 + float(times[xi1])
                    start_freq = float(freqs[yi0])
                    end_freq = float(freqs[yi1])
                    if end_time > start_time:
                        raw_boxes.append((start_time, end_time, start_freq, end_freq, float(score)))

        kept = _nms_by_time(raw_boxes)

        labels: list[Label] = []
        for start_time, end_time, start_freq, end_freq, _score in kept:
            if (end_time - start_time) < self.params.minEventDuration:
                continue
            labels.append(
                Label(
                    start_time=start_time,
                    end_time=end_time,
                    label="d",
                    start_frequency=start_freq,
                    end_frequency=end_freq,
                    start_index=round(start_time * fs),
                    stop_index=round(end_time * fs),
                )
            )
        labels.sort(key=lambda lbl: lbl.start_time)
        return labels
