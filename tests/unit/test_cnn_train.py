"""
End-to-end unit tests for squeak_peek.cnn.train and detectors.cnn.CNNDetector.
Requires torch — skipped automatically if the 'cnn' extra isn't installed.

pretrained_backbone=False throughout: keeps tests offline and fast, and a
single epoch is only enough to exercise the full train -> save -> load ->
detect pipeline, not to produce an accurate model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from squeak_peek.audio.io import save_wav  # noqa: E402
from squeak_peek.cnn.train import load_checkpoint, save_checkpoint, train_cnn  # noqa: E402
from squeak_peek.detectors.cnn import CNNDetector, CNNParams  # noqa: E402
from squeak_peek.labels.io import export_labels  # noqa: E402
from squeak_peek.labels.model import Label  # noqa: E402

FS = 20_000
FCUT_MIN, FCUT_MAX = 1_000, 9_000


def _make_pair(tmp_path: Path, name: str, duration_s: float, boxes: list[Label]) -> tuple[Path, Path]:
    rng = np.random.default_rng(hash(name) % (2**31))
    signal = (0.05 * rng.standard_normal(round(duration_s * FS))).astype(np.float32)
    for lbl in boxes:
        s, e = round(lbl.start_time * FS), round(lbl.end_time * FS)
        t = np.arange(s, e) / FS
        freq = (lbl.start_frequency + lbl.end_frequency) / 2
        signal[s:e] += 0.8 * np.sin(2 * np.pi * freq * t)

    wav_path = tmp_path / f"{name}.wav"
    save_wav(wav_path, signal, FS)
    label_path = tmp_path / f"{name}_labels.txt"
    export_labels(label_path, boxes)
    return wav_path, label_path


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cnn_train")
    pairs = [
        _make_pair(tmp_path, "rec1", 3.0, [
            Label(start_time=0.5, end_time=0.6, label="USV", start_frequency=4_000, end_frequency=5_000),
            Label(start_time=1.8, end_time=1.9, label="USV", start_frequency=3_000, end_frequency=4_500),
        ]),
        _make_pair(tmp_path, "rec2", 3.0, [
            Label(start_time=0.9, end_time=1.0, label="USV", start_frequency=5_000, end_frequency=6_500),
        ]),
    ]
    checkpoint = train_cnn(
        pairs,
        window_s=0.5, hop_s=0.25, fcut_min=FCUT_MIN, fcut_max=FCUT_MAX,
        segment_length=256, overlap_factor=0.5,
        backbone="mobilenet", pretrained_backbone=False,
        epochs=1, batch_size=2, val_fraction=0.2, seed=0,
    )
    return checkpoint, tmp_path


class TestTrainCnn:
    def test_checkpoint_structure(self, tiny_checkpoint):
        checkpoint, _ = tiny_checkpoint
        info = checkpoint["training_info"]
        assert len(info["epoch_losses"]) == 1
        assert np.isfinite(info["final_loss"])
        assert info["n_tiles_train"] > 0
        assert checkpoint["tile_params"]["window_s"] == 0.5

    def test_reported_loss_stays_finite(self, tiny_checkpoint):
        """
        A batch of entirely call-free tiles makes a trained RPN emit no
        proposals, and torchvision's ROI-head loss then divides by a zero
        proposal count, yielding NaN. Those terms must not reach the
        reported loss, or the whole training curve reads as NaN.
        """
        checkpoint, _ = tiny_checkpoint
        assert all(np.isfinite(x) for x in checkpoint["training_info"]["epoch_losses"])

    def test_save_and_load_roundtrip(self, tiny_checkpoint):
        checkpoint, tmp_path = tiny_checkpoint
        path = tmp_path / "model.pt"
        save_checkpoint(checkpoint, path)

        loaded = load_checkpoint(path, device="cpu")
        assert loaded["tile_params"] == checkpoint["tile_params"]
        assert loaded["model"].training is False


class TestCnnDetector:
    def test_detect_returns_valid_labels(self, tiny_checkpoint):
        checkpoint, tmp_path = tiny_checkpoint
        model_path = tmp_path / "model_for_detect.pt"
        save_checkpoint(checkpoint, model_path)

        params = CNNParams(modelPath=str(model_path), sensitivity=0.0, minEventDuration=0.0)
        detector = CNNDetector(params)
        assert detector.name == "CNN"

        rng = np.random.default_rng(1)
        signal = (0.05 * rng.standard_normal(2 * FS)).astype(np.float32)
        labels = detector.detect(signal, FS)

        assert isinstance(labels, list)
        for lbl in labels:
            assert lbl.end_time > lbl.start_time
            assert lbl.start_frequency < lbl.end_frequency

    def test_missing_model_path_raises(self):
        detector = CNNDetector(CNNParams(modelPath=""))
        with pytest.raises(ValueError, match="modelPath is not set"):
            detector.detect(np.zeros(100, dtype=np.float32), FS)
