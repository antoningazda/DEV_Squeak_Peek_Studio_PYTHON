"""
Unit tests for squeak_peek.cnn.dataset (USVBoxDataset). Requires torch —
skipped automatically if the 'cnn' extra isn't installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from squeak_peek.audio.io import save_wav  # noqa: E402
from squeak_peek.cnn.dataset import USVBoxDataset  # noqa: E402
from squeak_peek.labels.io import export_labels  # noqa: E402
from squeak_peek.labels.model import Label  # noqa: E402

FS = 20_000
FCUT_MIN, FCUT_MAX = 1_000, 9_000


def _make_pair(tmp_path: Path, duration_s: float, boxes: list[Label]) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    signal = (0.05 * rng.standard_normal(round(duration_s * FS))).astype(np.float32)
    for lbl in boxes:
        t = np.arange(round(lbl.start_time * FS), round(lbl.end_time * FS)) / FS
        freq = (lbl.start_frequency + lbl.end_frequency) / 2
        signal[round(lbl.start_time * FS):round(lbl.end_time * FS)] += 0.8 * np.sin(2 * np.pi * freq * t)

    wav_path = tmp_path / "rec.wav"
    save_wav(wav_path, signal, FS)
    label_path = tmp_path / "rec_labels.txt"
    export_labels(label_path, boxes)
    return wav_path, label_path


class TestUSVBoxDataset:
    def test_tiles_have_correct_shapes_and_boxes(self, tmp_path: Path):
        boxes = [Label(start_time=1.0, end_time=1.1, label="USV", start_frequency=4_000, end_frequency=5_000)]
        wav_path, label_path = _make_pair(tmp_path, duration_s=3.0, boxes=boxes)

        ds = USVBoxDataset(
            [(wav_path, label_path)],
            window_s=0.5, hop_s=0.25, fcut_min=FCUT_MIN, fcut_max=FCUT_MAX,
            segment_length=256, overlap_factor=0.5, negative_ratio=1.0,
        )
        assert len(ds) > 0

        found_box = False
        for i in range(len(ds)):
            image, target = ds[i]
            assert image.shape[0] == 3
            n_boxes = target["boxes"].shape[0]
            assert target["labels"].shape[0] == n_boxes
            for x0, y0, x1, y1 in target["boxes"].tolist():
                assert 0 <= x0 < x1 <= image.shape[2]
                assert 0 <= y0 < y1 <= image.shape[1]
                found_box = True
        assert found_box, "expected at least one tile to contain the labeled box"

    def test_raises_without_any_positive_tile(self, tmp_path: Path):
        wav_path, label_path = _make_pair(tmp_path, duration_s=1.0, boxes=[])
        with pytest.raises(ValueError, match="no tiles contain a labeled call"):
            USVBoxDataset([(wav_path, label_path)], window_s=0.5, hop_s=0.25, fcut_min=FCUT_MIN, fcut_max=FCUT_MAX)

    def test_negative_ratio_caps_negative_tiles(self, tmp_path: Path):
        boxes = [Label(start_time=0.1, end_time=0.15, label="USV", start_frequency=4_000, end_frequency=5_000)]
        wav_path, label_path = _make_pair(tmp_path, duration_s=5.0, boxes=boxes)

        ds = USVBoxDataset(
            [(wav_path, label_path)],
            window_s=0.2, hop_s=0.1, fcut_min=FCUT_MIN, fcut_max=FCUT_MAX, negative_ratio=1.0, seed=1,
        )
        n_pos = sum(1 for i in range(len(ds)) if ds[i][1]["boxes"].shape[0] > 0)
        n_neg = len(ds) - n_pos
        assert n_neg <= n_pos + 1  # ratio 1.0, allow rounding
