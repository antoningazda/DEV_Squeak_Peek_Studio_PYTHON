"""
Unit tests for squeak_peek.cnn.convert_usvseg — no torch dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squeak_peek.audio.io import save_wav
from squeak_peek.cnn.convert_usvseg import (
    convert_usvseg_csv,
    convert_usvseg_dir,
    estimate_freq_band,
)
from squeak_peek.labels.io import import_labels

FS = 250_000


def _tone_signal(freq_hz: float, duration_s: float, fs: int = FS, noise_amp: float = 0.02) -> np.ndarray:
    t = np.arange(round(duration_s * fs)) / fs
    tone = 0.8 * np.sin(2 * np.pi * freq_hz * t)
    rng = np.random.default_rng(0)
    return (tone + noise_amp * rng.standard_normal(len(t))).astype(np.float32)


class TestEstimateFreqBand:
    def test_finds_band_around_tone(self):
        signal = _tone_signal(freq_hz=60_000, duration_s=0.05)
        f_lo, f_hi = estimate_freq_band(signal, FS, fcut_min=40_000, fcut_max=120_000)
        assert f_lo <= 60_000 <= f_hi
        assert f_hi - f_lo < 20_000  # tight around the tone, not the whole band

    def test_short_segment_falls_back_to_full_band(self):
        signal = np.zeros(4, dtype=np.float32)
        f_lo, f_hi = estimate_freq_band(signal, FS, fcut_min=40_000, fcut_max=120_000)
        assert (f_lo, f_hi) == (40_000, 120_000)


class TestConvertUsvsegCsv:
    def test_converts_rows_to_labels(self, tmp_path: Path):
        signal = _tone_signal(freq_hz=70_000, duration_s=0.2)
        wav_path = tmp_path / "rec.wav"
        save_wav(wav_path, signal, FS)

        csv_path = tmp_path / "rec.csv"
        csv_path.write_text("0.05,0.08\n0.12,0.15\n")

        labels = convert_usvseg_csv(csv_path, wav_path, fcut_min=40_000, fcut_max=120_000)
        assert len(labels) == 2
        assert labels[0].start_time == pytest.approx(0.05)
        assert labels[0].end_time == pytest.approx(0.08)
        for lbl in labels:
            assert 40_000 <= lbl.start_frequency < lbl.end_frequency <= 120_000

    def test_skips_malformed_rows(self, tmp_path: Path):
        signal = _tone_signal(freq_hz=70_000, duration_s=0.1)
        wav_path = tmp_path / "rec.wav"
        save_wav(wav_path, signal, FS)

        csv_path = tmp_path / "rec.csv"
        csv_path.write_text("not,a,number\n0.02,0.01\n0.03,0.05\n")  # bad row + inverted + valid

        labels = convert_usvseg_csv(csv_path, wav_path)
        assert len(labels) == 1
        assert labels[0].start_time == pytest.approx(0.03)


class TestConvertUsvsegDir:
    def test_converts_matching_pairs_and_writes_importable_labels(self, tmp_path: Path):
        for name, freq in [("a", 55_000), ("b", 80_000)]:
            save_wav(tmp_path / f"{name}.wav", _tone_signal(freq, 0.1), FS)
            (tmp_path / f"{name}.csv").write_text("0.02,0.04\n")

        pairs = convert_usvseg_dir(tmp_path)
        assert len(pairs) == 2
        for wav_path, label_path in pairs:
            assert label_path.exists()
            labels = import_labels(label_path, FS)
            assert len(labels) == 1

    def test_ignores_wav_without_matching_csv(self, tmp_path: Path):
        save_wav(tmp_path / "orphan.wav", _tone_signal(60_000, 0.05), FS)
        pairs = convert_usvseg_dir(tmp_path)
        assert pairs == []
