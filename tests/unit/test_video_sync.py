"""Unit tests for squeak_peek.video.sync: snap-onset detection and
WAV<->video timeline alignment."""

from __future__ import annotations

import numpy as np
import pytest

from squeak_peek.config import AppSettings
from squeak_peek.labels.model import Label
from squeak_peek.video.sync import compute_sync_offset, find_first_transient


def _signal_with_click(fs: int, duration: float, click_t: float, seed: int) -> np.ndarray:
    n = int(fs * duration)
    rng = np.random.default_rng(seed)
    sig = (rng.standard_normal(n) * 0.005).astype(np.float32)
    start = int(click_t * fs)
    length = max(1, int(0.003 * fs))
    sig[start : start + length] += (rng.standard_normal(length) * 0.6).astype(np.float32)
    return sig


class TestFindFirstTransient:
    def test_finds_click_near_expected_time(self):
        fs = 44100
        sig = _signal_with_click(fs, duration=1.0, click_t=0.35, seed=0)
        t = find_first_transient(sig, fs, band=(2_000, 20_000))
        assert abs(t - 0.35) < 0.02

    def test_no_transient_raises(self):
        fs = 44100
        rng = np.random.default_rng(1)
        sig = (rng.standard_normal(fs) * 0.005).astype(np.float32)  # flat noise, no click
        with pytest.raises(ValueError):
            find_first_transient(sig, fs, band=(2_000, 20_000))

    def test_invalid_band_raises(self):
        fs = 44100
        sig = np.zeros(fs, dtype=np.float32)
        with pytest.raises(ValueError):
            find_first_transient(sig, fs, band=(20_000, 2_000))


class _FakeState:
    def __init__(self, samples, fs, reference_labels=None, detected_labels=None):
        self.samples = samples
        self.fs = fs
        self.reference_labels = reference_labels or []
        self.detected_labels = detected_labels or []
        self.settings = AppSettings.defaults()


class TestComputeSyncOffset:
    def test_auto_detect_both_sides(self):
        wav_fs = 250_000
        wav_click_t = 0.4
        wav = _signal_with_click(wav_fs, duration=1.0, click_t=wav_click_t, seed=2)

        video_fs = 44100
        video_click_t = 0.7
        video = _signal_with_click(video_fs, duration=1.0, click_t=video_click_t, seed=3)

        state = _FakeState(wav, wav_fs)
        result = compute_sync_offset(state, video, video_fs)

        expected = video_click_t - wav_click_t
        assert abs(result.offset - expected) < 0.03
        assert "auto-detected snap in WAV" in result.method
        assert "video" in result.method

    def test_prefers_sk_label_over_auto_detect(self):
        wav_fs = 250_000
        sk_time = 0.5
        wav = _signal_with_click(wav_fs, duration=1.0, click_t=0.1, seed=4)  # decoy click

        video_fs = 44100
        video_click_t = 0.8
        video = _signal_with_click(video_fs, duration=1.0, click_t=video_click_t, seed=5)

        labels = [Label(start_time=sk_time, end_time=sk_time + 0.01, label="sk")]
        state = _FakeState(wav, wav_fs, reference_labels=labels)
        result = compute_sync_offset(state, video, video_fs)

        expected = video_click_t - sk_time
        assert abs(result.offset - expected) < 0.03
        assert "'sk'" in result.method

    def test_no_wav_loaded_raises(self):
        state = _FakeState(None, 250_000)
        video = _signal_with_click(44100, 1.0, 0.5, seed=6)
        with pytest.raises(RuntimeError):
            compute_sync_offset(state, video, 44100)
