"""Tests for the pre-detection noise suppression (squeak_peek.audio.denoise)."""

from __future__ import annotations

import numpy as np

import squeak_peek.detectors  # noqa: F401  (registers detectors)
from squeak_peek.audio.denoise import DenoiseParams, preprocess, spectral_denoise
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector

FS = 250_000


def _noisy_tone(seconds: float = 2.0, seed: int = 0) -> tuple[np.ndarray, slice]:
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = (rng.normal(size=n) * 0.01).astype(np.float32)
    call = slice(n // 4, n // 4 + FS // 20)  # 50 ms, 60 kHz
    t = np.arange(call.stop - call.start) / FS
    x[call] += (0.2 * np.sin(2 * np.pi * 60_000 * t)).astype(np.float32)
    return x, call


def test_no_reduction_is_identity():
    x, _ = _noisy_tone()
    y = spectral_denoise(x, FS, DenoiseParams(maxReductionDb=0.0))
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_suppresses_noise_and_keeps_the_call():
    x, call = _noisy_tone()
    y = spectral_denoise(x, FS)
    quiet = slice(0, FS // 4)  # before the call (which starts at 0.5 s)
    assert y[quiet].std() < 0.2 * x[quiet].std()          # ≥ 14 dB less background
    assert y[call].std() > 0.9 * x[call].std()            # the call survives


def test_shape_and_dtype_preserved():
    x, _ = _noisy_tone(0.5)
    y = spectral_denoise(x, FS)
    assert y.shape == x.shape and y.dtype == x.dtype
    short = x[:100]
    np.testing.assert_array_equal(spectral_denoise(short, FS), short)


def test_preprocess_none_is_passthrough():
    x, _ = _noisy_tone(0.5)
    assert preprocess(x, FS, None) is x
    assert preprocess(x, FS, DenoiseParams()) is not x


def test_denoise_is_a_per_detector_choice():
    psd = AbstractDetector.get("PSD")
    bscd = AbstractDetector.get("BSCD")
    ml = AbstractDetector.get("ML")
    assert psd(psd.Params()).wants_denoise
    assert not bscd(bscd.Params()).wants_denoise
    assert bscd(bscd.Params(denoise=True)).wants_denoise
    # Learned detectors follow their model, never the pipeline switch.
    assert not ml(ml.Params()).wants_denoise


def test_settings_round_trip(tmp_path):
    s = AppSettings.defaults()
    s.detection.pre = DenoiseParams(noisePercentile=30.0, maxReductionDb=24.0)
    path = tmp_path / "s.json"
    s.save_json(path)
    loaded = AppSettings.from_json(path)
    assert loaded.detection.pre.noisePercentile == 30.0
    assert loaded.detection.pre.maxReductionDb == 24.0
