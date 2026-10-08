"""
Stationary-noise suppression applied to a recording before detection.

An ultrasonic recording is mostly background noise — fans, electronics, the
microphone's own hiss — that is roughly constant over the session, while the
calls themselves are sparse. Each frequency bin's noise level can therefore
be estimated from the recording itself, as a low percentile of that bin's
magnitude over time, and subtracted (spectral subtraction, the same idea as
Audacity's Noise Reduction). Energy detectors such as PSD otherwise see that
in-band noise as a large constant part of their envelope and lose precision.

The result is a waveform of the same length and sample rate, so any detector
can run on it unchanged. It is meant for detection only: export, review and
call-type classification keep using the original audio.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

#: Upper bound on the frames used to estimate the noise profile; spread
#: evenly over the recording, so the estimate costs the same for any length.
_PROFILE_FRAMES = 20_000


class DenoiseParams(BaseModel):
    """Parameters of the pre-detection noise suppression (shared by every
    detector whose own ``denoise`` parameter is on)."""

    nfft: int = Field(
        1024, ge=128, le=16_384,
        description="STFT window length used for noise estimation and suppression.",
        json_schema_extra={"unit": "samples"},
    )
    noisePercentile: float = Field(
        50.0, ge=1.0, le=90.0,
        description="Percentile of each frequency bin's magnitude over time taken as its noise level.",
        json_schema_extra={
            "decimals": 1,
            "caption": "Calls are sparse, so the median of a bin over a whole recording is its background.",
        },
    )
    oversubtraction: float = Field(
        1.25, ge=0.5, le=5.0,
        description="Multiple of the noise level subtracted from each bin.",
        json_schema_extra={"decimals": 2},
    )
    maxReductionDb: float = Field(
        30.0, ge=0.0, le=60.0,
        description="Largest attenuation applied to any bin (a floor rather than a hard gate).",
        json_schema_extra={"unit": "dB", "decimals": 1},
    )

    model_config = {"populate_by_name": True}


def _frames(xp: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]


def noise_profile(x: np.ndarray, n_fft: int, percentile: float) -> np.ndarray:
    """Per-bin noise magnitude: the ``percentile`` of each rfft bin over
    (an even subsample of) the recording's Hann-windowed frames."""
    hop = n_fft // 4
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    frames = _frames(np.asarray(x, dtype=np.float32), n_fft, hop)
    if len(frames) == 0:
        return np.zeros(n_fft // 2 + 1, dtype=np.float32)
    idx = np.unique(np.linspace(0, len(frames) - 1, min(len(frames), _PROFILE_FRAMES)).astype(int))
    mag = np.abs(np.fft.rfft(frames[idx] * win, axis=1))
    return np.percentile(mag, percentile, axis=0).astype(np.float32)


def spectral_denoise(
    signal: np.ndarray,
    fs: int,
    params: DenoiseParams | None = None,
    *,
    block_frames: int = 8192,
) -> np.ndarray:
    """
    Suppress stationary background noise by soft spectral subtraction.

    Each STFT bin is scaled by ``(|X| - a·N) / |X|``, clipped to
    ``[10^(-maxReductionDb/20), 1]`` and smoothed over 3 bins × 5 frames so
    the residual stays a quiet hiss instead of "musical noise"; ``N`` is the
    bin's noise profile (:func:`noise_profile`) and ``a`` the oversubtraction.
    Resynthesised by weighted overlap-add (periodic Hann, hop = nfft/4),
    block by block, so memory stays bounded on hour-long recordings.

    Returns a float array of the same length and dtype as ``signal``.
    ``fs`` is accepted for a uniform preprocessing signature; the
    suppression itself works in bins and frames.
    """
    from scipy.ndimage import uniform_filter

    p = params or DenoiseParams()
    x = np.asarray(signal)
    n = len(x)
    n_fft = int(p.nfft)
    hop = n_fft // 4
    m = n_fft // hop
    if n < n_fft:
        return x.copy()

    xf = x.astype(np.float32)
    profile = noise_profile(xf, n_fft, p.noisePercentile)
    floor = np.float32(10.0 ** (-p.maxReductionDb / 20.0))
    sub = np.float32(p.oversubtraction) * profile

    # Pad by a full window on both sides so every real sample is covered by
    # all m overlapping frames (constant overlap-add weight).
    xp = np.pad(xf, (n_fft, n_fft + hop))
    frames = _frames(xp, n_fft, hop)
    n_frames = len(frames)
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    out = np.zeros((n_frames + m) * hop, dtype=np.float32)

    # Overlapping blocks (2 frames each side) so the time smoothing of the
    # gain has real neighbours at block seams; only the core is written.
    ctx = 2
    for b0 in range(0, n_frames, block_frames):
        b1 = min(n_frames, b0 + block_frames)
        a0, a1 = max(0, b0 - ctx), min(n_frames, b1 + ctx)
        spec = np.fft.rfft(frames[a0:a1] * win, axis=1)
        mag = np.abs(spec)
        gain = (mag - sub) / (mag + np.float32(1e-30))
        np.clip(gain, floor, 1.0, out=gain)
        gain = uniform_filter(gain, size=(5, 3), mode="nearest")
        core = slice(b0 - a0, b0 - a0 + (b1 - b0))
        y = np.fft.irfft(spec[core] * gain[core], n=n_fft, axis=1).astype(np.float32) * win
        chunks = y.reshape(b1 - b0, m, hop)
        for j in range(m):
            out[(b0 + j) * hop:(b1 + j) * hop] += chunks[:, j, :].reshape(-1)

    wsum = (win ** 2).reshape(m, hop).sum(axis=0)  # constant 1.5 for this window/hop
    out = out[: (len(out) // hop) * hop].reshape(-1, hop) / wsum
    y = out.reshape(-1)[n_fft:n_fft + n]
    return y.astype(x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32)


def preprocess(signal: np.ndarray, fs: int, params: DenoiseParams | None) -> np.ndarray:
    """``signal`` denoised with ``params``, or ``signal`` itself (not a
    copy) when ``params`` is None — the "no denoising" case."""
    if params is None:
        return signal
    return spectral_denoise(signal, fs, params)
