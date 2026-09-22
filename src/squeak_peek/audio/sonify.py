"""
Sonification — bringing ultrasonic vocalizations down into human hearing.

Why the obvious implementation sounds bad
-----------------------------------------
The direct approach is ``librosa.effects.pitch_shift(y, sr, n_steps=-36)``.
Internally that is a phase-vocoder time-stretch by 2**(36/12) = 8x followed
by a resample. A phase vocoder stretching by 8x is far outside the range
where it sounds transparent: it loses vertical phase coherence between the
bins of a single partial, which is heard as the classic "phasey", metallic,
reverberant smear. Stacking a second vocoder pass on top for the slowdown
factor doubles the damage. On top of that the vocoder ran at the *analysis*
rate (250 kHz), where the default 2048-point window is only 8 ms long and
122 Hz wide — the worst possible resolution trade for a fast FM whistle —
and it materialised an 8x-longer intermediate signal at 250 kHz, which for
a 10-minute recording is over a gigasample.

What this module does instead
-----------------------------
1. **Tape-speed pitch shift.** Reinterpreting the same samples at a lower
   sample rate is a mathematically exact pitch shift: zero artefacts, zero
   cost. It also stretches time by the same ratio — which is exactly what a
   time-expansion bat detector does, and what makes a 30 ms squeak audible
   as a 240 ms whistle. So the pitch shift is free and perfect, and the
   time expansion that comes with it is a feature, not a side effect.
2. **Vocoder only for the residual.** A caller who wants a slowdown that
   differs from the pitch ratio pays for one vocoder pass over the small
   *residual* factor, not over the whole 8x. When the two match (the
   default), no vocoder runs at all and the result is pristine.
3. **Phase-locked vocoder when one is needed.** The residual pass uses
   identity phase locking (Laroche & Dolson 1999): bins are locked to the
   propagated phase of the spectral peak that owns them, which is what
   removes the phasiness. It runs at the *audible* rate with a window
   chosen in the audible domain (~16 ms), and it streams over blocks so
   peak memory stays bounded on hour-long files.
4. **Spectral denoising.** An ultrasonic recording is mostly broadband
   noise; shifted down, that noise becomes a loud hiss that buries the
   calls. A soft spectral gate keyed on a per-bin noise floor (with the
   gain smoothed in time and frequency so it does not turn into musical
   noise) makes the calls pop out of a quiet background.
5. **Mastering.** Subsonic high-pass, equal-power fades to kill boundary
   clicks, level set from a high percentile rather than the absolute peak
   so a single cage-rattle click cannot bury the calls, and a soft-knee
   ceiling instead of hard clipping.

:func:`sonify_segment` is for auditioning a stretch of spectrogram — it
expands time. :func:`sonify_full_track` keeps real-world duration so the
track stays locked to a video's clock, which does require the vocoder.
"""

from __future__ import annotations

import numpy as np

_EPS = float(np.finfo(np.float32).eps)

#: Ultrasonic band kept before shifting. Everything outside is noise.
DEFAULT_BAND_HZ = (40_000.0, 120_000.0)

#: STFT window length aimed for, in seconds, measured at the *audible*
#: (post-tape-speed) rate. ~16 ms resolves a shifted FM sweep to a couple
#: of bins while still being long enough for a clean phase estimate.
_WINDOW_S = 0.016


# ───────────────────────────────────────────────────────────────────────────
# Internals
# ───────────────────────────────────────────────────────────────────────────

def _stft_params(fs: float) -> tuple[int, int]:
    """Pick (n_fft, hop) for a signal at *fs*. hop divides n_fft exactly,
    which the block overlap-add below relies on."""
    target = max(1, int(_WINDOW_S * fs))
    n_fft = 1 << (target - 1).bit_length()
    n_fft = int(np.clip(n_fft, 256, 2048))
    return n_fft, n_fft // 4


def _noise_gain(
    mag: np.ndarray,
    percentile: float = 20.0,
    oversubtract: float = 1.5,
    floor_db: float = -18.0,
) -> np.ndarray:
    """Soft spectral-subtraction gain for a magnitude spectrogram.

    The noise floor is a low percentile over time per frequency bin —
    robust here because calls are sparse, so most frames in any bin are
    noise. The gain is floored (rather than gated to zero) and smoothed
    across neighbouring bins and frames, which is what keeps the residual
    sounding like quiet hiss instead of the bubbling "musical noise" a
    hard gate produces.
    """
    from scipy.ndimage import uniform_filter

    noise = np.percentile(mag, percentile, axis=1, keepdims=True)
    gain = (mag - oversubtract * noise) / (mag + _EPS)
    np.clip(gain, 10.0 ** (floor_db / 20.0), 1.0, out=gain)
    return uniform_filter(gain, size=(3, 5), mode="nearest")


def _lock_phase(propagated: np.ndarray, analysis: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Identity phase locking (Laroche & Dolson).

    A vocoder that propagates every bin's phase independently lets the
    bins of one partial drift apart, which is audible as phasiness. Here
    each bin is assigned to the spectral peak whose region of influence it
    falls in, and takes that peak's *propagated* phase plus its own
    original offset from the peak — so the partial's waveform stays
    internally coherent.
    """
    if mag.size < 5:
        return propagated

    is_peak = np.zeros(mag.size, dtype=bool)
    c = mag[2:-2]
    is_peak[2:-2] = (
        (c > mag[1:-3]) & (c > mag[3:-1]) & (c > mag[:-4]) & (c > mag[4:])
    )
    peaks = np.flatnonzero(is_peak)
    if peaks.size == 0:
        return propagated

    # Region of influence: each bin belongs to the nearer of the two peaks
    # bracketing it.
    edges = (peaks[:-1] + peaks[1:] + 1) // 2
    bounds = np.concatenate(([0], edges, [mag.size]))
    owner = np.repeat(peaks, np.diff(bounds))
    return propagated[owner] + (analysis - analysis[owner])


def _render(
    x: np.ndarray,
    fs: float,
    factor: float,
    denoise: bool,
    block_frames: int = 4096,
) -> np.ndarray:
    """Denoise *x* and time-scale it by *factor* (>1 longer, <1 shorter).

    A single streaming STFT pass does both, so the signal is analysed once.
    ``factor == 1`` leaves every phase untouched (gain-only denoising), which
    is what makes the default sonification path bit-for-bit artefact-free.

    Memory is bounded by *block_frames*: only that many synthesis frames'
    worth of spectra exist at a time, with the vocoder's phase accumulator
    carried across block boundaries so there is no seam.
    """
    import librosa
    from scipy.signal import get_window

    if factor == 1.0 and not denoise:
        return x

    n_fft, hop = _stft_params(fs)
    m = n_fft // hop
    pad = n_fft // 2

    xp = np.pad(x.astype(np.float32), (pad, pad))
    n_analysis = 1 + (len(xp) - n_fft) // hop
    if n_analysis < 2:
        return x

    rate = 1.0 / factor  # analysis frames consumed per synthesis frame
    n_out = int(np.floor((n_analysis - 1) / rate)) + 1
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)

    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    omega = 2.0 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft

    acc = np.zeros((n_out + m) * hop, dtype=np.float32)
    phase_acc: np.ndarray | None = None

    k = 0
    while k < n_out:
        kb = min(block_frames, n_out - k)
        a0 = int(np.floor(k * rate))
        a1 = min(n_analysis, int(np.floor((k + kb - 1) * rate)) + 2)
        seg = xp[a0 * hop: (a1 - 1) * hop + n_fft]

        D = librosa.stft(seg, n_fft=n_fft, hop_length=hop, window="hann", center=False)
        if denoise:
            mag = np.abs(D)
            D = D * _noise_gain(mag)

        if factor == 1.0:
            spec = D[:, :kb]
        else:
            mag = np.abs(D)
            phase = np.angle(D)
            if phase_acc is None:
                phase_acc = phase[:, 0].copy()
            spec = np.empty((D.shape[0], kb), dtype=np.complex64)
            last = D.shape[1] - 1
            for i in range(kb):
                step = (k + i) * rate
                t0 = min(int(step) - a0, last)
                t1 = min(t0 + 1, last)
                frac = step - np.floor(step)
                col = (1.0 - frac) * mag[:, t0] + frac * mag[:, t1]

                spec[:, i] = col * np.exp(1j * _lock_phase(phase_acc, phase[:, t0], col))

                dphi = phase[:, t1] - phase[:, t0] - omega
                dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
                phase_acc += omega + dphi

        # Overlap-add. Frame j covers output hop-blocks (k+j) .. (k+j+m-1),
        # so chunk i of every frame lands in row i+j — a shifted row add
        # rather than a per-frame Python loop.
        frames = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32) * win[:, None]
        chunks = frames.reshape(m, hop, kb)
        block = np.zeros((kb + m, hop), dtype=np.float32)
        for i in range(m):
            block[i:i + kb] += chunks[i].T
        acc[k * hop: (k + kb + m) * hop] += block.ravel()

        k += kb

    # Normalise by the summed squared window (Griffin-Lim / weighted OLA).
    w2 = (win ** 2).reshape(m, hop)
    wsum = np.zeros((n_out + m, hop), dtype=np.float32)
    for i in range(m):
        wsum[i:i + n_out] += w2[i]
    wsum = wsum.ravel()
    y = acc[:wsum.size] / np.maximum(wsum, 1e-3 * float(wsum.max()))

    n_want = int(round(len(x) * factor))
    y = y[pad:pad + n_want]
    if y.size < n_want:
        y = np.pad(y, (0, n_want - y.size))
    return y.astype(np.float32)


def _resample(y: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """High-quality sample-rate conversion that accepts fractional rates
    (the tape-speed rate is ``fs * 2**(semitones/12)``, rarely an integer)."""
    if y.size == 0 or orig_fs == target_fs:
        return y.astype(np.float32)
    import soxr

    return soxr.resample(y.astype(np.float32), orig_fs, target_fs, quality="VHQ")


def _finalize(y: np.ndarray, fs: int, fade_ms: float = 10.0) -> np.ndarray:
    """Subsonic high-pass, equal-power fades, percentile level, soft ceiling."""
    if y.size == 0:
        return y.astype(np.float32)
    y = np.asarray(y, dtype=np.float32).copy()

    if y.size > 64:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(2, 120.0 / (fs / 2.0), btype="highpass", output="sos")
        y = sosfiltfilt(sos, y).astype(np.float32)

    n = min(int(fade_ms * fs / 1000.0), y.size // 2)
    if n > 1:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)) ** 2
        y[:n] *= ramp
        y[-n:] *= ramp[::-1]

    # Level from the 99.9th percentile, not the peak: a single broadband
    # click (cage, scratching) would otherwise set the gain and leave the
    # calls themselves near-inaudible.
    a = np.abs(y)
    ref = float(np.percentile(a, 99.9)) if a.size > 1000 else float(a.max())
    if ref <= 0.0:
        ref = float(a.max())
    if ref > 0.0:
        y *= 0.89 / ref

    # Soft knee above 0.9 instead of hard clipping, so the rare overshoot
    # rounds off rather than buzzing.
    a = np.abs(y)
    over = a > 0.9
    if over.any():
        y[over] = np.sign(y[over]) * (0.9 + 0.1 * np.tanh((a[over] - 0.9) / 0.1))

    return y.astype(np.float32)


def _band_limit(x: np.ndarray, fs: int, band: tuple[float, float]) -> np.ndarray:
    from squeak_peek.audio.filters import bandpass_filter_filtfilt

    f_lo, f_hi = band
    f_hi = min(f_hi, fs / 2.0 - 1.0)
    if f_lo >= f_hi:
        return x.astype(np.float32)
    return bandpass_filter_filtfilt(x.astype(np.float32), fs, f_lo, f_hi, order=12).astype(
        np.float32
    )


# ───────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────

def natural_slowdown(semitones: float) -> float:
    """The time-expansion factor that a pure tape-speed shift of *semitones*
    gives for free — and therefore the slowdown at which sonification needs
    no phase vocoder at all and is mathematically exact.

    ``-36`` semitones (three octaves) → ``8.0``.
    """
    return 2.0 ** (-semitones / 12.0)


def sonify_segment(
    audio: np.ndarray,
    start_time: float,
    end_time: float,
    fs: int,
    semitones: float,
    slowdown: float | None = None,
    target_fs: int = 44_100,
    denoise: bool = True,
    band: tuple[float, float] = DEFAULT_BAND_HZ,
) -> tuple[np.ndarray, int]:
    """
    Bring a segment of ultrasonic recording into the audible range.

    The returned array is real audio at *target_fs*: playing it back at that
    rate gives a ``(end_time - start_time) * slowdown`` second clip pitched
    ``semitones`` below the original.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples (1-D float array).
    start_time, end_time : float
        Segment bounds in seconds.
    fs : int
        Sample rate of *audio* in Hz.
    semitones : float
        Pitch shift (negative = lower). ``-36`` is three octaves.
    slowdown : float | None
        Time expansion. ``None`` (recommended) uses
        :func:`natural_slowdown`, i.e. exactly the pitch ratio, which is a
        pure tape-speed transform: no phase vocoder runs and there are no
        stretching artefacts at all. Any other value costs one vocoder pass
        over the residual factor ``slowdown / natural_slowdown(semitones)``.
    target_fs : int
        Sample rate of the returned audio.
    denoise : bool
        Apply the spectral gate. Ultrasonic recordings are mostly broadband
        noise; leaving it on is what makes calls stand out of a quiet
        background rather than a wall of hiss.
    band : (float, float)
        Ultrasonic band kept before shifting, in Hz.

    Returns
    -------
    (signal, target_fs) : tuple[np.ndarray, int]
    """
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )
    if slowdown is not None and slowdown <= 0:
        raise ValueError(f"slowdown must be positive, got {slowdown}")

    start_idx = max(0, round(start_time * fs))
    end_idx = min(len(audio), round(end_time * fs))
    x = np.asarray(audio[start_idx:end_idx], dtype=np.float32)

    # A zero-phase filtfilt needs more samples than its padding length, and
    # the STFT needs at least one full analysis window. Fail clearly here
    # rather than letting scipy raise something opaque further down.
    min_samples = 2048
    if len(x) < min_samples:
        raise ValueError(
            f"Segment too short to sonify: {len(x)} samples "
            f"(need at least {min_samples} for filtering and STFT analysis). "
            "Choose a longer time range."
        )

    x = _band_limit(x, fs, band)

    ratio = 2.0 ** (semitones / 12.0)   # tape-speed pitch ratio
    tape_fs = fs * ratio                # rate at which those samples now "are"
    natural = 1.0 / ratio               # time expansion the tape shift gives free

    if slowdown is None:
        slowdown = natural
    factor = slowdown / natural
    if abs(factor - 1.0) < 0.02:        # close enough — stay on the pristine path
        factor = 1.0

    y = _render(x, tape_fs, factor, denoise)
    y = _resample(y, tape_fs, target_fs)
    return _finalize(y, target_fs), target_fs


def sonify_full_track(
    audio: np.ndarray,
    fs: int,
    semitones: float,
    target_fs: int = 48_000,
    denoise: bool = True,
    band: tuple[float, float] = DEFAULT_BAND_HZ,
) -> tuple[np.ndarray, int]:
    """
    Pitch-shift an entire recording into the audible range while keeping its
    real-world duration, so it stays locked to a video's clock.

    Duration preservation is the one case a vocoder cannot be avoided: the
    tape-speed shift stretches time by the pitch ratio, so that stretch has
    to be undone. It is undone *at the shifted rate* on the already-shifted
    signal, with an audible-domain window and identity phase locking —
    roughly an order of magnitude cheaper in time and memory than stretching
    at the 250 kHz analysis rate, and markedly cleaner.

    ``len(sonified) / target_fs == len(audio) / fs``.

    See :func:`sonify_segment` for the shared parameters.
    """
    x = _band_limit(np.asarray(audio), fs, band)

    ratio = 2.0 ** (semitones / 12.0)
    tape_fs = fs * ratio

    y = _render(x, tape_fs, factor=ratio, denoise=denoise)
    y = _resample(y, tape_fs, target_fs)
    return _finalize(y, target_fs), target_fs
