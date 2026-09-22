"""
Pitch-shift and time-stretch for audio sonification.

Uses librosa's phase-vocoder primitives (effects.pitch_shift,
effects.time_stretch) plus a true resample to a target playback rate —
the same approach sonify_full_track already used. An earlier version of
sonify_segment hand-rolled its own two-pass STFT/overlap-add vocoder
without ever resampling the result; mechanically, resampling an STFT
frame timeline and re-synthesizing via overlap-add at a *fixed* hop
length changes duration while preserving pitch (a time-stretch), no
matter which knob (semitones or slowdown) supplied the resample ratio.
Both of that version's labeled steps were therefore actually time-stretches,
and the *only* real pitch shift came from a separate, hardcoded, unlabeled
trick in the GUI layer: declaring a playback rate (44100 Hz) far lower
than the recording's true rate. That trick's shift amount is fixed by
whatever the recording's sample rate happens to be (~-30 semitones for a
250 kHz recording played at 44100 Hz) — the "Sonification semitones"
setting had no effect on pitch at all, only on duration/speed.
"""

from __future__ import annotations

import numpy as np


def sonify_segment(
    audio: np.ndarray,
    start_time: float,
    end_time: float,
    fs: int,
    semitones: float,
    slowdown: float,
    target_fs: int = 44_100,
) -> tuple[np.ndarray, int]:
    """
    Pitch-shift and time-stretch a segment of audio for audible playback.

    Bandpass-filters the segment (40-120 kHz), pitch-shifts it by
    `semitones` (duration-preserving), time-stretches it by `slowdown`
    (pitch-preserving — independent of the pitch shift), then resamples
    the result from the analysis rate `fs` down to `target_fs`. That
    resample is a real sample-rate conversion (not a playback-rate
    trick), so the returned array's real-world duration and pitch match
    exactly what `semitones`/`slowdown` specify: playing it back at
    `target_fs` reproduces a ``(end_time - start_time) * slowdown``
    second clip, shifted down by `semitones` semitones from the original.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples (1-D float array)
    start_time : float
        Segment start time in seconds
    end_time : float
        Segment end time in seconds
    fs : int
        Sample rate in Hz of `audio`
    semitones : float
        Pitch shift in semitones (negative = lower pitch)
    slowdown : float
        Time-stretch factor (> 1 = slower/longer; must be > 0)
    target_fs : int
        Sample rate of the returned audio (default 44100 Hz)

    Returns
    -------
    (signal_out, target_fs) : tuple[np.ndarray, int]
        Sonified audio, normalized to max absolute value <= 1.0, and the
        sample rate it must be played back at.
    """
    if end_time <= start_time:
        raise ValueError(
            f"end_time ({end_time}) must be greater than start_time ({start_time})"
        )
    if slowdown <= 0:
        raise ValueError(f"slowdown must be positive, got {slowdown}")

    start_idx = max(0, round(start_time * fs))
    end_idx = min(len(audio), round(end_time * fs))
    x = audio[start_idx:end_idx].astype(np.float32)

    # A zero-phase filtfilt-style filter needs more samples than its padding
    # length, and librosa's STFT-based pitch_shift/time_stretch need at
    # least one full analysis window. Fail with a clear message rather than
    # letting scipy/librosa raise an opaque error further down.
    min_samples = max(2048, 3 * (12 + 1))  # matches scipy sosfiltfilt's default padlen for a 12th-order SOS
    if len(x) < min_samples:
        raise ValueError(
            f"Segment too short to sonify: {len(x)} samples "
            f"(need at least {min_samples} for filtering and STFT analysis). "
            "Choose a longer time range."
        )

    import librosa

    from squeak_peek.audio.filters import bandpass_filter_filtfilt

    x = bandpass_filter_filtfilt(x, fs, 40_000, min(120_000, fs / 2 - 1), order=12).astype(np.float32)

    shifted = librosa.effects.pitch_shift(y=x, sr=fs, n_steps=semitones)
    stretched = librosa.effects.time_stretch(y=shifted, rate=1.0 / slowdown)
    resampled = librosa.resample(stretched, orig_sr=fs, target_sr=target_fs)

    peak = np.max(np.abs(resampled))
    if peak > 0:
        resampled = resampled / (peak + np.finfo(float).eps)

    return resampled.astype(np.float32), target_fs


def sonify_full_track(
    audio: np.ndarray,
    fs: int,
    semitones: float,
    target_fs: int = 48_000,
) -> tuple[np.ndarray, int]:
    """
    Pitch-shift an entire recording down into the audible range, preserving
    its real-world duration — unlike :func:`sonify_segment`, this applies
    no time-stretch, so the output stays locked to a real-time clock (e.g.
    a video being played back alongside it).

    Uses librosa's duration-preserving pitch shift (phase vocoder + resample
    back to the original length) rather than the manual two-pass phase
    vocoder in :func:`sonify_segment`, since that one deliberately also
    time-stretches. The result is then resampled from the analysis rate
    *fs* (e.g. 250 kHz) down to *target_fs* — a true resample, so playing
    it back at *target_fs* reproduces the same real-world duration as
    *audio* at *fs*.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio samples (1-D float array), full recording.
    fs : int
        Sample rate in Hz of *audio*.
    semitones : float
        Pitch shift in semitones (e.g. -35 to bring ultrasonic calls
        down into human hearing range).
    target_fs : int
        Sample rate of the returned track (default 48 kHz).

    Returns
    -------
    (sonified, target_fs) : tuple[np.ndarray, int]
        Pitch-shifted audio at *target_fs*, with
        ``len(sonified) / target_fs == len(audio) / fs`` (real duration
        preserved).
    """
    import librosa

    from squeak_peek.audio.filters import bandpass_filter_filtfilt

    x = bandpass_filter_filtfilt(audio.astype(np.float32), fs, 40_000, min(120_000, fs / 2 - 1))
    shifted = librosa.effects.pitch_shift(y=x, sr=fs, n_steps=semitones)
    resampled = librosa.resample(shifted, orig_sr=fs, target_sr=target_fs)

    peak = np.max(np.abs(resampled))
    if peak > 0:
        resampled = resampled / (peak + np.finfo(float).eps)

    return resampled.astype(np.float32), target_fs
