"""
Pitch tracking: where a call is actually playing, and at what frequency.

This is the algorithm behind the Visualization tab's orange pitch trace,
factored out of the GUI so the PITCH detector (squeak_peek.detectors.pitch)
and the spectrogram widget share one implementation — the detector's events
are exactly the stretches the trace is drawn over.

Each STFT bin is first referenced to its own median over the analysed range
(removing stationary noise lines) and each frame to its own median across
frequency; a frame counts as a call frame when its strongest bin stands
``prominence_db`` above that. Within each contiguous run of call frames the
contour is a Viterbi path that trades bin power against frequency jumps:
these calls often show several parallel bands of similar power, and a plain
per-frame argmax hops between them, drawing vertical zigzags.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: A frame is a call frame when its peak bin stands this far above the
#: background; runs shorter than MIN_RUN frames are noise.
PROMINENCE_DB = 12.0
MIN_RUN = 3
#: Viterbi transition cost per kHz of frequency change between frames, and
#: the largest change allowed in one frame (fast FM sweeps stay under this).
JUMP_PENALTY_DB_PER_KHZ = 2.0
MAX_JUMP_KHZ = 6.0
#: Bins above this are ignored: stray high-frequency noise above the typical
#: USV range otherwise wins and makes the contour jump wildly.
MAX_FREQ_KHZ = 100.0


@dataclass(frozen=True)
class PitchRun:
    """One contiguous stretch of call frames and its pitch contour.

    ``start`` and ``end`` are frame indices into the spectrogram, both
    inclusive. ``freq_khz`` holds one frequency per frame in that range.
    """

    start: int
    end: int
    freq_khz: np.ndarray

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


def viterbi_path(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Best bin index per frame for emission scores (F, T) under a
    transition score matrix (F, F) indexed [to, from]."""
    n_bins, n_frames = emission.shape
    rows = np.arange(n_bins)
    score = emission[:, 0].copy()
    back = np.zeros((n_bins, n_frames), dtype=np.int32)
    for t in range(1, n_frames):
        cand = transition + score[None, :]
        back[:, t] = np.argmax(cand, axis=1)
        score = cand[rows, back[:, t]] + emission[:, t]
    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def track_pitch(
    f_khz: np.ndarray,
    Sxx_db: np.ndarray,
    *,
    prominence_db: float = PROMINENCE_DB,
    min_run: int = MIN_RUN,
    jump_penalty_db_per_khz: float = JUMP_PENALTY_DB_PER_KHZ,
    max_jump_khz: float = MAX_JUMP_KHZ,
    max_freq_khz: float = MAX_FREQ_KHZ,
) -> list[PitchRun]:
    """Find the stretches where a call is playing, with their contours.

    Parameters
    ----------
    f_khz   : frequency of each spectrogram bin, in kHz, shape (F,)
    Sxx_db  : spectrogram power in dB, shape (F, T)

    Returns
    -------
    One PitchRun per run of at least ``min_run`` consecutive call frames,
    in time order. Empty when the input is too short or nothing stands out.
    """
    band = f_khz < max_freq_khz
    if Sxx_db.shape[1] < min_run or np.count_nonzero(band) < 2:
        return []

    f_band = f_khz[band]
    power = Sxx_db[band]
    power = power - np.median(power, axis=1, keepdims=True)
    power = power - np.median(power, axis=0, keepdims=True)
    is_signal = power.max(axis=0) >= prominence_db

    run_starts = np.where(is_signal & ~np.concatenate(([False], is_signal[:-1])))[0]
    run_ends = np.where(is_signal & ~np.concatenate((is_signal[1:], [False])))[0]

    df_khz = float(f_band[1] - f_band[0])
    jump = np.abs(np.subtract.outer(np.arange(len(f_band)), np.arange(len(f_band)))) * df_khz
    transition = np.where(jump <= max_jump_khz, -jump_penalty_db_per_khz * jump, -np.inf)

    runs: list[PitchRun] = []
    for start, end in zip(run_starts, run_ends):
        if end - start + 1 < min_run:
            continue
        path = viterbi_path(power[:, start : end + 1], transition)
        runs.append(PitchRun(int(start), int(end), f_band[path]))
    return runs
