"""
Audio I/O — load and save WAV files for Squeak Peek Studio.

MATLAB equivalents
------------------
    load_wav  ↔  [y, fs] = audioread(filepath)
    save_wav  ↔  audiowrite(filepath, y, fs)

Design notes
------------
* All audio is handled as float32 single-channel (mono) NumPy arrays.
* The application targets 250 kHz ultrasonic recordings; soundfile handles
  any sample rate without resampling.
* If a stereo file is loaded, only the first channel is returned (matching
  MATLAB's audioread default behaviour for multi-channel inputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass  # soundfile imported lazily below to allow import without it installed


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return a mono float32 array and its sample rate.

    Parameters
    ----------
    path:
        Absolute or relative path to a .wav file.

    Returns
    -------
    samples : np.ndarray, shape (N,), dtype float32
        Audio samples normalised to [-1, 1].
    fs : int
        Sample rate in Hz (e.g. 250_000 for ultrasonic recordings).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If soundfile cannot read the file (unsupported format, corrupted, etc.).

    Examples
    --------
    >>> samples, fs = load_wav("data/example/single/USV_Example_Short.wav")
    >>> samples.dtype
    dtype('float32')
    >>> fs
    250000
    """
    import soundfile as sf  # lazy import — not needed until function is called

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        data, fs = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not read audio file '{path}': {exc}") from exc

    # Take first channel only (mono)
    samples: np.ndarray = data[:, 0]
    return samples, int(fs)


def save_wav(path: str | Path, samples: np.ndarray, fs: int) -> None:
    """
    Write a mono float32 array to a WAV file.

    Parameters
    ----------
    path:
        Destination file path. Parent directories must exist.
    samples:
        1-D array of audio samples. Will be cast to float32 if needed.
    fs:
        Sample rate in Hz.

    Raises
    ------
    ValueError
        If *samples* is not 1-D.
    RuntimeError
        If soundfile fails to write the file.

    Examples
    --------
    >>> save_wav("output/detected.wav", samples, 250_000)
    """
    import soundfile as sf  # lazy import

    path = Path(path)
    if samples.ndim != 1:
        raise ValueError(
            f"samples must be a 1-D array; got shape {samples.shape}. "
            "Use samples[:, 0] to extract a single channel."
        )

    samples_f32 = samples.astype(np.float32)

    try:
        sf.write(str(path), samples_f32, fs, subtype="FLOAT")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not write audio file '{path}': {exc}") from exc


def audio_duration(samples: np.ndarray, fs: int) -> float:
    """Return duration of the audio in seconds."""
    return len(samples) / fs


def time_to_sample(time_s: float, fs: int) -> int:
    """Convert a time in seconds to the nearest sample index."""
    return int(round(time_s * fs))


def sample_to_time(sample_idx: int, fs: int) -> float:
    """Convert a sample index to time in seconds."""
    return sample_idx / fs
