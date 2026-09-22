"""
Align a behavior video's timeline to the ultrasonic WAV's timeline.

Typical setup: a behavior camera (with its own audible-range microphone)
and a separate ultrasonic microphone are started independently, and a
finger snap near the start of the recording is used as a shared sync
marker — audible as a broadband click in the camera's audio track, and
either hand-annotated as an "sk" label or itself detectable as a sharp
broadband transient in the ultrasonic recording.

``compute_sync_offset`` returns a single number: the offset to add to a
WAV-timeline time to get the corresponding video-timeline time (and vice
versa, by subtracting it).
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from squeak_peek.audio.filters import bandpass_filter_filtfilt
from squeak_peek.audio.io import load_wav

# Sync-click band in the ultrasonic recording: a finger snap is broadband,
# so it shows up even at the top of the USV analysis range. Using the top of
# the band (rather than the full 40-120 kHz USV band) keeps this detector
# from firing on ordinary USV calls, which concentrate lower in that range.
_WAV_SNAP_BAND = (100_000.0, 120_000.0)


@dataclass
class SyncResult:
    """Outcome of a sync alignment attempt."""

    offset: float          # seconds to ADD to a WAV time to get video time
    method: str             # human-readable description, shown in the UI
    wav_snap_time: float
    video_snap_time: float


def extract_audio_track(video_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Decode a video's audio stream to a mono float32 array via ffmpeg.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If ffmpeg fails, or the video has no usable audio stream.
    """
    import imageio_ffmpeg

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "audio.wav"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-vn",                 # no video
            "-ac", "1",             # mono
            "-acodec", "pcm_s16le",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0 or not wav_path.exists():
            raise RuntimeError(
                f"Could not extract an audio track from '{video_path.name}'. "
                "The video may have no audio stream, which is required for "
                "automatic sync detection.\n\nffmpeg output:\n"
                f"{proc.stderr.strip()[-800:]}"
            )

        samples, fs = load_wav(wav_path)

    min_samples = int(0.05 * fs)  # need at least ~50ms to look for a transient
    if len(samples) < min_samples:
        raise RuntimeError(
            f"'{video_path.name}' has no usable audio track "
            "(required for automatic sync detection)."
        )

    return samples, fs


def _energy_envelope(signal: np.ndarray, fs: int, win_ms: float = 5.0) -> np.ndarray:
    """RMS energy envelope over non-overlapping windows of ``win_ms``."""
    win = max(1, int(fs * win_ms / 1000.0))
    n_win = len(signal) // win
    if n_win == 0:
        return np.array([np.sqrt(np.mean(signal.astype(np.float64) ** 2))])
    trimmed = signal[: n_win * win].astype(np.float64)
    frames = trimmed.reshape(n_win, win)
    return np.sqrt(np.mean(frames**2, axis=1))


def find_first_transient(
    signal: np.ndarray,
    fs: int,
    band: tuple[float, float],
    k: float = 8.0,
    win_ms: float = 5.0,
) -> float:
    """
    Return the time (seconds) of the first sharp broadband transient
    (e.g. a finger snap) in *signal*, found by bandpass filtering then
    thresholding a short-window RMS energy envelope against its own
    noise floor (the median envelope value).

    Raises
    ------
    ValueError
        If no window's energy clears the threshold.
    """
    nyq = fs / 2.0
    f_low, f_high = band
    f_high = min(f_high, nyq * 0.99)
    if f_low >= f_high:
        raise ValueError(f"Invalid band {band} for sample rate {fs} Hz")

    filtered = bandpass_filter_filtfilt(signal, fs, f_low, f_high, order=4)
    envelope = _energy_envelope(filtered, fs, win_ms=win_ms)

    noise_floor = float(np.median(envelope)) + 1e-12
    threshold = noise_floor * k

    above = np.nonzero(envelope > threshold)[0]
    if len(above) == 0:
        raise ValueError(
            "No broadband transient (snap) found — try a louder/sharper "
            "snap, or annotate an 'sk' sync label instead."
        )

    win_samples = max(1, int(fs * win_ms / 1000.0))
    return float(above[0] * win_samples / fs)


def _wav_snap_time(state, k: float) -> tuple[float, str]:
    """Find the sync-click time in the loaded WAV: prefer an 'sk' label,
    else auto-detect a broadband transient near the top of the USV band."""
    for labels, source in (
        (state.reference_labels, "reference"),
        (state.detected_labels, "detected"),
    ):
        sk = sorted(
            (lbl for lbl in labels if lbl.label == "sk"),
            key=lambda lbl: lbl.start_time,
        )
        if sk:
            return sk[0].start_time, f"'sk' {source} label"

    if state.samples is None:
        raise RuntimeError("No WAV loaded — cannot compute sync offset.")

    t = find_first_transient(state.samples, state.fs, _WAV_SNAP_BAND, k=k)
    return t, "auto-detected snap in WAV"


def compute_sync_offset(state, video_samples: np.ndarray, video_fs: int) -> SyncResult:
    """
    Compute the offset between the loaded WAV's timeline and a video's
    timeline, using an 'sk' label when available, else auto-detected
    broadband transients on both sides.

    Band and threshold for the video-side snap detector come from
    ``state.settings.video`` (Settings tab -> Video).
    """
    vs = state.settings.video
    k = vs.snap_threshold_factor
    video_band = (vs.snap_band_min_hz, vs.snap_band_max_hz)

    wav_t, wav_method = _wav_snap_time(state, k=k)
    video_t = find_first_transient(video_samples, video_fs, video_band, k=k)

    offset = video_t - wav_t
    method = f"{wav_method} + auto-detected snap in video"
    return SyncResult(
        offset=offset, method=method, wav_snap_time=wav_t, video_snap_time=video_t
    )
