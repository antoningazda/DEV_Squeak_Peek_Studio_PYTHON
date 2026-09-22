"""
Export a video with its original frames but a new (sonified) soundtrack.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from squeak_peek.audio.io import save_wav


def _align_audio(samples: np.ndarray, fs: int, offset: float) -> np.ndarray:
    """
    Shift *samples* (timestamped on the WAV timeline) onto the video
    timeline, where ``video_time = wav_time + offset``.

    offset >= 0 : the WAV recording started *after* the video, so the
                   sonified audio must start ``offset`` seconds into the
                   video — silence is prepended.
    offset <  0 : the WAV recording started *before* the video, so the
                   leading ``-offset`` seconds (which predate the video)
                   are trimmed off.
    """
    if offset >= 0:
        pad = int(round(offset * fs))
        if pad == 0:
            return samples
        return np.concatenate([np.zeros(pad, dtype=samples.dtype), samples])
    else:
        trim = int(round(-offset * fs))
        return samples[trim:]


def export_synced_video(
    video_path: str | Path,
    sonified_samples: np.ndarray,
    sonified_fs: int,
    offset: float,
    out_path: str | Path,
) -> None:
    """
    Mux the original video's frames with a new sonified soundtrack,
    aligned by *offset* (video_time = wav_time + offset — see
    ``squeak_peek.video.sync.compute_sync_offset``).

    The video stream is copied without re-encoding (``-c:v copy``); only
    the audio track is replaced. Output is trimmed to the shorter of the
    two streams (``-shortest``).

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If ffmpeg fails to produce the output file.
    """
    import imageio_ffmpeg

    video_path = Path(video_path)
    out_path = Path(out_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    aligned = _align_audio(sonified_samples, sonified_fs, offset)
    if len(aligned) == 0:
        raise RuntimeError(
            "Nothing left to export: the sync offset trims away the entire "
            "sonified track. Check the sync alignment before exporting."
        )

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_wav = Path(tmp_dir) / "sonified.wav"
        save_wav(audio_wav, aligned, sonified_fs)

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(video_path),
            "-i", str(audio_wav),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(
            f"ffmpeg failed to export '{out_path.name}'.\n\nffmpeg output:\n"
            f"{proc.stderr.strip()[-800:]}"
        )


class VideoExportWorker(QThread):
    """Runs :func:`export_synced_video` off the UI thread."""

    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str | Path,
        sonified_samples: np.ndarray,
        sonified_fs: int,
        offset: float,
        out_path: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._sonified_samples = sonified_samples
        self._sonified_fs = sonified_fs
        self._offset = offset
        self._out_path = out_path

    def run(self) -> None:
        try:
            export_synced_video(
                self._video_path,
                self._sonified_samples,
                self._sonified_fs,
                self._offset,
                self._out_path,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.succeeded.emit()
