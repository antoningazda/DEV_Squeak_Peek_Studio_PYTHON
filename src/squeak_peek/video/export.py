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
from squeak_peek.labels.model import Label
from squeak_peek.video.spectrogram_panel import SpectrogramPanelRenderer

# Scrolling spectrogram panel: total time span visible at once, and where
# the fixed "now" needle sits across its width (0.5 = centered).
_PANEL_WINDOW_SECONDS = 4.0
_PANEL_NEEDLE_FRACTION = 0.5
# Panel height as a fraction of the source video's height, clamped to a
# sane pixel range so it stays legible on both tiny and huge source videos.
_PANEL_HEIGHT_FRACTION = 0.4
_PANEL_MIN_HEIGHT = 160
_PANEL_MAX_HEIGHT = 480


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


def export_synced_video_with_spectrogram(
    video_path: str | Path,
    out_path: str | Path,
    samples: np.ndarray,
    fs: int,
    sonified_samples: np.ndarray,
    sonified_fs: int,
    offset: float,
    detected_labels: list[Label],
    reference_labels: list[Label],
    fmin_hz: float,
    fmax_hz: float,
    nperseg: int,
    noverlap: int,
    colormap_name: str,
    detected_color_name: str | None,
    reference_color_name: str | None,
) -> None:
    """
    Mux the original video's frames — with a scrolling, labeled spectrogram
    panel composited underneath, driven by a fixed "now" position needle —
    against a new sonified soundtrack, aligned by *offset* (see
    ``squeak_peek.video.sync.compute_sync_offset``).

    Unlike :func:`export_synced_video`, this re-encodes the video (the
    panel has to be burned into every frame), so it's slower but produces
    a self-contained clip that plays the ultrasonic content back visually
    in real time alongside the sonified audio.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist.
    RuntimeError
        If ffmpeg fails to produce the output file, or the aligned audio
        or spectrogram would be empty.
    """
    import imageio_ffmpeg

    video_path = Path(video_path)
    out_path = Path(out_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    aligned_audio = _align_audio(sonified_samples, sonified_fs, offset)
    if len(aligned_audio) == 0:
        raise RuntimeError(
            "Nothing left to export: the sync offset trims away the entire "
            "sonified track. Check the sync alignment before exporting."
        )

    reader = imageio_ffmpeg.read_frames(str(video_path))
    meta = next(reader)
    fps = float(meta["fps"]) or 30.0
    width, height = meta["size"]
    video_duration = float(meta["duration"]) or (len(aligned_audio) / sonified_fs)

    target_duration = min(video_duration, len(aligned_audio) / sonified_fs)
    frame_count = max(1, int(round(target_duration * fps)))

    panel_height = int(round(height * _PANEL_HEIGHT_FRACTION))
    panel_height = min(max(panel_height, _PANEL_MIN_HEIGHT), _PANEL_MAX_HEIGHT)
    if panel_height % 2:
        panel_height += 1

    renderer = SpectrogramPanelRenderer(
        samples, fs, detected_labels, reference_labels,
        fmin_hz, fmax_hz, nperseg, noverlap, colormap_name,
        detected_color_name, reference_color_name,
        panel_width=width, panel_height=panel_height,
        window_seconds=_PANEL_WINDOW_SECONDS, needle_fraction=_PANEL_NEEDLE_FRACTION,
    )

    out_width, out_height = width, height + panel_height
    frame_bytes_len = width * height * 3

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_wav = Path(tmp_dir) / "sonified.wav"
        target_len = int(round(frame_count / fps * sonified_fs))
        trimmed = aligned_audio[:target_len]
        if len(trimmed) < target_len:
            trimmed = np.concatenate(
                [trimmed, np.zeros(target_len - len(trimmed), dtype=trimmed.dtype)]
            )
        save_wav(audio_wav, trimmed, sonified_fs)

        writer = imageio_ffmpeg.write_frames(
            str(out_path),
            (out_width, out_height),
            fps=fps,
            quality=8,
            macro_block_size=2,
            audio_path=str(audio_wav),
            audio_codec="aac",
            output_params=["-shortest"],
        )
        writer.send(None)
        try:
            for i in range(frame_count):
                try:
                    frame_bytes = next(reader)
                except StopIteration:
                    break
                if len(frame_bytes) != frame_bytes_len:
                    break
                video_frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                    height, width, 3
                )
                audio_time = i / fps - offset
                panel_frame = renderer.frame_at(audio_time)
                composite = np.ascontiguousarray(np.vstack([video_frame, panel_frame]))
                writer.send(composite)
        finally:
            writer.close()
            reader.close()

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg failed to export '{out_path.name}'.")


class VideoExportWorker(QThread):
    """Runs :func:`export_synced_video_with_spectrogram` off the UI thread."""

    succeeded = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        video_path: str | Path,
        samples: np.ndarray,
        fs: int,
        sonified_samples: np.ndarray,
        sonified_fs: int,
        offset: float,
        detected_labels: list[Label],
        reference_labels: list[Label],
        fmin_hz: float,
        fmax_hz: float,
        nperseg: int,
        noverlap: int,
        colormap_name: str,
        detected_color_name: str | None,
        reference_color_name: str | None,
        out_path: str | Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._samples = samples
        self._fs = fs
        self._sonified_samples = sonified_samples
        self._sonified_fs = sonified_fs
        self._offset = offset
        self._detected_labels = detected_labels
        self._reference_labels = reference_labels
        self._fmin_hz = fmin_hz
        self._fmax_hz = fmax_hz
        self._nperseg = nperseg
        self._noverlap = noverlap
        self._colormap_name = colormap_name
        self._detected_color_name = detected_color_name
        self._reference_color_name = reference_color_name
        self._out_path = out_path

    def run(self) -> None:
        try:
            export_synced_video_with_spectrogram(
                self._video_path,
                self._out_path,
                self._samples,
                self._fs,
                self._sonified_samples,
                self._sonified_fs,
                self._offset,
                self._detected_labels,
                self._reference_labels,
                self._fmin_hz,
                self._fmax_hz,
                self._nperseg,
                self._noverlap,
                self._colormap_name,
                self._detected_color_name,
                self._reference_color_name,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.succeeded.emit()
