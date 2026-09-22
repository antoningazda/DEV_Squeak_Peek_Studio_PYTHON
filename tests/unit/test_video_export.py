"""Unit tests for squeak_peek.video.export: audio/video alignment and muxing."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from squeak_peek.labels.model import Label
from squeak_peek.video.export import (
    _align_audio,
    export_synced_video,
    export_synced_video_with_spectrogram,
)


class TestAlignAudio:
    def test_positive_offset_prepends_silence(self):
        samples = np.arange(10, dtype=np.float32)
        out = _align_audio(samples, fs=10, offset=0.3)
        assert len(out) == 13
        np.testing.assert_array_equal(out[:3], 0.0)
        np.testing.assert_array_equal(out[3:], samples)

    def test_negative_offset_trims_start(self):
        samples = np.arange(10, dtype=np.float32)
        out = _align_audio(samples, fs=10, offset=-0.3)
        np.testing.assert_array_equal(out, samples[3:])

    def test_zero_offset_passthrough(self):
        samples = np.arange(10, dtype=np.float32)
        out = _align_audio(samples, fs=10, offset=0.0)
        np.testing.assert_array_equal(out, samples)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory):
    """A tiny 2s test video (color bars + a sine-tone audio track), made
    with the bundled ffmpeg binary — no network/system ffmpeg required."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    tmp_dir = tmp_path_factory.mktemp("video_export")
    video_path = tmp_dir / "src.mp4"
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=15",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip(f"could not generate synthetic test video: {proc.stderr[-500:]}")
    return video_path


class TestExportSyncedVideo:
    def test_export_produces_playable_file(self, synthetic_video, tmp_path):
        sfs = 48000
        sonified = (0.5 * np.sin(2 * np.pi * 300 * np.arange(int(2 * sfs)) / sfs)).astype(
            np.float32
        )
        out_path = tmp_path / "out.mp4"

        export_synced_video(synthetic_video, sonified, sfs, offset=0.2, out_path=out_path)

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_missing_video_raises(self, tmp_path):
        sonified = np.zeros(1000, dtype=np.float32)
        with pytest.raises(FileNotFoundError):
            export_synced_video(
                tmp_path / "nonexistent.mp4", sonified, 48000, 0.0, tmp_path / "out.mp4"
            )


class TestExportSyncedVideoWithSpectrogram:
    def test_export_produces_playable_file(self, synthetic_video, tmp_path):
        fs = 250_000
        duration = 2.0
        n = int(duration * fs)
        t = np.arange(n) / fs
        samples = (0.01 * np.random.randn(n)).astype(np.float32)
        call_mask = (t >= 0.5) & (t < 0.55)
        samples[call_mask] += 0.5 * np.sin(2 * np.pi * 60_000 * t[call_mask]).astype(np.float32)

        sfs = 48000
        sonified = (
            0.5 * np.sin(2 * np.pi * 300 * np.arange(int(duration * sfs)) / sfs)
        ).astype(np.float32)

        detected = [Label(start_time=0.5, end_time=0.55, label="d")]
        reference = [
            Label(
                start_time=1.0, end_time=1.05, label="5",
                start_frequency=55_000, end_frequency=65_000,
            )
        ]
        out_path = tmp_path / "out_spec.mp4"

        export_synced_video_with_spectrogram(
            synthetic_video, out_path,
            samples, fs,
            sonified, sfs, 0.0,
            detected, reference,
            fmin_hz=40_000, fmax_hz=120_000, nperseg=1024, noverlap=512,
            colormap_name="parula",
            detected_color_name="cyan", reference_color_name="magenta",
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_missing_video_raises(self, tmp_path):
        sonified = np.zeros(1000, dtype=np.float32)
        samples = np.zeros(1000, dtype=np.float32)
        with pytest.raises(FileNotFoundError):
            export_synced_video_with_spectrogram(
                tmp_path / "nonexistent.mp4", tmp_path / "out.mp4",
                samples, 250_000,
                sonified, 48000, 0.0,
                [], [],
                fmin_hz=40_000, fmax_hz=120_000, nperseg=64, noverlap=32,
                colormap_name="parula",
                detected_color_name=None, reference_color_name=None,
            )
