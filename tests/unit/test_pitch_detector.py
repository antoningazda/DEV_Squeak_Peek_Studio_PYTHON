"""
Unit tests for the shared pitch tracker and the PITCH detector.

The tracker (squeak_peek.features.pitch) is what the Visualization tab draws
its orange trace with; the detector turns the same runs into labels, so the
two can never disagree about where a call is.
"""

from __future__ import annotations

import numpy as np

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.detectors.pitch import PitchDetector, PitchParams
from squeak_peek.features.pitch import track_pitch, viterbi_path

FS = 250_000


def _tone(start_s: float, duration_s: float, f_hz: float, total_s: float,
          amplitude: float = 1.0, f_end_hz: float | None = None) -> np.ndarray:
    """A sweep (or steady tone) inside an otherwise silent recording."""
    signal = np.zeros(int(total_s * FS))
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    f1 = f_hz if f_end_hz is None else f_end_hz
    phase = 2 * np.pi * (f_hz * t + (f1 - f_hz) * t**2 / (2 * duration_s))
    signal[int(start_s * FS) : int(start_s * FS) + n] = amplitude * np.sin(phase)
    return signal


def _detect(signal: np.ndarray, **params):
    return PitchDetector(PitchParams(**params)).detect(signal, FS)


class TestViterbiPath:
    def test_prefers_a_steady_band_over_a_louder_jumpy_one(self):
        # Two bins: bin 0 is steady, bin 1 is slightly louder on one frame.
        emission = np.array([[5.0, 5.0, 5.0], [0.0, 6.0, 0.0]])
        # Changing bin costs 10; staying is free.
        transition = np.array([[0.0, -10.0], [-10.0, 0.0]])
        assert list(viterbi_path(emission, transition)) == [0, 0, 0]

    def test_follows_a_moving_peak_when_jumping_is_cheap(self):
        emission = np.array([[9.0, 0.0], [0.0, 9.0]])
        transition = np.zeros((2, 2))
        assert list(viterbi_path(emission, transition)) == [0, 1]


class TestTrackPitch:
    def test_finds_a_tone_and_its_frequency(self):
        signal = _tone(0.05, 0.04, 70_000, total_s=0.2)
        f_hz, t_rel, Sxx = compute_stft(signal, FS, 1024, 0.5)
        f_band, Sxx_band = band_restrict(f_hz, Sxx, 40_000, 120_000)
        runs = track_pitch(f_band / 1_000.0, Sxx_band)

        assert len(runs) == 1
        run = runs[0]
        assert run.n_frames == len(run.freq_khz)
        assert abs(np.median(run.freq_khz) - 70.0) < 2.0
        assert 0.04 < t_rel[run.start] < 0.06
        assert 0.08 < t_rel[run.end] < 0.10

    def test_silence_yields_nothing(self):
        f_hz, _t, Sxx = compute_stft(np.zeros(int(0.1 * FS)), FS, 1024, 0.5)
        f_band, Sxx_band = band_restrict(f_hz, Sxx, 40_000, 120_000)
        assert track_pitch(f_band / 1_000.0, Sxx_band) == []

    def test_too_short_to_track_yields_nothing(self):
        f_khz = np.linspace(40.0, 120.0, 50)
        assert track_pitch(f_khz, np.zeros((50, 2))) == []

    def test_prominence_controls_sensitivity(self):
        signal = _tone(0.05, 0.04, 70_000, total_s=0.2)
        f_hz, _t, Sxx = compute_stft(signal, FS, 1024, 0.5)
        f_band, Sxx_band = band_restrict(f_hz, Sxx, 40_000, 120_000)
        assert track_pitch(f_band / 1_000.0, Sxx_band, prominence_db=200.0) == []


class TestPitchDetector:
    def test_registers_as_a_detector(self):
        assert AbstractDetector.get("PITCH") is PitchDetector
        assert "PITCH" in [d.id for d in AbstractDetector.all()]

    def test_detects_a_tone_with_sane_bounds(self):
        labels = _detect(_tone(0.05, 0.04, 70_000, total_s=0.3))
        assert len(labels) == 1
        lbl = labels[0]
        assert abs(lbl.start_time - 0.05) < 0.01
        assert abs(lbl.end_time - 0.09) < 0.01
        assert lbl.label == "d"
        assert lbl.start_index == round(lbl.start_time * FS)
        assert 60_000 < lbl.start_frequency < 80_000

    def test_reports_sweep_endpoints(self):
        labels = _detect(_tone(0.05, 0.05, 50_000, total_s=0.3, f_end_hz=90_000))
        assert len(labels) == 1
        assert labels[0].start_frequency < labels[0].end_frequency

    def test_silence_yields_nothing(self):
        assert _detect(np.zeros(int(0.3 * FS))) == []

    def test_empty_signal_yields_nothing(self):
        assert _detect(np.array([])) == []

    def test_min_duration_discards_short_events(self):
        signal = _tone(0.05, 0.04, 70_000, total_s=0.3)
        assert _detect(signal) != []
        assert _detect(signal, minDurationMs=500.0) == []

    def test_call_band_filter_drops_out_of_band_calls(self):
        signal = (_tone(0.05, 0.04, 45_000, total_s=0.4)
                  + _tone(0.20, 0.04, 70_000, total_s=0.4))
        both = _detect(signal)
        assert len(both) == 2

        kept = _detect(signal, minCallFreqKhz=50.0, maxCallFreqKhz=100.0)
        assert len(kept) == 1
        assert abs(kept[0].start_time - 0.20) < 0.01

        assert _detect(signal, minCallFreqKhz=100.0, maxCallFreqKhz=250.0) == []

    def test_call_band_filter_does_not_disturb_tracking(self):
        """Filtering happens after the contour is found over the full search
        band, so the surviving event is bit-identical to the unfiltered one."""
        signal = (_tone(0.05, 0.04, 45_000, total_s=0.4)
                  + _tone(0.20, 0.04, 70_000, total_s=0.4))
        unfiltered = _detect(signal)[1]
        filtered = _detect(signal, minCallFreqKhz=50.0, maxCallFreqKhz=100.0)[0]
        assert filtered == unfiltered

    def test_a_call_crossing_a_block_seam_is_reported_once(self):
        """Blocks overlap, so a call on a seam is seen twice; the merge must
        collapse it back into one event with the full span."""
        signal = _tone(0.98, 0.04, 70_000, total_s=3.0)
        whole = _detect(signal)
        split = _detect(signal, blockSeconds=1.0)

        assert len(whole) == 1
        assert len(split) == 1
        assert abs(split[0].start_time - whole[0].start_time) < 0.01
        assert abs(split[0].end_time - whole[0].end_time) < 0.01

    def test_blocking_does_not_multiply_events(self):
        signal = sum(_tone(start, 0.03, 70_000, total_s=4.0)
                     for start in (0.5, 1.5, 2.5, 3.5))
        assert len(_detect(signal)) == 4
        assert len(_detect(signal, blockSeconds=1.0)) == 4
