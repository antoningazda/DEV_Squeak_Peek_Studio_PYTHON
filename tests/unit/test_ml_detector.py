"""
Unit tests for squeak_peek.detectors.ml.MLDetector (WP6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squeak_peek.config import MLParams
from squeak_peek.detectors.ml import MLDetector
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.ml.train import save_model, train_model


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory, example_wav_path: Path, example_ref_labels_path: Path) -> Path:
    """Train once on the real example WAV/labels; reused by every test in this module."""
    model_dict = train_model([(example_wav_path, example_ref_labels_path)], n_trees=100, noise_ratio=3, seed=42)
    path = tmp_path_factory.mktemp("ml_model") / "model.joblib"
    save_model(model_dict, path)
    return path


class TestMLDetectorBasics:
    def test_name(self):
        detector = MLDetector(MLParams())
        assert detector.name == "ML"

    def test_no_model_path_raises(self):
        detector = MLDetector(MLParams(modelPath=""))
        signal = np.random.randn(10_000)
        with pytest.raises(ValueError, match="modelPath"):
            detector.detect(signal, 250_000)

    def test_missing_model_file_raises(self):
        detector = MLDetector(MLParams(modelPath="/nonexistent/model.joblib"))
        signal = np.random.randn(10_000)
        with pytest.raises(FileNotFoundError):
            detector.detect(signal, 250_000)


class TestMLDetectorOnRealAudio:
    def test_detect_returns_labels(self, trained_model_path: Path, example_audio: tuple[np.ndarray, int]):
        samples, fs = example_audio
        detector = MLDetector(MLParams(modelPath=str(trained_model_path), sensitivity=0.5))
        labels = detector.detect(samples, fs)

        assert isinstance(labels, list)
        assert len(labels) > 0
        for lbl in labels:
            assert isinstance(lbl, Label)
            assert lbl.label == "d"
            assert lbl.start_frequency == 0.0
            assert lbl.end_frequency == 0.0
            assert lbl.start_time < lbl.end_time
            assert lbl.start_index < lbl.stop_index

    def test_detect_matches_reference_reasonably(
        self, trained_model_path: Path, example_audio: tuple[np.ndarray, int], example_ref_labels_path: Path
    ):
        from squeak_peek.labels.io import import_labels

        samples, fs = example_audio
        detector = MLDetector(MLParams(modelPath=str(trained_model_path), sensitivity=0.5))
        labels = detector.detect(samples, fs)

        ref = import_labels(example_ref_labels_path, fs)
        stats = compare_labels(labels, ref)
        # Trained and evaluated on the same recording, so this is an
        # in-sample sanity check (expect a strong fit), not a generalization
        # benchmark -- a held-out recording would score lower.
        assert stats.f1_score > 0.5

    def test_higher_sensitivity_reduces_detections(
        self, trained_model_path: Path, example_audio: tuple[np.ndarray, int]
    ):
        samples, fs = example_audio
        low = MLDetector(MLParams(modelPath=str(trained_model_path), sensitivity=0.2)).detect(samples, fs)
        high = MLDetector(MLParams(modelPath=str(trained_model_path), sensitivity=0.9)).detect(samples, fs)
        assert len(high) <= len(low)

    def test_min_event_duration_filters_short_events(
        self, trained_model_path: Path, example_audio: tuple[np.ndarray, int]
    ):
        samples, fs = example_audio
        lenient = MLDetector(MLParams(modelPath=str(trained_model_path), minEventDuration=0.0)).detect(samples, fs)
        strict = MLDetector(
            MLParams(modelPath=str(trained_model_path), minEventDuration=1.0)
        ).detect(samples, fs)
        assert len(strict) <= len(lenient)
        for lbl in strict:
            assert (lbl.end_time - lbl.start_time) >= 1.0

    def test_model_is_cached_across_detect_calls(
        self, trained_model_path: Path, example_audio: tuple[np.ndarray, int]
    ):
        samples, fs = example_audio
        detector = MLDetector(MLParams(modelPath=str(trained_model_path)))
        detector.detect(samples[: 5 * fs], fs)
        model_dict_after_first = detector._model_dict
        detector.detect(samples[: 5 * fs], fs)
        assert detector._model_dict is model_dict_after_first  # not reloaded


class TestMLDetectorEdgeCases:
    def test_short_signal_returns_empty_or_valid(self, trained_model_path: Path):
        detector = MLDetector(MLParams(modelPath=str(trained_model_path)))
        signal = np.random.randn(1000).astype(np.float32)  # far shorter than one frame
        labels = detector.detect(signal, 250_000)
        assert isinstance(labels, list)

    def test_silent_signal_returns_empty_list(self, trained_model_path: Path):
        detector = MLDetector(MLParams(modelPath=str(trained_model_path)))
        signal = np.zeros(50_000, dtype=np.float32)
        labels = detector.detect(signal, 250_000)
        assert isinstance(labels, list)
