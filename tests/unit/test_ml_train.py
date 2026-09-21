"""
Unit tests for squeak_peek.ml.train and squeak_peek.ml.optimize (WP7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squeak_peek.labels.model import Label
from squeak_peek.ml.optimize import sweep_noise_ratio, sweep_sensitivity
from squeak_peek.ml.train import (
    FEATURE_COLS,
    _balance_classes,
    _label_frames,
    collect_training_data,
    load_model,
    save_model,
    train_from_features,
    train_model,
)


class TestLabelFrames:
    def test_frame_inside_label_is_usv(self):
        mid_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        labels = [Label(start_time=0.4, end_time=1.1, label="d")]
        y = _label_frames(mid_times, labels)
        assert list(y) == [0, 1, 1, 0, 0]

    def test_boundary_inclusive(self):
        mid_times = np.array([1.0, 2.0])
        labels = [Label(start_time=1.0, end_time=2.0, label="d")]
        y = _label_frames(mid_times, labels)
        assert list(y) == [1, 1]

    def test_multiple_labels_union(self):
        mid_times = np.array([0.5, 1.5, 2.5])
        labels = [
            Label(start_time=0.0, end_time=0.6, label="d"),
            Label(start_time=2.4, end_time=2.6, label="d"),
        ]
        y = _label_frames(mid_times, labels)
        assert list(y) == [1, 0, 1]

    def test_no_labels(self):
        mid_times = np.array([0.0, 1.0])
        y = _label_frames(mid_times, [])
        assert list(y) == [0, 0]


class TestBalanceClasses:
    def test_undersamples_noise_to_ratio(self):
        y = np.array([1] * 10 + [0] * 1000)
        X = np.zeros((len(y), 3))
        X_bal, y_bal = _balance_classes(X, y, noise_ratio=3, seed=42)
        assert np.sum(y_bal == 1) == 10
        assert np.sum(y_bal == 0) == 30
        assert len(y_bal) == 40

    def test_ratio_capped_by_available_noise(self):
        y = np.array([1] * 10 + [0] * 15)
        X = np.zeros((len(y), 3))
        X_bal, y_bal = _balance_classes(X, y, noise_ratio=5, seed=42)
        assert np.sum(y_bal == 1) == 10
        assert np.sum(y_bal == 0) == 15  # only 15 noise frames exist

    def test_deterministic_with_seed(self):
        y = np.array([1] * 5 + [0] * 100)
        X = np.arange(len(y) * 2, dtype=float).reshape(len(y), 2)
        X1, y1 = _balance_classes(X, y, noise_ratio=2, seed=7)
        X2, y2 = _balance_classes(X, y, noise_ratio=2, seed=7)
        assert np.array_equal(X1, X2)
        assert np.array_equal(y1, y2)


class TestTrainFromFeatures:
    def test_too_few_usv_frames_raises(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        X = np.random.RandomState(0).randn(len(y), 12)
        with pytest.raises(ValueError, match="too few USV frames"):
            train_from_features(X, y, fcut_min=40_000, fcut_max=120_000, frame_len_s=0.02, hop_len_s=0.005)

    def test_synthetic_separable_data_trains_well(self):
        rng = np.random.RandomState(0)
        n_usv, n_noise = 60, 300
        X_usv = rng.randn(n_usv, 12) + 5.0
        X_noise = rng.randn(n_noise, 12)
        X = np.vstack([X_usv, X_noise])
        y = np.concatenate([np.ones(n_usv), np.zeros(n_noise)])

        model_dict = train_from_features(
            X, y, fcut_min=40_000, fcut_max=120_000, frame_len_s=0.02, hop_len_s=0.005,
            n_trees=50, noise_ratio=3, seed=42,
        )

        assert model_dict["feature_cols"] == FEATURE_COLS
        assert model_dict["frame_params"]["fcutMin"] == 40_000
        assert model_dict["training_info"]["oob_accuracy"] > 0.9
        assert set(model_dict["model"].classes_) == {0, 1}


class TestModelPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        rng = np.random.RandomState(0)
        X = np.vstack([rng.randn(30, 12) + 5.0, rng.randn(150, 12)])
        y = np.concatenate([np.ones(30), np.zeros(150)])
        model_dict = train_from_features(
            X, y, fcut_min=40_000, fcut_max=120_000, frame_len_s=0.02, hop_len_s=0.005, n_trees=20
        )

        path = tmp_path / "model.joblib"
        save_model(model_dict, path)
        assert path.exists()

        loaded = load_model(path)
        assert loaded["feature_cols"] == model_dict["feature_cols"]
        assert loaded["frame_params"] == model_dict["frame_params"]
        np.testing.assert_array_equal(
            loaded["model"].predict(X), model_dict["model"].predict(X)
        )

    def test_load_missing_model_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "does_not_exist.joblib")


@pytest.fixture(scope="module")
def real_model_dict(example_wav_path: Path, example_ref_labels_path: Path):
    """Train once on the real example WAV/labels; reused by several tests."""
    return train_model([(example_wav_path, example_ref_labels_path)], n_trees=100, noise_ratio=3, seed=42)


class TestTrainOnRealAudio:
    def test_train_model_end_to_end(self, real_model_dict):
        info = real_model_dict["training_info"]
        assert info["n_usv_total"] > 100  # USV_Example_Short.wav has 270 reference labels
        assert info["n_noise_total"] > info["n_usv_total"]
        assert 0.5 < info["oob_accuracy"] <= 1.0
        assert real_model_dict["frame_params"]["fcutMin"] == 40_000
        assert real_model_dict["frame_params"]["fcutMax"] == 120_000

    def test_collect_training_data_matches_train_model_frame_count(
        self, example_wav_path: Path, example_ref_labels_path: Path, real_model_dict
    ):
        X, y = collect_training_data([(example_wav_path, example_ref_labels_path)])
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == len(FEATURE_COLS)
        assert real_model_dict["training_info"]["n_frames_total"] == X.shape[0]


class TestOptimize:
    def test_sweep_sensitivity_on_real_audio(
        self, real_model_dict, example_wav_path: Path, example_ref_labels_path: Path
    ):
        result = sweep_sensitivity(
            real_model_dict, example_wav_path, example_ref_labels_path,
            sensitivity_range=(0.3, 0.5, 0.7),
        )
        assert 0.0 <= result["best_sensitivity"] <= 1.0
        assert result["best_stats"].f1_score > 0.0
        assert len(result["sweep"]) == 3
        # Best entry in the sweep should indeed be the reported best.
        best_in_sweep = max(result["sweep"], key=lambda e: e["stats"].f1_score)
        assert best_in_sweep["sensitivity"] == result["best_sensitivity"]

    def test_sweep_noise_ratio_picks_best_across_ratios(
        self, example_wav_path: Path, example_ref_labels_path: Path
    ):
        result = sweep_noise_ratio(
            [(example_wav_path, example_ref_labels_path)],
            example_wav_path, example_ref_labels_path,
            noise_ratio_range=(2, 4),
            sensitivity_range=(0.4, 0.5, 0.6),
            n_trees=50,
        )
        assert result["best_noise_ratio"] in (2, 4)
        assert result["best_stats"].f1_score > 0.0
        assert len(result["sweep"]) == 2
        for entry in result["sweep"]:
            assert entry["model"]["training_info"]["noise_ratio"] == entry["noise_ratio"]
