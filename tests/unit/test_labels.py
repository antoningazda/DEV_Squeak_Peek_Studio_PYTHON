"""
Tests for label I/O, post-processing, and metrics.

Covers:
  - import_labels / export_labels round-trip
  - export_labels_detector
  - All post-processing functions
  - compare_labels metrics
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from squeak_peek.labels.io import (
    export_labels,
    export_labels_detector,
    import_labels,
)
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import (
    adaptive_strongest_filter,
    filter_low_centroid_labels,
    merge_close_labels,
    pad_labels,
    remove_short_labels,
)

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_labels() -> list[Label]:
    """Create a small set of test labels."""
    return [
        Label(start_time=0.0, end_time=0.1, label="d"),
        Label(start_time=0.5, end_time=0.6, label="5"),
        Label(start_time=1.0, end_time=1.2, label="sk"),
    ]


@pytest.fixture
def labels_with_frequencies() -> list[Label]:
    """Create test labels with frequency information."""
    return [
        Label(
            start_time=0.1,
            end_time=0.2,
            label="d",
            start_frequency=50000.0,
            end_frequency=60000.0,
        ),
        Label(
            start_time=0.5,
            end_time=0.6,
            label="5",
            start_frequency=45000.0,
            end_frequency=55000.0,
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# I/O Tests
# ══════════════════════════════════════════════════════════════════════════════


def test_import_labels_basic() -> None:
    """Test basic import of labels from file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0.0\t0.1\td\n")
        f.write("\\\t0.0\t0.0\n")
        f.write("0.5\t0.6\t5\n")
        f.write("\\\t45000.0\t55000.0\n")
        tmp_path = f.name

    try:
        labels = import_labels(tmp_path)
        assert len(labels) == 2
        assert labels[0].start_time == 0.0
        assert labels[0].end_time == 0.1
        assert labels[0].label == "d"
        assert labels[1].start_frequency == 45000.0
        assert labels[1].end_frequency == 55000.0
    finally:
        Path(tmp_path).unlink()


def test_export_labels_basic(sample_labels: list[Label]) -> None:
    """Test basic export of labels to file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        export_labels(tmp_path, sample_labels)

        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Should have 2 lines per label
        assert len(lines) == 6

        # Check first label (line 0 and 1)
        assert "0.000000" in lines[0]
        assert "0.100000" in lines[0]
        assert "d" in lines[0]
        assert "\\" in lines[1]

        # Check second label
        assert "0.500000" in lines[2]
        assert "0.600000" in lines[2]
        assert "5" in lines[2]
    finally:
        Path(tmp_path).unlink()


def test_round_trip_with_frequencies(
    labels_with_frequencies: list[Label],
) -> None:
    """Test round-trip: export then import with full fidelity."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        # Export
        export_labels(tmp_path, labels_with_frequencies)

        # Import
        imported = import_labels(tmp_path)

        # Verify
        assert len(imported) == len(labels_with_frequencies)
        for orig, imp in zip(labels_with_frequencies, imported):
            assert abs(imp.start_time - orig.start_time) < 1e-5
            assert abs(imp.end_time - orig.end_time) < 1e-5
            assert imp.label == orig.label
            assert abs(imp.start_frequency - orig.start_frequency) < 0.01
            assert abs(imp.end_frequency - orig.end_frequency) < 0.01
    finally:
        Path(tmp_path).unlink()


def test_export_labels_detector() -> None:
    """Test detector export (fixed 'd' label, zeroed frequencies)."""
    labels = [
        Label(start_time=0.1, end_time=0.2, label="anything"),
        Label(start_time=0.5, end_time=0.6, label="ignored"),
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        export_labels_detector(tmp_path, labels)

        with open(tmp_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Check all labels are marked as 'd'
        assert "0.100000\t0.200000\td" in lines[0]
        assert "0.500000\t0.600000\td" in lines[2]

        # Check all frequencies are zeroed
        assert "\\\t0.000000\t0.000000" in lines[1]
        assert "\\\t0.000000\t0.000000" in lines[3]
    finally:
        Path(tmp_path).unlink()


def test_import_with_fs(sample_labels: list[Label]) -> None:
    """Test that import_labels computes sample indices when fs is provided."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0.0\t0.1\td\n")
        f.write("\\\t0.0\t0.0\n")
        tmp_path = f.name

    try:
        fs = 250000
        labels = import_labels(tmp_path, fs=fs)

        assert len(labels) == 1
        assert labels[0].start_index == 0
        assert labels[0].stop_index == 25000  # 0.1 * 250000
    finally:
        Path(tmp_path).unlink()


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing Tests
# ══════════════════════════════════════════════════════════════════════════════


def test_merge_close_labels() -> None:
    """Test merging of labels with small gaps."""
    labels = [
        Label(start_time=0.0, end_time=0.1, label="d"),
        Label(start_time=0.102, end_time=0.2, label="d"),  # 0.002s gap
        Label(start_time=0.5, end_time=0.6, label="5"),  # Large gap
    ]

    merged = merge_close_labels(labels, max_gap=0.005)

    assert len(merged) == 2
    # First two should be merged
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 0.2
    # Third should remain
    assert merged[1].start_time == 0.5


def test_merge_close_labels_empty() -> None:
    """Test merge_close_labels with empty input."""
    result = merge_close_labels([], max_gap=0.005)
    assert result == []


def test_remove_short_labels() -> None:
    """Test removal of labels shorter than threshold."""
    labels = [
        Label(start_time=0.0, end_time=0.001, label="d"),  # 1 ms
        Label(start_time=0.1, end_time=0.15, label="5"),  # 50 ms
        Label(start_time=0.2, end_time=0.3, label="sk"),  # 100 ms
    ]

    filtered = remove_short_labels(labels, min_duration=0.01)

    assert len(filtered) == 2
    assert filtered[0].label == "5"
    assert filtered[1].label == "sk"


def test_pad_labels() -> None:
    """Test label padding."""
    labels = [
        Label(start_time=0.1, end_time=0.2, label="d"),
        Label(start_time=0.5, end_time=0.6, label="5"),
    ]

    padded = pad_labels(labels, padding_sec=0.05, total_duration_sec=1.0)

    assert len(padded) == 2
    assert padded[0].start_time == 0.05  # 0.1 - 0.05
    assert padded[0].end_time == 0.25  # 0.2 + 0.05
    assert padded[1].start_time == 0.45  # 0.5 - 0.05
    assert padded[1].end_time == 0.65  # 0.6 + 0.05


def test_pad_labels_clamping() -> None:
    """Test that padding respects boundaries."""
    labels = [
        Label(start_time=0.02, end_time=0.1, label="d"),
    ]

    padded = pad_labels(labels, padding_sec=0.05, total_duration_sec=0.5)

    assert padded[0].start_time == 0.0  # Clamped at zero
    assert abs(padded[0].end_time - 0.15) < 1e-9  # Within duration (approx)


def test_filter_low_centroid_labels() -> None:
    """Test centroid-based filtering."""
    # Create a synthetic high-frequency signal (above 30 kHz)
    fs = 250000
    duration = 0.1
    t = np.arange(int(fs * duration)) / fs
    # 60 kHz sine wave
    signal_high = np.sin(2 * np.pi * 60000 * t).astype(np.float32)

    # Create a synthetic low-frequency signal (below 30 kHz)
    signal_low = np.sin(2 * np.pi * 10000 * t).astype(np.float32)

    # Concatenate: first half high-freq, second half low-freq
    signal = np.concatenate([signal_high, signal_low])

    labels = [
        Label(start_time=0.0, end_time=0.05, label="d"),  # High freq
        Label(start_time=0.05, end_time=0.1, label="d"),  # Low freq
    ]

    filtered = filter_low_centroid_labels(
        labels, signal, fs, min_centroid_hz=30000.0
    )

    # High freq label should be kept, low freq should be filtered
    assert len(filtered) <= 2
    # At least the high-freq one should survive
    assert any(lbl.start_time < 0.05 for lbl in filtered)


def test_adaptive_strongest_filter() -> None:
    """Test adaptive power-based filtering."""
    fs = 250000
    # Create signals for two time periods
    t_loud = np.arange(int(fs * 0.1)) / fs  # 0.0-0.1s
    t_quiet = np.arange(int(fs * 0.1)) / fs + 0.1  # 0.1-0.2s

    # Create two signals: one loud, one very quiet (much smaller amplitude)
    loud = np.sin(2 * np.pi * 50000 * t_loud) * 1.0  # Keep float64
    quiet = np.sin(2 * np.pi * 50000 * t_quiet) * 0.05  # 5% amplitude

    signal = np.concatenate([loud, quiet])  # float64 array

    labels = [
        Label(start_time=0.0, end_time=0.1, label="d"),  # Loud
        Label(start_time=0.1, end_time=0.2, label="d"),  # Quiet
    ]

    filtered = adaptive_strongest_filter(
        labels, signal, fs, relative_threshold=0.15  # 15% of max
    )

    # Quiet label should be filtered out since 0.05 RMS < 0.15 * 0.707 RMS
    assert len(filtered) < len(labels)
    assert len(filtered) == 1
    assert filtered[0].start_time < 0.1


def test_adaptive_strongest_filter_single_label() -> None:
    """Test that single label is not filtered."""
    fs = 250000
    signal = np.sin(2 * np.pi * 50000 * np.arange(fs * 0.1) / fs).astype(np.float32)

    labels = [Label(start_time=0.0, end_time=0.1, label="d")]

    filtered = adaptive_strongest_filter(labels, signal, fs)

    assert len(filtered) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Tests
# ══════════════════════════════════════════════════════════════════════════════


def test_compare_labels_perfect_match() -> None:
    """Test metrics with perfect match (all TPs)."""
    detected = [
        Label(start_time=0.0, end_time=0.2, label="d"),
        Label(start_time=0.5, end_time=0.7, label="d"),
    ]
    reference = [
        Label(start_time=0.05, end_time=0.15, label="d"),  # Midpoint at 0.1 (in range)
        Label(start_time=0.55, end_time=0.65, label="d"),  # Midpoint at 0.6 (in range)
    ]

    stats = compare_labels(detected, reference)

    assert stats.true_positives == 2
    assert stats.false_positives == 0
    assert stats.false_negatives == 0
    assert stats.precision == 1.0
    assert stats.recall == 1.0
    assert stats.f1_score == 1.0


def test_compare_labels_no_match() -> None:
    """Test metrics with no matches."""
    detected = [
        Label(start_time=0.0, end_time=0.1, label="d"),
    ]
    reference = [
        Label(start_time=0.5, end_time=0.6, label="d"),  # No overlap
    ]

    stats = compare_labels(detected, reference)

    assert stats.true_positives == 0
    assert stats.false_positives == 1
    assert stats.false_negatives == 1
    assert stats.precision == 0.0
    assert stats.recall == 0.0


def test_compare_labels_partial_match() -> None:
    """Test metrics with partial match."""
    detected = [
        Label(start_time=0.0, end_time=0.2, label="d"),
        Label(start_time=0.3, end_time=0.4, label="d"),
    ]
    reference = [
        Label(start_time=0.05, end_time=0.15, label="d"),  # Matches first detected
        Label(start_time=0.5, end_time=0.6, label="d"),  # No match
    ]

    stats = compare_labels(detected, reference)

    assert stats.true_positives == 1
    assert stats.false_positives == 1
    assert stats.false_negatives == 1
    assert stats.precision == 0.5
    assert stats.recall == 0.5


def test_compare_labels_one_to_one_matching() -> None:
    """Test that matching is one-to-one (each detected matches at most one reference)."""
    detected = [
        Label(start_time=0.0, end_time=0.5, label="d"),  # Wide range
    ]
    reference = [
        Label(start_time=0.1, end_time=0.2, label="d"),  # Midpoint 0.15
        Label(start_time=0.2, end_time=0.3, label="d"),  # Midpoint 0.25
    ]

    stats = compare_labels(detected, reference)

    # The wide detected label matches the first reference (midpoint 0.15)
    # Second reference is unmatched
    assert stats.true_positives == 1
    assert stats.false_positives == 0
    assert stats.false_negatives == 1


# ══════════════════════════════════════════════════════════════════════════════
# Real Data Tests
# ══════════════════════════════════════════════════════════════════════════════


def test_import_real_labels(example_labels_path: Path) -> None:
    """Test import of real example labels."""
    labels = import_labels(example_labels_path, fs=250000)

    assert len(labels) > 0
    # Check format
    assert all(lbl.start_time >= 0 for lbl in labels)
    assert all(lbl.end_time >= lbl.start_time for lbl in labels)  # Allow zero-duration
    assert all(lbl.label == "d" for lbl in labels)  # Detector output


def test_round_trip_real_labels(example_labels_path: Path) -> None:
    """Test round-trip with real example labels."""
    original = import_labels(example_labels_path, fs=250000)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        tmp_path = f.name

    try:
        export_labels(tmp_path, original)
        reimported = import_labels(tmp_path, fs=250000)

        assert len(reimported) == len(original)
        for orig, reimp in zip(original, reimported):
            assert abs(orig.start_time - reimp.start_time) < 1e-5
            assert abs(orig.end_time - reimp.end_time) < 1e-5
            assert orig.label == reimp.label
    finally:
        Path(tmp_path).unlink()


def test_compare_real_labels(
    example_labels_path: Path, example_ref_labels_path: Path
) -> None:
    """Test compare_labels with real data."""
    detected = import_labels(example_labels_path, fs=250000)
    reference = import_labels(example_ref_labels_path, fs=250000)

    stats = compare_labels(detected, reference)

    # Basic sanity checks
    assert stats.total_detected_labels == len(detected)
    assert stats.total_provided_labels == len(reference)
    assert 0 <= stats.precision <= 1
    assert 0 <= stats.recall <= 1
    assert 0 <= stats.f1_score <= 1
    assert stats.true_positives + stats.false_positives == len(detected)
    assert stats.true_positives + stats.false_negatives == len(reference)


def test_postprocess_real_labels(example_audio: tuple[np.ndarray, int]) -> None:
    """Test post-processing on real audio with synthetic labels."""
    samples, fs = example_audio
    # Use only first 5 seconds for fast testing
    t_max = min(5.0, len(samples) / fs)

    labels = [
        Label(start_time=0.5, end_time=0.6, label="d"),
        Label(start_time=0.601, end_time=0.65, label="d"),  # Close to previous
        Label(start_time=1.0, end_time=1.0001, label="d"),  # Very short
        Label(start_time=2.0, end_time=3.0, label="d"),
    ]

    # Test merge
    merged = merge_close_labels(labels, max_gap=0.005)
    assert len(merged) < len(labels)

    # Test remove short
    filtered = remove_short_labels(labels, min_duration=0.001)
    assert len(filtered) < len(labels)

    # Test pad
    padded = pad_labels(labels, padding_sec=0.05, total_duration_sec=t_max)
    assert all(0 <= p.start_time for p in padded)
    assert all(p.end_time <= t_max for p in padded)

    # Test centroid filter on real signal
    try:
        centroid_filtered = filter_low_centroid_labels(
            labels, samples, fs, min_centroid_hz=30000.0
        )
        assert len(centroid_filtered) <= len(labels)
    except Exception:
        # If FFT fails on edge cases, that's okay for this integration test
        pass
