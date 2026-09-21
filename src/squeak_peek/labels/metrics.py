"""
Label comparison and evaluation metrics.

Ported from MATLAB:
  - compareLabels.m → compare_labels()

Uses midpoint-matching criterion: a detected label is a true positive if
its time range contains the midpoint of a ground-truth label.
"""

from __future__ import annotations

from dataclasses import dataclass

from squeak_peek.labels.model import Label


@dataclass
class ComparisonStats:
    """Detection evaluation metrics."""

    total_provided_labels: int
    total_detected_labels: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


def compare_labels(
    detected: list[Label],
    reference: list[Label],
) -> ComparisonStats:
    """
    Compare detected labels to ground-truth labels using midpoint matching.

    A detected label matches a reference label if the detected range contains
    the midpoint of the reference label. Matching is one-to-one (each detected
    label can match at most one reference label).

    Args:
        detected: List of detected Label objects
        reference: List of ground-truth Label objects

    Returns:
        ComparisonStats with TP/FP/FN counts and derived metrics
    """
    total_ref = len(reference)
    total_det = len(detected)

    # Track which detected labels have been matched
    matched_detected = [False] * total_det

    true_positives = 0
    false_negatives = 0

    # For each reference label, try to find a matching detected label
    for ref_lbl in reference:
        ref_midpoint = ref_lbl.midpoint
        matched = False

        for j, det_lbl in enumerate(detected):
            if matched_detected[j]:
                # Already matched to another reference
                continue

            # Check if reference midpoint falls within detected range
            if det_lbl.start_time <= ref_midpoint <= det_lbl.end_time:
                matched_detected[j] = True
                matched = True
                break

        if matched:
            true_positives += 1
        else:
            false_negatives += 1

    false_positives = sum(1 for m in matched_detected if not m)

    # Compute metrics
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return ComparisonStats(
        total_provided_labels=total_ref,
        total_detected_labels=total_det,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
    )
