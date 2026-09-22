r"""
Label I/O: import and export of USV event annotations.

Implements the canonical 2-line-per-label text format:
  Line 1: StartTime    EndTime     Label
  Line 2: \            StartFreq   EndFreq

State encoding: labels may have a 2-char suffix (_<detCode><clsCode>) appended
to the label text where:
  - detCode: D=Accepted, d=Rejected, x=other
  - clsCode: C=Accepted, c=Rejected, x=other
Example: "d_Dc" means label "d" with Accepted detection, Rejected classification.

Ported from MATLAB:
  - importLabels.m → import_labels()
  - exportLabels.m → export_labels()
  - exportLabelsDetector.m → export_labels_detector()
  - decodeStateSuffix.m → decode_state_suffix()
"""

from __future__ import annotations

from pathlib import Path

from squeak_peek.labels.model import Label


def _encode_state_suffix(detection_state: str, classification_state: str) -> str:
    """
    Encode detection and classification states into a 2-char suffix.

    Args:
        detection_state: "Accepted", "Rejected", or "None"
        classification_state: "Accepted", "Rejected", or "None"

    Returns:
        2-char suffix like "Dc" or "xx"
    """
    det_code = "D" if detection_state == "Accepted" else ("d" if detection_state == "Rejected" else "x")
    cls_code = "C" if classification_state == "Accepted" else ("c" if classification_state == "Rejected" else "x")
    return det_code + cls_code


def _decode_state_suffix(suffix: str) -> tuple[str, str]:
    """
    Decode a 2-char state suffix into detection and classification states.

    Args:
        suffix: exactly 2-char string like "Dc" or "xx"

    Returns:
        (detection_state, classification_state) tuple
    """
    if len(suffix) != 2:
        return "None", "None"

    det_code = suffix[0]
    cls_code = suffix[1]

    det_state = "Accepted" if det_code == "D" else ("Rejected" if det_code == "d" else "None")
    cls_state = "Accepted" if cls_code == "C" else ("Rejected" if cls_code == "c" else "None")

    return det_state, cls_state


def import_labels(path: str | Path, fs: int | None = None) -> list[Label]:
    """
    Load labels from a 2-line-per-label text file.

    Each label occupies two lines:
      Line 1: StartTime    EndTime     Label
      Line 2: \\           StartFreq   EndFreq

    Labels may include a 2-char state suffix (e.g., "_Dc") after the last underscore,
    which will be decoded into detection_state and classification_state fields.

    Args:
        path: Path to the label file
        fs: Sampling rate (Hz). If provided, converts times to sample indices.

    Returns:
        List of Label objects
    """
    labels: list[Label] = []

    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line1 = lines[i].strip()
        i += 1

        # Skip empty lines and comments
        if not line1 or line1.startswith("#"):
            continue

        # Parse line 1: StartTime EndTime Label
        parts1 = line1.split("\t")
        if len(parts1) < 3:
            continue

        try:
            start_time = float(parts1[0])
            end_time = float(parts1[1])
            label_name = parts1[2]
            detection_state = "None"
            classification_state = "None"

            # Check for state suffix: exactly 2 chars after LAST underscore
            if "_" in label_name:
                last_us_idx = label_name.rfind("_")
                suffix = label_name[last_us_idx + 1:]

                # Only decode if suffix is exactly 2 chars
                if len(suffix) == 2:
                    detection_state, classification_state = _decode_state_suffix(suffix)
                    # Strip the suffix and underscore from the label text
                    label_name = label_name[:last_us_idx]

            # Parse line 2: \ StartFreq EndFreq (if present)
            start_freq = 0.0
            end_freq = 0.0
            if i < len(lines):
                line2 = lines[i].strip()
                if line2.startswith("\\") or line2.startswith("backslash"):
                    parts2 = line2.split("\t")
                    if len(parts2) >= 3:
                        try:
                            start_freq = float(parts2[1])
                            end_freq = float(parts2[2])
                        except ValueError:
                            pass
                    i += 1

            # Calculate sample indices
            start_index = 0
            stop_index = 0
            if fs is not None and fs > 0:
                start_index = round(start_time * fs)
                stop_index = round(end_time * fs)

            labels.append(
                Label(
                    start_time=start_time,
                    end_time=end_time,
                    label=label_name,
                    start_frequency=start_freq,
                    end_frequency=end_freq,
                    start_index=start_index,
                    stop_index=stop_index,
                    detection_state=detection_state,
                    classification_state=classification_state,
                )
            )
        except (ValueError, IndexError):
            continue

    return labels


def export_labels(path: str | Path, labels: list[Label]) -> None:
    """
    Export labels to the canonical 2-line-per-label text format.

    Appends a 2-char state suffix (e.g., "_Dc") to each label text encoding
    the detection_state and classification_state.

    Args:
        path: Output file path
        labels: List of Label objects to export
    """
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            # Encode state suffix into label text
            suffix = _encode_state_suffix(lbl.detection_state, lbl.classification_state)
            label_with_suffix = f"{lbl.label}_{suffix}"

            # Line 1: StartTime EndTime Label_<detCode><clsCode>
            fh.write(f"{lbl.start_time:.6f}\t{lbl.end_time:.6f}\t{label_with_suffix}\n")
            # Line 2: \ StartFreq EndFreq
            fh.write(f"\\\t{lbl.start_frequency:.6f}\t{lbl.end_frequency:.6f}\n")


def export_labels_detector(path: str | Path, labels: list[Label]) -> None:
    """
    Export detected labels in simplified format (detector output).

    The label type is fixed as 'd' (detected), and frequency bounds are zeroed.

    Args:
        path: Output file path
        labels: List of Label objects to export
    """
    with open(path, "w", encoding="utf-8") as fh:
        for lbl in labels:
            # Line 1: StartTime EndTime d
            fh.write(f"{lbl.start_time:.6f}\t{lbl.end_time:.6f}\td\n")
            # Line 2: \ 0.0 0.0
            fh.write("\\\t0.000000\t0.000000\n")
