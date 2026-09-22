"""
Headless CLI for Squeak Peek Studio.

Usage:
    squeak-peek-cli detect audio.wav --detector psd
    squeak-peek-cli batch wav_dir/ --detector bscd
    squeak-peek-cli evaluate detected.txt reference.txt
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.detectors.bscd import BSCDDetector
from squeak_peek.detectors.ml import MLDetector
from squeak_peek.detectors.psd import PSDDetector
from squeak_peek.detectors.rbd import RBDDetector
from squeak_peek.labels.io import export_labels_detector, import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import (
    filter_broadband_labels,
    merge_close_labels,
    remove_short_labels,
)

_DETECTOR_CHOICES = ["psd", "bscd", "rbd", "ml", "cnn"]


def _build_detector(name: str, settings: AppSettings) -> AbstractDetector:
    """Instantiate a detector by name from the loaded settings."""
    det = settings.detection
    if name == "psd":
        return PSDDetector(det.psd)
    if name == "bscd":
        return BSCDDetector(det.bscd)
    if name == "rbd":
        return RBDDetector(det.rbd)
    if name == "ml":
        if not det.ml.modelPath:
            raise click.ClickException(
                "The 'ml' detector needs a trained model. Train one with "
                "squeak_peek.ml.train.train_model() and set Detection.ML.modelPath "
                "in your settings JSON, or pass --settings pointing at one that does."
            )
        return MLDetector(det.ml)
    if name == "cnn":
        if not det.cnn.modelPath:
            raise click.ClickException(
                "The 'cnn' detector needs a trained model. Train one with "
                "'squeak-peek-cli train-cnn' and set Detection.CNN.modelPath "
                "in your settings JSON, or pass --settings pointing at one that does."
            )
        try:
            from squeak_peek.detectors.cnn import CNNDetector
        except ImportError as e:
            raise click.ClickException(
                "The 'cnn' detector needs torch/torchvision. Install with: "
                "pip install squeak-peek-studio[cnn]"
            ) from e
        return CNNDetector(det.cnn)
    raise click.ClickException(f"Unknown detector: {name!r}")


def _detector_band(detector_name: str, settings: AppSettings) -> tuple[float, float]:
    """The detector's own analysis band, for band-limited post-processing."""
    params = getattr(settings.detection, detector_name, None)
    return (
        getattr(params, "fcutMin", 40_000.0),
        getattr(params, "fcutMax", 120_000.0),
    )


def _postprocess(
    labels: list[Label],
    settings: AppSettings,
    signal: np.ndarray | None = None,
    fs: int | None = None,
    band: tuple[float, float] = (40_000.0, 120_000.0),
) -> list[Label]:
    """Apply the tonality (optional) + merge + remove-short post-processing pipeline."""
    post = settings.detection.post

    # Before merging: merging spans the gap between two calls, which would
    # dilute the tonality of an otherwise clean detection.
    if post.minTonality > 0 and signal is not None and fs:
        labels = filter_broadband_labels(
            labels, signal, fs, post.minTonality, fcut_min=band[0], fcut_max=band[1]
        )

    labels = merge_close_labels(labels, post.maxGapToMerge)
    labels = remove_short_labels(labels, post.minLabelLength)
    return labels


def _run_detection(wav_path: Path, detector_name: str, settings: AppSettings) -> list[Label]:
    signal, fs = load_wav(wav_path)
    detector = _build_detector(detector_name, settings)
    labels = detector.detect(signal, fs)
    return _postprocess(labels, settings, signal, fs, _detector_band(detector_name, settings))


def _default_output_path(wav_path: Path, detector_name: str) -> Path:
    return wav_path.with_name(f"{wav_path.stem}_{detector_name}_detected.txt")


@click.group()
@click.version_option("1.0.0", prog_name="squeak-peek-cli")
def cli() -> None:
    """Squeak Peek Studio — headless batch processing CLI."""


@cli.command()
@click.argument("wav_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--detector", "-d",
    type=click.Choice(_DETECTOR_CHOICES, case_sensitive=False),
    default="psd",
    show_default=True,
    help="Detection algorithm to use.",
)
@click.option(
    "--settings", "-s",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="settings/default.json",
    show_default=True,
    help="Path to settings JSON file.",
)
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None, help="Output label file path.")
@click.option(
    "--min-tonality", type=float, default=None,
    help="Drop broadband (non-USV) detections below this tonality score, 0-1. "
         "Overrides Detection.POST.minTonality. Try 0.5.",
)
def detect(wav_path: Path, detector: str, settings: Path, output: Path | None, min_tonality: float | None) -> None:
    """Run a detector on a single WAV file."""
    app_settings = AppSettings.from_json(settings)
    if min_tonality is not None:
        app_settings.detection.post.minTonality = min_tonality
    labels = _run_detection(wav_path, detector.lower(), app_settings)

    out_path = output or _default_output_path(wav_path, detector.lower())
    export_labels_detector(out_path, labels)

    click.echo(f"[{detector.upper()}] {len(labels)} events detected in: {wav_path}")
    click.echo(f"  Output: {out_path}")


@cli.command()
@click.argument("wav_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--detector", "-d",
    type=click.Choice(_DETECTOR_CHOICES, case_sensitive=False),
    default="psd",
    show_default=True,
)
@click.option(
    "--settings", "-s",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="settings/default.json",
    show_default=True,
)
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), default=None)
@click.option(
    "--min-tonality", type=float, default=None,
    help="Drop broadband (non-USV) detections below this tonality score, 0-1. "
         "Overrides Detection.POST.minTonality. Try 0.5.",
)
def batch(wav_dir: Path, detector: str, settings: Path, output_dir: Path | None, min_tonality: float | None) -> None:
    """Run a detector on all WAV files in a directory."""
    app_settings = AppSettings.from_json(settings)
    if min_tonality is not None:
        app_settings.detection.post.minTonality = min_tonality
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        click.echo(f"No .wav files found in: {wav_dir}")
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_files:
        labels = _run_detection(wav_path, detector.lower(), app_settings)
        out_path = (
            output_dir / f"{wav_path.stem}_{detector.lower()}_detected.txt"
            if output_dir is not None
            else _default_output_path(wav_path, detector.lower())
        )
        export_labels_detector(out_path, labels)
        click.echo(f"[{detector.upper()}] {wav_path.name}: {len(labels)} events -> {out_path}")


@cli.command()
@click.argument("detected", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def evaluate(detected: Path, reference: Path) -> None:
    """Compare detected labels against a reference file (TP/FP/FN/F1)."""
    detected_labels = import_labels(detected)
    reference_labels = import_labels(reference)

    stats = compare_labels(detected_labels, reference_labels)

    click.echo(f"Detected:  {detected} ({stats.total_detected_labels} labels)")
    click.echo(f"Reference: {reference} ({stats.total_provided_labels} labels)")
    click.echo(f"  True Positives:  {stats.true_positives}")
    click.echo(f"  False Positives: {stats.false_positives}")
    click.echo(f"  False Negatives: {stats.false_negatives}")
    click.echo(f"  Precision: {stats.precision:.3f}")
    click.echo(f"  Recall:    {stats.recall:.3f}")
    click.echo(f"  F1 Score:  {stats.f1_score:.3f}")


@cli.command()
@click.argument("wav_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--labels", "-l", "label_paths", multiple=True, required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Label file for each WAV, same order and count as WAV_PATHS.",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path),
    default=Path("data/models/ml_detector_model.joblib"), show_default=True,
    help="Where to save the trained model (.joblib).",
)
@click.option("--noise-ratio", default=3.0, show_default=True, type=float, help="Noise:USV frame ratio for class balancing.")
@click.option("--n-trees", default=200, show_default=True, type=int, help="Random forest tree count.")
def train(wav_paths: tuple[Path, ...], label_paths: tuple[Path, ...], output: Path, noise_ratio: float, n_trees: int) -> None:
    """Train an ML detector model from WAV + label file pairs."""
    from squeak_peek.ml.train import save_model, train_model

    if len(wav_paths) != len(label_paths):
        raise click.ClickException(
            f"Got {len(wav_paths)} WAV_PATHS but {len(label_paths)} --labels options; "
            "pass one --labels per WAV, in the same order."
        )

    model_dict = train_model(list(zip(wav_paths, label_paths)), noise_ratio=noise_ratio, n_trees=n_trees)
    save_model(model_dict, output)

    info = model_dict["training_info"]
    click.echo(
        f"Trained on {info['n_frames_trained']} frames "
        f"({info['n_usv_total']} USV / {info['n_noise_total']} noise available)"
    )
    click.echo(f"OOB accuracy: {info['oob_accuracy']:.3f}")
    click.echo(f"Saved model: {output}")


@cli.command("train-cnn")
@click.argument("wav_paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--labels", "-l", "label_paths", multiple=True, required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Label file for each WAV, same order and count as WAV_PATHS.",
)
@click.option(
    "--output", "-o", type=click.Path(path_type=Path),
    default=Path("data/models/cnn_detector_model.pt"), show_default=True,
    help="Where to save the trained model checkpoint (.pt).",
)
@click.option("--epochs", default=10, show_default=True, type=int)
@click.option("--batch-size", default=4, show_default=True, type=int)
@click.option("--lr", default=1e-4, show_default=True, type=float)
@click.option("--window-s", default=1.0, show_default=True, type=float, help="Spectrogram tile length (s).")
@click.option("--hop-s", default=0.5, show_default=True, type=float, help="Tile hop during training (s).")
@click.option(
    "--backbone", type=click.Choice(["mobilenet", "resnet50"]), default="mobilenet", show_default=True,
    help="mobilenet = fast, CPU-friendly. resnet50 = heavier, more accurate, wants a GPU.",
)
def train_cnn_cmd(
    wav_paths: tuple[Path, ...], label_paths: tuple[Path, ...], output: Path,
    epochs: int, batch_size: int, lr: float, window_s: float, hop_s: float, backbone: str,
) -> None:
    """Train a CNN (Faster R-CNN) detector model from WAV + label file pairs."""
    try:
        from squeak_peek.cnn.train import save_checkpoint, train_cnn
    except ImportError as e:
        raise click.ClickException(
            "train-cnn needs torch/torchvision. Install with: pip install squeak-peek-studio[cnn]"
        ) from e

    if len(wav_paths) != len(label_paths):
        raise click.ClickException(
            f"Got {len(wav_paths)} WAV_PATHS but {len(label_paths)} --labels options; "
            "pass one --labels per WAV, in the same order."
        )

    checkpoint = train_cnn(
        list(zip(wav_paths, label_paths)),
        window_s=window_s, hop_s=hop_s, epochs=epochs, batch_size=batch_size, lr=lr, backbone=backbone,
        progress=True,
    )
    save_checkpoint(checkpoint, output)

    info = checkpoint["training_info"]
    click.echo(f"Trained on {info['n_tiles_train']} tiles ({info['n_tiles_val']} held out for validation)")
    click.echo(f"Final epoch loss: {info['final_loss']:.4f}")
    if info["n_batches_without_proposals"]:
        click.echo(
            f"Note: {info['n_batches_without_proposals']} batches of entirely call-free tiles produced no "
            "region proposals; their ROI-head loss terms were dropped (expected, not an error)."
        )
    if info["val_ground_truth_boxes"]:
        click.echo(
            f"Validation: {info['val_detections_at_0.5']} detections vs. "
            f"{info['val_ground_truth_boxes']} ground-truth boxes (score>0.5, unmatched)"
        )
    click.echo(f"Saved model: {output}")


@cli.command("convert-usvseg")
@click.argument("src_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--output-dir", "-o", type=click.Path(path_type=Path), default=None,
    help="Where to write converted label files (default: next to each WAV).",
)
def convert_usvseg_cmd(src_dir: Path, output_dir: Path | None) -> None:
    """
    Convert a USVSEG dataset directory (Zenodo 3428024, extracted — matching
    <name>.wav/<name>.csv pairs) into this app's label format, ready for
    'train-cnn' or 'train'.
    """
    from squeak_peek.cnn.convert_usvseg import convert_usvseg_dir

    pairs = convert_usvseg_dir(src_dir, output_dir)
    if not pairs:
        click.echo(f"No matching <name>.wav/<name>.csv pairs found in: {src_dir}")
        return
    for wav_path, label_path in pairs:
        click.echo(f"{wav_path.name} -> {label_path}")
    click.echo(f"Converted {len(pairs)} recordings.")


if __name__ == "__main__":
    cli()
