"""
Headless CLI for Squeak Peek Studio.

Usage:
    squeak-peek-cli detect audio.wav --detector psd
    squeak-peek-cli batch wav_dir/ --detector bscd
    squeak-peek-cli evaluate detected.txt reference.txt
    squeak-peek-cli classify-calls model/ results/ -r rec.wav rec_detected.txt
    squeak-peek-cli train-classifier training.csv model_run/
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek import __version__ as APP_VERSION
from squeak_peek.audio.denoise import preprocess
from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import export_labels_detector, import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import (
    filter_broadband_labels,
    merge_close_labels,
    remove_short_labels,
)

_DETECTOR_CHOICES = sorted(cls.id.lower() for cls in AbstractDetector.all())


def _build_detector(name: str, settings: AppSettings) -> AbstractDetector:
    """Instantiate a detector by name from the loaded settings."""
    det_id = name.upper()
    try:
        cls = AbstractDetector.get(det_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    return cls(settings.detection.params_for(det_id))


def _detector_band(detector_name: str, settings: AppSettings) -> tuple[float, float]:
    """The detector's own analysis band, for band-limited post-processing."""
    try:
        params = settings.detection.params_for(detector_name.upper())
    except KeyError:
        return 40_000.0, 120_000.0
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
    det_input = preprocess(signal, fs, settings.detection.pre) if detector.wants_denoise else signal
    try:
        labels = detector.detect(det_input, fs)
    except (ValueError, RuntimeError) as exc:
        # e.g. MLDetector/CNNDetector when Params.modelPath isn't configured,
        # or the CNN extra isn't installed.
        raise click.ClickException(str(exc)) from exc
    return _postprocess(labels, settings, signal, fs, _detector_band(detector_name, settings))


def _set_denoise(settings: AppSettings, detector_name: str, denoise: bool) -> None:
    """Override a detector's own ``denoise`` parameter (if it has one)."""
    det_id = detector_name.upper()
    params = settings.detection.params_for(det_id)
    if "denoise" in type(params).model_fields:
        settings.detection.set_params(det_id, params.model_copy(update={"denoise": denoise}))


def _default_output_path(wav_path: Path, detector_name: str) -> Path:
    return wav_path.with_name(f"{wav_path.stem}_{detector_name}_detected.txt")


@click.group()
@click.version_option(APP_VERSION, prog_name="squeak-peek-cli")
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
@click.option(
    "--denoise/--no-denoise", default=None,
    help="Suppress stationary background noise before detection. Overrides the "
         "detector's own 'denoise' setting (PSD/BSCD/RBD; ML/CNN follow their model).",
)
def detect(
    wav_path: Path, detector: str, settings: Path, output: Path | None,
    min_tonality: float | None, denoise: bool | None,
) -> None:
    """Run a detector on a single WAV file."""
    app_settings = AppSettings.from_json(settings)
    if denoise is not None:
        _set_denoise(app_settings, detector, denoise)
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
@click.option(
    "--denoise/--no-denoise", default=None,
    help="Suppress stationary background noise before detection. Overrides the "
         "detector's own 'denoise' setting (PSD/BSCD/RBD; ML/CNN follow their model).",
)
def batch(
    wav_dir: Path, detector: str, settings: Path, output_dir: Path | None,
    min_tonality: float | None, denoise: bool | None,
) -> None:
    """Run a detector on all WAV files in a directory."""
    app_settings = AppSettings.from_json(settings)
    if denoise is not None:
        _set_denoise(app_settings, detector, denoise)
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
            "train-cnn needs torch/torchvision, which are not installed in this "
            f"Python environment ({sys.executable}). Install with: "
            f'"{sys.executable}" -m pip install torch torchvision'
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


@cli.command("classify-calls")
@click.argument("model", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option(
    "--recording", "-r", "recordings", nargs=2, multiple=True, required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A WAV and its detected-label file. Repeat for more recordings.",
)
@click.option("--device", default="cpu", show_default=True, type=click.Choice(["cpu", "auto", "mps"]))
@click.option("--drop-noise", is_flag=True, help="Leave NOISE calls out of the classified label files.")
def classify_calls_cmd(model: Path, output: Path, recordings, device: str, drop_noise: bool) -> None:
    """
    Classify detected calls with a trained USV model (CNN USV/NOISE + Random
    Forest call types). MODEL is a model folder (manifest.json); OUTPUT must
    be a new folder.
    """
    from squeak_peek.usv_classifier import api

    inputs = [api.ClassifyInput(wav, api.read_label_file(txt)) for wav, txt in recordings]
    try:
        result = api.classify_recordings(
            inputs, model, output, device=device, drop_noise=drop_noise,
            progress=lambda _f, msg: click.echo(msg),
        )
    except (api.TorchMissingError, ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    for name, n in result.counts().items():
        click.echo(f"  {name}: {n}")
    click.echo(f"Results: {result.out_dir}")


@cli.command("train-classifier")
@click.argument("recordings_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.option("--epochs", default=12, show_default=True, type=int, help="CNN epochs.")
@click.option("--trees", default=200, show_default=True, type=int, help="Random Forest trees.")
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--min-examples", default=10, show_default=True, type=int,
              help="Fewer training examples than this: a call type is learned as plain USV.")
@click.option("--generic-labels", default="d", show_default=True, help="Labels meaning 'USV without a type'.")
@click.option("--device", default="cpu", show_default=True, type=click.Choice(["cpu", "auto", "mps"]))
def train_classifier_cmd(
    recordings_csv: Path, output: Path, epochs: int, trees: int, seed: int,
    min_examples: int, generic_labels: str, device: str,
) -> None:
    """
    Train a USV call-type model. RECORDINGS_CSV has columns WavFile and
    LabelFile (call types), optionally DetectedFile (unmatched detections
    become NOISE), GroupID and Split; paths are relative to the CSV. OUTPUT
    must be a new folder; the model is written to OUTPUT/run/model.
    """
    import pandas as pd

    from squeak_peek.usv_classifier import api

    table = pd.read_csv(recordings_csv, dtype=str, keep_default_na=False)
    missing = {"WavFile", "LabelFile"} - set(table.columns)
    if missing:
        raise click.ClickException(f"{recordings_csv} lacks column(s): {', '.join(sorted(missing))}")
    root = recordings_csv.resolve().parent
    inputs = [
        api.TrainingInput(
            wav=root / row["WavFile"],
            labels=root / row["LabelFile"],
            detected=root / row["DetectedFile"] if row.get("DetectedFile") else None,
            group=row.get("GroupID", ""),
            split=row.get("Split", "") or "auto",
        )
        for row in table.to_dict("records")
    ]
    options = api.TrainingOptions(
        epochs=epochs, trees=trees, seed=seed, min_examples=min_examples,
        generic_labels=tuple(s.strip() for s in generic_labels.split(",") if s.strip()),
        device=device,
    )
    try:
        result = api.train_model(inputs, output, options, progress=lambda _f, msg: click.echo(msg))
    except (api.TorchMissingError, ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(result.training_set.counts().to_string())
    for note in result.training_set.notes:
        click.echo(f"Note: {note}")
    if result.info:
        click.echo(result.info.summary())
    if result.metrics:
        m = result.metrics
        click.echo(f"Test: accuracy {m['Accuracy']:.3f}, balanced accuracy {m['BalancedAccuracy']:.3f}, "
                   f"macro-F1 {m['MacroF1']:.3f}")
    click.echo(f"Model: {result.model_dir}")


if __name__ == "__main__":
    cli()
