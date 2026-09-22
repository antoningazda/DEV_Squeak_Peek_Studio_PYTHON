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

import squeak_peek.detectors  # noqa: F401  (registers built-in detectors)
from squeak_peek.audio.io import load_wav
from squeak_peek.config import AppSettings
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.labels.io import export_labels_detector, import_labels
from squeak_peek.labels.metrics import compare_labels
from squeak_peek.labels.model import Label
from squeak_peek.labels.postprocess import merge_close_labels, remove_short_labels

_DETECTOR_CHOICES = sorted(cls.id.lower() for cls in AbstractDetector.all())


def _build_detector(name: str, settings: AppSettings) -> AbstractDetector:
    """Instantiate a detector by name from the loaded settings."""
    det_id = name.upper()
    try:
        cls = AbstractDetector.get(det_id)
    except KeyError as exc:
        raise click.ClickException(str(exc)) from exc
    return cls(settings.detection.params_for(det_id))


def _postprocess(labels: list[Label], settings: AppSettings) -> list[Label]:
    """Apply the standard merge + remove-short post-processing pipeline."""
    post = settings.detection.post
    labels = merge_close_labels(labels, post.maxGapToMerge)
    labels = remove_short_labels(labels, post.minLabelLength)
    return labels


def _run_detection(wav_path: Path, detector_name: str, settings: AppSettings) -> list[Label]:
    signal, fs = load_wav(wav_path)
    detector = _build_detector(detector_name, settings)
    try:
        labels = detector.detect(signal, fs)
    except ValueError as exc:
        # e.g. MLDetector when its Params.modelPath isn't configured.
        raise click.ClickException(str(exc)) from exc
    return _postprocess(labels, settings)


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
def detect(wav_path: Path, detector: str, settings: Path, output: Path | None) -> None:
    """Run a detector on a single WAV file."""
    app_settings = AppSettings.from_json(settings)
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
def batch(wav_dir: Path, detector: str, settings: Path, output_dir: Path | None) -> None:
    """Run a detector on all WAV files in a directory."""
    app_settings = AppSettings.from_json(settings)
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


if __name__ == "__main__":
    cli()
