"""
Headless CLI for Squeak Peek Studio.

Phase 5/6 will add full sub-commands. This stub registers the CLI group
so the 'squeak-peek-cli' entry point works from Phase 1 onward.

Usage (Phase 1):
    squeak-peek-cli --help
"""

import click


@click.group()
@click.version_option("1.0.0", prog_name="squeak-peek-cli")
def cli() -> None:
    """Squeak Peek Studio — headless batch processing CLI."""


@cli.command()
@click.argument("wav_path", type=click.Path(exists=True))
@click.option(
    "--detector", "-d",
    type=click.Choice(["psd", "bscd", "rbd", "ml"], case_sensitive=False),
    default="psd",
    show_default=True,
    help="Detection algorithm to use.",
)
@click.option(
    "--settings", "-s",
    type=click.Path(exists=True),
    default="settings/default.json",
    show_default=True,
    help="Path to settings JSON file.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output label file path.")
def detect(wav_path: str, detector: str, settings: str, output: str | None) -> None:
    """Run a detector on a single WAV file. (Phase 2)"""
    click.echo(
        f"[Phase 2 pending] Would run '{detector.upper()}' detector on: {wav_path}"
    )
    click.echo(f"  Settings: {settings}")
    click.echo(f"  Output:   {output or '<auto>'}")


@cli.command()
@click.argument("wav_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--detector", "-d", default="psd", show_default=True)
@click.option("--settings", "-s", default="settings/default.json", show_default=True)
@click.option("--output-dir", "-o", type=click.Path(), default=None)
def batch(wav_dir: str, detector: str, settings: str, output_dir: str | None) -> None:
    """Run a detector on all WAV files in a directory. (Phase 2)"""
    click.echo(f"[Phase 2 pending] Would batch-process: {wav_dir}")


@cli.command()
@click.argument("detected", type=click.Path(exists=True))
@click.argument("reference", type=click.Path(exists=True))
def evaluate(detected: str, reference: str) -> None:
    """Compare detected labels against a reference file (TP/FP/FN/F1). (Phase 4)"""
    click.echo(f"[Phase 4 pending] Would compare:\n  Detected:  {detected}\n  Reference: {reference}")


if __name__ == "__main__":
    cli()
