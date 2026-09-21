"""
Integration tests — full pipeline end-to-end.

Phase 2–6 will add tests here as each detector and pipeline stage lands.
Phase 1: only a smoke test verifying the project structure is importable.
"""



def test_package_importable():
    import squeak_peek  # noqa: F401
    assert squeak_peek.__version__ == "1.0.0"


def test_config_importable():
    from squeak_peek.config import AppSettings  # noqa: F401


def test_audio_io_importable():
    from squeak_peek.audio.io import load_wav, save_wav  # noqa: F401


def test_filters_importable():
    from squeak_peek.audio.filters import bandpass_filter, compute_stft  # noqa: F401


def test_label_model_importable():
    from squeak_peek.labels.model import Label  # noqa: F401


def test_detector_base_importable():
    from squeak_peek.detectors.base import AbstractDetector  # noqa: F401


def test_cli_importable():
    from cli.main import cli  # noqa: F401
