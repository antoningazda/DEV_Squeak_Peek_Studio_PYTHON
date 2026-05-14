"""
Shared pytest fixtures for Squeak Peek Studio tests.

All fixtures that reference audio or label files point at the example data
from the original MATLAB project, which lives at a known relative path.
Tests remain valid as long as that directory is present alongside this repo.

Fixture overview
----------------
matlab_data_dir     — Path to the MATLAB project's data/example/single folder
example_wav_path    — Path object to USV_Example_Short.wav
example_labels_path — Path object to Detected_Labels_Example.txt
example_ref_labels_path — Path object to Reference_Labels_Example.txt
example_audio       — (samples: np.ndarray, fs: int) loaded from example_wav_path
settings            — AppSettings loaded from settings/default.json
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ── Resolve paths relative to this file ────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent

# The MATLAB project is expected to sit alongside the Python repo.
# Adjust MATLAB_PROJECT if your directory layout differs.
MATLAB_PROJECT = Path(__file__).parent.parent.parent / "DEV_Squeak-Peek-Studio"
MATLAB_EXAMPLE_SINGLE = MATLAB_PROJECT / "data" / "example" / "single"
MATLAB_SETTINGS = MATLAB_PROJECT / "settings" / "default.json"


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def matlab_data_dir() -> Path:
    """Path to the MATLAB project's single-file example data directory."""
    if not MATLAB_EXAMPLE_SINGLE.exists():
        pytest.skip(
            f"MATLAB example data not found at {MATLAB_EXAMPLE_SINGLE}. "
            "Ensure the MATLAB project is present alongside this repo."
        )
    return MATLAB_EXAMPLE_SINGLE


@pytest.fixture(scope="session")
def example_wav_path(matlab_data_dir: Path) -> Path:
    p = matlab_data_dir / "USV_Example_Short.wav"
    assert p.exists(), f"Example WAV not found: {p}"
    return p


@pytest.fixture(scope="session")
def example_labels_path(matlab_data_dir: Path) -> Path:
    p = matlab_data_dir / "Detected_Labels_Example.txt"
    assert p.exists(), f"Example labels not found: {p}"
    return p


@pytest.fixture(scope="session")
def example_ref_labels_path(matlab_data_dir: Path) -> Path:
    p = matlab_data_dir / "Reference_Labels_Example.txt"
    assert p.exists(), f"Reference labels not found: {p}"
    return p


@pytest.fixture(scope="session")
def example_audio(example_wav_path: Path) -> tuple[np.ndarray, int]:
    """Load the example WAV file once per test session."""
    from squeak_peek.audio.io import load_wav
    return load_wav(example_wav_path)


@pytest.fixture(scope="session")
def settings():
    """Load AppSettings from the MATLAB project's default.json."""
    from squeak_peek.config import AppSettings
    if not MATLAB_SETTINGS.exists():
        pytest.skip(f"Settings file not found: {MATLAB_SETTINGS}")
    return AppSettings.from_json(MATLAB_SETTINGS)
