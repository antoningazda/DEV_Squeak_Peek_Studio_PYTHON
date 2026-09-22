"""
Application configuration — Pydantic v2 models mirroring settings/default.json.

Usage
-----
    from squeak_peek.config import AppSettings

    settings = AppSettings.from_json("settings/default.json")

    # Access any parameter with full IDE type support:
    print(settings.detection.psd.fcutMin)       # 40000
    print(settings.visualization.colormap)       # "parula"

Note: the app's visual theme (light/dark/system) is handled entirely by
squeak_peek.gui._theme's design-token system, not by this module — MATLAB's
Light/Gray/Custom raw-RGB theme model was deliberately not ported (see
GUI_PARITY_PLAN.md, WP27).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Section: DataInput
# ═══════════════════════════════════════════════════════════════════════════

class DataInputSettings(BaseModel):
    """Mirrors the 'DataInput' section of default.json."""

    default_usv_single: str = Field(
        alias="DefaultUSVFieldSingle",
        default="",
        description="Default WAV path for single-file mode.",
    )
    default_label_single: str = Field(
        alias="DefaultLabelFieldSingle",
        default="",
        description="Default detected-label path for single-file mode.",
    )
    default_reference_label_single: str = Field(
        alias="DefaultReferenceLabelFieldSingle",
        default="",
        description="Default reference-label path for single-file mode.",
    )
    batch_mode: bool = Field(
        alias="BatchMode",
        default=False,
    )
    default_usv_batch: str = Field(
        alias="DefaultUSVFieldBatch",
        default="",
    )
    default_label_batch: str = Field(
        alias="DefaultLabelFieldBatch",
        default="",
    )
    default_reference_label_batch: str = Field(
        alias="DefaultReferenceLabelFieldBatch",
        default="",
    )

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# Section: Visualization
# ═══════════════════════════════════════════════════════════════════════════

class VisualizationSettings(BaseModel):
    """Mirrors the 'Visualization' section of default.json."""

    show_loading_dialog: bool = Field(alias="ShowLoadingDialog", default=False)

    segment_start_seconds: float = Field(alias="SegmentStartSeconds", default=0.0)
    segment_length_seconds: float = Field(alias="SegmentLengthSeconds", default=1.0)

    spectrogram_window: int = Field(alias="SpectrogramWindow", default=1024)
    spectrogram_overlap: int = Field(alias="SpectrogramOverlap", default=512)
    spectrogram_min_freq_khz: float = Field(
        alias="SpectrogramMinFrequency", default=40,
        description="kHz — lower bound of the display frequency axis.",
    )
    spectrogram_max_freq_khz: float = Field(
        alias="SpectrogramMaxFrequency", default=120,
        description="kHz — upper bound of the display frequency axis.",
    )
    colormap: str = Field(alias="SpectrogramColormap", default="parula")

    show_reference_labels: bool = Field(alias="ShowReferenceLabels", default=True)
    show_labels: bool = Field(alias="ShowLabels", default=True)

    reference_label_color: str = Field(alias="ReferenceLabelColor", default="white")
    label_color: str = Field(alias="LabelColor", default="cyan")

    manual_label_length: float = Field(
        alias="ManualLabelLength", default=0.075,
        description="Default duration (s) of a manually-placed label.",
    )

    # Sonification
    sonification_st: float = Field(
        alias="SonificationST", default=-35,
        description="Semitone shift applied during sonification playback.",
    )
    sonification_slowdown: int = Field(
        alias="SonificationSlowdown", default=4,
        description="Integer factor by which playback is slowed down.",
    )

    model_config = {"populate_by_name": True}

    @property
    def spectrogram_min_freq_hz(self) -> float:
        return self.spectrogram_min_freq_khz * 1_000

    @property
    def spectrogram_max_freq_hz(self) -> float:
        return self.spectrogram_max_freq_khz * 1_000


# ═══════════════════════════════════════════════════════════════════════════
# Section: LabelEdit
# ═══════════════════════════════════════════════════════════════════════════

class LabelEditSettings(BaseModel):
    """Mirrors the 'LabelEdit' section of default.json."""

    spectrogram_window: int = Field(alias="SpectrogramWindow", default=1024)
    spectrogram_overlap: int = Field(alias="SpectrogramOverlap", default=512)
    spectrogram_min_freq_khz: float = Field(alias="SpectrogramMinFrequency", default=40)
    spectrogram_max_freq_khz: float = Field(alias="SpectrogramMaxFrequency", default=120)
    colormap: str = Field(alias="SpectrogramColormap", default="hsv")
    classifications: str = Field(
        alias="Classifications",
        default="d,sk,5,5t,5w,c5",
        description="Comma-separated list of valid USV call-type labels.",
    )

    model_config = {"populate_by_name": True}

    @property
    def classification_list(self) -> list[str]:
        """Return classifications as a Python list."""
        return [c.strip() for c in self.classifications.split(",") if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════
# Detector-specific parameter blocks
# ═══════════════════════════════════════════════════════════════════════════

class PSDParams(BaseModel):
    """Parameters for the Power Spectral Density detector."""

    fcutMin: float = 40_000          # Hz
    fcutMax: float = 120_000         # Hz
    ROIstart: float = 60             # seconds — start of Region of Interest
    ROIlength: float = 10            # seconds — length of ROI
    runWholeSignal: bool = True      # ignore ROI and process full signal
    segmentLength: int = 8192        # STFT window length (samples)
    overlapFactor: float = 0.59      # STFT overlap fraction
    maWindow: int = 3                # moving-average window (frames)
    noiseWindow: int = 240           # noise-floor estimation window (frames)
    localWindow: int = 194           # local smoothing window (frames)
    k: float = 0.023                 # threshold scaling factor
    w: float = 0.994                 # noise-floor weighting factor
    minEffectivePower: float = 8.5e-5  # minimum power for event acceptance

    model_config = {"populate_by_name": True}


class BSCDParams(BaseModel):
    """Parameters for the Bayesian Sequential Change Detection detector."""

    fcutMin: float = 40_000
    fcutMax: float = 120_000
    wlen: float = 0.01               # analysis window length (s)
    maWindow: int = 5_000            # moving-average window (samples)
    noiseWindow: int = 256           # noise estimation window (samples)
    localWindow: int = 256           # local smoothing window (samples)
    k: float = 0.023
    w: float = 0.994

    model_config = {"populate_by_name": True}


class RBDParams(BaseModel):
    """Parameters for the Relative Bayesian Difference detector."""

    fcutMin: float = 40_000
    fcutMax: float = 120_000
    wlen: float = 0.04               # analysis window length (s)
    AR_order_left: int = 4           # AR model order for the left segment
    AR_order_right: int = 4          # AR model order for the right segment
    Bayesian_Evidence_order: int = 4
    dynamicScaling: float = 0.3      # dynamic threshold scaling
    smoothingWindowRBD: float = 0.02 # RBD output smoothing window (s)
    smoothingWindowThr: float = 0.02 # threshold smoothing window (s)
    amplitudeThreshold: float = 0.02 # minimum amplitude for detection

    model_config = {"populate_by_name": True}


class PostProcessParams(BaseModel):
    """Post-processing parameters applied after any detector."""

    maxGapToMerge: float = 0.005     # merge events closer than this gap (s)
    minLabelLength: float = 0.001    # discard events shorter than this (s)

    model_config = {"populate_by_name": True}


class MLParams(BaseModel):
    """Runtime parameters for the ML (Random Forest) detector."""

    modelPath: str = ""              # path to a saved .joblib model file
    minEventDuration: float = 0.003  # minimum event duration after merging (s)
    sensitivity: float = 0.5         # frame-probability threshold (0-1); higher = more selective

    model_config = {"populate_by_name": True}


class DetectionSettings(BaseModel):
    """Mirrors the 'Detection' section of default.json."""

    export_path: str = Field(alias="ExportPath", default="")
    psd: PSDParams   = Field(alias="PSD",  default_factory=PSDParams)
    bscd: BSCDParams = Field(alias="BSCD", default_factory=BSCDParams)
    rbd: RBDParams   = Field(alias="RBD",  default_factory=RBDParams)
    post: PostProcessParams = Field(alias="POST", default_factory=PostProcessParams)
    ml: MLParams     = Field(alias="ML",   default_factory=MLParams)

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# Root settings model
# ═══════════════════════════════════════════════════════════════════════════

class AppSettings(BaseModel):
    """
    Root configuration model for Squeak Peek Studio.

    All fields map 1-to-1 to keys in settings/default.json.
    """

    data_input: DataInputSettings = Field(
        alias="DataInput", default_factory=DataInputSettings
    )
    visualization: VisualizationSettings = Field(
        alias="Visualization", default_factory=VisualizationSettings
    )
    label_edit: LabelEditSettings = Field(
        alias="LabelEdit", default_factory=LabelEditSettings
    )
    detection: DetectionSettings = Field(
        alias="Detection", default_factory=DetectionSettings
    )

    model_config = {"populate_by_name": True}

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> AppSettings:
        """Load settings from a JSON file (e.g. settings/default.json)."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    @classmethod
    def defaults(cls) -> AppSettings:
        """Return a settings object with all default values (no file needed)."""
        return cls()

    # ── Persistence ─────────────────────────────────────────────────────

    def save_json(self, path: str | Path) -> None:
        """Serialise current settings back to JSON."""
        data = self.model_dump(by_alias=True)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
