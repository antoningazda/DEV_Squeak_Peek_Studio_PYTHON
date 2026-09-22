"""
Application configuration — Pydantic v2 models mirroring settings/default.json.

Usage
-----
    from squeak_peek.config import AppSettings

    settings = AppSettings.from_json("settings/default.json")

    # Access any parameter with full IDE type support:
    print(settings.detection.params_for("PSD").fcutMin)   # 40000
    print(settings.visualization.colormap)                 # "invgray"

Detector and classifier parameters are stored generically, keyed by each
plugin's `id` (see squeak_peek.detectors / squeak_peek.classifiers) —
adding a new detector or classifier never requires touching this module.

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
    colormap: str = Field(alias="SpectrogramColormap", default="invgray")

    show_reference_labels: bool = Field(alias="ShowReferenceLabels", default=True)
    show_labels: bool = Field(alias="ShowLabels", default=True)

    reference_label_color: str = Field(alias="ReferenceLabelColor", default="magenta")
    label_color: str = Field(alias="LabelColor", default="cyan")

    manual_label_length: float = Field(
        alias="ManualLabelLength", default=0.075,
        description="Default duration (s) of a manually-placed label.",
    )

    # Sonification
    sonification_st: float = Field(
        alias="SonificationST", default=-36,
        description="Semitone shift applied during sonification playback. "
                    "-36 is exactly three octaves (a division by 8).",
    )
    sonification_slowdown: int = Field(
        alias="SonificationSlowdown", default=8,
        description="Integer factor by which playback is slowed down. "
                    "Only used when SonificationNaturalSpeed is off.",
    )
    sonification_natural_speed: bool = Field(
        alias="SonificationNaturalSpeed", default=True,
        description="Slow playback down by exactly the pitch ratio. This is a "
                    "pure tape-speed transform (like a time-expansion bat "
                    "detector) and needs no phase vocoder, so it is free of "
                    "stretching artefacts. Turning it off honours "
                    "SonificationSlowdown at the cost of one vocoder pass.",
    )
    sonification_denoise: bool = Field(
        alias="SonificationDenoise", default=True,
        description="Apply a spectral gate before shifting, so the recording's "
                    "broadband noise floor does not come down as a wall of hiss.",
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

    spectrogram_window: int = Field(alias="SpectrogramWindow", default=4096)
    spectrogram_overlap: int = Field(alias="SpectrogramOverlap", default=2048)
    spectrogram_min_freq_khz: float = Field(alias="SpectrogramMinFrequency", default=40)
    spectrogram_max_freq_khz: float = Field(alias="SpectrogramMaxFrequency", default=120)
    colormap: str = Field(alias="SpectrogramColormap", default="invgray")
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
# Section: Video
# ═══════════════════════════════════════════════════════════════════════════

class VideoSettings(BaseModel):
    """Mirrors the 'Video' section of default.json — sync-click detection
    used when no 'sk' label is available to align a video's timeline."""

    snap_band_min_hz: float = Field(
        alias="SnapBandMinHz", default=2_000,
        description="Lower bound (Hz) of the band searched for a finger-snap "
        "transient in the video's own audio track.",
    )
    snap_band_max_hz: float = Field(
        alias="SnapBandMaxHz", default=20_000,
        description="Upper bound (Hz) of the band searched for a finger-snap "
        "transient in the video's own audio track.",
    )
    snap_threshold_factor: float = Field(
        alias="SnapThresholdFactor", default=8.0,
        description="How many times the noise-floor energy a window must "
        "exceed to be flagged as the snap transient.",
    )

    model_config = {"populate_by_name": True}


# ═══════════════════════════════════════════════════════════════════════════
# Detector / classifier parameter storage
# ═══════════════════════════════════════════════════════════════════════════
#
# Each detector/classifier plugin owns its own Params pydantic model,
# defined alongside its class (see e.g. squeak_peek.detectors.psd.PSDParams,
# squeak_peek.classifiers.duration.DurationClassifierParams). DetectionSettings
# and ClassificationSettings below store the *raw* per-plugin dicts, keyed by
# plugin id, and only validate them against a Params model on demand via
# params_for() — so adding a new detector or classifier never requires
# touching this module or default.json (missing entries just fall back to
# that plugin's own defaults).


class PostProcessParams(BaseModel):
    """Post-processing parameters applied after any detector."""

    maxGapToMerge: float = 0.005     # merge events closer than this gap (s)
    minLabelLength: float = 0.001    # discard events shorter than this (s)

    model_config = {"populate_by_name": True}


class _PluginParamStore(BaseModel):
    """Shared behavior for DetectionSettings/ClassificationSettings: generic,
    per-plugin-id raw parameter dicts, captured via pydantic's `extra`
    mechanism so any key round-trips through JSON without a schema change."""

    model_config = {"populate_by_name": True, "extra": "allow"}

    def params_for(self, plugin_id: str) -> BaseModel:
        """Return the validated Params instance for one registered plugin.

        Falls back to that plugin's own defaults if no raw data is stored
        for it yet (e.g. it was added after this settings file was saved).
        """
        params_cls = self._plugin_registry().get(plugin_id).Params
        raw = self.model_extra.get(plugin_id, {})
        return params_cls.model_validate(raw)

    def set_params(self, plugin_id: str, params: BaseModel) -> None:
        """Store a plugin's Params instance (e.g. after a Settings-tab Apply)."""
        self.model_extra[plugin_id] = params.model_dump()

    def _plugin_registry(self):
        raise NotImplementedError


class DetectionSettings(_PluginParamStore):
    """Mirrors the 'Detection' section of default.json."""

    export_path: str = Field(alias="ExportPath", default="")
    post: PostProcessParams = Field(alias="POST", default_factory=PostProcessParams)

    def _plugin_registry(self):
        from squeak_peek.detectors.base import AbstractDetector
        return AbstractDetector


class ClassificationSettings(_PluginParamStore):
    """Mirrors the (optional) 'Classification' section of default.json —
    generic per-classifier parameter storage, keyed by classifier plugin id."""

    def _plugin_registry(self):
        from squeak_peek.classifiers.base import AbstractClassifier
        return AbstractClassifier


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
    classification: ClassificationSettings = Field(
        alias="Classification", default_factory=ClassificationSettings
    )
    video: VideoSettings = Field(
        alias="Video", default_factory=VideoSettings
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
