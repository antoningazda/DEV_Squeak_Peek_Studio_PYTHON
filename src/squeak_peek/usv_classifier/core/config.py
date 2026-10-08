from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class TrackerConfig:
    baseline_bins: int = 21
    min_prominence_db: float = 3.0
    min_frame_level_db: float = -35.0
    max_candidates: int = 5
    jump_penalty_per_khz: float = 0.22
    preferred_max_jump_hz: float = 15000
    large_jump_penalty_per_khz: float = 0.45
    gap_emission_penalty: float = 3.0
    gap_transition_penalty: float = 1.5
    gap_continue_penalty: float = 0.75
    max_interp_gap_frames: int = 2
    smooth_frames: int = 3
    min_coverage: float = 0.60
    min_frames: int = 4
    min_median_prominence_db: float = 2.0

    def __post_init__(self):
        if not 0 <= self.min_coverage <= 1 or self.min_frame_level_db >= 0:
            raise ValueError("Invalid tracker coverage / frame level")
        for name, value in asdict(self).items():
            if not math.isfinite(value) or (name != "min_frame_level_db" and value < 0):
                raise ValueError(f"Invalid tracker parameter: {name}")
        for name in ("baseline_bins", "max_candidates", "smooth_frames", "min_frames", "max_interp_gap_frames"):
            if int(getattr(self, name)) != getattr(self, name):
                raise ValueError(f"{name} must be an integer")
        if self.baseline_bins < 3 or self.max_candidates < 1 or self.smooth_frames < 1 or self.min_frames < 2:
            raise ValueError("Invalid tracker dimensions")


@dataclass(frozen=True)
class FeatureConfig:
    freq_range_hz: tuple = (20000.0, 120000.0)
    min_call_s: float = 0.003
    max_call_s: float = 0.5
    preemph: float = 0.9
    filter_order: int = 6
    win_len: int = 1024
    hop_len: int = 256
    nfft: int = 2048
    nfft_policy: str = "matlab_v03"
    min_window_ms: float = 0.8
    short_call_window_fraction: float = 1/3
    smooth_ms: float = 3.0
    background_window_s: float = 0.020
    min_direction_step_hz: float = 500.0
    tracker: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self):
        if isinstance(self.tracker, dict):
            object.__setattr__(self, "tracker", TrackerConfig(**self.tracker))
        validate_range(self.freq_range_hz, "frequency", positive=True)
        if not 0 <= self.min_call_s < self.max_call_s or not 0 <= self.preemph <= 1:
            raise ValueError("Invalid duration / preemphasis")
        if self.nfft_policy not in ("matlab_v03", "fixed"):
            raise ValueError("nfft_policy must be matlab_v03 or fixed")
        for name in ("filter_order", "win_len", "hop_len", "nfft"):
            v = getattr(self, name)
            if int(v) != v or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.win_len < 16 or self.nfft < 128 or not 0 < self.short_call_window_fraction <= 1:
            raise ValueError("Invalid STFT configuration")
        if min(self.min_window_ms, self.smooth_ms, self.background_window_s) <= 0 or self.min_direction_step_hz < 0:
            raise ValueError("Invalid time/step parameters")


@dataclass(frozen=True)
class ImageConfig:
    size: tuple = (96, 96)
    pad_s: float = 0.015
    min_call_s: float = 0.006
    max_call_s: float = 0.200
    win_ms: float = 1.0
    hop_ms: float = 0.25
    nfft: int = 1024
    freq_range_hz: tuple = (15000.0, 120000.0)
    db_range: tuple = (-85.0, -15.0)
    gamma: float = 1.0
    invert: bool = True
    channels: int = 1

    def __post_init__(self):
        validate_range(self.freq_range_hz, "frequency", positive=True)
        validate_range(self.db_range, "dB")
        if len(self.size) != 2 or any(int(x) != x or x < 8 for x in self.size):
            raise ValueError("Image size must contain two integers >= 8")
        if self.channels not in (1, 3) or min(self.win_ms, self.hop_ms, self.gamma) <= 0 or self.pad_s < 0:
            raise ValueError("Invalid image parameters")
        if not 0 <= self.min_call_s < self.max_call_s or self.nfft < 128:
            raise ValueError("Invalid image support/NFFT")


def validate_range(values, name, positive=False):
    if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] >= values[1] or (positive and values[0] <= 0):
        raise ValueError(f"Invalid {name} range: {values}")

