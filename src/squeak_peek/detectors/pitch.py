"""
Pitch-trace detector: one event per stretch where pitch tracking holds.

The Visualization tab draws an orange pitch contour wherever a call is
actually playing. This detector reports exactly those stretches as events —
what you see traced is what you get — by calling the same tracker
(squeak_peek.features.pitch.track_pitch) rather than a copy of it.

Because it keys on a *coherent, prominent* frequency contour rather than on
band power, it ignores broadband transients (knocks, rustle) that fool
power-threshold detectors, but it also misses calls too faint for the
contour to lock on. Useful as a second opinion next to PSD/BSCD, and as a
boundary refiner: its start/end times are the first and last frame the
contour survives.

The signal is processed in overlapping blocks (see ``blockSeconds``) so a
long recording never needs its whole spectrogram in memory at once; events
spanning a block seam are stitched back together afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.detectors.base import AbstractDetector
from squeak_peek.features import pitch as pitch_tracking

if TYPE_CHECKING:
    from squeak_peek.labels.model import Label


@dataclass
class _Span:
    """One tracked contour, in absolute recording time and Hz."""

    start: float
    end: float
    f_start: float
    f_end: float
    f_median: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class PitchParams(BaseModel):
    """Parameters for the pitch-trace detector."""

    fcutMin: float = Field(
        40_000, ge=0, le=250_000,
        description="Lower bound of the frequency band searched for a pitch contour.",
        json_schema_extra={"unit": "Hz", "group": "freq_band"},
    )
    fcutMax: float = Field(
        120_000, ge=0, le=250_000,
        description="Upper bound of the frequency band searched for a pitch contour.",
        json_schema_extra={
            "unit": "Hz", "group": "freq_band",
            "caption": "Bins outside [fcutMin, fcutMax] are ignored, as is anything above "
                       "Max contour frequency.",
        },
    )
    denoise: bool = Field(
        False,
        description="Suppress stationary background noise before this detector runs.",
        json_schema_extra={
            "caption": "Off by default: the tracker already references each frequency bin to its "
                       "own median, which removes stationary noise lines on its own.",
        },
    )
    segmentLength: int = Field(
        1024, ge=256, le=16_384,
        description="STFT window length used to compute the spectrogram.",
        json_schema_extra={
            "unit": "samples",
            "caption": "Matches the Visualization tab's default, so the events line up with the "
                       "pitch trace you see. Large windows slow the contour search considerably.",
        },
    )
    overlapFactor: float = Field(
        0.5, ge=0.0, le=0.99,
        description="Fraction of each analysis window that overlaps with the next.",
        json_schema_extra={
            "decimals": 3,
            "caption": "Higher overlap gives finer event boundaries at the cost of speed.",
        },
    )
    prominenceDb: float = Field(
        pitch_tracking.PROMINENCE_DB, ge=0.0, le=60.0,
        description="How far a frame's strongest bin must stand above the background to count "
                    "as part of a call.",
        json_schema_extra={
            "unit": "dB", "decimals": 1,
            "caption": "Lower -> more, fainter events. Higher -> only strong, clean contours.",
        },
    )
    minRunFrames: int = Field(
        pitch_tracking.MIN_RUN, ge=1, le=1_000,
        description="Shortest run of consecutive call frames accepted as an event.",
        json_schema_extra={"unit": "frames"},
    )
    maxJumpKhz: float = Field(
        pitch_tracking.MAX_JUMP_KHZ, ge=0.1, le=100.0,
        description="Largest frequency change the contour may make between two frames.",
        json_schema_extra={
            "unit": "kHz", "decimals": 1,
            "caption": "Fast FM sweeps stay under this; raise it only if steep sweeps are "
                       "being cut short.",
        },
    )
    jumpPenaltyDbPerKhz: float = Field(
        pitch_tracking.JUMP_PENALTY_DB_PER_KHZ, ge=0.0, le=100.0,
        description="Cost charged per kHz of frequency change when choosing the contour.",
        json_schema_extra={
            "unit": "dB/kHz", "decimals": 1,
            "caption": "Higher keeps the contour on one band instead of hopping between "
                       "parallel harmonics.",
        },
    )
    maxContourFreqKhz: float = Field(
        pitch_tracking.MAX_FREQ_KHZ, ge=1.0, le=250.0,
        description="Bins above this are ignored when searching for the contour.",
        json_schema_extra={
            "unit": "kHz", "decimals": 1,
            "caption": "Stray high-frequency noise above the usual USV range otherwise wins "
                       "and drags the contour with it.",
        },
    )
    minCallFreqKhz: float = Field(
        0.0, ge=0.0, le=250.0,
        description="Discard events whose contour sits below this frequency.",
        json_schema_extra={
            "unit": "kHz", "decimals": 1, "group": "call_band",
        },
    )
    maxCallFreqKhz: float = Field(
        250.0, ge=0.0, le=250.0,
        description="Discard events whose contour sits above this frequency.",
        json_schema_extra={
            "unit": "kHz", "decimals": 1, "group": "call_band",
            "caption": "Judged on the median frequency of each event's contour, after the "
                       "contour has been found over the full search band — so narrowing this "
                       "throws calls away without disturbing the tracking, which narrowing "
                       "the search band would. Defaults 0-250 keep everything.",
        },
    )
    minDurationMs: float = Field(
        0.0, ge=0.0, le=10_000.0,
        description="Discard events shorter than this.",
        json_schema_extra={
            "unit": "ms", "decimals": 1,
            "caption": "0 keeps every run that passed Min. run length.",
        },
    )
    blockSeconds: float = Field(
        20.0, ge=1.0, le=600.0,
        description="Length of the blocks the recording is processed in.",
        json_schema_extra={
            "unit": "s", "decimals": 1,
            "caption": "Bounds memory on long recordings. Each block's noise background is "
                       "estimated from that block, so shorter blocks adapt faster to drifting "
                       "noise. Events crossing a block seam are rejoined.",
        },
    )

    model_config = {"populate_by_name": True}


class PitchDetector(AbstractDetector):
    """Reports one event per stretch of audio where a pitch contour holds."""

    id = "PITCH"
    display_name = "Pitch trace"
    description = (
        "Pitch trace — marks every stretch where a coherent frequency contour "
        "stands out from the background, i.e. exactly where Visualization draws "
        "its orange trace. Ignores broadband noise; misses faint calls."
    )
    Params = PitchParams

    def detect(self, signal: np.ndarray, fs: int) -> list[Label]:
        p = self.params

        x = np.asarray(signal, dtype=np.float64).ravel()
        if x.size == 0:
            return []

        hop = max(1, int(round(p.segmentLength * (1.0 - p.overlapFactor))))
        block = max(p.segmentLength, int(round(p.blockSeconds * fs)))
        # Overlap blocks by a whole window so a call sitting on a seam is seen
        # intact in at least one of them; the seam-merge below removes the
        # duplicate it produces.
        step = max(hop, block - p.segmentLength)

        spans: list[_Span] = []
        for offset in range(0, x.size, step):
            chunk = x[offset : offset + block]
            if chunk.size < p.segmentLength:
                break
            spans.extend(self._spans_in(chunk, fs, offset / fs))
            if offset + block >= x.size:
                break

        return self._to_labels(self._merge(spans, hop / fs), fs)

    # ── Internals ─────────────────────────────────────────────────────────

    def _spans_in(self, chunk: np.ndarray, fs: int, t_offset: float) -> list[_Span]:
        """Every tracked contour in one block, in absolute recording time."""
        p = self.params
        f_hz, t_rel, Sxx_db = compute_stft(chunk, fs, p.segmentLength, p.overlapFactor)
        f_band_hz, Sxx_band = band_restrict(f_hz, Sxx_db, p.fcutMin, p.fcutMax)
        if f_band_hz.size < 2 or Sxx_band.shape[1] == 0:
            return []

        runs = pitch_tracking.track_pitch(
            f_band_hz / 1_000.0,
            Sxx_band,
            prominence_db=p.prominenceDb,
            min_run=p.minRunFrames,
            jump_penalty_db_per_khz=p.jumpPenaltyDbPerKhz,
            max_jump_khz=p.maxJumpKhz,
            max_freq_khz=p.maxContourFreqKhz,
        )
        return [
            _Span(
                start=t_offset + float(t_rel[run.start]),
                end=t_offset + float(t_rel[run.end]),
                f_start=float(run.freq_khz[0]) * 1_000.0,
                f_end=float(run.freq_khz[-1]) * 1_000.0,
                f_median=float(np.median(run.freq_khz)) * 1_000.0,
            )
            for run in runs
        ]

    @staticmethod
    def _merge(spans: list[_Span], tolerance_s: float) -> list[_Span]:
        """Join spans that overlap or sit within one hop of each other.

        Blocks overlap, so the same call is usually reported twice; and a
        call split by a seam arrives as two touching halves. Both collapse
        here. The merged span keeps the earliest start frequency and the
        latest end frequency, so the contour's endpoints survive, and a
        duration-weighted median so the band filter sees the whole call.
        """
        merged: list[_Span] = []
        for span in sorted(spans, key=lambda s: (s.start, s.end)):
            prev = merged[-1] if merged else None
            if prev is not None and span.start - prev.end <= tolerance_s:
                if span.end <= prev.end:
                    continue        # wholly inside the previous span
                total = prev.duration + span.duration
                if total > 0:
                    prev.f_median = (prev.f_median * prev.duration
                                     + span.f_median * span.duration) / total
                prev.end = span.end
                prev.f_end = span.f_end
            else:
                merged.append(span)
        return merged

    def _to_labels(self, spans: list[_Span], fs: int) -> list[Label]:
        from squeak_peek.labels.model import Label

        p = self.params
        min_duration = p.minDurationMs / 1_000.0
        f_low, f_high = p.minCallFreqKhz * 1_000.0, p.maxCallFreqKhz * 1_000.0
        return [
            Label(
                start_time=span.start,
                end_time=span.end,
                label="d",
                start_frequency=span.f_start,
                end_frequency=span.f_end,
                start_index=int(round(span.start * fs)),
                stop_index=int(round(span.end * fs)),
            )
            for span in spans
            if span.duration >= min_duration and f_low <= span.f_median <= f_high
        ]
