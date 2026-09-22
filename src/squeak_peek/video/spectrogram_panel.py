"""
Render a scrolling spectrogram-with-labels panel to composite under an
exported behavior video (see ``video.export.export_synced_video_with_spectrogram``).

Strategy: the full-recording spectrogram is colorized ONCE into a single
wide RGB image (``SpectrogramPanelRenderer._pil_img``). Per output video
frame, a sub-pixel-accurate window is cropped out of that image and
resized to the panel's on-screen size in a single ``Image.transform``
call (smooth, continuous scrolling regardless of the source spectrogram's
own time resolution) — a fixed "now" needle is then drawn over it, with
label boxes/lines/text for whatever calls are visible in that window.
Drawing overlays after the crop (rather than baking them into the big
source image) keeps line widths and font sizes at a constant on-screen
pixel size for the whole video, independent of the source-to-panel scale
factor.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from squeak_peek.audio.colormaps import get_colormap_lut, get_named_color_rgba
from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

_NEEDLE_COLOR = (255, 40, 40)
_DEFAULT_DET_COLOR = (0, 255, 255)      # cyan
_DEFAULT_REF_COLOR = (255, 0, 255)      # magenta
_BADGE_FILL = (0, 0, 0, 220)


def _load_font(size: int) -> ImageFont.ImageFont:
    # matplotlib is a hard dependency and always bundles DejaVu Sans, so this
    # is a reliable cross-platform source of a real (scalable) font — plain
    # names like "Arial.ttf" only resolve on systems where PIL's own font
    # search path happens to include that font, which isn't guaranteed.
    try:
        import matplotlib
        from pathlib import Path

        font_path = (
            Path(matplotlib.__file__).parent
            / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        )
        return ImageFont.truetype(str(font_path), size)
    except Exception:
        return ImageFont.load_default()


class SpectrogramPanelRenderer:
    """Precomputes the whole recording's colorized spectrogram once, then
    produces per-frame panel images for an arbitrary "now" time via
    :meth:`frame_at`.

    Parameters
    ----------
    samples, fs
        The original (ultrasonic) WAV recording — label times and the
        spectrogram itself are both on this timeline.
    detected_labels, reference_labels
        Drawn as colored boxes (when a label carries real frequency
        bounds) or full-height start/end lines, with a text badge.
    fmin_hz, fmax_hz
        Display frequency band.
    nperseg, noverlap
        STFT window/overlap in samples (matches the Visualization tab's
        settings so the panel looks the same as the interactive view).
    colormap_name
        One of the 15 names supported by ``audio.colormaps.get_colormap_lut``.
    panel_width, panel_height
        Output panel size in pixels (must match the final video's frame
        width so it can be stacked under the video frame directly).
    window_seconds
        Total time span visible in the panel at once.
    needle_fraction
        Where the fixed "now" position indicator sits across the panel's
        width, 0 (left edge) – 1 (right edge). 0.5 centers it.
    """

    def __init__(
        self,
        samples: np.ndarray,
        fs: int,
        detected_labels: list[Label] | None,
        reference_labels: list[Label] | None,
        fmin_hz: float,
        fmax_hz: float,
        nperseg: int,
        noverlap: int,
        colormap_name: str,
        detected_color_name: str | None,
        reference_color_name: str | None,
        panel_width: int,
        panel_height: int,
        window_seconds: float = 4.0,
        needle_fraction: float = 0.5,
    ) -> None:
        if panel_width < 2 or panel_height < 2:
            raise ValueError("panel_width/panel_height must be at least 2px")

        noverlap = min(noverlap, nperseg - 1)
        overlap_factor = noverlap / nperseg
        f, t_rel, Sxx_db = compute_stft(samples, fs, nperseg, overlap_factor)
        f_sub, Sxx_sub = band_restrict(f, Sxx_db, fmin_hz, fmax_hz)
        if Sxx_sub.shape[1] < 2 or f_sub.shape[0] < 2:
            raise RuntimeError(
                "Recording is too short to render a spectrogram panel for export."
            )

        dt = float(t_rel[1] - t_rel[0])

        vmin = float(np.percentile(Sxx_sub, 2))
        vmax = float(np.percentile(Sxx_sub, 99.5))
        span = max(vmax - vmin, 1e-6)
        normalized = np.clip((Sxx_sub - vmin) / span, 0.0, 1.0)
        idx = (normalized * 255.0).astype(np.uint8)
        idx = idx[::-1, :]  # row 0 → highest frequency (top of panel)

        lut = get_colormap_lut(colormap_name)
        rgb = lut[idx]  # (F, T, 3) uint8

        n_freq, n_time, _ = rgb.shape
        pad_px = int(math.ceil(window_seconds / dt)) + 4
        bg = lut[0]
        padded = np.empty((n_freq, pad_px + n_time + pad_px, 3), dtype=np.uint8)
        padded[:, :, :] = bg
        padded[:, pad_px : pad_px + n_time, :] = rgb

        self._pil_img = Image.fromarray(padded, mode="RGB")
        self._bg_color = (int(bg[0]), int(bg[1]), int(bg[2]))
        self._dt = dt
        self._t_origin = float(t_rel[0]) - pad_px * dt
        self._n_freq = n_freq

        self._fmin_khz = fmin_hz / 1000.0
        self._fmax_khz = fmax_hz / 1000.0

        self._detected_labels = list(detected_labels or [])
        self._reference_labels = list(reference_labels or [])

        det_rgba = get_named_color_rgba(detected_color_name)
        ref_rgba = get_named_color_rgba(reference_color_name)
        self._det_color = (
            tuple(int(c * 255) for c in det_rgba[:3]) if det_rgba else _DEFAULT_DET_COLOR
        )
        self._ref_color = (
            tuple(int(c * 255) for c in ref_rgba[:3]) if ref_rgba else _DEFAULT_REF_COLOR
        )

        self._panel_width = panel_width
        self._panel_height = panel_height
        self._window_seconds = window_seconds
        self._needle_fraction = min(max(needle_fraction, 0.0), 1.0)
        self._line_width = max(2, panel_height // 130)
        self._font = _load_font(max(12, panel_height // 16))

    def frame_at(self, audio_time: float) -> np.ndarray:
        """Return an RGB uint8 ``(panel_height, panel_width, 3)`` array for
        the panel when "now" (the fixed needle position) is *audio_time*
        seconds on the original WAV timeline."""
        col_center = (audio_time - self._t_origin) / self._dt
        window_px_src = self._window_seconds / self._dt
        left_px = window_px_src * self._needle_fraction
        right_px = window_px_src - left_px
        col_left = col_center - left_px
        col_right = col_center + right_px

        scale_x = (col_right - col_left) / self._panel_width
        scale_y = self._n_freq / self._panel_height
        frame_img = self._pil_img.transform(
            (self._panel_width, self._panel_height),
            Image.AFFINE,
            (scale_x, 0.0, col_left, 0.0, scale_y, 0.0),
            resample=Image.BICUBIC,
            fillcolor=self._bg_color,
        )

        draw = ImageDraw.Draw(frame_img)
        t_win_start = col_left * self._dt + self._t_origin
        t_win_end = col_right * self._dt + self._t_origin
        span = max(t_win_end - t_win_start, 1e-9)

        def x_of(t: float) -> float:
            return (t - t_win_start) / span * self._panel_width

        def y_of(freq_hz: float) -> float:
            frac = (freq_hz / 1000.0 - self._fmin_khz) / (self._fmax_khz - self._fmin_khz)
            return (1.0 - min(max(frac, 0.0), 1.0)) * self._panel_height

        for labels, color in (
            (self._reference_labels, self._ref_color),
            (self._detected_labels, self._det_color),
        ):
            for lbl in labels:
                if lbl.end_time < t_win_start or lbl.start_time > t_win_end:
                    continue
                self._draw_label(draw, lbl, color, x_of, y_of)

        needle_x = self._needle_fraction * self._panel_width
        draw.line(
            [(needle_x, 0), (needle_x, self._panel_height)],
            fill=_NEEDLE_COLOR,
            width=self._line_width,
        )

        return np.asarray(frame_img.convert("RGB"), dtype=np.uint8)

    def _draw_label(self, draw, lbl: Label, color, x_of, y_of) -> None:
        x0, x1 = x_of(lbl.start_time), x_of(lbl.end_time)
        has_freq_range = lbl.start_frequency != 0.0 or lbl.end_frequency != 0.0

        if has_freq_range:
            f_lo = min(lbl.start_frequency, lbl.end_frequency)
            f_hi = max(lbl.start_frequency, lbl.end_frequency)
            y_hi, y_lo = y_of(f_hi), y_of(f_lo)
            draw.rectangle([x0, y_hi, x1, y_lo], outline=color, width=self._line_width)
            top_y = y_hi
        else:
            draw.line([(x0, 0), (x0, self._panel_height)], fill=color, width=self._line_width)
            draw.line([(x1, 0), (x1, self._panel_height)], fill=color, width=self._line_width)
            top_y = self._panel_height * 0.04

        if lbl.label:
            tx, ty = max(0.0, x0 + 2), max(0.0, top_y - self._font.size - 4)
            bbox = draw.textbbox((tx, ty), lbl.label, font=self._font)
            pad = 2
            draw.rectangle(
                [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=_BADGE_FILL
            )
            draw.text((tx, ty), lbl.label, font=self._font, fill=color)
