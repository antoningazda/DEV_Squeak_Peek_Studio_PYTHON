from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

from . import _theme as theme

# Reversed grayscale: low power → light gray, high power → dark  (matches MATLAB)
_GRAY_CM = pg.ColorMap(
    pos=np.array([0.0, 1.0]),
    color=np.array([[210, 210, 210, 255], [15, 15, 15, 255]], dtype=np.uint8),
)


class SpectrogramWidget(QWidget):
    """Waveform (top) + spectrogram (bottom) with optional label overlays.

    Layout mirrors the MATLAB app: waveform on top, spectrogram below,
    both sharing the same time axis.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # ── Waveform plot — row 0 (TOP) ───────────────────────────────────
        self._wave_plot: pg.PlotItem = self._glw.addPlot(row=0, col=0)
        self._wave_plot.showGrid(x=True, y=False, alpha=0.3)
        self._wave_plot.getAxis("bottom").setStyle(showValues=False)
        self._wave_curve = self._wave_plot.plot()

        # ── Spectrogram plot — row 1 (BOTTOM) ────────────────────────────
        self._spec_plot: pg.PlotItem = self._glw.addPlot(row=1, col=0)
        self._spec_plot.showGrid(x=False, y=True, alpha=0.2)

        self._img = pg.ImageItem()
        self._img.setColorMap(_GRAY_CM)
        self._spec_plot.addItem(self._img)

        # Share x-axis so both plots pan/zoom together
        self._spec_plot.setXLink(self._wave_plot)

        # Height ratio 1:4  (waveform is narrow, spectrogram is tall)
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 4)

        self._label_items: list = []   # items added to spec or wave plot

        # Resolved here (after QApplication exists) so colors match the
        # current mode; call again via refresh_theme() if it changes live.
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """(Re-)apply theme-dependent colors. Existing label overlays keep
        their old pens until the next display() call redraws them."""
        self._pen_det = pg.mkPen(theme.DETECTED_COLOR, width=2.0)
        self._pen_ref = pg.mkPen(theme.REFERENCE_COLOR, width=2.0)
        self._col_det = pg.mkColor(theme.DETECTED_COLOR).getRgb()
        self._col_ref = pg.mkColor(theme.REFERENCE_COLOR).getRgb()

        self._glw.setBackground(theme.SURFACE)
        label_style = {"color": theme.TEXT_SECONDARY, "font-size": "11px"}
        self._wave_plot.setLabel("left", "Amplitude", **label_style)
        self._spec_plot.setLabel("left", "Frequency (kHz)", **label_style)
        self._spec_plot.setLabel("bottom", "Time (s)", **label_style)
        for plot in (self._wave_plot, self._spec_plot):
            for axis in ("left", "bottom"):
                plot.getAxis(axis).setPen(theme.BORDER_HOVER)
                plot.getAxis(axis).setTextPen(theme.TEXT_SECONDARY)
        self._wave_curve.setPen(pg.mkPen(theme.CURVE, width=1))

    # ── Public API ────────────────────────────────────────────────────────

    def display(
        self,
        samples: np.ndarray,
        fs: int,
        t_start: float,
        t_end: float,
        fmin_hz: float = 40_000.0,
        fmax_hz: float = 120_000.0,
        nperseg: int = 1024,
        noverlap: int = 512,
        detected_labels: list[Label] | None = None,
        reference_labels: list[Label] | None = None,
        show_detected: bool = True,
        show_reference: bool = True,
    ) -> None:
        i0 = max(0, int(t_start * fs))
        i1 = min(len(samples), int(t_end * fs))
        chunk = samples[i0:i1]
        if len(chunk) < nperseg:
            return

        # ── Spectrogram ───────────────────────────────────────────────────
        noverlap = min(noverlap, nperseg - 1)
        overlap_factor = noverlap / nperseg
        f, t_rel, Sxx_db = compute_stft(chunk, fs, nperseg, overlap_factor)
        f_sub, Sxx_sub = band_restrict(f, Sxx_db, fmin_hz, fmax_hz)
        f_khz = f_sub / 1_000.0
        t_abs = t_rel + t_start  # absolute time axis

        # ImageItem column-major: shape (T, F)
        img_data = Sxx_sub.T.astype(np.float32)
        vmin = float(np.percentile(img_data, 2))
        vmax = float(np.percentile(img_data, 99.5))

        self._img.setImage(img_data, autoLevels=False, levels=[vmin, vmax])
        self._img.setRect(
            QRectF(
                float(t_abs[0]),
                float(f_khz[0]),
                float(t_abs[-1] - t_abs[0]),
                float(f_khz[-1] - f_khz[0]),
            )
        )
        self._spec_plot.setRange(
            xRange=[t_start, t_end],
            yRange=[fmin_hz / 1_000.0, fmax_hz / 1_000.0],
            padding=0.0,
        )

        # ── Waveform ──────────────────────────────────────────────────────
        n = len(chunk)
        t_wave = np.linspace(t_start, t_start + n / fs, n, endpoint=False)
        step = max(1, n // 50_000)
        self._wave_curve.setData(t_wave[::step], chunk[::step])
        self._wave_plot.setRange(xRange=[t_start, t_end], padding=0.0)

        # ── Label overlays ────────────────────────────────────────────────
        self._clear_labels()
        fmax_khz = fmax_hz / 1_000.0
        fmin_khz = fmin_hz / 1_000.0

        if show_detected and detected_labels:
            self._draw_labels(detected_labels, t_start, t_end, self._pen_det, self._col_det, fmin_khz, fmax_khz)
        if show_reference and reference_labels:
            self._draw_labels(reference_labels, t_start, t_end, self._pen_ref, self._col_ref, fmin_khz, fmax_khz)

    def clear(self) -> None:
        self._img.clear()
        self._wave_curve.setData([], [])
        self._clear_labels()

    # ── Private ───────────────────────────────────────────────────────────

    def _clear_labels(self) -> None:
        for plot, item in self._label_items:
            plot.removeItem(item)
        self._label_items.clear()

    def _draw_labels(
        self,
        labels: list[Label],
        t_start: float,
        t_end: float,
        pen: object,
        color: tuple,
        fmin_khz: float,
        fmax_khz: float,
    ) -> None:
        """Draw start/end lines + text for each label visible in the segment."""
        text_y = fmax_khz - (fmax_khz - fmin_khz) * 0.04  # just inside top edge

        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue

            # Vertical lines at start and end on spectrogram
            for t in (lbl.start_time, lbl.end_time):
                line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                self._spec_plot.addItem(line)
                self._label_items.append((self._spec_plot, line))

            # Horizontal tick at the top connecting start → end
            top_line = pg.PlotDataItem(
                [lbl.start_time, lbl.end_time],
                [fmax_khz * 0.995, fmax_khz * 0.995],
                pen=pen,
            )
            self._spec_plot.addItem(top_line)
            self._label_items.append((self._spec_plot, top_line))

            # Text label
            label_text = lbl.label if lbl.label else ""
            if label_text:
                txt = pg.TextItem(text=label_text, color=color, anchor=(0.0, 1.0))
                txt.setPos(lbl.start_time, text_y)
                self._spec_plot.addItem(txt)
                self._label_items.append((self._spec_plot, txt))

            # Thin lines on waveform plot too
            for t in (lbl.start_time, lbl.end_time):
                wline = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                self._wave_plot.addItem(wline)
                self._label_items.append((self._wave_plot, wline))
