from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

# Reversed grayscale: low power → light gray, high power → dark  (matches MATLAB)
_GRAY_CM = pg.ColorMap(
    pos=np.array([0.0, 1.0]),
    color=np.array([[210, 210, 210, 255], [15, 15, 15, 255]], dtype=np.uint8),
)

# Detected labels: cyan dashed  |  Reference labels: orange dashed
_PEN_DET = pg.mkPen("#4CC8CC", width=1.5, style=Qt.PenStyle.DashLine)
_PEN_REF = pg.mkPen("#F9A030", width=1.5, style=Qt.PenStyle.DashLine)


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
        self._wave_plot.setLabel("left", "Amplitude")
        self._wave_plot.showGrid(x=True, y=False, alpha=0.3)
        self._wave_plot.getAxis("bottom").setStyle(showValues=False)
        self._wave_curve = self._wave_plot.plot(pen=pg.mkPen("#1F77B4", width=1))

        # ── Spectrogram plot — row 1 (BOTTOM) ────────────────────────────
        self._spec_plot: pg.PlotItem = self._glw.addPlot(row=1, col=0)
        self._spec_plot.setLabel("left", "Frequency (kHz)")
        self._spec_plot.setLabel("bottom", "Time (s)")
        self._spec_plot.showGrid(x=False, y=True, alpha=0.2)

        self._img = pg.ImageItem()
        self._img.setColorMap(_GRAY_CM)
        self._spec_plot.addItem(self._img)

        # Share x-axis so both plots pan/zoom together
        self._spec_plot.setXLink(self._wave_plot)

        # Height ratio 1:4  (waveform is narrow, spectrogram is tall)
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 4)

        self._label_items: list[pg.InfiniteLine] = []

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

        # ── Label overlays (dashed vertical lines) ────────────────────────
        for item in self._label_items:
            self._spec_plot.removeItem(item)
        self._label_items.clear()

        if show_detected and detected_labels:
            self._draw_label_lines(detected_labels, t_start, t_end, _PEN_DET)
        if show_reference and reference_labels:
            self._draw_label_lines(reference_labels, t_start, t_end, _PEN_REF)

    def clear(self) -> None:
        self._img.clear()
        self._wave_curve.setData([], [])
        for item in self._label_items:
            self._spec_plot.removeItem(item)
        self._label_items.clear()

    # ── Private ───────────────────────────────────────────────────────────

    def _draw_label_lines(
        self,
        labels: list[Label],
        t_start: float,
        t_end: float,
        pen: object,
    ) -> None:
        """Draw a pair of dashed vertical lines for each label in view."""
        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue
            for t in (lbl.start_time, lbl.end_time):
                line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                self._spec_plot.addItem(line)
                self._label_items.append(line)
