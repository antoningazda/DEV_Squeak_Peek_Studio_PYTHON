from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

# Suppress pyqtgraph OpenGL warnings on headless systems
pg.setConfigOptions(antialias=False)


class SpectrogramWidget(QWidget):
    """Stacked spectrogram (top) + waveform (bottom) with label overlays.

    Call display() to render a time segment.  Both plots share a linked x-axis
    so zooming/panning on one moves the other.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        # ── Spectrogram plot ──────────────────────────────────────────────
        self._spec_plot: pg.PlotItem = self._glw.addPlot(row=0, col=0)
        self._spec_plot.setLabel("left", "Frequency (kHz)")
        self._spec_plot.showGrid(x=False, y=True, alpha=0.25)
        self._spec_plot.getAxis("bottom").setStyle(showValues=False)

        self._img = pg.ImageItem()
        self._img.setColorMap(pg.colormap.get("viridis"))
        self._spec_plot.addItem(self._img)

        # ── Waveform plot ─────────────────────────────────────────────────
        self._wave_plot: pg.PlotItem = self._glw.addPlot(row=1, col=0)
        self._wave_plot.setLabel("left", "Amp")
        self._wave_plot.setLabel("bottom", "Time (s)")
        self._wave_plot.showGrid(x=True, y=False, alpha=0.25)
        self._wave_curve = self._wave_plot.plot(pen=pg.mkPen("w", width=1))
        self._wave_plot.setXLink(self._spec_plot)

        # Row height ratio 4:1
        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 1)

        self._label_items: list[pg.LinearRegionItem] = []

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
        t_abs = t_rel + t_start  # shape (T,)

        # ImageItem default column-major: data shape (T, F) → x=time, y=freq
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
        for item in self._label_items:
            self._spec_plot.removeItem(item)
        self._label_items.clear()

        if show_detected and detected_labels:
            self._draw_regions(detected_labels, t_start, t_end, (0, 200, 200, 70))
        if show_reference and reference_labels:
            self._draw_regions(reference_labels, t_start, t_end, (220, 220, 220, 70))

    def clear(self) -> None:
        self._img.clear()
        self._wave_curve.setData([], [])
        for item in self._label_items:
            self._spec_plot.removeItem(item)
        self._label_items.clear()

    # ── Private ───────────────────────────────────────────────────────────

    def _draw_regions(
        self,
        labels: list[Label],
        t_start: float,
        t_end: float,
        rgba: tuple[int, int, int, int],
    ) -> None:
        r, g, b, a = rgba
        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue
            region = pg.LinearRegionItem(
                values=[lbl.start_time, lbl.end_time],
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(r, g, b, a),
                pen=pg.mkPen((r, g, b), width=1),
            )
            self._spec_plot.addItem(region)
            self._label_items.append(region)
