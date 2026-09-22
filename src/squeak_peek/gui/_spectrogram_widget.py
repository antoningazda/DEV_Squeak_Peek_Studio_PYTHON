"""Waveform + spectrogram display with draggable label boundaries and right-click label creation.

Three public additions to the SpectrogramWidget API:

1. COLORMAPS: 15 selectable colormaps (parula, turbo, hsv, hot, cool, spring, summer,
   autumn, winter, gray, bone, copper, pink, jet, invgray). Call set_colormap(name) to
   change, or pass colormap_name kwarg to display(). Persists across display() calls.

2. LABEL COLORS: 7 selectable colors (red, green, blue, cyan, magenta, yellow, white)
   for both detected and reference label overlays. Pass detected_color_name and/or
   reference_color_name kwargs to display() to override theme defaults. None/omitted
   falls back to current theme colors.

3. BOUNDARY DRAG & RIGHT-CLICK: Two new signals—
   - boundary_dragged(str, float): emits ("start" or "end", new_time) when dragging
     movable start/end lines (enable_boundary_drag() draws them, disable_boundary_drag()
     removes them).
   - spectrogram_right_clicked(float): emits time value on right-click in either plot.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from scipy.ndimage import median_filter

from squeak_peek.audio.colormaps import get_colormap_lut, get_named_color_rgba
from squeak_peek.audio.filters import band_restrict, compute_stft
from squeak_peek.labels.model import Label

from . import _theme as theme

# Reversed grayscale: low power → light gray, high power → dark  (matches MATLAB)
_GRAY_CM = pg.ColorMap(
    pos=np.array([0.0, 1.0]),
    color=np.array([[210, 210, 210, 255], [15, 15, 15, 255]], dtype=np.uint8),
)

# Pitch trace: fixed bright orange, distinct from the 7 selectable label
# colors and both light/dark theme palettes, drawn on top of the spectrogram.
_PITCH_WIDTH = 3.0
_PEN_PITCH = pg.mkPen("#FFA500", width=_PITCH_WIDTH)

# Detected/reference label lines and boxes: twice as thick as the pitch trace.
_LABEL_WIDTH = _PITCH_WIDTH * 2

# The raw colormap LUTs and named label colors live in squeak_peek.audio.colormaps
# (shared, non-GUI) so the headless video-export spectrogram-panel renderer can
# reuse them without depending on PyQt/pyqtgraph.
_get_color_for_name = get_named_color_rgba


def _build_colormap(name: str) -> pg.ColorMap:
    """Build a pyqtgraph ColorMap from a MATLAB/matplotlib-compatible name.

    Supports 15 names: parula, turbo, hsv, hot, cool, spring, summer, autumn,
    winter, gray, bone, copper, pink, jet, invgray. See
    squeak_peek.audio.colormaps.get_colormap_lut for the underlying LUTs.
    """
    lut = get_colormap_lut(name)
    return pg.ColorMap(pos=np.linspace(0, 1, len(lut)), color=lut)


class SpectrogramWidget(QWidget):
    """Waveform (top) + spectrogram (bottom) with optional label overlays.

    Layout mirrors the MATLAB app: waveform on top, spectrogram below,
    both sharing the same time axis.
    """

    # Signals for interactive features
    boundary_dragged = pyqtSignal(str, float)  # ("start" or "end", new_time)
    spectrogram_right_clicked = pyqtSignal(float)  # time value

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

        # Boundary drag lines (for label edit mode)
        self._boundary_lines: dict[str, pg.InfiniteLine] = {}  # "start" → line, "end" → line
        self._current_colormap_name: str = "parula"
        self._current_colormap: pg.ColorMap = _build_colormap("parula")

        # Connect mouse click handlers for right-click label creation
        self._spec_plot.scene().sigMouseClicked.connect(self._on_plot_mouse_click)
        self._wave_plot.scene().sigMouseClicked.connect(self._on_plot_mouse_click)

        # Resolved here (after QApplication exists) so colors match the
        # current mode; call again via refresh_theme() if it changes live.
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """(Re-)apply theme-dependent colors. Existing label overlays keep
        their old pens until the next display() call redraws them."""
        self._pen_det = pg.mkPen(theme.DETECTED_COLOR, width=_LABEL_WIDTH, style=Qt.PenStyle.DashLine)
        self._pen_ref = pg.mkPen(theme.REFERENCE_COLOR, width=_LABEL_WIDTH, style=Qt.PenStyle.DashLine)
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
        show_pitch: bool = True,
        colormap_name: str | None = None,
        detected_color_name: str | None = None,
        reference_color_name: str | None = None,
    ) -> None:
        # ── Handle colormap change ────────────────────────────────────────
        if colormap_name is not None:
            self.set_colormap(colormap_name)

        # ── Handle label color overrides ───────────────────────────────────
        pen_det = self._pen_det
        col_det = self._col_det
        pen_ref = self._pen_ref
        col_ref = self._col_ref

        if detected_color_name is not None:
            rgba = _get_color_for_name(detected_color_name)
            if rgba:
                col_det = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))
                pen_det = pg.mkPen(col_det, width=_LABEL_WIDTH, style=Qt.PenStyle.DashLine)

        if reference_color_name is not None:
            rgba = _get_color_for_name(reference_color_name)
            if rgba:
                col_ref = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), int(rgba[3]*255))
                pen_ref = pg.mkPen(col_ref, width=_LABEL_WIDTH, style=Qt.PenStyle.DashLine)

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
        self._img.setColorMap(self._current_colormap)  # Apply current colormap
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
            self._draw_labels(detected_labels, t_start, t_end, pen_det, col_det, fmin_khz, fmax_khz)
        if show_reference and reference_labels:
            self._draw_labels(reference_labels, t_start, t_end, pen_ref, col_ref, fmin_khz, fmax_khz)

        if show_pitch:
            self._draw_pitch_trace(f_khz, Sxx_sub, t_abs)

    def clear(self) -> None:
        self._img.clear()
        self._wave_curve.setData([], [])
        self._clear_labels()

    def set_colormap(self, name: str) -> None:
        """Set the spectrogram colormap by name.

        Supported names: parula (default), turbo, hsv, hot, cool, spring, summer,
        autumn, winter, gray, bone, copper, pink, jet, invgray.

        The colormap persists across display() calls until changed again.
        """
        self._current_colormap_name = name
        self._current_colormap = _build_colormap(name)
        self._img.setColorMap(self._current_colormap)

    def enable_boundary_drag(
        self,
        start_time: float,
        end_time: float,
        fmin_khz: float,
        fmax_khz: float,
    ) -> None:
        """Draw two draggable vertical lines at start_time and end_time.

        The lines span the full label frequency range [fmin_khz, fmax_khz] and
        are movable=True. Emits boundary_dragged signal on drag completion.

        Calling this again replaces any previous drag lines (idempotent).
        """
        # Clean up any existing lines
        self.disable_boundary_drag()

        # Create start line
        start_line = pg.InfiniteLine(
            pos=start_time,
            angle=90,
            pen=pg.mkPen(theme.DETECTED_COLOR, width=2.0, style=Qt.PenStyle.DashLine),
            movable=True,
            name="start",
        )
        start_line.sigPositionChangeFinished.connect(
            lambda: self._on_boundary_line_moved("start", start_line)
        )
        self._spec_plot.addItem(start_line)
        self._boundary_lines["start"] = start_line

        # Create end line
        end_line = pg.InfiniteLine(
            pos=end_time,
            angle=90,
            pen=pg.mkPen(theme.REFERENCE_COLOR, width=2.0, style=Qt.PenStyle.DashDotLine),
            movable=True,
            name="end",
        )
        end_line.sigPositionChangeFinished.connect(
            lambda: self._on_boundary_line_moved("end", end_line)
        )
        self._spec_plot.addItem(end_line)
        self._boundary_lines["end"] = end_line

    def disable_boundary_drag(self) -> None:
        """Remove any draggable boundary lines."""
        for key, line in list(self._boundary_lines.items()):
            self._spec_plot.removeItem(line)
            self._boundary_lines.pop(key)

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
        """Draw start/end lines + text for each label visible in the segment.

        A label carrying real frequency bounds (start_frequency/end_frequency
        not both 0 — e.g. reference labels imported from a file with a
        StartFreq/EndFreq line) gets a full bounding box on the spectrogram
        instead of full-height start/end lines, since the frequency range is
        actually known. Detector output currently zeroes both fields (no
        frequency estimate), so it keeps the full-height line style.
        """
        text_y = fmax_khz - (fmax_khz - fmin_khz) * 0.04  # just inside top edge

        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue

            has_freq_range = lbl.start_frequency != 0.0 or lbl.end_frequency != 0.0

            if has_freq_range:
                f_lo = min(lbl.start_frequency, lbl.end_frequency) / 1000.0
                f_hi = max(lbl.start_frequency, lbl.end_frequency) / 1000.0
                box = pg.PlotCurveItem(
                    [lbl.start_time, lbl.end_time, lbl.end_time, lbl.start_time, lbl.start_time],
                    [f_lo, f_lo, f_hi, f_hi, f_lo],
                    pen=pen,
                )
                box.setZValue(10)
                self._spec_plot.addItem(box)
                self._label_items.append((self._spec_plot, box))
                top_y = f_hi
            else:
                # Vertical dashed lines at start and end on spectrogram —
                # no horizontal tie-line at the top; just the two lines.
                for t in (lbl.start_time, lbl.end_time):
                    line = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                    line.setZValue(10)
                    self._spec_plot.addItem(line)
                    self._label_items.append((self._spec_plot, line))
                top_y = text_y

            # Text label: a small opaque badge (dark fill, colored border/
            # text matching the line color) rather than bare text, so it
            # reads clearly against any part of the spectrogram.
            label_text = lbl.label if lbl.label else ""
            if label_text:
                txt = pg.TextItem(
                    text=label_text,
                    color=color,
                    fill=pg.mkBrush(0, 0, 0, 220),
                    border=pg.mkPen(color, width=1),
                    anchor=(0.0, 1.0),
                )
                txt.setZValue(11)
                txt.setPos(lbl.start_time, top_y)
                self._spec_plot.addItem(txt)
                self._label_items.append((self._spec_plot, txt))

            # Thin lines on waveform plot too (amplitude-vs-time only, so
            # start/end markers regardless of whether a frequency box was drawn)
            for t in (lbl.start_time, lbl.end_time):
                wline = pg.InfiniteLine(pos=t, angle=90, pen=pen, movable=False)
                wline.setZValue(10)
                self._wave_plot.addItem(wline)
                self._label_items.append((self._wave_plot, wline))

    def _draw_pitch_trace(
        self,
        f_khz: np.ndarray,
        Sxx_sub: np.ndarray,
        t_abs: np.ndarray,
    ) -> None:
        """Overlay the dominant-frequency contour wherever a call is
        actually playing in the visible segment.

        Traces the peak-power frequency bin per STFT column (the same
        "DomFreq" measure used for ML features), gated by each frame's
        power relative to this segment's own noise floor — independent of
        any detected/reference label, so an untagged call still gets a
        trace. Frames that don't clear the gate (background noise, where
        argmax picks an arbitrary — often very high — frequency bin) are
        dropped rather than traced, and each surviving contiguous run of
        frames is drawn as its own line so separate calls, or a call's
        onset after a quiet lead-in, aren't bridged by a line through
        silence.
        """
        if Sxx_sub.shape[1] < 4:
            return

        peak_idx = np.argmax(Sxx_sub, axis=0)
        pitch_khz = f_khz[peak_idx]
        peak_power = Sxx_sub[peak_idx, np.arange(Sxx_sub.shape[1])]

        noise_floor = np.percentile(peak_power, 20)
        ceiling = np.percentile(peak_power, 95)
        threshold = noise_floor + 0.5 * (ceiling - noise_floor)
        is_signal = peak_power >= threshold

        run_starts = np.where(is_signal & ~np.concatenate(([False], is_signal[:-1])))[0]
        run_ends = np.where(is_signal & ~np.concatenate((is_signal[1:], [False])))[0]

        for start, end in zip(run_starts, run_ends):
            trace_t = t_abs[start : end + 1]
            trace_f = pitch_khz[start : end + 1]
            if len(trace_f) < 2:
                continue
            if len(trace_f) >= 3:
                trace_f = median_filter(trace_f, size=3, mode="nearest")

            curve = pg.PlotDataItem(trace_t, trace_f, pen=_PEN_PITCH)
            self._spec_plot.addItem(curve)
            self._label_items.append((self._spec_plot, curve))

    def _on_boundary_line_moved(self, line_type: str, line: pg.InfiniteLine) -> None:
        """Handle boundary line drag completion."""
        new_time = line.value()
        self.boundary_dragged.emit(line_type, new_time)

    def _on_plot_mouse_click(self, event: object) -> None:
        """Handle right-click on spectrogram or waveform plots."""
        # event is a MouseClickEvent from pyqtgraph
        # Check if it's a right-click
        if event.button() != Qt.MouseButton.RightButton:
            return

        # Map scene position to view coordinates of the spectrogram plot
        # (both plots share x-axis, so time coordinate is valid for both)
        try:
            # Get the scene position and map to the spectrogram viewbox
            scene_pos = event.scenePos()
            vb = self._spec_plot.getViewBox()
            data_pos = vb.mapSceneToView(scene_pos)
            clicked_time = data_pos.x()
            self.spectrogram_right_clicked.emit(clicked_time)
        except Exception:
            # Silently ignore if mapping fails
            pass
