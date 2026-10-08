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

Mouse drag/wheel pans and zooms the time axis only; the widget re-renders the
newly visible range itself and emits view_range_changed(t_start, t_end).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

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
# Pitch-trace tracking (see _draw_pitch_trace): a frame is a call frame when
# its peak bin stands this far above the background; shorter runs are noise.
_PITCH_PROMINENCE_DB = 12.0
_PITCH_MIN_RUN = 3
# Viterbi transition cost per kHz of frequency change between frames, and
# the largest change allowed in one frame (fast FM sweeps stay under this).
_PITCH_JUMP_PENALTY_DB_PER_KHZ = 2.0
_PITCH_MAX_JUMP_KHZ = 6.0

# Mouse pan/zoom: re-render at most this often during a drag, and render this
# fraction of the visible span beyond each edge so panning reveals real data
# before the next re-render lands.
_VIEW_RENDER_INTERVAL_MS = 40
_RENDER_MARGIN = 0.5
_LEFT_AXIS_WIDTH = 60
_MAX_VIEW_SPAN_S = 60.0  # matches the Visualization tab's max segment length

# Detected/reference label lines and boxes: twice as thick as the pitch trace.
_LABEL_WIDTH = _PITCH_WIDTH * 2

# The raw colormap LUTs and named label colors live in squeak_peek.audio.colormaps
# (shared, non-GUI) so the headless video-export spectrogram-panel renderer can
# reuse them without depending on PyQt/pyqtgraph.
_get_color_for_name = get_named_color_rgba


def _viterbi_path(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Best bin index per frame for emission scores (F, T) under a
    transition score matrix (F, F) indexed [to, from]."""
    n_bins, n_frames = emission.shape
    rows = np.arange(n_bins)
    score = emission[:, 0].copy()
    back = np.zeros((n_bins, n_frames), dtype=np.int32)
    for t in range(1, n_frames):
        cand = transition + score[None, :]
        back[:, t] = np.argmax(cand, axis=1)
        score = cand[rows, back[:, t]] + emission[:, t]
    path = np.empty(n_frames, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


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
    view_range_changed = pyqtSignal(float, float)  # (t_start, t_end) after mouse pan/zoom

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

        # Share x-axis so both plots pan/zoom together. The link maps by
        # pixels, so both left axes need the same width or a pan offsets
        # the waveform against the spectrogram.
        self._spec_plot.setXLink(self._wave_plot)
        for plot in (self._wave_plot, self._spec_plot):
            plot.getAxis("left").setWidth(_LEFT_AXIS_WIDTH)

        # Height ratio 1:4  (waveform is narrow, spectrogram is tall)
        self._glw.ci.layout.setRowStretchFactor(0, 1)
        self._glw.ci.layout.setRowStretchFactor(1, 4)

        self._label_items: list = []   # items added to spec or wave plot

        # Boundary drag lines (for label edit mode)
        self._boundary_lines: dict[str, pg.InfiniteLine] = {}  # "start" → line, "end" → line
        self._current_colormap_name: str = "invgray"
        self._current_colormap: pg.ColorMap = _build_colormap("invgray")

        # Mouse pan/zoom re-renders the visible range (see _on_view_x_changed).
        for plot in (self._wave_plot, self._spec_plot):
            plot.setMouseEnabled(x=True, y=False)
        self._render_args: dict | None = None
        self._rendered_range: tuple[float, float] = (0.0, 0.0)
        self._view_range: tuple[float, float] = (0.0, 0.0)
        self._rendering = False
        self._pending_view_render = QTimer(self)
        self._pending_view_render.setSingleShot(True)
        self._pending_view_render.setInterval(_VIEW_RENDER_INTERVAL_MS)
        self._pending_view_render.timeout.connect(self._render_current_view)
        self._wave_plot.getViewBox().sigXRangeChanged.connect(self._on_view_x_changed)

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

        # Remembered so a mouse pan/zoom can re-render the newly visible range.
        self._render_args = dict(
            samples=samples, fs=fs, fmin_hz=fmin_hz, fmax_hz=fmax_hz,
            nperseg=nperseg, noverlap=noverlap,
            detected_labels=detected_labels if show_detected else None,
            reference_labels=reference_labels if show_reference else None,
            show_pitch=show_pitch,
            pen_det=pen_det, col_det=col_det, pen_ref=pen_ref, col_ref=col_ref,
        )
        self._pending_view_render.stop()

        # Mouse pan/zoom: time axis only, clamped to the recording.
        duration = len(samples) / fs
        min_span = min(duration, max(4 * nperseg / fs, 0.005))
        for plot in (self._wave_plot, self._spec_plot):
            plot.setLimits(xMin=0.0, xMax=duration, minXRange=min_span, maxXRange=min(duration, _MAX_VIEW_SPAN_S))

        self._view_range = (t_start, t_end)
        self._rendering = True
        try:
            self._spec_plot.setRange(
                xRange=[t_start, t_end],
                yRange=[fmin_hz / 1_000.0, fmax_hz / 1_000.0],
                padding=0.0,
            )
            self._wave_plot.setRange(xRange=[t_start, t_end], padding=0.0)
        finally:
            self._rendering = False
        self._render(t_start, t_end)

    def _render(self, t_start: float, t_end: float) -> None:
        """Draw spectrogram, waveform and overlays for [t_start, t_end] plus a
        margin on each side, so panning shows real data immediately while the
        throttled re-render catches up. Does not touch the view range."""
        a = self._render_args
        if a is None:
            return
        samples, fs = a["samples"], a["fs"]
        fmin_hz, fmax_hz = a["fmin_hz"], a["fmax_hz"]
        nperseg, noverlap = a["nperseg"], a["noverlap"]

        margin = (t_end - t_start) * _RENDER_MARGIN
        r_start = max(0.0, t_start - margin)
        r_end = min(len(samples) / fs, t_end + margin)
        self._rendered_range = (r_start, r_end)

        i0 = max(0, int(r_start * fs))
        i1 = min(len(samples), int(r_end * fs))
        chunk = samples[i0:i1]
        if len(chunk) < nperseg:
            return
        r_start = i0 / fs

        # ── Spectrogram ───────────────────────────────────────────────────
        noverlap = min(noverlap, nperseg - 1)
        overlap_factor = noverlap / nperseg
        f, t_rel, Sxx_db = compute_stft(chunk, fs, nperseg, overlap_factor)
        f_sub, Sxx_sub = band_restrict(f, Sxx_db, fmin_hz, fmax_hz)
        f_khz = f_sub / 1_000.0
        t_abs = t_rel + r_start  # absolute time axis

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

        # ── Waveform ──────────────────────────────────────────────────────
        n = len(chunk)
        t_wave = np.linspace(r_start, r_start + n / fs, n, endpoint=False)
        step = max(1, n // 50_000)
        self._wave_curve.setData(t_wave[::step], chunk[::step])

        # ── Label overlays ────────────────────────────────────────────────
        self._clear_labels()
        fmax_khz = fmax_hz / 1_000.0
        fmin_khz = fmin_hz / 1_000.0

        if a["detected_labels"]:
            self._draw_labels(a["detected_labels"], r_start, r_end, a["pen_det"], a["col_det"], fmin_khz, fmax_khz)
        if a["reference_labels"]:
            self._draw_labels(a["reference_labels"], r_start, r_end, a["pen_ref"], a["col_ref"], fmin_khz, fmax_khz)

        if a["show_pitch"]:
            self._draw_pitch_trace(f_khz, Sxx_sub, t_abs)

    def _on_view_x_changed(self, _vb: object, x_range: tuple) -> None:
        """User pan/zoom moved the time axis: schedule a re-render (throttled,
        so a continuous drag redraws at a steady rate instead of per event)."""
        if self._rendering or self._render_args is None:
            return
        if not self._pending_view_render.isActive():
            self._pending_view_render.start()

    def _render_current_view(self) -> None:
        t0, t1 = self._wave_plot.getViewBox().viewRange()[0]
        # Layout/resize can re-emit the range display() just set; only a
        # real change of view needs a re-render.
        tol = (t1 - t0) * 1e-6
        if abs(t0 - self._view_range[0]) <= tol and abs(t1 - self._view_range[1]) <= tol:
            return
        self._view_range = (t0, t1)
        self._render(t0, t1)
        self.view_range_changed.emit(t0, t1)

    def clear(self) -> None:
        self._render_args = None
        self._pending_view_render.stop()
        self._img.clear()
        self._wave_curve.setData([], [])
        self._clear_labels()

    def set_colormap(self, name: str) -> None:
        """Set the spectrogram colormap by name.

        Supported names: invgray (default), parula, turbo, hsv, hot, cool, spring, summer,
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

        Always full-height start/end lines (not a frequency-range box) —
        label frequency data isn't reliable enough across example/reference
        files to anchor a box's vertical extent.
        """
        text_y = fmax_khz - (fmax_khz - fmin_khz) * 0.04  # just inside top edge

        for lbl in labels:
            if lbl.end_time < t_start or lbl.start_time > t_end:
                continue

            # Vertical dashed lines at start and end on spectrogram — no
            # horizontal tie-line at the top; just the two lines.
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
        """Overlay a pitch contour wherever a call is actually playing.

        Independent of any detected/reference label, so an untagged call
        still gets a trace. Each STFT bin is first referenced to its own
        median over the rendered range (removing stationary noise lines)
        and each frame to its own median across frequency; a frame counts
        as a call when its strongest bin stands _PITCH_PROMINENCE_DB above
        that. Within each contiguous run of call frames the contour is a
        Viterbi path that trades bin power against frequency jumps: these
        calls often show several parallel bands of similar power, and a
        plain per-frame argmax hops between them, drawing vertical zigzags.
        """
        # Ignore bins above 100 kHz: stray high-frequency noise above the
        # typical USV range otherwise wins and makes the trace jump wildly.
        band = f_khz < 100.0
        if Sxx_sub.shape[1] < _PITCH_MIN_RUN or np.count_nonzero(band) < 2:
            return
        f_band = f_khz[band]
        power = Sxx_sub[band]
        power = power - np.median(power, axis=1, keepdims=True)
        power = power - np.median(power, axis=0, keepdims=True)
        is_signal = power.max(axis=0) >= _PITCH_PROMINENCE_DB

        run_starts = np.where(is_signal & ~np.concatenate(([False], is_signal[:-1])))[0]
        run_ends = np.where(is_signal & ~np.concatenate((is_signal[1:], [False])))[0]

        df_khz = float(f_band[1] - f_band[0])
        jump = np.abs(np.subtract.outer(np.arange(len(f_band)), np.arange(len(f_band)))) * df_khz
        transition = np.where(jump <= _PITCH_MAX_JUMP_KHZ, -_PITCH_JUMP_PENALTY_DB_PER_KHZ * jump, -np.inf)

        for start, end in zip(run_starts, run_ends):
            if end - start + 1 < _PITCH_MIN_RUN:
                continue
            path = _viterbi_path(power[:, start : end + 1], transition)
            curve = pg.PlotDataItem(t_abs[start : end + 1], f_band[path], pen=_PEN_PITCH)
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
