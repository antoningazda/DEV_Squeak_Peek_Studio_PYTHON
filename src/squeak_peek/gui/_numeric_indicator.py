"""A tiny, purely-explanatory bar that shows where a numeric setting's
current value (or a min/max pair) sits within its allowed range.

It never accepts input itself — the spin box next to it is what the user
edits — it just gives numbers a visual anchor alongside the text caption.
"""

from __future__ import annotations

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QDoubleSpinBox, QSizePolicy, QSpinBox, QWidget

from . import _theme as t


class RangeIndicator(QWidget):
    """Point mode shows one marker at ``value``; range mode shades the
    band between ``lo`` and ``hi``. Both draw against the widget's full
    ``[minimum, maximum]`` span, labeled at each end.
    """

    def __init__(
        self,
        minimum: float = 0.0,
        maximum: float = 1.0,
        unit: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._unit = unit
        self._lo: float | None = None
        self._hi: float | None = None
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        t.signal.changed.connect(self.update)

    def set_bounds(self, minimum: float, maximum: float) -> None:
        self._min, self._max = minimum, maximum
        self.update()

    def set_value(self, value: float) -> None:
        self._lo, self._hi = value, None
        self.update()

    def set_range(self, lo: float, hi: float) -> None:
        self._lo, self._hi = lo, hi
        self.update()

    def _frac(self, value: float) -> float:
        span = self._max - self._min
        if span <= 0:
            return 0.0
        return max(0.0, min(1.0, (value - self._min) / span))

    def _fmt(self, value: float) -> str:
        text = f"{int(value)}" if float(value).is_integer() else f"{value:g}"
        return f"{text}{self._unit}"

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        left, right = 3, self.width() - 3
        track_w = max(1, right - left)
        track_y = 15.0

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(t.BORDER))
        p.drawRoundedRect(QRectF(left, track_y, track_w, 4), 2, 2)

        if self._lo is not None:
            p.setBrush(QColor(t.ACCENT))
            if self._hi is not None:
                x0 = left + self._frac(self._lo) * track_w
                x1 = left + self._frac(self._hi) * track_w
                p.drawRoundedRect(QRectF(x0, track_y, max(3.0, x1 - x0), 4), 2, 2)
            else:
                x = left + self._frac(self._lo) * track_w
                p.drawEllipse(QRectF(x - 4, track_y - 2, 8, 8))

        font = p.font()
        font.setPointSize(max(7, font.pointSize() - 2))
        p.setFont(font)
        p.setPen(QColor(t.TEXT_MUTED))
        p.drawText(QRectF(left, 0, 90, 13), Qt.AlignmentFlag.AlignLeft, self._fmt(self._min))
        p.drawText(QRectF(right - 90, 0, 90, 13), Qt.AlignmentFlag.AlignRight, self._fmt(self._max))


def bind_value(spin: QSpinBox | QDoubleSpinBox, indicator: RangeIndicator) -> None:
    """Point mode: indicator tracks a single spin box's current value."""
    indicator.set_bounds(spin.minimum(), spin.maximum())
    indicator.set_value(spin.value())
    spin.valueChanged.connect(indicator.set_value)


def bind_range(
    spin_lo: QSpinBox | QDoubleSpinBox,
    spin_hi: QSpinBox | QDoubleSpinBox,
    indicator: RangeIndicator,
) -> None:
    """Range mode: indicator shades the band between two spin boxes."""
    def _update() -> None:
        indicator.set_range(spin_lo.value(), spin_hi.value())

    spin_lo.valueChanged.connect(_update)
    spin_hi.valueChanged.connect(_update)
    _update()
