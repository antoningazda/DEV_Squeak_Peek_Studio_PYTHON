from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from . import _theme as t


class InfoTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(t.SP_2)
        layout.setContentsMargins(t.SP_6, t.SP_6, t.SP_6, t.SP_6)

        def _centered(html: str) -> QLabel:
            lbl = QLabel(html)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            return lbl

        self._title = QLabel("Squeak Peek Studio")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._version = QLabel("Version 0.0.6 · Python port")
        self._version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._version)

        self._divider = QFrame()
        self._divider.setFrameShape(QFrame.Shape.HLine)
        self._divider.setFixedWidth(80)
        layout.addSpacing(t.SP_4)
        layout.addWidget(self._divider, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(t.SP_4)

        layout.addWidget(
            _centered(
                "Visualization, segmentation and sonification of "
                "ultrasonic rodent vocalizations (USVs)."
            )
        )
        layout.addSpacing(t.SP_5)
        layout.addWidget(_centered("<b>Author:</b> Antonín Gazda"))
        layout.addWidget(
            _centered(
                "Czech Technical University in Prague<br>"
                "National Institute of Mental Health (NUDZ)"
            )
        )
        self._year = _centered("2026")
        layout.addWidget(self._year)
        layout.addStretch()
        self._footer = _centered(
            "Original MATLAB app: SqueakPeekStudio v2.1.2026.02.03<br>"
            "Python port implemented with PyQt6 and pyqtgraph"
        )
        layout.addWidget(self._footer)

        self._apply_theme()
        t.signal.changed.connect(self._apply_theme)

    def _apply_theme(self) -> None:
        self._title.setStyleSheet(
            f"font-size: {t.TEXT_XXL}px; font-weight: 600; color: {t.TEXT_PRIMARY};"
        )
        self._version.setStyleSheet(f"color: {t.TEXT_SECONDARY}; font-size: {t.TEXT_SM}px;")
        self._divider.setStyleSheet(
            f"background: {t.BORDER}; border: none; min-height: 1px; max-height: 1px;"
        )
        caption_style = f"color: {t.TEXT_MUTED}; font-size: {t.TEXT_XS}px;"
        self._year.setStyleSheet(caption_style)
        self._footer.setStyleSheet(caption_style)
