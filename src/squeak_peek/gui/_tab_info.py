from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class InfoTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.setSpacing(10)
        layout.setContentsMargins(40, 40, 40, 40)

        def _centered(html: str) -> QLabel:
            lbl = QLabel(html)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setWordWrap(True)
            return lbl

        layout.addWidget(_centered("<h1>Squeak Peek Studio</h1>"))
        layout.addWidget(_centered("<b>Version 1.0.0  —  Python Port</b>"))
        layout.addSpacing(8)
        layout.addWidget(
            _centered(
                "Visualization, segmentation and sonification of "
                "ultrasonic rodent vocalizations (USVs)."
            )
        )
        layout.addSpacing(16)
        layout.addWidget(_centered("<b>Author:</b> Antonín Gazda"))
        layout.addWidget(
            _centered(
                "Czech Technical University in Prague<br>"
                "National Institute of Mental Health (NUDZ)"
            )
        )
        layout.addWidget(_centered("2025"))
        layout.addStretch()
        layout.addWidget(
            _centered(
                "<small>Original MATLAB app: SqueakPeekStudio v2.1.2026.02.03<br>"
                "Python port implemented with PyQt6 and pyqtgraph</small>"
            )
        )
