from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from squeak_peek import __version__ as APP_VERSION

from . import _theme as t

# URLs for the buttons
GITHUB_REPO_URL = "https://github.com/antoningazda/DEV_Squeak_Peek_Studio_PYTHON"
# The documentation site is built from docs/ in the Python repo and published
# to the public releases repo's GitHub Pages (.github/workflows/docs.yml), so
# it sits next to the installers users download.
DOCUMENTATION_URL = "https://antoningazda.github.io/squeak-peek-studio-releases/"
THESIS_URL = "https://dspace.cvut.cz/entities/publication/dede5152-081b-41cf-a4ba-55dacad884d5"
AUTHOR_EMAIL = "antonin.gazda@gmail.com"

# Get the assets directory relative to this file
_ASSETS_DIR = Path(__file__).parent / "assets" / "logo"


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

        # Main logo with Easter egg support
        logo_path = _ASSETS_DIR / "squeakpeak_logo.png"
        if logo_path.exists():
            self._logo = QLabel()
            self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._logo_pixmap = QPixmap(str(logo_path)).scaledToWidth(120, Qt.TransformationMode.SmoothTransformation)
            self._logo_pixmap_rat = QPixmap(str(_ASSETS_DIR / "rat.gif")).scaledToWidth(120, Qt.TransformationMode.SmoothTransformation)
            self._logo.setPixmap(self._logo_pixmap)
            self._logo.setCursor(Qt.CursorShape.PointingHandCursor)
            self._logo.setToolTip("Click me.")
            self._is_rat_shown = False
            self._logo.mousePressEvent = self._on_logo_clicked
            layout.addWidget(self._logo)

        self._title = QLabel("Squeak Peek Studio")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._title)

        self._version = QLabel(f"Version {APP_VERSION} · Python port")
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

        # Author email link
        email_label = QLabel(f'<a href="mailto:{AUTHOR_EMAIL}" style="color: inherit; text-decoration: none;">{AUTHOR_EMAIL}</a>')
        email_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        email_label.setOpenExternalLinks(True)
        layout.addWidget(email_label)

        layout.addWidget(
            _centered(
                "Czech Technical University in Prague<br>"
                "National Institute of Mental Health (NUDZ)"
            )
        )
        self._year = _centered("2026")
        layout.addWidget(self._year)

        # Action buttons (Documentation, Master's Thesis, Source Code)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(t.SP_3)
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._doc_button = QPushButton("Open Documentation")
        self._doc_button.setToolTip(f"Open the user documentation in your browser.\n{DOCUMENTATION_URL}")
        self._doc_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(DOCUMENTATION_URL)))
        buttons_layout.addWidget(self._doc_button)

        self._thesis_button = QPushButton("Open Master's Thesis")
        self._thesis_button.setToolTip(f"Open the thesis this app is based on.\n{THESIS_URL}")
        self._thesis_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(THESIS_URL)))
        buttons_layout.addWidget(self._thesis_button)

        self._source_button = QPushButton("Open Source Code")
        self._source_button.setToolTip(f"Open the GitHub repository in your browser.\n{GITHUB_REPO_URL}")
        self._source_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL)))
        buttons_layout.addWidget(self._source_button)

        layout.addSpacing(t.SP_3)
        layout.addLayout(buttons_layout)

        # Logos section
        logos_layout = QHBoxLayout()
        logos_layout.setSpacing(t.SP_4)
        logos_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        nimh_logo_path = _ASSETS_DIR / "nimh_logo.png"
        if nimh_logo_path.exists():
            nimh_label = QLabel()
            nimh_pixmap = QPixmap(str(nimh_logo_path)).scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            nimh_label.setPixmap(nimh_pixmap)
            logos_layout.addWidget(nimh_label)

        ctu_logo_path = _ASSETS_DIR / "fee_logo.jpg"
        if ctu_logo_path.exists():
            ctu_label = QLabel()
            ctu_pixmap = QPixmap(str(ctu_logo_path)).scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            ctu_label.setPixmap(ctu_pixmap)
            logos_layout.addWidget(ctu_label)

        if nimh_logo_path.exists() or ctu_logo_path.exists():
            layout.addSpacing(t.SP_3)
            layout.addLayout(logos_layout)

        layout.addStretch()
        self._footer = _centered(
            "Original MATLAB app: SqueakPeekStudio v2.1.2026.02.03<br>"
            "Python port implemented with PyQt6 and pyqtgraph"
        )
        layout.addWidget(self._footer)

        self._apply_theme()
        t.signal.changed.connect(self._apply_theme)

    def _on_logo_clicked(self, event: object) -> None:
        """Toggle between main logo and rat GIF on click (Easter egg)."""
        if self._is_rat_shown:
            self._logo.setPixmap(self._logo_pixmap)
            self._is_rat_shown = False
        else:
            self._logo.setPixmap(self._logo_pixmap_rat)
            self._is_rat_shown = True

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

        # Apply theme to buttons
        button_style = (
            f"QPushButton {{"
            f"  background-color: {t.SURFACE};"
            f"  color: {t.TEXT_PRIMARY};"
            f"  border: 1px solid {t.BORDER};"
            f"  border-radius: 4px;"
            f"  padding: 6px 12px;"
            f"  font-size: {t.TEXT_SM}px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {t.SURFACE_HOVER};"
            f"}}"
        )
        self._doc_button.setStyleSheet(button_style)
        self._thesis_button.setStyleSheet(button_style)
        self._source_button.setStyleSheet(button_style)
