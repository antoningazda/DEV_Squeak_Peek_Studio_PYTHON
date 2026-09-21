"""
Squeak Peek Studio — PyQt6 main window and application entry point.
"""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar, QStyleFactory, QTabWidget

from . import _theme as t
from ._state import AppState
from ._tab_data_input import DataInputTab
from ._tab_detection import DetectionTab
from ._tab_info import InfoTab
from ._tab_label_edit import LabelEditTab
from ._tab_metrics import MetricsTab
from ._tab_settings import SettingsTab
from ._tab_visualization import VisualizationTab


def _build_qss() -> str:
    """Built lazily (after QApplication exists) so ``t.ACCENT`` etc. resolve
    against the OS's current light/dark appearance."""
    return f"""
/* ── Base ────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {{ background: {t.BG_APP}; }}
QWidget {{
    background: transparent;
    color: {t.TEXT_PRIMARY};
    font-size: {t.TEXT_SM}px;
    selection-background-color: {t.ACCENT_SUBTLE};
    selection-color: {t.TEXT_PRIMARY};
}}
QMainWindow > QWidget, QTabWidget#mainTabs > QWidget {{ background: {t.BG_APP}; }}
QToolTip {{
    background: {t.GRAY_800}; color: {t.GRAY_0}; border: none;
    padding: {t.SP_1}px {t.SP_2}px; border-radius: {t.RADIUS_CONTROL}px;
}}

/* ── Main tab bar — flat, underline indicator ─────────────────────────── */
QTabWidget#mainTabs::pane {{
    border: none; border-top: 1px solid {t.BORDER}; background: {t.BG_APP}; top: -1px;
}}
QTabWidget#mainTabs QTabBar {{
    background: {t.SURFACE};
    border-bottom: 1px solid {t.BORDER};
}}
QTabWidget#mainTabs QTabBar::tab {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    padding: {t.SP_3}px {t.SP_4}px;
    margin: 0px;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabWidget#mainTabs QTabBar::tab:selected {{
    color: {t.TEXT_PRIMARY};
    font-weight: 600;
    border-bottom: 2px solid {t.ACCENT};
}}
QTabWidget#mainTabs QTabBar::tab:hover:!selected {{ color: {t.TEXT_PRIMARY}; }}

/* ── Inner tab bars (Settings sub-tabs — named "innerTabs") ───────────── */
QTabWidget#innerTabs::pane {{ border: none; border-top: 1px solid {t.BORDER}; top: -1px; }}
QTabWidget#innerTabs QTabBar::tab {{
    background: transparent; color: {t.TEXT_SECONDARY};
    padding: {t.SP_2}px {t.SP_3}px; margin-right: {t.SP_3}px;
    border-bottom: 2px solid transparent;
}}
QTabWidget#innerTabs QTabBar::tab:selected {{
    color: {t.TEXT_PRIMARY}; font-weight: 600;
    border-bottom: 2px solid {t.ACCENT};
}}
QTabWidget#innerTabs QTabBar::tab:hover:!selected {{ color: {t.TEXT_PRIMARY}; }}

/* ── Group boxes (panels — one hairline border, no extra decoration) ────*/
QGroupBox {{
    background: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PANEL}px;
    margin-top: {t.SP_4}px; padding: {t.SP_4}px {t.SP_3}px {t.SP_3}px {t.SP_3}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: {t.SP_2}px; padding: 0 {t.SP_1}px;
    color: {t.TEXT_SECONDARY}; font-weight: 600; font-size: {t.TEXT_XS}px;
}}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{
    background: {t.SURFACE}; border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px; padding: {t.SP_2}px {t.SP_4}px;
    color: {t.TEXT_PRIMARY};
}}
QPushButton:hover    {{ background: {t.SURFACE_HOVER}; border-color: {t.BORDER_HOVER}; }}
QPushButton:pressed  {{ background: {t.SURFACE_PRESSED}; }}
QPushButton:disabled {{ color: {t.TEXT_MUTED}; background: {t.SURFACE_MUTED}; }}
QPushButton:focus    {{ outline: none; border: 1px solid {t.ACCENT}; }}

QPushButton#primaryBtn {{
    background: {t.ACCENT}; border: 1px solid {t.ACCENT};
    color: {t.TEXT_ON_ACCENT}; font-weight: 600;
}}
QPushButton#primaryBtn:hover   {{ background: {t.ACCENT_HOVER}; border-color: {t.ACCENT_HOVER}; }}
QPushButton#primaryBtn:pressed {{ background: {t.ACCENT_ACTIVE}; border-color: {t.ACCENT_ACTIVE}; }}
QPushButton#primaryBtn:disabled {{ background: {t.SURFACE_MUTED}; border-color: {t.SURFACE_MUTED}; color: {t.TEXT_MUTED}; }}

QPushButton#dangerBtn {{
    background: {t.SURFACE}; border: 1px solid {t.BORDER};
    color: {t.DANGER}; font-weight: 500;
}}
QPushButton#dangerBtn:hover   {{ background: {t.DANGER_SUBTLE}; border-color: {t.DANGER}; }}
QPushButton#dangerBtn:pressed {{ background: {t.DANGER_SUBTLE}; }}

/* ── Input fields ────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {t.SURFACE}; border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_CONTROL}px; padding: {t.SP_1}px {t.SP_2}px;
    color: {t.TEXT_PRIMARY};
    selection-background-color: {t.ACCENT_SUBTLE};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {t.ACCENT};
}}
QLineEdit:disabled {{ color: {t.TEXT_MUTED}; background: {t.SURFACE_MUTED}; }}
QLineEdit:read-only {{ background: {t.SURFACE_MUTED}; color: {t.TEXT_SECONDARY}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {t.SURFACE}; border: 1px solid {t.BORDER};
    selection-background-color: {t.ACCENT_SUBTLE}; selection-color: {t.TEXT_PRIMARY};
    outline: none; padding: {t.SP_1}px;
}}

/* ── Checkboxes / radio buttons ──────────────────────────────────────── */
QCheckBox, QRadioButton, QLabel {{ background: transparent; color: {t.TEXT_PRIMARY}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {t.BORDER_HOVER}; background: {t.SURFACE};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {t.ACCENT}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {t.ACCENT}; border-color: {t.ACCENT};
}}

/* ── Form layout labels ──────────────────────────────────────────────── */
QFormLayout QLabel {{ color: {t.TEXT_SECONDARY}; }}

/* ── Scroll bars ─────────────────────────────────────────────────────── */
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {t.BORDER_HOVER}; border-radius: 5px; min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {t.TEXT_MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {t.BORDER_HOVER}; border-radius: 5px; min-width: 24px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}

/* ── Misc ────────────────────────────────────────────────────────────── */
QStatusBar {{ background: {t.SURFACE}; color: {t.TEXT_SECONDARY}; border-top: 1px solid {t.BORDER}; }}
QMenuBar   {{ background: {t.SURFACE}; color: {t.TEXT_PRIMARY}; border-bottom: 1px solid {t.BORDER}; }}
QMenuBar::item {{ padding: {t.SP_1}px {t.SP_3}px; background: transparent; border-radius: {t.RADIUS_CONTROL}px; }}
QMenuBar::item:selected {{ background: {t.SURFACE_HOVER}; }}
QMenu {{ background: {t.SURFACE}; color: {t.TEXT_PRIMARY}; border: 1px solid {t.BORDER}; padding: {t.SP_1}px; }}
QMenu::item {{ padding: {t.SP_1}px {t.SP_5}px; border-radius: {t.RADIUS_CONTROL}px; }}
QMenu::item:selected {{ background: {t.SURFACE_HOVER}; color: {t.TEXT_PRIMARY}; }}
QMenu::separator {{ height: 1px; background: {t.BORDER}; margin: {t.SP_1}px {t.SP_2}px; }}
QMessageBox {{ background: {t.SURFACE}; }}
"""


class MainWindow(QMainWindow):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state
        self.setWindowTitle("Squeak Peek Studio")
        self.resize(1280, 800)
        self._build_ui()
        self._build_menu()

        state.wav_loaded.connect(self._on_wav_loaded)

    def _build_ui(self) -> None:
        s = self._state

        self._data_tab = DataInputTab(s)

        tabs = QTabWidget()
        tabs.setObjectName("mainTabs")
        tabs.setDocumentMode(True)
        tabs.addTab(self._data_tab,       "Data Input")
        tabs.addTab(VisualizationTab(s),  "Visualization")
        tabs.addTab(DetectionTab(s),      "Detection")
        tabs.addTab(LabelEditTab(s),      "Label Edit")
        tabs.addTab(MetricsTab(s),        "Metrics")
        tabs.addTab(SettingsTab(s),       "Settings")
        tabs.addTab(InfoTab(),            "Info")

        self.setCentralWidget(tabs)
        self.setStatusBar(QStatusBar())

    def _build_menu(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        open_wav = QAction("&Open WAV…", self)
        open_wav.setShortcut("Ctrl+O")
        open_wav.triggered.connect(self._data_tab.open_wav)
        file_menu.addAction(open_wav)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def _on_wav_loaded(self) -> None:
        p = self._state.wav_path
        name = p.name if p else ""
        self.setWindowTitle(f"Squeak Peek Studio — {name}")
        self.statusBar().showMessage(
            f"Loaded: {p}  |  "
            f"{self._state.duration:.3f} s  |  "
            f"{self._state.fs:,} Hz",
            6_000,
        )


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Squeak Peek Studio")
    app.setOrganizationName("NUDZ")
    # Force Qt's cross-platform style: on macOS, native widget styling
    # (e.g. QComboBox popups rendered as real Cocoa menus) bypasses our
    # QSS entirely, which looks broken in dark mode. Fusion respects it.
    app.setStyle(QStyleFactory.create("Fusion"))

    def apply_theme() -> None:
        # Resolved fresh each call, so it matches the current mode
        # (user preference, falling back to the OS's light/dark setting).
        pg.setConfigOptions(antialias=True, background=t.SURFACE, foreground=t.TEXT_PRIMARY)
        app.setStyleSheet(_build_qss())

    apply_theme()
    # Re-polish the global stylesheet when the user switches mode from the
    # Settings tab, or the OS appearance changes while on "System".
    t.signal.changed.connect(apply_theme)
    app.styleHints().colorSchemeChanged.connect(lambda _: apply_theme())

    state = AppState()
    window = MainWindow(state)
    window.show()

    sys.exit(app.exec())
