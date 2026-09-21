"""
Squeak Peek Studio — PyQt6 main window and application entry point.
"""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QStatusBar, QTabWidget

from ._state import AppState
from ._tab_data_input import DataInputTab
from ._tab_detection import DetectionTab
from ._tab_info import InfoTab
from ._tab_label_edit import LabelEditTab
from ._tab_metrics import MetricsTab
from ._tab_settings import SettingsTab
from ._tab_visualization import VisualizationTab

# ── Light theme colours (mirror AppSettings defaults) ─────────────────────
_TEAL   = "#7DCED2"   # primary  (0.490, 0.808, 0.824)
_ORANGE = "#F9C06E"   # accent   (0.976, 0.753, 0.431)

_QSS = f"""
/* ── Base ────────────────────────────────────────────────────────────── */
QMainWindow, QDialog {{ background: white; }}
QWidget  {{ background: white; color: #111111; font-size: 13px; }}

/* ── Main tab bar — teal background, orange active tab ───────────────── */
QTabWidget#mainTabs QTabBar {{
    background: {_TEAL};
    padding: 4px 4px 0px 4px;
}}
QTabWidget#mainTabs QTabBar::tab {{
    background: white;
    color: #111111;
    padding: 7px 18px;
    margin: 3px 2px 0px 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-width: 72px;
}}
QTabWidget#mainTabs QTabBar::tab:selected {{
    background: {_ORANGE};
    font-weight: bold;
}}
QTabWidget#mainTabs QTabBar::tab:hover:!selected {{ background: #D8F4F6; }}
QTabWidget#mainTabs::pane {{ border: none; background: white; }}

/* ── Inner tab bars (Settings sub-tabs — named "innerTabs") ──────────── */
QTabWidget#innerTabs QTabBar::tab {{
    background: #EEEEEE; color: #111; padding: 5px 14px;
    margin: 1px; border-radius: 3px;
}}
QTabWidget#innerTabs QTabBar::tab:selected {{
    background: {_TEAL}; color: white;
}}

/* ── Group boxes ─────────────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid #D8D8D8; border-radius: 5px;
    margin-top: 8px; padding-top: 12px; background: white;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 4px; color: #666666;
}}

/* ── Buttons ─────────────────────────────────────────────────────────── */
QPushButton {{
    background: #F0F0F0; border: 1px solid #C8C8C8;
    border-radius: 5px; padding: 5px 14px; color: #111111;
}}
QPushButton:hover   {{ background: #E4E4E4; }}
QPushButton:pressed {{ background: #D0D0D0; }}

/* ── Input fields ────────────────────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: white; border: 1px solid #C8C8C8;
    border-radius: 4px; padding: 3px 6px; color: #111111;
}}

/* ── Misc ────────────────────────────────────────────────────────────── */
QStatusBar  {{ background: #F4F4F4; color: #555555; }}
QMenuBar    {{ background: white;   color: #111111; }}
QMenuBar::item:selected {{ background: #E8E8E8; }}
QMenu       {{ background: white;   color: #111111; }}
QMenu::item:selected {{ background: {_TEAL}; color: white; }}
QCheckBox, QRadioButton, QLabel {{ background: transparent; color: #111111; }}
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
        tabs.addTab(self._data_tab,           "Data Input")
        tabs.addTab(VisualizationTab(s),       "Visualization")
        tabs.addTab(DetectionTab(s),           "Detection")
        tabs.addTab(LabelEditTab(s),           "Label Edit")
        tabs.addTab(MetricsTab(s),             "Metrics")
        tabs.addTab(SettingsTab(s),            "Settings")
        tabs.addTab(InfoTab(),                 "Info")

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
    # White background for all pyqtgraph plots before any widget is created
    pg.setConfigOptions(antialias=True, background="w", foreground="k")

    app = QApplication(sys.argv)
    app.setApplicationName("Squeak Peek Studio")
    app.setOrganizationName("NUDZ")
    app.setStyleSheet(_QSS)

    state = AppState()
    window = MainWindow(state)
    window.show()

    sys.exit(app.exec())
