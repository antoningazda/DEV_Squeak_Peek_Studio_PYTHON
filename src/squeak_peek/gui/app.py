"""
Squeak Peek Studio — PyQt6 main window and application entry point.
"""

from __future__ import annotations

import sys

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
    app = QApplication(sys.argv)
    app.setApplicationName("Squeak Peek Studio")
    app.setOrganizationName("NUDZ")

    state = AppState()
    window = MainWindow(state)
    window.show()

    sys.exit(app.exec())
