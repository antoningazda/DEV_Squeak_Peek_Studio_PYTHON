"""Render the documentation screenshots in docs/assets/screenshots/.

Drives the real application offscreen against the bundled example
recording, so the images cannot drift from the UI the way hand-taken ones
do. Every tab is shot twice, light and dark, because the docs site follows
the reader's colour scheme (MkDocs Material swaps the pair through the
``#only-light`` / ``#only-dark`` URL fragment).

    QT_QPA_PLATFORM=offscreen python tools/make_docs_screenshots.py

Needs the GUI dependencies plus matplotlib (for the colormaps) and a
``settings/default.json`` pointing at ``data/example/single``. It writes to
its own QSettings domain, so it never touches your own theme or shortcuts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "assets" / "screenshots"
sys.path.insert(0, str(REPO / "src"))

from PyQt6.QtCore import QCoreApplication, QLocale, QSettings  # noqa: E402
from PyQt6.QtWidgets import QApplication, QStyleFactory, QTabWidget  # noqa: E402

# A QSettings domain of our own: the theme we force below must not land in
# the user's real preferences.
QCoreApplication.setOrganizationName("NUDZ-docs")
QCoreApplication.setApplicationName("SqueakPeekStudioDocs")
# English number formatting, so the screenshots never show a decimal comma
# in an English-language manual.
QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))

import pyqtgraph as pg  # noqa: E402

from squeak_peek.gui import _theme as t  # noqa: E402
from squeak_peek.gui._state import AppState  # noqa: E402
from squeak_peek.gui.app import MainWindow, _autoload_defaults, _build_qss  # noqa: E402

WIDTH = 1440
# Window height per shot: enough for the tab's content, no acre of empty space.
HEIGHTS = {
    "data-input": 520,
    "metrics": 480,
    "detection": 740,
    "settings-preprocessing": 620,
}
# A one-second window of the example recording holding six calls.
SEGMENT_START = 70.4
# A 121 ms complex call, for the single-call Label Edit view.
LABEL_EDIT_AT = 84.78


def build(app: QApplication, theme: str) -> tuple[MainWindow, AppState]:
    QSettings().setValue("appearance/mode", theme)
    t.signal.changed.emit()
    pg.setConfigOptions(antialias=True, background=t.SURFACE, foreground=t.TEXT_PRIMARY)
    app.setStyleSheet(_build_qss())
    state = AppState()
    win = MainWindow(state)
    win.resize(WIDTH, 900)
    win.show()
    app.processEvents()
    _autoload_defaults(state, win._data_tab)
    app.processEvents()
    if state.samples is None:
        raise SystemExit(
            "No recording loaded — check settings/default.json's DataInput paths."
        )
    return win, state


def shoot(app: QApplication, win: MainWindow, state: AppState, suffix: str) -> None:
    tabs = win._tabs

    def show_tab(name: str):
        tabs.setCurrentIndex(
            next(i for i in range(tabs.count()) if tabs.tabText(i) == name)
        )
        app.processEvents()
        return tabs.currentWidget()

    def shot(name: str) -> None:
        win.resize(WIDTH, HEIGHTS.get(name, 900))
        for _ in range(5):  # let the re-layout and any re-render settle
            app.processEvents()
        win.grab().save(str(OUT / f"{name}{suffix}.png"))
        print("wrote", name + suffix)

    def inner_tabs(widget) -> QTabWidget:
        return widget.findChildren(QTabWidget)[0]

    show_tab("Data Input")
    shot("data-input")

    vis = show_tab("Visualization")
    vis._start_spin.setValue(SEGMENT_START)
    vis._len_spin.setValue(1.0)
    app.processEvents()
    shot("visualization")

    det = show_tab("Detection")
    sub = inner_tabs(det)
    sub.setCurrentIndex(0)
    shot("detection")
    sub.setCurrentIndex(1)
    det._training_page._add_loaded()
    app.processEvents()
    shot("detection-train")
    sub.setCurrentIndex(0)

    cls = show_tab("Classification")
    inner_tabs(cls).setCurrentIndex(0)
    cls._cls_add_loaded()
    app.processEvents()
    shot("classification")

    # Partway through a review pass, so the tallies are not all zero.
    le = show_tab("Label Edit")
    target = next(
        i for i, lab in enumerate(state.detected_labels) if lab.start_time > LABEL_EDIT_AT
    )
    for i, lab in enumerate(state.detected_labels[:target]):
        reject = i % 12 == 0
        lab.detection_state = "Rejected" if reject else "Accepted"
        lab.classification_state = "Rejected" if reject or i % 9 == 0 else "Accepted"
    le._idx = target
    le._show_current()
    le._update_counters()
    shot("label-edit")

    met = show_tab("Metrics")
    met._compute()
    shot("metrics")

    setg = show_tab("Settings")
    nav = setg._nav
    titles = [nav.item(i).text() for i in range(nav.count())]
    for section, name, scroll in [
        # Scrolled far enough to show the manual-label fields.
        ("Visualization", "settings-visualization", 430),
        ("Pre-processing", "settings-preprocessing", 0),
        ("BSCD detector", "settings-bscd", 0),
        ("Shortcuts", "settings-shortcuts", 0),
    ]:
        nav.setCurrentRow(titles.index(section))
        app.processEvents()
        page = setg._pages.currentWidget()
        if hasattr(page, "verticalScrollBar"):
            page.verticalScrollBar().setValue(scroll)
        app.processEvents()
        shot(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)  # settings/default.json uses repo-relative data paths
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        win, state = build(app, theme)
        shoot(app, win, state, suffix)
        win.close()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
