"""Keyboard shortcut registry.

Mirrors the pattern in ``_theme.py``: a fixed set of ids with shipped
defaults, user overrides persisted via ``QSettings`` (survive restarts,
independent of the JSON settings file), and a signal so open windows can
react when the Settings tab changes a binding.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, QSettings, pyqtSignal
from PyQt6.QtGui import QKeySequence

_SETTINGS_PREFIX = "shortcuts/"


@dataclass(frozen=True)
class ShortcutSpec:
    id: str
    label: str
    description: str
    default: str  # QKeySequence-parseable string, e.g. "Ctrl+O"
    context: str  # where it's active — for display only


SHORTCUTS: list[ShortcutSpec] = [
    ShortcutSpec(
        "open_wav", "Open WAV…",
        "Open a WAV file for analysis.",
        "Ctrl+O", "File menu",
    ),
    ShortcutSpec(
        "quit", "Quit",
        "Close Squeak Peek Studio.",
        "Ctrl+Q", "File menu",
    ),
    ShortcutSpec(
        "prev_segment", "Previous segment / label",
        "Step to the previous visualization segment, or the previous label in Label Edit.",
        "Left", "Visualization / Label Edit",
    ),
    ShortcutSpec(
        "next_segment", "Next segment / label",
        "Step to the next visualization segment, or the next label in Label Edit.",
        "Right", "Visualization / Label Edit",
    ),
    ShortcutSpec(
        "accept_detection", "Accept detection",
        "Mark the current label's detection as accepted.",
        "D", "Label Edit",
    ),
    ShortcutSpec(
        "reject_detection", "Reject detection",
        "Mark the current label's detection as rejected.",
        "Shift+D", "Label Edit",
    ),
    ShortcutSpec(
        "accept_classification", "Accept classification",
        "Mark the current label's call-type classification as accepted.",
        "C", "Label Edit",
    ),
    ShortcutSpec(
        "reject_classification", "Reject classification",
        "Mark the current label's call-type classification as rejected.",
        "Shift+C", "Label Edit",
    ),
    ShortcutSpec(
        "accept_both_advance", "Accept both & advance",
        "Accept both detection and classification for the current label, then move to the next one.",
        "Space", "Label Edit",
    ),
    ShortcutSpec(
        "undo_label_edit", "Undo",
        "Undo the last label edit (accept/reject, drag-resize, manual create).",
        "Ctrl+Z", "Label Edit / Visualization",
    ),
    ShortcutSpec(
        "redo_label_edit", "Redo",
        "Redo the last undone label edit.",
        "Ctrl+Shift+Z", "Label Edit / Visualization",
    ),
]

_BY_ID = {s.id: s for s in SHORTCUTS}


class _ShortcutSignal(QObject):
    changed = pyqtSignal()


signal = _ShortcutSignal()  # emitted whenever any binding changes


def spec(action_id: str) -> ShortcutSpec:
    return _BY_ID[action_id]


def label_for(action_id: str) -> str:
    return _BY_ID[action_id].label


def get_shortcut(action_id: str) -> QKeySequence:
    default = _BY_ID[action_id].default
    value = QSettings().value(_SETTINGS_PREFIX + action_id, default)
    seq = QKeySequence(value)
    return seq if not seq.isEmpty() else QKeySequence(default)


def set_shortcut(action_id: str, sequence: QKeySequence) -> None:
    QSettings().setValue(_SETTINGS_PREFIX + action_id, sequence.toString())
    signal.changed.emit()


def reset_shortcut(action_id: str) -> None:
    QSettings().remove(_SETTINGS_PREFIX + action_id)
    signal.changed.emit()


def reset_all() -> None:
    for s in SHORTCUTS:
        QSettings().remove(_SETTINGS_PREFIX + s.id)
    signal.changed.emit()


def matches(action_id: str, sequence: QKeySequence) -> bool:
    return get_shortcut(action_id).matches(sequence) == QKeySequence.SequenceMatch.ExactMatch


def conflicts_with(action_id: str, sequence: QKeySequence) -> list[str]:
    """Ids of other actions currently bound to the same key sequence."""
    if sequence.isEmpty():
        return []
    return [
        other.id for other in SHORTCUTS
        if other.id != action_id
        and get_shortcut(other.id).matches(sequence) == QKeySequence.SequenceMatch.ExactMatch
    ]
