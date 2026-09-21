"""Design tokens shared across the GUI.

Neutral-first palette with a single accent, used only for primary actions
and active/selected state. Spacing, radius and type are a fixed scale,
independent of light/dark mode. Color tokens (BG_APP, SURFACE, ACCENT, ...)
resolve dynamically to the current mode via module-level ``__getattr__`` —
callers just write ``theme.ACCENT`` and get the right value, without
threading a palette object through every call site.

Every interactive-state color (hover/pressed/disabled/read-only) must come
from the palette, never a raw neutral-scale constant directly — a raw
``GRAY_100`` is fixed regardless of mode and will render as a light patch
in dark mode (and vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from PyQt6.QtCore import QObject, QSettings, pyqtSignal

# ── Neutral scale (palette-independent building blocks) ─────────────────
GRAY_0   = "#FFFFFF"
GRAY_50  = "#FAFAFA"
GRAY_100 = "#F4F4F5"
GRAY_200 = "#E4E4E7"
GRAY_300 = "#D4D4D8"
GRAY_400 = "#A1A1AA"
GRAY_500 = "#71717A"
GRAY_600 = "#52525B"
GRAY_700 = "#3F3F46"
GRAY_800 = "#27272A"
GRAY_900 = "#18181B"

# ── Type scale (px) ────────────────────────────────────────────────────
TEXT_XS   = 12
TEXT_SM   = 13
TEXT_MD   = 14
TEXT_LG   = 16
TEXT_XL   = 20
TEXT_XXL  = 24

# ── Spacing scale (px) ─────────────────────────────────────────────────
SP_1 = 4
SP_2 = 8
SP_3 = 12
SP_4 = 16
SP_5 = 24
SP_6 = 32
SP_7 = 48

# ── Radius ─────────────────────────────────────────────────────────────
RADIUS_CONTROL = 6
RADIUS_PANEL   = 8


@dataclass(frozen=True)
class _Palette:
    BG_APP: str
    SURFACE: str
    SURFACE_HOVER: str    # button hover / raised state
    SURFACE_PRESSED: str  # button pressed state
    SURFACE_MUTED: str    # disabled controls, read-only fields
    BORDER: str
    BORDER_HOVER: str
    TEXT_PRIMARY: str
    TEXT_SECONDARY: str
    TEXT_MUTED: str
    TEXT_ON_ACCENT: str
    ACCENT: str
    ACCENT_HOVER: str
    ACCENT_ACTIVE: str
    ACCENT_SUBTLE: str
    SUCCESS: str
    DANGER: str
    DANGER_HOVER: str
    DANGER_SUBTLE: str
    DETECTED_COLOR: str   # spectrogram overlay legend, not chrome
    REFERENCE_COLOR: str  # spectrogram overlay legend, not chrome
    CURVE: str            # waveform line


LIGHT = _Palette(
    BG_APP=GRAY_50, SURFACE=GRAY_0,
    SURFACE_HOVER=GRAY_100, SURFACE_PRESSED=GRAY_200, SURFACE_MUTED=GRAY_50,
    BORDER=GRAY_200, BORDER_HOVER=GRAY_300,
    TEXT_PRIMARY=GRAY_900, TEXT_SECONDARY=GRAY_500, TEXT_MUTED=GRAY_400,
    TEXT_ON_ACCENT=GRAY_0,
    ACCENT="#0E7C86", ACCENT_HOVER="#0B6871", ACCENT_ACTIVE="#095860",
    ACCENT_SUBTLE="#E6F1F2",
    SUCCESS="#15803D", DANGER="#B42318", DANGER_HOVER="#912016",
    DANGER_SUBTLE="#FBEAE8",
    DETECTED_COLOR="#0E7C86", REFERENCE_COLOR=GRAY_600, CURVE=GRAY_700,
)

DARK = _Palette(
    BG_APP="#1C1C1F", SURFACE="#242428",
    SURFACE_HOVER="#2C2C31", SURFACE_PRESSED="#343439", SURFACE_MUTED="#1F1F22",
    BORDER="#38383D", BORDER_HOVER="#4A4A50",
    TEXT_PRIMARY="#F2F2F3", TEXT_SECONDARY="#A0A0A6", TEXT_MUTED="#6E6E76",
    TEXT_ON_ACCENT="#0B1416",
    ACCENT="#3FC4CC", ACCENT_HOVER="#5ED0D6", ACCENT_ACTIVE="#2FA8AF",
    ACCENT_SUBTLE="#1B3A3C",
    SUCCESS="#3FBF6B", DANGER="#F0645A", DANGER_HOVER="#F58880",
    DANGER_SUBTLE="#3A211F",
    DETECTED_COLOR="#3FC4CC", REFERENCE_COLOR="#9A9AA2", CURVE="#B8B8BE",
)

_PALETTE_FIELDS = {f.name for f in fields(_Palette)}

_MODES = ("system", "light", "dark")
_SETTINGS_KEY = "appearance/mode"


class _ThemeSignal(QObject):
    changed = pyqtSignal()


signal = _ThemeSignal()  # emitted whenever the effective mode changes


def get_mode() -> str:
    """Persisted user preference: 'system' (default), 'light' or 'dark'."""
    value = QSettings().value(_SETTINGS_KEY, "system")
    return value if value in _MODES else "system"


def set_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"invalid mode {mode!r}, expected one of {_MODES}")
    QSettings().setValue(_SETTINGS_KEY, mode)
    signal.changed.emit()


def is_dark() -> bool:
    """Whether the effective mode is dark: the user's explicit choice, or
    the OS setting when the preference is 'system' (best-effort; False
    before a QApplication exists or on Qt < 6.5, where colorScheme() is
    unavailable).
    """
    mode = get_mode()
    if mode == "light":
        return False
    if mode == "dark":
        return True
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:
        pass
    return False


def current_palette() -> _Palette:
    return DARK if is_dark() else LIGHT


def __getattr__(name: str) -> str:
    if name in _PALETTE_FIELDS:
        return getattr(current_palette(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
