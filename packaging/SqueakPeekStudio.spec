# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Squeak Peek Studio.

Build with:
    .venv/bin/pyinstaller packaging/SqueakPeekStudio.spec --noconfirm
Output:
    macOS:   dist/Squeak Peek Studio.app
    Windows: dist/SqueakPeekStudio/SqueakPeekStudio.exe
    Linux:   dist/SqueakPeekStudio/SqueakPeekStudio
"""

import sys
from pathlib import Path

SPEC_DIR = Path(SPECPATH)
SRC_DIR = SPEC_DIR.parent / "src"

sys.path.insert(0, str(SRC_DIR))
from squeak_peek import __version__ as APP_VERSION  # noqa: E402

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SqueakPeekStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="SqueakPeekStudio",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Squeak Peek Studio.app",
        icon=None,
        bundle_identifier="cz.nudz.squeakpeekstudio",
        info_plist={
            "CFBundleName": "Squeak Peek Studio",
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "NUDZ",
        },
    )
