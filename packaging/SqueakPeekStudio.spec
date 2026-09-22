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

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH)
SRC_DIR = SPEC_DIR.parent / "src"

sys.path.insert(0, str(SRC_DIR))
from squeak_peek import __version__ as APP_VERSION  # noqa: E402

# squeak_peek.detectors/.classifiers self-register by dynamically importing
# every module in their package (pkgutil.iter_modules) rather than via
# static `import` statements, so PyInstaller's static analysis can't see
# psd.py/bscd.py/rbd.py/ml.py/duration.py etc. and silently drops them from
# the build (only base.py gets bundled) unless listed here explicitly.
# collect_submodules re-walks each package's real directory at build time,
# so a new detector/classifier module is picked up automatically — nothing
# else needs updating when one is added.
plugin_hidden_imports = (
    collect_submodules("squeak_peek.detectors")
    + collect_submodules("squeak_peek.classifiers")
)

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=[],
    hiddenimports=plugin_hidden_imports,
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
