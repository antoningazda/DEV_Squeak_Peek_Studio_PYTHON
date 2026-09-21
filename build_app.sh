#!/usr/bin/env bash
# Build Squeak Peek Studio as a standalone macOS .app bundle (PyInstaller).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "→ .venv not found, run ./setup_dev.sh first" >&2
    exit 1
fi

.venv/bin/pip show pyinstaller > /dev/null 2>&1 || .venv/bin/pip install -q "pyinstaller>=6.0"

rm -rf build dist
.venv/bin/pyinstaller packaging/SqueakPeekStudio.spec --noconfirm

# PyInstaller's own ad-hoc signing step fails here because macOS tags copied
# files with a com.apple.provenance xattr ("resource fork ... not allowed").
# Strip xattrs and re-sign manually.
xattr -cr "dist/Squeak Peek Studio.app"
codesign -s - --force --deep --timestamp "dist/Squeak Peek Studio.app"

# ── DMG installer (drag-to-Applications, like any other Mac app) ───────────
DMG_STAGE="$(mktemp -d)"
cp -R "dist/Squeak Peek Studio.app" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"
rm -f "dist/Squeak Peek Studio.dmg"
hdiutil create -volname "Squeak Peek Studio" -srcfolder "$DMG_STAGE" -ov -format UDZO "dist/Squeak Peek Studio.dmg"
rm -rf "$DMG_STAGE"

echo
echo "=== Build complete ==="
echo "App bundle: dist/Squeak Peek Studio.app"
echo "Installer:  dist/Squeak Peek Studio.dmg"
echo "Launch with: open 'dist/Squeak Peek Studio.app'"
