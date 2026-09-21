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

echo
echo "=== Build complete ==="
echo "App bundle: dist/Squeak Peek Studio.app"
echo "Launch with: open 'dist/Squeak Peek Studio.app'"
