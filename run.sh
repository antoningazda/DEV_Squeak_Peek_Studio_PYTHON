#!/usr/bin/env bash
# Launch Squeak Peek Studio with the correct Qt framework path for macOS 26+.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QT_LIB="$SCRIPT_DIR/.venv/lib/python3.11/site-packages/PyQt6/Qt6/lib"
export DYLD_FALLBACK_FRAMEWORK_PATH="$QT_LIB"
export QT_PLUGIN_PATH="$SCRIPT_DIR/.venv/lib/python3.11/site-packages/PyQt6/Qt6/plugins"
exec "$SCRIPT_DIR/.venv/bin/squeak-peek" "$@"
