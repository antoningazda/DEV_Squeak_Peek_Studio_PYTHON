#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_dev.sh — One-shot development environment setup for Squeak Peek Studio
#
# Run once after cloning:
#   chmod +x setup_dev.sh && ./setup_dev.sh
# ---------------------------------------------------------------------------
set -e

echo "=== Squeak Peek Studio — Python Port : Dev Setup ==="
echo

# 1. Create virtual environment
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment (.venv)..."
    python3 -m venv .venv
else
    echo "→ .venv already exists, skipping creation."
fi

# 2. Activate
source .venv/bin/activate
echo "→ Activated: $(python --version)"

# 3. Install package in editable mode with dev extras
echo "→ Installing squeak-peek-studio[dev]..."
pip install --upgrade pip -q
pip install -e ".[dev]" -q

echo
echo "=== Setup complete ==="
echo
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo
echo "To run tests:"
echo "  pytest"
echo "  pytest --cov=squeak_peek      # with coverage"
echo "  pytest tests/unit/            # unit tests only"
echo
echo "To use the CLI:"
echo "  squeak-peek-cli --help"
