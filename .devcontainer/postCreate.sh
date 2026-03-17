#!/usr/bin/env bash
set -euo pipefail
cd /home/dev/src

# Marks this mounted repo from host as safe so Git can be used 
git config --global --add safe.directory /home/dev/src

sudo mkdir -p /home/vscode/.cache/uv
sudo chown -R vscode:vscode /home/vscode/.cache/uv

# Sync project deps from uv.lock (creates .venv automatically)
uv sync --frozen

# Install *this repo* as a package (src-layout)
# uv pip install -e .

# NLTK punkt required for chunking
# uv run python -c "import nltk; nltk.download('punkt_tab')"

echo "✅ Devcontainer ready."