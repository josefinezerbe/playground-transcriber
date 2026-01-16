#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-transcriber-env}
REQ=${REQ:-requirements.txt}

# Create env if missing
if [ ! -d "$VENV_DIR" ]; then
  "$PY" -m venv "$VENV_DIR"
fi

# Upgrade pip and install
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel
pip install -r "$REQ"

echo
echo "Done."
echo "Activate your env with: source $VENV_DIR/bin/activate"
echo "Then run: vid2txt path/to/video.mp4"