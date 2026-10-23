#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

export DEVELOPER_DIR="/Library/Developer/CommandLineTools"

if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    "$DEVELOPER_DIR/usr/bin/python3" -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

echo "Starting ProjectForge web application on http://127.0.0.1:5001 ..."
.venv/bin/python app.py
