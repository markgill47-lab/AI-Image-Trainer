#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export QT_QPA_PLATFORM=xcb
python main.py "$@"
