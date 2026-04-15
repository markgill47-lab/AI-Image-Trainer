#!/bin/bash
# install_olympus.sh — One-shot installer for Olympus lab machines
# Run this on Apollo/Ares to clone, set up venv, install deps, and apply patches.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/markgill47-lab/AI-Image-Trainer/master/install_olympus.sh | bash
#   — or —
#   wget -qO- https://raw.githubusercontent.com/markgill47-lab/AI-Image-Trainer/master/install_olympus.sh | bash

set -e

REPO="https://github.com/markgill47-lab/AI-Image-Trainer.git"
INSTALL_DIR="$HOME/AI-Image-Trainer"

echo "========================================"
echo "  AI Image Trainer — Olympus Installer"
echo "========================================"
echo ""

# ── 1. Clone the repo ──────────────────────────────────────────────
echo "[1/7] Cloning repository..."
if [ -d "$INSTALL_DIR" ]; then
    echo "  $INSTALL_DIR already exists. Pulling latest..."
    cd "$INSTALL_DIR"
    git pull
else
    git clone "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi
echo ""

# ── 2. Check Python ────────────────────────────────────────────────
echo "[2/7] Checking Python..."
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$($cmd -c 'import sys; print(sys.version_info[:2])')
        PYTHON_CMD="$cmd"
        break
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10+ required. Install with:"
    echo "  sudo apt install python3.10 python3.10-venv"
    exit 1
fi
echo "  Using: $($PYTHON_CMD --version)"
echo ""

# ── 3. System dependencies for PyQt6 ──────────────────────────────
echo "[3/7] Checking system packages..."
MISSING=""
for pkg in libxcb-xinerama0 libxcb-cursor0 libgl1 libegl1 python3-venv; do
    dpkg -s "$pkg" &>/dev/null 2>&1 || MISSING="$MISSING $pkg"
done
if [ -n "$MISSING" ]; then
    echo "  Installing:$MISSING"
    sudo apt-get update -qq && sudo apt-get install -y -qq $MISSING
else
    echo "  All present."
fi
echo ""

# ── 4. Create venv ─────────────────────────────────────────────────
echo "[4/7] Creating Python venv..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv exists, reusing."
fi
source .venv/bin/activate
pip install --upgrade pip -q
echo ""

# ── 5. Install PyTorch + dependencies ─────────────────────────────
echo "[5/7] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

echo "  Installing Python dependencies..."
pip install -r requirements.txt -q
echo ""

# ── 6. Clone ai-toolkit and apply patches ─────────────────────────
echo "[6/7] Setting up ai-toolkit..."
if [ ! -d "ai-toolkit" ]; then
    echo "  Cloning ai-toolkit..."
    git clone --depth 1 https://github.com/ostris/ai-toolkit.git ai-toolkit
else
    echo "  ai-toolkit exists, skipping clone."
fi

if [ -d "ai-toolkit-patches" ]; then
    echo "  Applying Klein patches..."
    cp -r ai-toolkit-patches/. ai-toolkit/
    echo "  Done."
fi
echo ""

# ── 7. Make launcher executable ────────────────────────────────────
echo "[7/7] Finalizing..."
chmod +x start_gui.sh setup_olympus.sh 2>/dev/null || true

echo ""
echo "========================================"
echo "  Installation complete!"
echo "========================================"
echo ""
echo "  Location: $INSTALL_DIR"
echo ""
echo "  To launch the trainer GUI:"
echo "    cd $INSTALL_DIR"
echo "    ./start_gui.sh"
echo ""
echo "  First-run setup:"
echo "    1. Set ComfyUI Models Base to your ComfyUI/models/ path"
echo "    2. Click 'Auto-detect from ComfyUI'"
echo "    3. The text encoder (~8-10GB) downloads from HuggingFace"
echo "       on first training run (cached for future runs)"
echo ""
