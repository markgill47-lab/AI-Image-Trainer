#!/bin/bash
# setup_olympus.sh - Setup script for Olympus lab machines (Ubuntu 24.04, RTX 4090)
# Creates a separate Python venv for LoRA training (does not interfere with ComfyUI)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  AI Image Trainer - Olympus Lab Setup"
echo "========================================"
echo ""

# 1. Check Python 3.10+
echo "[1/6] Checking Python..."
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10+ not found. Install with: sudo apt install python3.10"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  Found: $PYTHON_VERSION"

# 2. Check system dependencies for PyQt6
echo ""
echo "[2/6] Checking system dependencies..."
MISSING_DEPS=""
for pkg in libxcb-xinerama0 libxcb-cursor0 libgl1-mesa-glx libegl1; do
    if ! dpkg -s "$pkg" &>/dev/null 2>&1; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "  Installing missing packages:$MISSING_DEPS"
    sudo apt-get install -y $MISSING_DEPS
else
    echo "  All system dependencies present."
fi

# 3. Create venv
echo ""
echo "[3/6] Creating Python virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_CMD -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists, reusing."
fi
source .venv/bin/activate

# 4. Install PyTorch with CUDA 12.1
echo ""
echo "[4/6] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install requirements
echo ""
echo "[5/6] Installing Python dependencies..."
pip install -r requirements.txt

# 6. Clone ai-toolkit and apply local patches
echo ""
echo "[6/6] Setting up ai-toolkit..."
if [ ! -d "ai-toolkit" ]; then
    echo "  Cloning ai-toolkit..."
    git clone https://github.com/ostris/ai-toolkit.git ai-toolkit
else
    echo "  ai-toolkit exists, skipping clone."
fi

# Apply local patches (Klein model support + fixes)
if [ -d "ai-toolkit-patches" ]; then
    echo "  Applying local patches..."
    cp -rv ai-toolkit-patches/. ai-toolkit/
    echo "  Patches applied."
fi

# Create launcher script
cat > start_gui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"
source .venv/bin/activate
export QT_QPA_PLATFORM=xcb
python main.py "$@"
EOF
chmod +x start_gui.sh

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start the trainer GUI:"
echo "  ./start_gui.sh"
echo ""
echo "First-run notes:"
echo "  - Set 'ComfyUI Models Base' in the GUI to point at your ComfyUI models/ dir"
echo "  - Click 'Auto-detect from ComfyUI' to find transformer and VAE files"
echo "  - The text encoder (~8-10GB) will download from HuggingFace on first training run"
echo "    and be cached in ~/.cache/huggingface/ for future runs"
echo ""
