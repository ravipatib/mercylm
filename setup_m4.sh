#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Mercy LLM — Apple Silicon setup script
# Run once from inside the project folder:  bash setup_m4.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo ""
echo "  Mercy: The Only Human Left with Pigeon Gerald"
echo "  Apple Silicon setup"
echo ""

# ── 1. Check we're on Apple Silicon ──────────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "ERROR: This script is for Apple Silicon (arm64). Detected: $ARCH"
    echo "If you're running under Rosetta, open a native terminal instead."
    exit 1
fi
echo "✓ Apple Silicon confirmed ($ARCH)"

# ── 2. Check macOS version (needs 12.3+ for MPS) ─────────────────────────────
MACOS=$(sw_vers -productVersion)
MACOS_MAJOR=$(echo $MACOS | cut -d. -f1)
MACOS_MINOR=$(echo $MACOS | cut -d. -f2)
echo "✓ macOS $MACOS"

if [ "$MACOS_MAJOR" -lt 12 ] || ([ "$MACOS_MAJOR" -eq 12 ] && [ "$MACOS_MINOR" -lt 3 ]); then
    echo "WARNING: MPS requires macOS 12.3+. Please update your system."
fi

if [ "$MACOS_MAJOR" -ge 15 ]; then
    echo "✓ macOS 15+ — MPS non-contiguous tensor bug is fixed on this version"
else
    echo "  Note: macOS 15+ fixes a known MPS AdamW bug. Updating is recommended."
    echo "  Our scripts work around this bug regardless."
fi

# ── 3. Check Python ───────────────────────────────────────────────────────────
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if command -v $cmd &>/dev/null; then
        VER=$($cmd --version 2>&1 | cut -d' ' -f2)
        MAJOR=$(echo $VER | cut -d. -f1)
        MINOR=$(echo $VER | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo ""
    echo "ERROR: Python 3.10+ not found."
    echo "Install with pyenv:"
    echo "  brew install pyenv"
    echo "  pyenv install 3.12.11"
    echo "  pyenv local 3.12.11"
    exit 1
fi

echo "✓ Python $($PYTHON_CMD --version 2>&1 | cut -d' ' -f2) ($PYTHON_CMD)"

# Check it's native arm64, not Rosetta
PYTHON_ARCH=$($PYTHON_CMD -c "import platform; print(platform.machine())")
if [ "$PYTHON_ARCH" != "arm64" ]; then
    echo ""
    echo "ERROR: Python is running under Rosetta ($PYTHON_ARCH), not native arm64."
    echo "MPS will not be available. Install a native arm64 Python:"
    echo "  brew install pyenv"
    echo "  pyenv install 3.12.11"
    exit 1
fi
echo "✓ Python is native arm64"

# ── 4. Create virtual environment ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
    echo "✓ Virtual environment created at .venv/"
else
    echo "✓ Virtual environment already exists"
fi

source .venv/bin/activate

# ── 5. Upgrade pip ────────────────────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── 6. Install PyTorch ────────────────────────────────────────────────────────
echo ""
echo "Installing PyTorch (stable, MPS included for Apple Silicon)..."
# Standard pip install includes MPS on macOS arm64 — no special index needed
pip install torch torchvision torchaudio --quiet
echo "✓ PyTorch installed"

# ── 7. Install project dependencies ───────────────────────────────────────────
echo "Installing project dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"

# ── 8. Verify MPS ─────────────────────────────────────────────────────────────
echo ""
echo "Verifying MPS..."
python3 -c "
import torch
print(f'  PyTorch version:  {torch.__version__}')
print(f'  MPS built:        {torch.backends.mps.is_built()}')
print(f'  MPS available:    {torch.backends.mps.is_available()}')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    print(f'  MPS test tensor:  OK ({y.device})')
    print()
    print('  MPS is working correctly.')
else:
    print()
    print('  WARNING: MPS not available. Training will use CPU.')
    print('  Check: macOS 12.3+, native arm64 Python, PyTorch 2.0+')
"

# ── 9. Verify model ───────────────────────────────────────────────────────────
echo ""
echo "Verifying Mercy model..."
python3 -c "
import sys; sys.path.insert(0, '.')
from tlha.config import TLHAConfig
from tlha.model import MercyLLM
cfg = TLHAConfig()
model = MercyLLM(cfg.model)
print(f'  Character:    {cfg.character_name}')
print(f'  Parameters:   {model.num_parameters():,}')
print(f'  Layers:       {cfg.model.num_layers}')
print(f'  Embed dim:    {cfg.model.embed_dim}')
print(f'  Batch size:   {cfg.train.batch_size}')
print(f'  Epochs:       {cfg.train.num_epochs}')
print()
print('  Model OK.')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "─────────────────────────────────────────────────────────"
echo "  Setup complete. Run these three commands:"
echo ""
echo "  source .venv/bin/activate"
echo "  python -m tlha prepare    # ~1 min"
echo "  python -m tlha train      # ~15-20 min"
echo "  python -m tlha chat       # talk to Mercy"
echo "─────────────────────────────────────────────────────────"
echo ""
