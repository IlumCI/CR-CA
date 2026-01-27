#!/bin/bash
# DeepSpeed installation script for WSL2 Ubuntu 24.04

set -e

VENV_DIR="${VENV_DIR:-venv}"
USE_VENV="${USE_VENV:-1}"

echo "=== Checking environment ==="
python3 -V || python3 --version
python3 -m pip --version || pip3 -V || pip3 --version

echo -e "\n=== Checking PyTorch installation ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "PyTorch not installed"

if [ "$USE_VENV" = "1" ]; then
    echo -e "\n=== Setting up virtual environment ==="
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi
    
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    echo -e "\n=== Upgrading pip in virtual environment ==="
    python3 -m pip install --upgrade pip setuptools wheel
else
    echo -e "\n=== Upgrading pip (system-wide with --break-system-packages) ==="
    python3 -m pip install --upgrade pip setuptools wheel --break-system-packages
fi

echo -e "\n=== Installing build dependencies ==="
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

echo -e "\n=== Installing DeepSpeed ==="
# Install DeepSpeed with CUDA support
# Disable custom CUDA ops building for WSL2 compatibility
if [ "$USE_VENV" = "1" ]; then
    DS_BUILD_OPS=0 python3 -m pip install deepspeed --upgrade
else
    DS_BUILD_OPS=0 python3 -m pip install deepspeed --upgrade --break-system-packages
fi

echo -e "\n=== Verifying DeepSpeed installation ==="
python3 -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"

if [ "$USE_VENV" = "1" ]; then
    echo -e "\n=== Virtual environment info ==="
    echo "Virtual environment created at: $VENV_DIR"
    echo "To activate it in the future, run: source $VENV_DIR/bin/activate"
    deactivate
fi

echo -e "\n=== Installation complete! ==="
