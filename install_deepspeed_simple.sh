#!/bin/bash
# Simple DeepSpeed installation for WSL2 Ubuntu 24.04 (system-wide)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Installing build dependencies ==="
sudo apt-get update
sudo apt-get install -y build-essential python3-dev

echo -e "\n=== Checking CUDA setup ==="
# Check if CUDA toolkit is installed
if ! command -v nvcc &> /dev/null; then
    echo "CUDA toolkit not found. Installing CUDA toolkit for WSL2..."
    
    # Use existing keyring file if available
    if [ -f "cuda-keyring_1.1-1_all.deb" ]; then
        echo "Using existing CUDA keyring file..."
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
    else
        echo "Downloading CUDA keyring..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
        sudo dpkg -i cuda-keyring_1.1-1_all.deb
    fi
    
    sudo apt-get update
    
    # Try to install CUDA toolkit 12.x (matching PyTorch's CUDA 12.8)
    echo "Installing CUDA toolkit 12.x..."
    if sudo apt-get install -y cuda-toolkit-12-6 2>/dev/null; then
        export CUDA_HOME=/usr/local/cuda-12.6
    elif sudo apt-get install -y cuda-toolkit-12-5 2>/dev/null; then
        export CUDA_HOME=/usr/local/cuda-12.5
    elif sudo apt-get install -y cuda-toolkit-12-4 2>/dev/null; then
        export CUDA_HOME=/usr/local/cuda-12.4
    elif sudo apt-get install -y cuda-toolkit-12-3 2>/dev/null; then
        export CUDA_HOME=/usr/local/cuda-12.3
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        echo "Warning: Could not install CUDA toolkit. DeepSpeed installation may fail."
        echo "You may need to install CUDA toolkit manually or use a different installation method."
    fi
    
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        echo "CUDA_HOME set to: $CUDA_HOME"
        echo "Add to ~/.bashrc:"
        echo "  export CUDA_HOME=$CUDA_HOME"
        echo "  export PATH=\$CUDA_HOME/bin:\$PATH"
        echo "  export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    fi
else
    echo "CUDA toolkit found at: $(which nvcc)"
    if [ -z "$CUDA_HOME" ]; then
        # Try to find CUDA_HOME
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        export CUDA_HOME
        echo "Setting CUDA_HOME to: $CUDA_HOME"
    fi
fi

echo -e "\n=== Installing DeepSpeed (system-wide) ==="
# Disable custom CUDA ops building for WSL2 compatibility
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    echo "Installing DeepSpeed with CUDA_HOME=$CUDA_HOME"
    DS_BUILD_OPS=0 python3 -m pip install deepspeed --upgrade --break-system-packages
else
    echo "Warning: CUDA_HOME not set. Attempting installation with workaround..."
    # Create a minimal CUDA_HOME structure to satisfy DeepSpeed's setup.py check
    TMP_CUDA=/tmp/cuda_home
    mkdir -p "$TMP_CUDA/bin" "$TMP_CUDA/include"
    export CUDA_HOME=$TMP_CUDA
    DS_BUILD_OPS=0 python3 -m pip install deepspeed --upgrade --break-system-packages || {
        echo "Installation failed. Please install CUDA toolkit first using install_cuda_wsl2.sh"
        exit 1
    }
fi

echo -e "\n=== Verifying DeepSpeed installation ==="
python3 -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"

echo -e "\n=== Installation complete! ==="
