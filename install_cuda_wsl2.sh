#!/bin/bash
# Install CUDA toolkit for WSL2 Ubuntu 24.04

set -e

echo "=== Installing CUDA toolkit for WSL2 ==="

# Check if CUDA keyring already exists
if [ -f "cuda-keyring_1.1-1_all.deb" ]; then
    echo "CUDA keyring file found, using existing..."
else
    echo "Downloading CUDA keyring..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
fi

echo "Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

echo "Installing CUDA toolkit 12.x..."
# Try to install the latest 12.x version available
sudo apt-get install -y cuda-toolkit-12-6 || \
sudo apt-get install -y cuda-toolkit-12-5 || \
sudo apt-get install -y cuda-toolkit-12-4 || \
sudo apt-get install -y cuda-toolkit-12-3 || \
sudo apt-get install -y cuda-toolkit-12-2 || \
sudo apt-get install -y cuda-toolkit-12-1 || \
echo "Failed to install CUDA toolkit. Please check available versions with: apt-cache search cuda-toolkit"

# Set CUDA_HOME
if [ -d "/usr/local/cuda-12.6" ]; then
    export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d "/usr/local/cuda-12.5" ]; then
    export CUDA_HOME=/usr/local/cuda-12.5
elif [ -d "/usr/local/cuda-12.4" ]; then
    export CUDA_HOME=/usr/local/cuda-12.4
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

if [ -n "$CUDA_HOME" ]; then
    echo "CUDA_HOME set to: $CUDA_HOME"
    echo ""
    echo "Add these to your ~/.bashrc:"
    echo "export CUDA_HOME=$CUDA_HOME"
    echo "export PATH=\$CUDA_HOME/bin:\$PATH"
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
    
    # Add to current session
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    echo "Warning: Could not determine CUDA_HOME"
fi

echo -e "\n=== Verifying CUDA installation ==="
nvcc --version || echo "nvcc not found in PATH (may need to restart shell or source ~/.bashrc)"

echo -e "\n=== CUDA toolkit installation complete! ==="
