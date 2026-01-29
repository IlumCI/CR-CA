# Installing DeepSpeed on WSL2 Ubuntu 24.04

## Prerequisites

- Python 3.12+ (already installed)
- PyTorch with CUDA support (already installed: PyTorch 2.10.0+cu128)
- CUDA 12.8+ (already available)

## Installation Steps

### Option 1: Using the Installation Script (Recommended)

```bash
# Navigate to project directory
cd /mnt/c/Users/ilum/Documents/Work/agents/CR-CA

# Run the installation script
sudo bash install_deepspeed.sh
```

### Option 2: Manual Installation

```bash
# 1. Update package lists
sudo apt-get update

# 2. Install build dependencies
sudo apt-get install -y build-essential python3-dev

# 3. Upgrade pip
python3 -m pip install --upgrade pip setuptools wheel

# 4. Install DeepSpeed
python3 -m pip install deepspeed --upgrade

# 5. Verify installation
python3 -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"
```

### Option 3: Install with Specific CUDA Version

If you need to match your PyTorch CUDA version (12.8):

```bash
# Install DeepSpeed with CUDA 12.x support
DS_BUILD_OPS=0 python3 -m pip install deepspeed --upgrade
```

The `DS_BUILD_OPS=0` flag prevents DeepSpeed from trying to build custom CUDA operations, which can be problematic in WSL2.

## Troubleshooting

### Issue: "No package metadata was found for deepspeed"

This means DeepSpeed isn't installed. Run the installation steps above.

### Issue: Build errors during installation

If you encounter build errors:

```bash
# Disable CUDA operation building (uses PyTorch's CUDA ops instead)
export DS_BUILD_OPS=0
python3 -m pip install deepspeed --upgrade
```

### Issue: CUDA not detected

Make sure:
1. NVIDIA drivers are installed in Windows
2. WSL2 has access to GPU (check with `nvidia-smi` in WSL2)
3. PyTorch can see CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Issue: Permission errors

Use `python3 -m pip` instead of `pip3`, or install in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install deepspeed --upgrade
```

## Verification

After installation, verify DeepSpeed works:

```bash
python3 -c "import deepspeed; print('DeepSpeed version:', deepspeed.__version__)"
```

You should see output like: `DeepSpeed version: 0.x.x`

## Using DeepSpeed in Your Code

Once installed, your training script should work. The code in `training/finetune.py` will automatically use DeepSpeed if:
1. DeepSpeed is installed
2. A DeepSpeed config file is provided (e.g., `training/deepspeed_zero3_offload.json`)

## MoE (Switch/Flan-MoE) Training

Switch/Flan-MoE models are **Seq2Seq (encoder-decoder)**. The training pipeline now supports a dual path:
- **Causal LM** (Qwen/Qwen2.5 and similar)
- **Seq2Seq** (Switch/Flan-MoE)

Example command for Switch MoE:

```bash
deepspeed --num_gpus=3 scripts/run_full_finetune.py \
  --model-size 1.5b \
  --model-id google/switch-base-8 \
  --train-file training_data/react_train.jsonl \
  --output-dir lrm_switch_base_8_full_finetune \
  --deepspeed-config training/deepspeed_zero3_h100_3gpu.json
```

## CRCA MoE Model Selection

To have CRCA use a MoE model for LLM orchestration, set:

```bash
export CRCA_MOE_MODEL=google/switch-base-8
```

If `CRCA_MOE_MODEL` is set, it overrides the default LLM model. You can also set `CRCA_LLM_MODEL` for a non-MoE override.
