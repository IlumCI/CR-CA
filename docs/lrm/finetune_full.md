---
title: Full Finetune (Qwen2.5-0.5B)
---

# Full Finetune (Qwen2.5-0.5B-Instruct)

This guide targets finetuning the **Qwen2.5-0.5B-Instruct** model using **DeepSpeed ZeRO-3 CPU offload** and a **4k context length**. The 0.5B model is optimized for faster training with larger batch sizes and higher learning rates compared to larger models.

## Requirements
- Python deps: `transformers`, `datasets`, `accelerate`, `deepspeed`
- GPU driver + CUDA compatible with your PyTorch build
- Fast SSD/NVMe recommended (CPU offload is heavy on disk I/O)

## 1) Build the dataset (hybrid)
```bash
python scripts/build_lrm_dataset.py \
  --trace-jsonl path/to/traces.jsonl \
  --output training_data/react_train.jsonl
```

Optional: provide a public dataset config JSON:
```bash
python scripts/build_lrm_dataset.py \
  --trace-jsonl path/to/traces.jsonl \
  --public-config configs/public_datasets.json \
  --output training_data/react_train.jsonl
```

## 2) Run full finetune
```bash
python scripts/run_full_finetune.py \
  --train-file training_data/react_train.jsonl \
  --output-dir lrm_qwen25_0_5b_full_finetune
```

Notes:
- Uses `training/deepspeed_zero3_0_5b.json` (optimized for 0.5B model)
- Defaults to micro-batch 8 with gradient accumulation of 4
- Higher learning rate (3e-4) optimized for smaller model size
- Larger batch sizes possible due to smaller model memory footprint

### Hyperparameter Comparison

| Parameter | 0.5B Model (Default) | 7B Model (Reference) |
|-----------|----------------------|----------------------|
| per_device_batch_size | 8 | 1 |
| gradient_accumulation_steps | 4 | 32 |
| learning_rate | 3e-4 | 1e-5 |
| max_seq_length | 4096 | 4096 |
| DeepSpeed config | `deepspeed_zero3_0_5b.json` | `deepspeed_zero3_offload.json` |

The 0.5B model configuration uses:
- **Larger batch sizes**: The model is ~14x smaller, allowing for batch size 8 instead of 1
- **Higher learning rates**: Smaller models can handle higher learning rates (3e-4 vs 1e-5)
- **Reduced gradient accumulation**: Larger batch size means less accumulation needed
- **Optimized DeepSpeed buckets**: Smaller bucket sizes tuned for the 0.5B model

## 3) Evaluate traces
```bash
python scripts/run_lrm_eval.py \
  --trace path/to/trace.jsonl \
  --output eval_results/lrm_eval.json
```

## 4) Export to HuggingFace
```bash
python scripts/export_hf_lrm.py \
  --checkpoint-dir lrm_qwen25_0_5b_full_finetune \
  --output-dir hf_export/crca_lrm_0_5b
```

## 5) Export to Ollama (GGUF)
```bash
python scripts/export_ollama_gguf.py \
  --checkpoint-dir hf_export/crca_lrm \
  --output ollama/crca_lrm.gguf \
  --llama-cpp-dir /path/to/llama.cpp
```

Then update `ollama/Modelfile` and run:
```bash
ollama create crca-lrm -f ollama/Modelfile
```
