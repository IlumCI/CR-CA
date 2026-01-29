"""Run full finetune for Qwen2.5 models with CRCA-optimized configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.finetune import (
    FinetuneConfig,
    full_finetune_qwen25_0_5b_config,
    full_finetune_qwen25_0_5b_config_cloud,
    full_finetune_qwen25_1_5b_config,
    full_finetune_qwen25_7b_config,
    full_finetune_qwen25_14b_config,
    run_finetune,
)


def _infer_model_key(base_model: str) -> str:
    model = base_model.lower()
    if "14b" in model:
        return "14b"
    if "7b" in model:
        return "7b"
    if "1.5b" in model:
        return "1.5b"
    if "0.5b" in model:
        return "0.5b"
    return "unknown"


def _apply_auto_config(cfg: FinetuneConfig, args: argparse.Namespace) -> None:
    if getattr(args, "no_auto_config", False):
        return
    try:
        import torch
    except Exception:
        return

    if not torch.cuda.is_available():
        return

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()

    if bf16_supported:
        cfg.bf16 = True
        cfg.fp16 = False

    # DeepSpeed config selection for multi-GPU NVIDIA setups
    if device_count > 1 and args.deepspeed_config is None:
        if total_mem_gb >= 60:
            cfg.deepspeed_config = str((REPO_ROOT / "training" / "deepspeed_zero3_h100_3gpu.json").resolve())
        else:
            cfg.deepspeed_config = str((REPO_ROOT / "training" / "deepspeed_zero3_offload.json").resolve())

    # Batch/grad/seq tuning for high-memory NVIDIA GPUs
    model_key = _infer_model_key(cfg.base_model)
    if total_mem_gb >= 60 and model_key != "unknown":
        if model_key == "1.5b":
            cfg.per_device_batch_size = 16
            if args.grad_accum is None:
                cfg.gradient_accumulation_steps = 8
            cfg.max_seq_length = 8192
        elif model_key == "7b":
            cfg.per_device_batch_size = 8
            if args.grad_accum is None:
                cfg.gradient_accumulation_steps = 16
            cfg.max_seq_length = 4096
        elif model_key == "14b":
            cfg.per_device_batch_size = 4
            if args.grad_accum is None:
                cfg.gradient_accumulation_steps = 32
            cfg.max_seq_length = 2048
        elif model_key == "0.5b":
            cfg.per_device_batch_size = 32
            if args.grad_accum is None:
                cfg.gradient_accumulation_steps = 4
            cfg.max_seq_length = 4096

    print(
        f"Auto-config: gpu={device_name}, count={device_count}, mem={total_mem_gb:.0f}GB, "
        f"bf16={'on' if cfg.bf16 else 'off'}, ds={'on' if cfg.deepspeed_config else 'off'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full finetune for Qwen2.5 models (1.5B, 7B, 14B) with CRCA-optimized configurations."
    )
    parser.add_argument("--train-file", type=str, required=True, help="Path to training JSONL.")
    parser.add_argument("--eval-file", type=str, default=None, help="Optional eval JSONL.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_train_epochs.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation.")
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["0.5b", "1.5b", "7b", "14b"],
        default="0.5b",
        help="Model size to finetune: 0.5b, 1.5b, 7b, or 14b. Default: 0.5b",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use cloud-optimized config (only for 0.5B model). For other sizes, configs are already cloud-optimized.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Override base model ID (e.g. google/switch-base-8 for MoE Seq2Seq training).",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Override DeepSpeed config path (e.g. training/deepspeed_zero3_h100_3gpu.json for 3x H100 ZeRO-3).",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=-1,
        dest="local_rank",
        help="Local rank for distributed training (set by DeepSpeed launcher).",
    )
    parser.add_argument(
        "--no-auto-config",
        action="store_true",
        help="Disable automatic GPU-based configuration tuning.",
    )
    args = parser.parse_args()

    # Map model sizes to config functions
    config_map = {
        "0.5b": full_finetune_qwen25_0_5b_config_cloud if args.cloud else full_finetune_qwen25_0_5b_config,
        "1.5b": full_finetune_qwen25_1_5b_config,
        "7b": full_finetune_qwen25_7b_config,
        "14b": full_finetune_qwen25_14b_config,
    }
    
    # Get the appropriate config function
    config_func = config_map.get(args.model_size.lower())
    if config_func is None:
        raise ValueError(f"Invalid model size: {args.model_size}. Choose from: 0.5b, 1.5b, 7b, 14b")
    
    cfg = config_func()
    if args.model_id:
        cfg.base_model = args.model_id
    cfg.train_file = args.train_file
    # Only set eval_file if explicitly provided and file exists
    if args.eval_file:
        if not Path(args.eval_file).exists():
            raise FileNotFoundError(f"Eval file not found: {args.eval_file}")
        cfg.eval_file = args.eval_file
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs is not None:
        cfg.num_train_epochs = args.epochs
    if args.grad_accum is not None:
        cfg.gradient_accumulation_steps = args.grad_accum
    _apply_auto_config(cfg, args)
    if args.deepspeed_config is None and cfg.deepspeed_config:
        args.deepspeed_config = cfg.deepspeed_config
    if args.deepspeed_config:
        ds_path = Path(args.deepspeed_config)
        if not ds_path.is_absolute():
            ds_path = REPO_ROOT / ds_path
        if not ds_path.exists():
            raise FileNotFoundError(f"DeepSpeed config not found: {args.deepspeed_config}")
        cfg.deepspeed_config = str(ds_path.resolve())
        try:
            ds_config = json.loads(Path(cfg.deepspeed_config).read_text())
            bf16_enabled = bool(ds_config.get("bf16", {}).get("enabled", False))
            fp16_enabled = bool(ds_config.get("fp16", {}).get("enabled", False))
            if bf16_enabled and not fp16_enabled:
                cfg.bf16 = True
                cfg.fp16 = False
        except Exception:
            pass

    if cfg.deepspeed_config and not Path(cfg.deepspeed_config).exists():
        raise FileNotFoundError(f"Missing deepspeed config: {cfg.deepspeed_config}")

    run_finetune(cfg)


if __name__ == "__main__":
    main()
