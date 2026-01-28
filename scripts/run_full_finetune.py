"""Run full finetune for Qwen2.5 models with CRCA-optimized configurations."""

from __future__ import annotations

import argparse
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

    if cfg.deepspeed_config and not Path(cfg.deepspeed_config).exists():
        raise FileNotFoundError(f"Missing deepspeed config: {cfg.deepspeed_config}")

    run_finetune(cfg)


if __name__ == "__main__":
    main()
