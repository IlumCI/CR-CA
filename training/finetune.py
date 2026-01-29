"""Low-compute finetuning pipeline (LoRA/QLoRA when available)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

# Disable DeepSpeed op building if CUDA_HOME not set (prevents MissingCUDAException)
if "CUDA_HOME" not in os.environ and "DS_BUILD_OPS" not in os.environ:
    os.environ["DS_BUILD_OPS"] = "0"

MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "google/switch-base-8": {"arch": "seq2seq", "moe": True},
    "google/switch-large-16": {"arch": "seq2seq", "moe": True},
}


def _resolve_model_info(base_model: str) -> Dict[str, object]:
    model = base_model.lower()
    for model_id, info in MODEL_REGISTRY.items():
        if model == model_id.lower() or model.startswith(model_id.lower()):
            return {"arch": info.get("arch", "causal"), "moe": info.get("moe", False)}
    if "switch" in model:
        return {"arch": "seq2seq", "moe": True}
    return {"arch": "causal", "moe": False}


@dataclass
class FinetuneConfig:
    base_model: str = "microsoft/phi-2"
    output_dir: str = "lrm_finetune_out"
    train_file: str = "training_data/react_train.jsonl"
    eval_file: Optional[str] = None
    num_train_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    use_lora: bool = True
    max_seq_length: int = 512
    gradient_checkpointing: bool = False
    fp16: bool = True
    bf16: bool = False
    deepspeed_config: Optional[str] = None


def full_finetune_qwen25_1_5b_config() -> FinetuneConfig:
    """
    Full finetune configuration for Qwen2.5-1.5B-Instruct.
    
    Aggressively optimized for CRCA reasoning:
    - Higher learning rate (5e-4) for smaller models to avoid boilerplate
    - Longer sequences (8192) to capture full reasoning chains
    - Full finetune (no LoRA) for maximum reasoning capability
    - DeepSpeed ZeRO-2 for memory efficiency
    - 20 epochs for thorough convergence
    
    Compatible with NVIDIA GPUs (e.g. H100 SXM).
    """
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="lrm_qwen25_1_5b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=20,  # More epochs for CRCA reasoning convergence
        per_device_batch_size=8,  # Cloud GPU optimized
        gradient_accumulation_steps=16,  # Effective batch size: 128
        learning_rate=5e-4,  # Higher LR for smaller models, aggressive for CRCA
        use_lora=False,  # Full finetune for maximum reasoning capability
        max_seq_length=8192,  # Longer sequences for reasoning chains
        gradient_checkpointing=True,  # Enable for memory efficiency
        fp16=True,
        bf16=False,
        deepspeed_config="training/deepspeed_zero2_1_5b.json",
    )


def full_finetune_qwen25_7b_config() -> FinetuneConfig:
    """
    Full finetune configuration for Qwen2.5-7B-Instruct.
    
    Aggressively optimized for CRCA reasoning:
    - Moderate learning rate (2e-4) for stability with reasoning tasks
    - 20 epochs for thorough convergence on CRCA tasks
    - Full finetune (no LoRA) for maximum reasoning capability
    - DeepSpeed ZeRO-3 with CPU offload for memory efficiency
    - BF16 for better numerical stability on larger models
    
    Compatible with NVIDIA GPUs (e.g. H100 SXM).
    """
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        output_dir="lrm_qwen25_7b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=20,  # Increased from 1 for CRCA reasoning convergence
        per_device_batch_size=4,  # Cloud GPU optimized
        gradient_accumulation_steps=32,  # Effective batch size: 128
        learning_rate=2e-4,  # Optimized for CRCA reasoning, higher than default
        use_lora=False,  # Full finetune for maximum reasoning capability
        max_seq_length=4096,  # Full context for reasoning chains
        gradient_checkpointing=True,  # Enable for memory efficiency
        fp16=False,
        bf16=True,  # Better numerical stability for larger models
        deepspeed_config="training/deepspeed_zero3_offload.json",
    )


def full_finetune_qwen25_14b_config() -> FinetuneConfig:
    """
    Full finetune configuration for Qwen2.5-14B-Instruct.
    
    Aggressively optimized for CRCA reasoning:
    - Lower learning rate (1e-4) for stability on large models
    - 20 epochs for thorough convergence on CRCA tasks
    - Full finetune (no LoRA) for maximum reasoning capability
    - DeepSpeed ZeRO-3 with CPU offload for memory efficiency
    - BF16 required for numerical stability
    - Longer gradient accumulation for effective batch size
    
    Compatible with NVIDIA GPUs (e.g. H100 SXM).
    """
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-14B-Instruct",
        output_dir="lrm_qwen25_14b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=20,  # Thorough convergence for CRCA reasoning
        per_device_batch_size=2,  # Cloud GPU optimized (memory constrained)
        gradient_accumulation_steps=64,  # Effective batch size: 128
        learning_rate=1e-4,  # Lower LR for stability, still aggressive for CRCA
        use_lora=False,  # Full finetune for maximum reasoning capability
        max_seq_length=2048,  # Memory constraints on 14B model
        gradient_checkpointing=True,  # Critical for memory efficiency
        fp16=False,
        bf16=True,  # Required for numerical stability on large models
        deepspeed_config="training/deepspeed_zero3_14b.json",
    )


def full_finetune_switch_base_8_config() -> FinetuneConfig:
    """
    Full finetune configuration for Switch MoE base (Seq2Seq).

    Optimized for Switch MoE (encoder-decoder):
    - BF16 for H100 stability
    - ZeRO-3 without CPU offload (H100-class GPUs)
    - Moderate batch sizes for Seq2Seq memory footprint
    """
    return FinetuneConfig(
        base_model="google/switch-base-8",
        output_dir="lrm_switch_base_8_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=10,
        per_device_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        use_lora=False,
        max_seq_length=1024,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        deepspeed_config="training/deepspeed_zero3_h100_3gpu.json",
    )


def full_finetune_switch_large_16_config() -> FinetuneConfig:
    """
    Full finetune configuration for Switch MoE large (Seq2Seq).

    - BF16 for numerical stability
    - ZeRO-3 without CPU offload (H100-class GPUs)
    - Conservative batch sizes to keep memory stable
    """
    return FinetuneConfig(
        base_model="google/switch-large-16",
        output_dir="lrm_switch_large_16_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=10,
        per_device_batch_size=2,
        gradient_accumulation_steps=32,
        learning_rate=1e-4,
        use_lora=False,
        max_seq_length=1024,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        deepspeed_config="training/deepspeed_zero3_h100_3gpu.json",
    )


def full_finetune_qwen25_0_5b_config_cloud() -> FinetuneConfig:
    """
    Cloud GPU optimized configuration for Qwen2.5-0.5B-Instruct.
    
    For GPUs with 16GB+ VRAM (RTX 3090, A4000, A100, etc.):
    - Much larger batch sizes
    - Longer sequences
    - Full finetune (no LoRA needed)
    """
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="lrm_qwen25_0_5b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=20,
        per_device_batch_size=16,  # Cloud GPUs can handle this
        gradient_accumulation_steps=8,  # Adjusted for effective batch size
        learning_rate=4e-4,
        use_lora=False,  # Full finetune on cloud GPU
        max_seq_length=4096,  # Full context length on cloud
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        deepspeed_config=None,  # Not needed on cloud GPUs
    )


def full_finetune_qwen25_0_5b_config() -> FinetuneConfig:
    """
    Full finetune configuration for Qwen2.5-0.5B-Instruct.

    Optimized for smaller model size:
    - Larger batch sizes (0.5B fits easily in memory)
    - Higher learning rates (smaller models can handle higher LRs)
    - Reduced gradient accumulation (larger batch size means less accumulation needed)
    - Uses Accelerate (simpler than DeepSpeed for 0.5B model)
    """
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        output_dir="lrm_qwen25_0_5b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=20,
        per_device_batch_size=1,  # Must be 1 for 4GB GPU
        gradient_accumulation_steps=128,  # Large accumulation to maintain effective batch size
        learning_rate=4e-4,  # Smaller models can handle higher learning rates
        use_lora=True,  # Use LoRA to avoid CPU offload - trains only ~1% of parameters
        max_seq_length=512,  # Must be 512 or less for 4GB GPU
        gradient_checkpointing=False,  # Not needed with LoRA + 8-bit
        fp16=True,
        bf16=False,
        deepspeed_config=None,  # No DeepSpeed needed with LoRA - stays on GPU
    )


def run_finetune(cfg: FinetuneConfig) -> None:
    """
    Run finetuning on NVIDIA GPUs (e.g. H100 SXM).
    
    Uses CUDA with NCCL for distributed training. Supports 4-bit/8-bit quantization
    for LoRA and full finetune with BF16/FP16. DeepSpeed ZeRO-2/ZeRO-3 for multi-GPU.
    """
    print("CUDA (NVIDIA GPU) detected - using CUDA settings")
    
    # Configure environment for single GPU DeepSpeed (if using DeepSpeed)
    if cfg.deepspeed_config:
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
    
    try:
        from datasets import load_dataset  # type: ignore
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install: transformers, datasets, accelerate, peft"
        ) from exc

    model_info = _resolve_model_info(cfg.base_model)
    is_seq2seq = model_info.get("arch") == "seq2seq"
    if model_info.get("moe"):
        print("MoE model detected - using Seq2Seq pipeline")

    # Load tokenizer with error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load tokenizer from {cfg.base_model}: {exc}") from exc
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token is None:
            raise ValueError(f"Tokenizer from {cfg.base_model} has no pad_token or eos_token")
    
    model_cls = AutoModelForSeq2SeqLM if is_seq2seq else AutoModelForCausalLM

    # Load model (CUDA: 4-bit/8-bit for LoRA, full precision for full finetune)
    if cfg.use_lora:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = model_cls.from_pretrained(
                cfg.base_model,
                quantization_config=quantization_config,
                device_map="auto",
            )
            print("Using 4-bit quantization (CUDA)")
        except (ImportError, Exception):
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                )
                model = model_cls.from_pretrained(
                    cfg.base_model,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                print("Using 8-bit quantization (4-bit not available)")
            except (ImportError, Exception):
                model = model_cls.from_pretrained(
                    cfg.base_model,
                    torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
                    low_cpu_mem_usage=True,
                )
                print("Using full precision (quantization not available)")
    else:
        # Full finetune: use BF16/FP16 based on config
        model = model_cls.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            low_cpu_mem_usage=True,
        )
        precision_str = "BF16" if cfg.bf16 else "FP16"
        print(f"Using full finetune with {precision_str} precision")

    if cfg.use_lora:
        try:
            from peft import LoraConfig, get_peft_model  # type: ignore
        except Exception as exc:
            raise RuntimeError("LoRA requested but peft not installed. Install peft.") from exc

        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        # LoRA doesn't need gradient checkpointing - it's already memory efficient
    elif cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not Path(cfg.train_file).exists():
        raise FileNotFoundError(f"Training file not found: {cfg.train_file}")
    if cfg.eval_file and not Path(cfg.eval_file).exists():
        raise FileNotFoundError(f"Eval file not found: {cfg.eval_file}")

    data_files = {"train": cfg.train_file}
    if cfg.eval_file:
        data_files["validation"] = cfg.eval_file

    try:
        dataset = load_dataset("json", data_files=data_files)
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset from {data_files}: {exc}") from exc
    
    if "train" not in dataset:
        raise ValueError(f"Dataset must contain 'train' split, got: {list(dataset.keys())}")
    if len(dataset["train"]) == 0:
        raise ValueError("Training dataset is empty")

    def _tokenize(examples):
        if is_seq2seq:
            inputs = tokenizer(
                examples["prompt"],
                truncation=True,
                padding="max_length",
                max_length=cfg.max_seq_length,
                return_tensors=None,
            )
            targets = tokenizer(
                text_target=examples["response"],
                truncation=True,
                padding="max_length",
                max_length=cfg.max_seq_length,
                return_tensors=None,
            )
            pad_token_id = tokenizer.pad_token_id
            labels = [
                [token_id if token_id != pad_token_id else -100 for token_id in seq]
                for seq in targets["input_ids"]
            ]
            inputs["labels"] = labels
            return inputs

        texts = [p + "\n" + r for p, r in zip(examples["prompt"], examples["response"])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_seq_length,
            return_tensors=None,  # Return lists, not tensors
        )
        # For causal LM, labels are the same as input_ids
        # Set padding tokens to -100 so they're ignored in loss calculation
        labels = []
        pad_token_id = tokenizer.pad_token_id
        for input_ids in tokenized["input_ids"]:
            label = [token_id if token_id != pad_token_id else -100 for token_id in input_ids]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    # Get column names before tokenization (handle both train and validation)
    original_columns = dataset["train"].column_names
    
    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=original_columns,
    )

    # Validate configuration before training
    if cfg.per_device_batch_size < 1:
        raise ValueError(f"per_device_batch_size must be >= 1, got {cfg.per_device_batch_size}")
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError(f"gradient_accumulation_steps must be >= 1, got {cfg.gradient_accumulation_steps}")
    if cfg.learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {cfg.learning_rate}")
    if cfg.max_seq_length < 1:
        raise ValueError(f"max_seq_length must be >= 1, got {cfg.max_seq_length}")
    if cfg.fp16 and cfg.bf16:
        raise ValueError("Cannot use both fp16 and bf16 simultaneously")
    
    # Resolve DeepSpeed config path if provided
    deepspeed_config_path = None
    if cfg.deepspeed_config:
        deepspeed_config_path = str(Path(cfg.deepspeed_config).resolve())
        if not Path(deepspeed_config_path).exists():
            raise FileNotFoundError(f"DeepSpeed config file not found: {deepspeed_config_path}")
    
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing if not cfg.use_lora else False,  # LoRA doesn't need it
        deepspeed=deepspeed_config_path if deepspeed_config_path else None,
        logging_steps=50,
        save_steps=200,
        eval_strategy="no" if cfg.eval_file is None else "steps",
        eval_steps=200 if cfg.eval_file else None,  # Evaluate every 200 steps if eval_file provided
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Save memory
        dataloader_num_workers=0,  # Reduce memory overhead
        optim="adamw_torch",  # Use standard AdamW (more memory efficient than fused variants)
        max_grad_norm=1.0,  # Gradient clipping
        warmup_steps=100,  # Add warmup for better convergence
        lr_scheduler_type="cosine",  # Cosine learning rate schedule for better convergence
    )

    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation") if cfg.eval_file else None
    
    if len(train_dataset) == 0:
        raise ValueError("Tokenized training dataset is empty")
    
    data_collator = None
    if is_seq2seq:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        )

    # Data collator is optional when using max_length padding, but helps ensure consistency
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print(f"Starting training with {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Evaluation dataset has {len(eval_dataset)} examples")
    print(f"Effective batch size: {cfg.per_device_batch_size * cfg.gradient_accumulation_steps}")
    print(f"Total training steps: {len(train_dataset) // (cfg.per_device_batch_size * cfg.gradient_accumulation_steps) * cfg.num_train_epochs}")
    
    try:
        trainer.train()
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc
    
    try:
        trainer.save_model(cfg.output_dir)
        print(f"Model saved to {cfg.output_dir}")
    except Exception as exc:
        raise RuntimeError(f"Failed to save model to {cfg.output_dir}: {exc}") from exc

