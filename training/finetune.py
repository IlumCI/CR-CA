"""Low-compute finetuning pipeline (LoRA/QLoRA when available)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

# Disable DeepSpeed op building if CUDA_HOME not set (prevents MissingCUDAException)
if "CUDA_HOME" not in os.environ and "DS_BUILD_OPS" not in os.environ:
    os.environ["DS_BUILD_OPS"] = "0"


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


def full_finetune_qwen25_7b_config() -> FinetuneConfig:
    return FinetuneConfig(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        output_dir="lrm_qwen25_7b_full_finetune",
        train_file="training_data/react_train.jsonl",
        eval_file=None,
        num_train_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-5,
        use_lora=False,
        max_seq_length=4096,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        deepspeed_config="training/deepspeed_zero3_offload.json",
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
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install: transformers, datasets, accelerate, peft"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Load model - use 4-bit if available for LoRA, otherwise fp16
    if cfg.use_lora:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4-bit uses less memory than 8-bit
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                quantization_config=quantization_config,
                device_map="auto",
            )
        except (ImportError, Exception):
            # Fallback: regular fp16 if bitsandbytes not available
            model = AutoModelForCausalLM.from_pretrained(
                cfg.base_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            low_cpu_mem_usage=True,
        )

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

    dataset = load_dataset("json", data_files=data_files)

    def _tokenize(examples):
        """Tokenize the examples, combining prompt and response."""
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
        deepspeed=cfg.deepspeed_config if cfg.deepspeed_config else None,
        logging_steps=50,
        save_steps=200,
        eval_strategy="no" if cfg.eval_file is None else "steps",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # Save memory on 4GB GPU
        dataloader_num_workers=0,  # Reduce memory overhead
        optim="adamw_torch",  # Use standard AdamW (more memory efficient than fused variants)
        max_grad_norm=1.0,  # Gradient clipping
    )

    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation") if cfg.eval_file else None
    
    # Data collator is optional when using max_length padding, but helps ensure consistency
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # data_collator not needed when padding to max_length, but can help with label handling
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)

