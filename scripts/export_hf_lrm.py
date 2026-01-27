"""Export finetuned LRM model for HuggingFace."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export finetuned LRM model to HF format.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to finetuned checkpoint.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for HF upload.")
    parser.add_argument("--model-card", type=str, default="MODEL_CARD.md", help="Model card path.")
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        raise RuntimeError("transformers is required to export model.") from exc

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    card_path = Path(args.model_card)
    if card_path.exists():
        shutil.copy(card_path, output_dir / "README.md")


if __name__ == "__main__":
    main()
