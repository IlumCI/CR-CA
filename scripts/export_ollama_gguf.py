"""Export a finetuned HF model to GGUF for Ollama."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF via llama.cpp.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="HF checkpoint directory.")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path.")
    parser.add_argument("--llama-cpp-dir", type=str, default="", help="Path to llama.cpp repo.")
    args = parser.parse_args()

    llama_cpp_dir = Path(args.llama_cpp_dir) if args.llama_cpp_dir else None
    if llama_cpp_dir is None:
        env_path = os.environ.get("LLAMA_CPP_DIR")
        if env_path:
            llama_cpp_dir = Path(env_path)
    if llama_cpp_dir is None:
        raise RuntimeError("Provide --llama-cpp-dir or set LLAMA_CPP_DIR.")

    converter = llama_cpp_dir / "convert-hf-to-gguf.py"
    if not converter.exists():
        raise FileNotFoundError(f"Missing convert script: {converter}")

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        str(converter),
        str(checkpoint_dir),
        "--outfile",
        str(output_path),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
