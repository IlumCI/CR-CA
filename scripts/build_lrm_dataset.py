"""Build a hybrid LRM dataset from internal traces and public datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.datasets import (
    ReActExample,
    examples_from_traces,
    filter_examples,
    merge_examples,
    save_jsonl,
)
from training.public_datasets import PublicDatasetConfig, default_public_configs, load_public_examples


def _load_public_config(path: Path) -> List[PublicDatasetConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    configs: List[PublicDatasetConfig] = []
    for item in payload:
        configs.append(PublicDatasetConfig(**item))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hybrid LRM dataset JSONL.")
    parser.add_argument("--trace-jsonl", action="append", default=[], help="Path to LRM plan trace JSONL.")
    parser.add_argument("--public-config", type=str, default="", help="Path to public dataset config JSON.")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--max-internal", type=int, default=None, help="Max internal examples to include.")
    parser.add_argument("--max-public", type=int, default=None, help="Max public examples to include.")
    parser.add_argument("--min-response-len", type=int, default=1)
    parser.add_argument("--max-prompt-len", type=int, default=None)
    parser.add_argument("--max-response-len", type=int, default=None)
    args = parser.parse_args()

    internal_examples: List[ReActExample] = []
    if args.trace_jsonl:
        trace_paths = [Path(p) for p in args.trace_jsonl]
        internal_examples = examples_from_traces(trace_paths)

    if args.public_config:
        public_configs = _load_public_config(Path(args.public_config))
    else:
        public_configs = default_public_configs()
    public_examples = load_public_examples(public_configs)

    internal_examples = filter_examples(
        internal_examples,
        min_response_len=args.min_response_len,
        max_prompt_len=args.max_prompt_len,
        max_response_len=args.max_response_len,
    )
    public_examples = filter_examples(
        public_examples,
        min_response_len=args.min_response_len,
        max_prompt_len=args.max_prompt_len,
        max_response_len=args.max_response_len,
    )

    merged = merge_examples(
        internal_examples=internal_examples,
        public_examples=public_examples,
        max_internal=args.max_internal,
        max_public=args.max_public,
    )

    save_jsonl(merged, Path(args.output))


if __name__ == "__main__":
    main()
