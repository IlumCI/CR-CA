"""Run LRM evaluation on trace files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.datasets import load_plan_results
from training.eval import EvalConfig, run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LRM eval on trace JSON/JSONL.")
    parser.add_argument("--trace", action="append", default=[], help="Path to trace JSON or JSONL.")
    parser.add_argument("--output", type=str, default="eval_results/lrm_eval.json")
    args = parser.parse_args()

    if not args.trace:
        raise ValueError("At least one --trace file is required.")

    plans = load_plan_results([Path(p) for p in args.trace])
    run_eval(plans, EvalConfig(output_path=args.output))


if __name__ == "__main__":
    main()
