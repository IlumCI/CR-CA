"""Dataset assembly for ReAct training traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from crca_reasoning.types import LRMPlanResult


@dataclass
class ReActExample:
    prompt: str
    response: str
    tags: Dict[str, str]
    refusal: bool = False


def plan_result_to_examples(plan: LRMPlanResult) -> List[ReActExample]:
    examples: List[ReActExample] = []
    for cycle in plan.cycle_traces:
        prompt = cycle.reasoning
        response = ""
        if cycle.actions:
            response += "Actions:\n"
            for act in cycle.actions:
                response += f"- {act.tool_name}: {act.payload}\n"
        if cycle.critique:
            response += f"Critique: {cycle.critique}\n"
        examples.append(
            ReActExample(
                prompt=prompt,
                response=response,
                tags={"type": "react_cycle"},
                refusal=any(obs.refusal is not None for obs in cycle.observations),
            )
        )
    if plan.rationale_trace and plan.rationale_trace.steps:
        examples.append(
            ReActExample(
                prompt="RationaleTrace",
                response="\n".join(plan.rationale_trace.steps),
                tags={"type": "rationale_trace"},
                refusal=False,
            )
        )
    return examples


def load_plan_results(paths: Sequence[Path]) -> List[LRMPlanResult]:
    """Load LRMPlanResult objects from JSON or JSONL files."""
    results: List[LRMPlanResult] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    results.append(LRMPlanResult.model_validate(payload))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                results.extend(LRMPlanResult.model_validate(item) for item in payload)
            else:
                results.append(LRMPlanResult.model_validate(payload))
    return results


def examples_from_traces(paths: Sequence[Path]) -> List[ReActExample]:
    """Load plan results from trace files and convert to ReActExamples."""
    plans = load_plan_results(paths)
    examples: List[ReActExample] = []
    for plan in plans:
        examples.extend(plan_result_to_examples(plan))
    return examples


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def filter_examples(
    examples: Iterable[ReActExample],
    *,
    min_response_len: int = 1,
    max_prompt_len: Optional[int] = None,
    max_response_len: Optional[int] = None,
) -> List[ReActExample]:
    filtered: List[ReActExample] = []
    for ex in examples:
        prompt = normalize_text(ex.prompt)
        response = normalize_text(ex.response)
        if len(response) < min_response_len:
            continue
        if max_prompt_len is not None and len(prompt) > max_prompt_len:
            continue
        if max_response_len is not None and len(response) > max_response_len:
            continue
        filtered.append(
            ReActExample(
                prompt=prompt,
                response=response,
                tags=dict(ex.tags),
                refusal=ex.refusal,
            )
        )
    return filtered


def merge_examples(
    *,
    internal_examples: Iterable[ReActExample],
    public_examples: Iterable[ReActExample],
    max_internal: Optional[int] = None,
    max_public: Optional[int] = None,
) -> List[ReActExample]:
    merged: List[ReActExample] = []
    if max_internal is None:
        merged.extend(list(internal_examples))
    else:
        merged.extend(list(internal_examples)[: max_internal])
    if max_public is None:
        merged.extend(list(public_examples))
    else:
        merged.extend(list(public_examples)[: max_public])
    return merged


def save_jsonl(examples: Iterable[ReActExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")

