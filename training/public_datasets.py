"""Public dataset ingestion for hybrid LRM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from training.datasets import ReActExample, normalize_text


@dataclass
class PublicDatasetConfig:
    name: str
    split: str
    prompt_field: str
    response_field: str
    config_name: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    system_field: Optional[str] = None
    max_samples: Optional[int] = None
    license_tag: Optional[str] = None
    source_tag: Optional[str] = None


def default_public_configs() -> List[PublicDatasetConfig]:
    """Conservative defaults with known field names."""
    return [
        PublicDatasetConfig(
            name="openai/gsm8k",
            config_name="main",
            split="train",
            prompt_field="question",
            response_field="answer",
            prompt_template="Question: {question}\nAnswer:",
            response_template="{answer}",
            license_tag="unknown",
            source_tag="gsm8k",
        )
    ]


def _format_with_template(template: Optional[str], row: Dict[str, Any], field: str) -> str:
    if template:
        return template.format(**row)
    value = row.get(field, "")
    return str(value) if value is not None else ""


def load_public_examples(
    configs: Iterable[PublicDatasetConfig],
    *,
    seed: int = 7,
) -> List[ReActExample]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"datasets library is required to load public datasets: {exc}") from exc

    examples: List[ReActExample] = []
    for cfg in configs:
        if cfg.config_name:
            dataset = load_dataset(cfg.name, cfg.config_name, split=cfg.split)
        else:
            dataset = load_dataset(cfg.name, split=cfg.split)
        rows = list(dataset)
        if cfg.max_samples is not None:
            rows = rows[: cfg.max_samples]
        for row in rows:
            prompt = _format_with_template(cfg.prompt_template, row, cfg.prompt_field)
            response = _format_with_template(cfg.response_template, row, cfg.response_field)
            if cfg.system_field and row.get(cfg.system_field):
                system = str(row[cfg.system_field])
                prompt = f"System: {system}\n{prompt}"
            prompt = normalize_text(prompt)
            response = normalize_text(response)
            if not prompt or not response:
                continue
            tags = {
                "type": "public_reasoning",
                "dataset": cfg.name,
            }
            if cfg.license_tag:
                tags["license"] = cfg.license_tag
            if cfg.source_tag:
                tags["source"] = cfg.source_tag
            examples.append(ReActExample(prompt=prompt, response=response, tags=tags, refusal=False))
    return examples

