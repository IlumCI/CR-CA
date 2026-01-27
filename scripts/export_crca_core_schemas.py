"""Export `crca_core` Pydantic JSON schemas for downstream tooling.

This supports the "structured object only" requirement: downstream systems can
validate inputs/outputs against stable schemas.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crca_core.models.refusal import RefusalResult
from crca_core.models.result import (
    CounterfactualResult,
    DiscoveryHypothesisResult,
    EstimateResult,
    IdentificationResult,
    InterventionDesignResult,
    ValidationReport,
)
from crca_core.models.spec import DraftSpec, LockedSpec


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "schemas_export" / "crca_core"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("DraftSpec", DraftSpec),
        ("LockedSpec", LockedSpec),
        ("ValidationReport", ValidationReport),
        ("RefusalResult", RefusalResult),
        ("DiscoveryHypothesisResult", DiscoveryHypothesisResult),
        ("InterventionDesignResult", InterventionDesignResult),
        ("CounterfactualResult", CounterfactualResult),
        ("IdentificationResult", IdentificationResult),
        ("EstimateResult", EstimateResult),
    ]

    for name, model in models:
        schema = model.model_json_schema()
        (out_dir / f"{name}.schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print(f"Wrote {len(models)} schemas to {out_dir}")


if __name__ == "__main__":
    main()

