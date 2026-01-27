"""Structured result types for crca_core.

All results are structured objects. Human-readable reports must be generated
by rendering these objects, not by mixing narrative into scientific fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from crca_core.models.provenance import ProvenanceManifest
from crca_core.models.refusal import RefusalResult


class ValidationIssue(BaseModel):
    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    path: Optional[str] = None


class ValidationReport(BaseModel):
    """Returned by `validate_spec`."""

    ok: bool
    errors: List[ValidationIssue] = Field(default_factory=list)
    warnings: List[ValidationIssue] = Field(default_factory=list)


class BaseResult(BaseModel):
    """Base result type with mandatory provenance."""

    result_type: str
    provenance: ProvenanceManifest
    assumptions: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)


class DiscoveryHypothesisResult(BaseResult):
    result_type: Literal["DiscoveryHypothesis"] = "DiscoveryHypothesis"
    graph_hypothesis: Dict[str, Any] = Field(default_factory=dict)
    stability_report: Dict[str, Any] = Field(default_factory=dict)


class InterventionDesignResult(BaseResult):
    result_type: Literal["InterventionDesign"] = "InterventionDesign"
    designs: List[Dict[str, Any]] = Field(default_factory=list)


class CounterfactualResult(BaseResult):
    result_type: Literal["CounterfactualResult"] = "CounterfactualResult"
    counterfactual: Dict[str, Any] = Field(default_factory=dict)


class IdentificationResult(BaseResult):
    result_type: Literal["IdentificationResult"] = "IdentificationResult"
    method: str
    scope: Literal["conservative", "partial", "complete"] = "conservative"
    confidence: Literal["low", "medium", "high"] = "low"
    estimand_expression: str
    assumptions_used: List[str] = Field(default_factory=list)
    witnesses: Dict[str, Any] = Field(default_factory=dict)
    proof: Dict[str, Any] = Field(default_factory=dict)


class EstimateResult(BaseResult):
    result_type: Literal["EstimateResult"] = "EstimateResult"
    estimate: Dict[str, Any] = Field(default_factory=dict)
    refutations: Dict[str, Any] = Field(default_factory=dict)


AnyResult = (
    RefusalResult
    | ValidationReport
    | DiscoveryHypothesisResult
    | InterventionDesignResult
    | CounterfactualResult
    | IdentificationResult
    | EstimateResult
)

