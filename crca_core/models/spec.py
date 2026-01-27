"""Typed causal specification objects (Draft → Locked)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class SpecStatus(str, Enum):
    draft = "draft"
    locked = "locked"


class AssumptionStatus(str, Enum):
    declared = "declared"
    contested = "contested"
    violated = "violated"
    unknown = "unknown"


class DataColumnSpec(BaseModel):
    name: str = Field(..., min_length=1)
    dtype: str = Field(..., min_length=1)
    allowed_range: Optional[Tuple[float, float]] = None
    missingness_expected: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    unit: Optional[str] = None
    description: Optional[str] = None


class TimeIndexSpec(BaseModel):
    column: str = Field(..., min_length=1)
    frequency: Optional[str] = None
    timezone: Optional[str] = None
    irregular_sampling_policy: Optional[str] = None


class EntityIndexSpec(BaseModel):
    entity_id_column: str = Field(..., min_length=1)
    time_column: str = Field(..., min_length=1)


class DataSpec(BaseModel):
    dataset_name: Optional[str] = None
    dataset_hash: Optional[str] = None
    columns: List[DataColumnSpec] = Field(default_factory=list)
    time_index: Optional[TimeIndexSpec] = None
    entity_index: Optional[EntityIndexSpec] = None
    measurement_error_notes: Optional[str] = None
    proxy_variables: Dict[str, str] = Field(default_factory=dict)


class NodeSpec(BaseModel):
    name: str = Field(..., min_length=1)
    observed: bool = True
    unit: Optional[str] = None
    description: Optional[str] = None


class EdgeSpec(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    lag: Optional[int] = None
    description: Optional[str] = None


class CausalGraphSpec(BaseModel):
    nodes: List[NodeSpec] = Field(default_factory=list)
    edges: List[EdgeSpec] = Field(default_factory=list)
    latent_confounders: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class RoleSpec(BaseModel):
    treatments: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    mediators: List[str] = Field(default_factory=list)
    instruments: List[str] = Field(default_factory=list)
    adjustment_candidates: List[str] = Field(default_factory=list)
    prohibited_controls: List[str] = Field(default_factory=list)


class AssumptionItem(BaseModel):
    name: str = Field(..., min_length=1)
    status: AssumptionStatus = AssumptionStatus.unknown
    description: Optional[str] = None
    evidence: Optional[str] = None


class AssumptionSpec(BaseModel):
    items: List[AssumptionItem] = Field(default_factory=list)
    falsification_plan: List[str] = Field(default_factory=list)


class NoiseSpec(BaseModel):
    distribution: Literal["gaussian"] = "gaussian"
    params: Dict[str, Any] = Field(default_factory=dict)


class StructuralEquationSpec(BaseModel):
    """Represents one structural equation V = f(Pa(V), U_V).

    v0.1: store both a human-readable formula and an executable parameterization
    for supported SCM families.
    """

    variable: str = Field(..., min_length=1)
    parents: List[str] = Field(default_factory=list)
    form: Literal["linear_gaussian"] = "linear_gaussian"
    coefficients: Dict[str, float] = Field(default_factory=dict)  # parent -> beta
    intercept: float = 0.0
    noise: NoiseSpec = Field(default_factory=NoiseSpec)


class SCMSpec(BaseModel):
    """Explicit SCM required for counterfactuals."""

    scm_type: Literal["linear_gaussian"] = "linear_gaussian"
    equations: List[StructuralEquationSpec] = Field(default_factory=list)
    # Optional correlated noise for linear-Gaussian SCMs (advanced; v0.1 may require diagonal)
    noise_cov: Optional[List[List[float]]] = None
    intervention_semantics: Dict[str, str] = Field(default_factory=dict)  # var -> set/shift/mechanism-change


class DraftSpec(BaseModel):
    """Draft spec (may be LLM-generated; never authorizes numeric causal outputs)."""

    status: SpecStatus = Field(default=SpecStatus.draft, frozen=True)
    data: DataSpec = Field(default_factory=DataSpec)
    graph: CausalGraphSpec = Field(default_factory=CausalGraphSpec)
    roles: RoleSpec = Field(default_factory=RoleSpec)
    assumptions: AssumptionSpec = Field(default_factory=AssumptionSpec)
    scm: Optional[SCMSpec] = None
    draft_notes: Optional[str] = None


class LockedSpec(BaseModel):
    """Locked spec (authoritative for identification/estimation/simulation semantics)."""

    status: SpecStatus = Field(default=SpecStatus.locked, frozen=True)
    spec_hash: str
    approvals: List[str] = Field(default_factory=list)
    locked_at_utc: str

    data: DataSpec
    graph: CausalGraphSpec
    roles: RoleSpec
    assumptions: AssumptionSpec
    scm: Optional[SCMSpec] = None

