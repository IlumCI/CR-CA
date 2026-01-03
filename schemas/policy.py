"""Policy schema definitions for the temporal policy engine.

This module defines Pydantic v2 models for:
- DoctrineV1: Policy specification (epochs, metrics, objectives, invariants, levers, risk budget)
- CompiledPolicy: Normalized, compiled version with resolved IDs and deterministic selectors
- LedgerEvent: Event types for the event-sourced ledger
- InterventionSpec + Intervention: Typed intervention parameters
- ModelState: Learned parameters θ with covariance and versioning

This is the "system of record" for doctrine + ledger + interventions (R1 requirement).
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class EpochConfig(BaseModel):
    """Epoch configuration for temporal policy execution.
    
    Attributes:
        unit: Time unit ("seconds", "minutes", "hours", "days")
        length: Length of one epoch in the specified unit
        timezone: Timezone policy (e.g., "UTC", "America/New_York")
    """
    unit: Literal["seconds", "minutes", "hours", "days"] = "hours"
    length: float = Field(default=1.0, gt=0.0)
    timezone: str = "UTC"


class MetricSpec(BaseModel):
    """Specification for a metric to track.
    
    Attributes:
        extractor_key: Key to extract metric from state snapshot
        unit: Unit of measurement (optional)
        description: Human-readable description
    """
    extractor_key: str
    unit: Optional[str] = None
    description: Optional[str] = None


class Objective(BaseModel):
    """Multi-objective target with deadline.
    
    Attributes:
        metric_name: Name of the metric to optimize
        target_value: Target value (None means minimize/maximize)
        direction: "minimize" or "maximize"
        deadline_epoch: Epoch by which objective should be met (None = no deadline)
        weight: Weight in multi-objective optimization (default: 1.0)
    """
    metric_name: str
    target_value: Optional[float] = None
    direction: Literal["minimize", "maximize"] = "minimize"
    deadline_epoch: Optional[int] = None
    weight: float = Field(default=1.0, gt=0.0)


class Invariant(BaseModel):
    """Hard constraint that must never be violated.
    
    Attributes:
        name: Name of the invariant
        condition: Condition expression (e.g., "metric_name > threshold")
        description: Human-readable description
    """
    name: str
    condition: str  # Expression to evaluate
    description: Optional[str] = None


class LeverSpec(BaseModel):
    """Specification for an allowed intervention lever.
    
    Attributes:
        lever_type: Type identifier (e.g., "ThrottleCPU", "StopService")
        bounds: Parameter bounds as dict (e.g., {"min": 0.0, "max": 1.0})
        cost_function: Cost function expression (optional)
        risk_function: Risk function expression (optional)
        rollback_required: Whether rollback is required for this lever
        description: Human-readable description
    """
    lever_type: str
    bounds: Dict[str, Any] = Field(default_factory=dict)
    cost_function: Optional[str] = None
    risk_function: Optional[str] = None
    rollback_required: bool = False
    description: Optional[str] = None


class RiskBudget(BaseModel):
    """Risk budget constraints.
    
    Attributes:
        max_actions_per_epoch: Maximum number of actions allowed per epoch
        max_risk_per_epoch: Maximum risk score allowed per epoch
        rollback_required: Whether rollback is required when budget exceeded
    """
    max_actions_per_epoch: int = Field(default=10, ge=0)
    max_risk_per_epoch: float = Field(default=1.0, ge=0.0)
    rollback_required: bool = False


class DoctrineV1(BaseModel):
    """Policy doctrine specification (version 1).
    
    This is the formal policy plan ("five-year plan") that defines:
    - Epoch configuration
    - Metrics to track
    - Objectives to optimize
    - Invariants (hard constraints)
    - Levers (allowed interventions)
    - Risk budget
    
    Attributes:
        epoch: Epoch configuration
        metrics: Dictionary mapping metric names to MetricSpec
        objectives: List of objectives to optimize
        invariants: List of hard constraints
        levers: Dictionary mapping lever IDs to LeverSpec
        risk_budget: Risk budget constraints
        version: Doctrine version (default: "1.0")
        created_at: Creation timestamp
    """
    epoch: EpochConfig
    metrics: Dict[str, MetricSpec]
    objectives: List[Objective]
    invariants: List[Invariant]
    levers: Dict[str, LeverSpec]
    risk_budget: RiskBudget
    version: str = "1.0"
    created_at: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_path: str) -> "DoctrineV1":
        """Load doctrine from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            DoctrineV1: Loaded doctrine
        """
        import json
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str) -> None:
        """Save doctrine to JSON file.
        
        Args:
            json_path: Path to save JSON file
        """
        import json
        with open(json_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)


class CompiledPolicy(BaseModel):
    """Compiled policy with normalized weights and resolved IDs.
    
    This is the compiled version of a DoctrineV1 that has been:
    - Normalized (weights sum to 1.0)
    - Resolved (metric IDs resolved, deadline epochs computed)
    - Compiled (deterministic selector rules)
    
    Attributes:
        normalized_weights: Dictionary mapping objective names to normalized weights
        resolved_metric_ids: List of resolved metric IDs
        deadline_epochs: Dictionary mapping objective names to deadline epochs
        compiled_selectors: Dictionary of compiled deterministic selector rules
        policy_hash: Stable hash of the original doctrine
        compiled_at: Compilation timestamp
    """
    normalized_weights: Dict[str, float]
    resolved_metric_ids: List[str]
    deadline_epochs: Dict[str, int]
    compiled_selectors: Dict[str, Any]
    policy_hash: str
    compiled_at: str
    
    @classmethod
    def compile(cls, doctrine: DoctrineV1) -> "CompiledPolicy":
        """Compile a DoctrineV1 into a CompiledPolicy.
        
        Args:
            doctrine: The doctrine to compile
            
        Returns:
            CompiledPolicy: Compiled policy
        """
        from utils.canonical import stable_hash
        from datetime import timezone
        
        # Compute policy hash
        policy_hash = stable_hash(doctrine.model_dump())
        
        # Normalize objective weights
        total_weight = sum(obj.weight for obj in doctrine.objectives)
        normalized_weights = {
            obj.metric_name: obj.weight / total_weight if total_weight > 0 else 1.0 / len(doctrine.objectives)
            for obj in doctrine.objectives
        }
        
        # Resolve metric IDs
        resolved_metric_ids = list(doctrine.metrics.keys())
        
        # Compute deadline epochs
        deadline_epochs = {
            obj.metric_name: obj.deadline_epoch
            for obj in doctrine.objectives
            if obj.deadline_epoch is not None
        }
        
        # Compiled selectors (deterministic rules)
        # For now, this is a placeholder - can be extended with rule compilation
        compiled_selectors = {
            "invariant_checker": {
                "invariants": [inv.model_dump() for inv in doctrine.invariants]
            },
            "lever_validator": {
                "levers": {k: v.model_dump() for k, v in doctrine.levers.items()}
            }
        }
        
        return cls(
            normalized_weights=normalized_weights,
            resolved_metric_ids=resolved_metric_ids,
            deadline_epochs=deadline_epochs,
            compiled_selectors=compiled_selectors,
            policy_hash=policy_hash,
            compiled_at=datetime.now(timezone.utc).isoformat()
        )


class ObservationEvent(BaseModel):
    """Observation event - state snapshot at epoch t.
    
    Attributes:
        epoch: Epoch number
        metrics: Dictionary of metric values
        timestamp: Observation timestamp
    """
    epoch: int
    metrics: Dict[str, float]
    timestamp: str


class DecisionEvent(BaseModel):
    """Decision event - chosen interventions at epoch t.
    
    Attributes:
        epoch: Epoch number
        interventions: List of intervention specifications
        rationale: Human-readable rationale
        decision_hash: Stable hash of the decision
        score: Optimization score
    """
    epoch: int
    interventions: List[Dict[str, Any]]
    rationale: str
    decision_hash: str
    score: float


class OutcomeEvent(BaseModel):
    """Outcome event - observed results after interventions.
    
    Attributes:
        epoch: Epoch number
        metrics: Dictionary of metric values after interventions
        timestamp: Outcome timestamp
    """
    epoch: int
    metrics: Dict[str, float]
    timestamp: str


class PolicyEvent(BaseModel):
    """Policy event - policy changes or updates.
    
    Attributes:
        epoch: Epoch number
        event_type: Type of policy event ("update", "rollback", etc.)
        payload: Event payload
    """
    epoch: int
    event_type: str
    payload: Dict[str, Any]


class LedgerEvent(BaseModel):
    """Union type for all ledger events.
    
    Attributes:
        type: Event type ("observation", "decision", "outcome", "policy")
        epoch: Epoch number
        hash: Stable hash of event payload
        payload: Event payload (ObservationEvent, DecisionEvent, OutcomeEvent, or PolicyEvent)
    """
    type: Literal["observation", "decision", "outcome", "policy"]
    epoch: int
    hash: str
    payload: Dict[str, Any]
    
    @classmethod
    def from_observation(cls, event: ObservationEvent) -> "LedgerEvent":
        """Create LedgerEvent from ObservationEvent.
        
        Args:
            event: ObservationEvent instance
            
        Returns:
            LedgerEvent: Ledger event
        """
        from utils.canonical import stable_hash
        payload_dict = event.model_dump()
        return cls(
            type="observation",
            epoch=event.epoch,
            hash=stable_hash(payload_dict),
            payload=payload_dict
        )
    
    @classmethod
    def from_decision(cls, event: DecisionEvent) -> "LedgerEvent":
        """Create LedgerEvent from DecisionEvent.
        
        Args:
            event: DecisionEvent instance
            
        Returns:
            LedgerEvent: Ledger event
        """
        from utils.canonical import stable_hash
        payload_dict = event.model_dump()
        return cls(
            type="decision",
            epoch=event.epoch,
            hash=stable_hash(payload_dict),
            payload=payload_dict
        )
    
    @classmethod
    def from_outcome(cls, event: OutcomeEvent) -> "LedgerEvent":
        """Create LedgerEvent from OutcomeEvent.
        
        Args:
            event: OutcomeEvent instance
            
        Returns:
            LedgerEvent: Ledger event
        """
        from utils.canonical import stable_hash
        payload_dict = event.model_dump()
        return cls(
            type="outcome",
            epoch=event.epoch,
            hash=stable_hash(payload_dict),
            payload=payload_dict
        )
    
    @classmethod
    def from_policy(cls, event: PolicyEvent) -> "LedgerEvent":
        """Create LedgerEvent from PolicyEvent.
        
        Args:
            event: PolicyEvent instance
            
        Returns:
            LedgerEvent: Ledger event
        """
        from utils.canonical import stable_hash
        payload_dict = event.model_dump()
        return cls(
            type="policy",
            epoch=event.epoch,
            hash=stable_hash(payload_dict),
            payload=payload_dict
        )


class InterventionSpec(BaseModel):
    """Specification for an intervention.
    
    Attributes:
        lever_id: ID of the lever to use
        parameters: Dictionary of intervention parameters
        cost: Estimated cost (optional)
        risk: Estimated risk (optional)
        rollback_descriptor: Descriptor for rollback (optional)
    """
    lever_id: str
    parameters: Dict[str, Any]
    cost: Optional[float] = None
    risk: Optional[float] = None
    rollback_descriptor: Optional[Dict[str, Any]] = None


class Intervention(BaseModel):
    """Executable intervention with typed parameters.
    
    Attributes:
        spec: Intervention specification
        created_at: Creation timestamp
        executed_at: Execution timestamp (None if not executed)
    """
    spec: InterventionSpec
    created_at: str
    executed_at: Optional[str] = None


class ModelState(BaseModel):
    """Learned parameters θ with covariance and versioning.
    
    Attributes:
        parameters: Dictionary of learned parameters (e.g., A, B matrices for RLS)
        covariance: Parameter covariance matrix (for uncertainty quantification)
        version: Model version identifier
        model_hash: Stable hash of the model state
        updated_at: Last update timestamp
    """
    parameters: Dict[str, Any]
    covariance: Optional[Dict[str, Any]] = None
    version: str = "1.0"
    model_hash: str
    updated_at: str
    
    @classmethod
    def create(cls, parameters: Dict[str, Any], covariance: Optional[Dict[str, Any]] = None) -> "ModelState":
        """Create a new ModelState.
        
        Args:
            parameters: Learned parameters
            covariance: Parameter covariance (optional)
            
        Returns:
            ModelState: New model state
        """
        from utils.canonical import stable_hash
        from datetime import timezone
        
        state_dict = {
            "parameters": parameters,
            "covariance": covariance or {}
        }
        model_hash = stable_hash(state_dict)
        
        return cls(
            parameters=parameters,
            covariance=covariance,
            model_hash=model_hash,
            updated_at=datetime.now(timezone.utc).isoformat()
        )

