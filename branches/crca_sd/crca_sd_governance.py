"""
CRCA-SD Governance: Boards, Logistics, Visualization, Config, Metrics

This module implements:
- Board-of-agents governance (adapted from CorporateSwarm)
- Multi-board arbitration with weighted voting
- Logistics network optimization
- Visualization and dashboard
- Configuration management (Pydantic + YAML)
- Risk assessment
- Metrics collection

Incorporates patterns from CorporateSwarm for board structure, voting, and config management.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import time
import uuid
import os
from datetime import datetime
from loguru import logger

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("yaml not available, config file loading disabled")

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic not available, using dataclass fallback")

try:
    import cvxpy as cp
    import networkx as nx
    CVXPY_AVAILABLE = True
    NETWORKX_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    NETWORKX_AVAILABLE = False
    logger.warning("cvxpy/networkx not available, logistics optimization limited")

from crca_sd.crca_sd_core import StateVector, ControlVector
from crca_sd.crca_sd_mpc import ObjectiveVector


def _generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class BoardType(str, Enum):
    """Types of governance boards."""
    GROWTH = "growth"
    WELFARE = "welfare"
    SUSTAINABILITY = "sustainability"
    STABILITY = "stability"


class VoteResult(str, Enum):
    """Voting result outcomes (from CorporateSwarm pattern)."""
    APPROVED = "approved"
    REJECTED = "rejected"
    TABLED = "tabled"
    FAILED = "failed"
    UNANIMOUS = "unanimous"
    MAJORITY = "majority"
    MINORITY = "minority"
    ABSTAINED = "abstained"


# BoardMember: Adapted from CorporateMember (lines 353-369 in CorporateSwarm)
if PYDANTIC_AVAILABLE:
    class BoardMember(BaseModel):
        """Board member with role, expertise, and objective preferences."""
        
        member_id: str = Field(default_factory=_generate_uuid)
        name: str = Field(default="")
        board_type: BoardType = Field(default=BoardType.GROWTH)
        expertise_areas: List[str] = Field(default_factory=list)
        voting_weight: float = Field(default=1.0, ge=0.0, le=5.0)
        preferences: Dict[str, float] = Field(default_factory=dict)  # Objective weights
        independence_status: bool = Field(default=False)
        metadata: Dict[str, Any] = Field(default_factory=dict)
else:
    @dataclass
    class BoardMember:
        """Board member with role, expertise, and objective preferences."""
        member_id: str = field(default_factory=_generate_uuid)
        name: str = ""
        board_type: BoardType = BoardType.GROWTH
        expertise_areas: List[str] = field(default_factory=list)
        voting_weight: float = 1.0
        preferences: Dict[str, float] = field(default_factory=dict)
        independence_status: bool = False
        metadata: Dict[str, Any] = field(default_factory=dict)


class Board:
    """
    Governance board with members and objective priorities.
    
    Inspired by CorporateSwarm board structure.
    Each board type has different objective priorities:
    - Growth: Prioritizes Y (output), then U (unemployment)
    - Welfare: Prioritizes U (unemployment), inequality, literacy
    - Sustainability: Prioritizes C (ecological damage), long-term resilience
    - Stability: Prioritizes low variance, low CVaR
    
    Args:
        board_id: Unique board identifier
        board_type: Type of board
        members: List of board members
    """
    
    def __init__(
        self,
        board_id: str,
        board_type: BoardType,
        members: List[BoardMember]
    ) -> None:
        """Initialize board."""
        self.board_id = board_id
        self.board_type = board_type
        self.members = members
    
    def get_preference_weights(self) -> Dict[str, float]:
        """
        Get objective weight vector for this board type.
        
        Returns:
            Dict[str, float]: Mapping from objective names to weights
        """
        if self.board_type == BoardType.GROWTH:
            return {
                "J_U": 0.2,      # Unemployment
                "J_ℓ": 0.1,      # Literacy
                "J_Y": 0.5,      # Output (high priority)
                "J_ineq": 0.1,   # Inequality
                "J_C": 0.05,     # Ecological damage
                "J_risk": 0.05,  # Risk
            }
        elif self.board_type == BoardType.WELFARE:
            return {
                "J_U": 0.4,      # Unemployment (high priority)
                "J_ℓ": 0.3,      # Literacy (high priority)
                "J_Y": 0.1,      # Output
                "J_ineq": 0.15,  # Inequality (high priority)
                "J_C": 0.03,     # Ecological damage
                "J_risk": 0.02,  # Risk
            }
        elif self.board_type == BoardType.SUSTAINABILITY:
            return {
                "J_U": 0.1,      # Unemployment
                "J_ℓ": 0.1,      # Literacy
                "J_Y": 0.1,      # Output
                "J_ineq": 0.1,   # Inequality
                "J_C": 0.5,      # Ecological damage (high priority)
                "J_risk": 0.1,   # Risk
            }
        elif self.board_type == BoardType.STABILITY:
            return {
                "J_U": 0.2,      # Unemployment
                "J_ℓ": 0.1,      # Literacy
                "J_Y": 0.1,      # Output
                "J_ineq": 0.1,   # Inequality
                "J_C": 0.1,      # Ecological damage
                "J_risk": 0.4,   # Risk (high priority)
            }
        else:
            # Default: equal weights
            return {f"J_{i}": 1.0/6 for i in ["U", "ℓ", "Y", "ineq", "C", "risk"]}
    
    def evaluate_policies(
        self,
        policies: List[ControlVector],
        objectives: np.ndarray
    ) -> List[Tuple[ControlVector, float]]:
        """
        Rank policies by board's objective priorities.
        
        Args:
            policies: List of candidate policies
            objectives: Objective matrix (n_policies x n_objectives)
            
        Returns:
            List[Tuple[ControlVector, float]]: Ranked policies with scores
        """
        if len(policies) == 0:
            return []
        
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)
        
        # Get board preference weights
        weights = self.get_preference_weights()
        weight_vec = np.array([
            weights.get("J_U", 0.0),
            weights.get("J_ℓ", 0.0),
            weights.get("J_Y", 0.0),
            weights.get("J_ineq", 0.0),
            weights.get("J_C", 0.0),
            weights.get("J_risk", 0.0),
        ])
        
        # Normalize weights
        if weight_vec.sum() > 0:
            weight_vec = weight_vec / weight_vec.sum()
        
        # Compute weighted scores
        scores = objectives @ weight_vec
        
        # Rank policies (lower score = better for minimization)
        ranked = list(zip(policies, scores))
        ranked.sort(key=lambda x: x[1])
        
        return ranked


class Arbitration:
    """
    Multi-board arbitration using weighted voting.
    
    Adapted from CorporateSwarm's _analyze_vote_results pattern (lines 2026-2062).
    Implements multiple arbitration methods:
    - Pareto intersection
    - Minimax regret
    - Borda voting
    - Weighted voting
    """
    
    @staticmethod
    def pareto_intersection(
        boards: List[Board],
        policies: List[ControlVector],
        objectives: np.ndarray
    ) -> List[ControlVector]:
        """
        Find policies that are non-dominated under all boards.
        
        Args:
            boards: List of boards
            policies: Candidate policies
            objectives: Objective matrix
            
        Returns:
            List[ControlVector]: Consensus policies
        """
        if not boards or not policies:
            return []
        
        # Get Pareto-efficient policies for each board
        board_pareto_sets = []
        for board in boards:
            ranked = board.evaluate_policies(policies, objectives)
            # Take top 50% as "preferred"
            n_top = max(1, len(ranked) // 2)
            top_policies = [p for p, _ in ranked[:n_top]]
            board_pareto_sets.append(set(id(p) for p in top_policies))
        
        # Intersection: policies preferred by all boards
        consensus_ids = set.intersection(*board_pareto_sets) if board_pareto_sets else set()
        consensus_policies = [p for p in policies if id(p) in consensus_ids]
        
        return consensus_policies
    
    @staticmethod
    def minimax_regret(
        boards: List[Board],
        policies: List[ControlVector],
        objectives: np.ndarray
    ) -> Optional[ControlVector]:
        """
        Minimize worst-case regret across boards.
        
        Args:
            boards: List of boards
            policies: Candidate policies
            objectives: Objective matrix
            
        Returns:
            Optional[ControlVector]: Optimal policy
        """
        if not boards or not policies:
            return None
        
        # For each board, find best policy
        board_best_scores = []
        for board in boards:
            ranked = board.evaluate_policies(policies, objectives)
            if ranked:
                best_score = ranked[0][1]  # Best score for this board
                board_best_scores.append(best_score)
        
        # For each policy, compute maximum regret
        max_regrets = []
        for i, policy in enumerate(policies):
            regrets = []
            for j, board in enumerate(boards):
                ranked = board.evaluate_policies(policies, objectives)
                policy_score = next((s for p, s in ranked if p is policy), float('inf'))
                regret = policy_score - board_best_scores[j]
                regrets.append(regret)
            max_regrets.append(max(regrets))
        
        # Policy with minimum maximum regret
        min_regret_idx = np.argmin(max_regrets)
        return policies[min_regret_idx]
    
    @staticmethod
    def borda_vote(
        boards: List[Board],
        policies: List[ControlVector],
        objectives: np.ndarray
    ) -> Optional[ControlVector]:
        """
        Borda count aggregation across boards.
        
        Args:
            boards: List of boards
            policies: Candidate policies
            objectives: Objective matrix
            
        Returns:
            Optional[ControlVector]: Selected policy
        """
        if not boards or not policies:
            return None
        
        # Borda scores: rank 0 gets n-1 points, rank 1 gets n-2, etc.
        borda_scores = {id(p): 0.0 for p in policies}
        
        for board in boards:
            ranked = board.evaluate_policies(policies, objectives)
            n = len(ranked)
            for rank, (policy, _) in enumerate(ranked):
                borda_scores[id(policy)] += (n - rank - 1) * board.members[0].voting_weight if board.members else (n - rank - 1)
        
        # Policy with highest Borda score
        best_policy_id = max(borda_scores, key=borda_scores.get)
        return next(p for p in policies if id(p) == best_policy_id)
    
    @staticmethod
    def weighted_vote(
        boards: List[Board],
        policies: List[ControlVector],
        objectives: np.ndarray
    ) -> Optional[ControlVector]:
        """
        Weighted voting using CorporateSwarm's _analyze_vote_results pattern.
        
        Adapted from CorporateSwarm lines 2026-2062.
        
        Args:
            boards: List of boards
            policies: Candidate policies
            objectives: Objective matrix
            
        Returns:
            Optional[ControlVector]: Selected policy
        """
        if not boards or not policies:
            return None
        
        # Collect individual "votes" from each board
        individual_votes: Dict[str, Dict[str, Any]] = {}
        
        for board in boards:
            ranked = board.evaluate_policies(policies, objectives)
            if not ranked:
                continue
            
            # Board votes for top policy
            top_policy, top_score = ranked[0]
            
            # Compute board voting weight (sum of member weights)
            board_weight = sum(m.voting_weight for m in board.members)
            
            individual_votes[board.board_id] = {
                "vote": "APPROVE" if top_score < float('inf') else "REJECT",
                "policy": top_policy,
                "score": top_score,
                "voting_weight": board_weight,
                "board_type": board.board_type.value,
            }
        
        if not individual_votes:
            return None
        
        # Aggregate votes by policy (weighted)
        policy_scores: Dict[int, float] = {}  # policy_id -> weighted score
        
        for vote_data in individual_votes.values():
            policy = vote_data["policy"]
            weight = vote_data["voting_weight"]
            score = vote_data["score"]
            
            policy_id = id(policy)
            if policy_id not in policy_scores:
                policy_scores[policy_id] = 0.0
            
            # Lower score is better, so we use negative for voting
            policy_scores[policy_id] += weight / (1.0 + abs(score))
        
        # Policy with highest weighted score
        if not policy_scores:
            return None
        
        best_policy_id = max(policy_scores, key=policy_scores.get)
        return next(p for p in policies if id(p) == best_policy_id)


class GovernanceSystem:
    """
    Governance system orchestrating boards and arbitration.
    
    Pattern similar to CorporateSwarm's conduct_corporate_vote (lines 1035-1057).
    
    Args:
        boards: List of governance boards
        arbitration_method: Method for policy selection
    """
    
    def __init__(
        self,
        boards: List[Board],
        arbitration_method: str = "weighted_vote",
        major_change_threshold: float = 0.10,  # 10% = major change
    ) -> None:
        """Initialize governance system."""
        self.boards = boards
        self.arbitration_method = arbitration_method
        self.major_change_threshold = major_change_threshold
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.approval_history: List[Dict[str, Any]] = []
        self.emergency_stop_active = False
    
    def select_policy(
        self,
        candidate_policies: List[ControlVector],
        objectives: np.ndarray
    ) -> Optional[ControlVector]:
        """
        Select final policy using arbitration method.
        
        Args:
            candidate_policies: List of candidate policies
            objectives: Objective matrix
            
        Returns:
            Optional[ControlVector]: Selected policy
        """
        if not candidate_policies:
            return None
        
        if self.arbitration_method == "pareto_intersection":
            consensus = Arbitration.pareto_intersection(self.boards, candidate_policies, objectives)
            return consensus[0] if consensus else None
        
        elif self.arbitration_method == "minimax_regret":
            return Arbitration.minimax_regret(self.boards, candidate_policies, objectives)
        
        elif self.arbitration_method == "borda":
            return Arbitration.borda_vote(self.boards, candidate_policies, objectives)
        
        elif self.arbitration_method == "weighted_vote":
            return Arbitration.weighted_vote(self.boards, candidate_policies, objectives)
        
        else:
            logger.warning(f"Unknown arbitration method: {self.arbitration_method}, using weighted_vote")
            return Arbitration.weighted_vote(self.boards, candidate_policies, objectives)
    
    def requires_approval(
        self,
        policy: ControlVector,
        previous_policy: Optional[ControlVector]
    ) -> Tuple[bool, str]:
        """
        Check if policy requires human approval (major change > 10%).
        
        Args:
            policy: Proposed policy
            previous_policy: Previous policy (None if first)
            
        Returns:
            Tuple[bool, str]: (requires_approval, reason)
        """
        if self.emergency_stop_active:
            return True, "Emergency stop active"
        
        if previous_policy is None:
            return True, "First policy requires approval"
        
        # Check if major change (> 10% in any category)
        for cat in set(list(policy.budget_shares.keys()) + list(previous_policy.budget_shares.keys())):
            prev_val = previous_policy.budget_shares.get(cat, 0.0)
            curr_val = policy.budget_shares.get(cat, 0.0)
            change = abs(curr_val - prev_val)
            
            if change > self.major_change_threshold:
                return True, f"Major change in {cat}: {change:.1%} > {self.major_change_threshold:.1%}"
        
        return False, "Minor change, automated execution allowed"
    
    def request_approval(
        self,
        policy: ControlVector,
        reason: str,
        user_id: str,
        objectives: Optional[np.ndarray] = None
    ) -> str:
        """
        Request human approval for a policy.
        
        Args:
            policy: Policy requiring approval
            reason: Reason for approval request
            user_id: User requesting approval
            objectives: Optional objective values for context
            
        Returns:
            str: Approval request ID
        """
        approval_id = str(uuid.uuid4())
        
        self.pending_approvals[approval_id] = {
            "approval_id": approval_id,
            "policy": policy,
            "reason": reason,
            "requested_by": user_id,
            "objectives": objectives.tolist() if objectives is not None else None,
            "timestamp": time.time(),
            "status": "pending",
            "approved_by": [],
        }
        
        logger.info(f"Approval requested: {approval_id} by {user_id} - {reason}")
        
        return approval_id
    
    def approve_policy(
        self,
        approval_id: str,
        user_id: str,
        comment: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Approve a pending policy.
        
        Args:
            approval_id: Approval request ID
            user_id: User approving
            comment: Optional comment
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if approval_id not in self.pending_approvals:
            return False, "Approval request not found"
        
        approval = self.pending_approvals[approval_id]
        
        if approval["status"] != "pending":
            return False, f"Approval already {approval['status']}"
        
        approval["approved_by"].append(user_id)
        approval["status"] = "approved"
        approval["approval_comment"] = comment
        
        # Move to history
        self.approval_history.append(approval.copy())
        del self.pending_approvals[approval_id]
        
        logger.info(f"Policy approved: {approval_id} by {user_id}")
        
        return True, "Policy approved"
    
    def reject_policy(
        self,
        approval_id: str,
        user_id: str,
        reason: str
    ) -> Tuple[bool, str]:
        """
        Reject a pending policy.
        
        Args:
            approval_id: Approval request ID
            user_id: User rejecting
            reason: Rejection reason
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if approval_id not in self.pending_approvals:
            return False, "Approval request not found"
        
        approval = self.pending_approvals[approval_id]
        approval["status"] = "rejected"
        approval["rejected_by"] = user_id
        approval["rejection_reason"] = reason
        
        # Move to history
        self.approval_history.append(approval.copy())
        del self.pending_approvals[approval_id]
        
        logger.info(f"Policy rejected: {approval_id} by {user_id} - {reason}")
        
        return True, "Policy rejected"
    
    def override_policy(
        self,
        policy: ControlVector,
        user_id: str,
        reason: str
    ) -> Tuple[bool, str]:
        """
        Override automated decision with manual policy.
        
        Args:
            policy: Override policy
            user_id: User overriding
            reason: Reason for override
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        override_id = str(uuid.uuid4())
        
        override_entry = {
            "override_id": override_id,
            "policy": policy,
            "user_id": user_id,
            "reason": reason,
            "timestamp": time.time(),
        }
        
        self.approval_history.append(override_entry)
        
        logger.warning(f"Policy override: {override_id} by {user_id} - {reason}")
        
        return True, f"Override executed: {override_id}"
    
    def emergency_stop(self, user_id: str, reason: str) -> None:
        """
        Activate emergency stop (halt all automation).
        
        Args:
            user_id: User activating emergency stop
            reason: Reason for emergency stop
        """
        self.emergency_stop_active = True
        
        logger.critical(f"EMERGENCY STOP activated by {user_id}: {reason}")
    
    def resume_operations(self, user_id: str) -> None:
        """
        Resume operations after emergency stop.
        
        Args:
            user_id: User resuming operations
        """
        self.emergency_stop_active = False
        
        logger.info(f"Operations resumed by {user_id}")
    
    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get list of pending approvals."""
        return list(self.pending_approvals.values())


# RiskAssessment: Adapted from CorporateSwarm (lines 427-440)
if PYDANTIC_AVAILABLE:
    class RiskAssessment(BaseModel):
        """Risk assessment model for constraint violations."""
        
        risk_id: str = Field(default_factory=_generate_uuid)
        risk_category: str = Field(default="constraint_violation")
        risk_level: str = Field(default="medium")
        probability: float = Field(default=0.5, ge=0.0, le=1.0)
        impact: float = Field(default=0.5, ge=0.0, le=1.0)
        risk_score: float = Field(default=0.25, ge=0.0, le=1.0)
        mitigation_strategies: List[str] = Field(default_factory=list)
        owner: str = Field(default="")
        status: str = Field(default="active")
        last_reviewed: float = Field(default_factory=time.time)
else:
    @dataclass
    class RiskAssessment:
        """Risk assessment model for constraint violations."""
        risk_id: str = field(default_factory=_generate_uuid)
        risk_category: str = "constraint_violation"
        risk_level: str = "medium"
        probability: float = 0.5
        impact: float = 0.5
        risk_score: float = 0.25
        mitigation_strategies: List[str] = field(default_factory=list)
        owner: str = ""
        status: str = "active"
        last_reviewed: float = field(default_factory=time.time)


class LogisticsNetwork:
    """
    Multi-commodity flow optimization network.
    
    Graph G=(V,E) with nodes (regions, ports, warehouses) and edges (transport links).
    Solves linear program for optimal flow allocation.
    
    Args:
        nodes: List of node identifiers
        edges: List of (source, target) edge tuples
        capacities: Dictionary mapping edges to capacities
        costs: Dictionary mapping edges to transport costs
    """
    
    def __init__(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        capacities: Optional[Dict[Tuple[str, str], float]] = None,
        costs: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> None:
        """Initialize logistics network."""
        self.nodes = nodes
        self.edges = edges
        self.capacities = capacities or {}
        self.costs = costs or {}
        
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(nodes)
            self.graph.add_edges_from(edges)
    
    def solve_flow_problem(
        self,
        demands: Dict[str, Dict[str, float]],  # node -> commodity -> demand
        commodities: List[str],
        capacities: Optional[Dict[Tuple[str, str], float]] = None,
        costs: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Solve multi-commodity flow optimization problem.
        
        Args:
            demands: Node demands by commodity
            commodities: List of commodity types
            capacities: Edge capacities (default: use instance)
            costs: Edge costs (default: use instance)
            
        Returns:
            Dict[Tuple[str, str, str], float]: Flow values (source, target, commodity) -> flow
        """
        if not CVXPY_AVAILABLE:
            logger.warning("cvxpy not available, using heuristic flow allocation")
            return self._heuristic_flow(demands, commodities)
        
        capacities = capacities or self.capacities
        costs = costs or self.costs
        
        # Decision variables: flow[e, q] for each edge and commodity
        flows = {}
        for edge in self.edges:
            for commodity in commodities:
                flows[(edge[0], edge[1], commodity)] = cp.Variable(nonneg=True)
        
        # Objective: minimize total cost
        objective = cp.Minimize(
            sum(
                costs.get((edge[0], edge[1]), 1.0) * flows[(edge[0], edge[1], commodity)]
                for edge in self.edges
                for commodity in commodities
            )
        )
        
        # Constraints
        constraints = []
        
        # Flow capacity per edge
        for edge in self.edges:
            total_flow = sum(flows[(edge[0], edge[1], q)] for q in commodities)
            cap = capacities.get(edge, float('inf'))
            constraints.append(total_flow <= cap)
        
        # Flow balance per node and commodity
        for node in self.nodes:
            for commodity in commodities:
                inflow = sum(
                    flows.get((src, node, commodity), 0)
                    for src in self.nodes
                    if (src, node) in self.edges
                )
                outflow = sum(
                    flows.get((node, tgt, commodity), 0)
                    for tgt in self.nodes
                    if (node, tgt) in self.edges
                )
                demand = demands.get(node, {}).get(commodity, 0.0)
                constraints.append(inflow - outflow >= demand)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS if hasattr(cp, 'ECOS') else cp.SCS)
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(f"Flow problem status: {problem.status}, using heuristic")
                return self._heuristic_flow(demands, commodities)
            
            # Extract solution
            solution = {}
            for key, var in flows.items():
                solution[key] = float(var.value) if var.value is not None else 0.0
            
            return solution
        
        except Exception as e:
            logger.error(f"Flow optimization failed: {e}, using heuristic")
            return self._heuristic_flow(demands, commodities)
    
    def _heuristic_flow(
        self,
        demands: Dict[str, Dict[str, float]],
        commodities: List[str]
    ) -> Dict[Tuple[str, str, str], float]:
        """Heuristic flow allocation (fallback)."""
        flows = {}
        for edge in self.edges:
            for commodity in commodities:
                # Simple heuristic: allocate based on demand
                source_demand = demands.get(edge[0], {}).get(commodity, 0.0)
                flows[(edge[0], edge[1], commodity)] = source_demand * 0.1  # 10% flow
        return flows
    
    def compute_shadow_prices(self) -> Dict[Tuple[str, str], float]:
        """
        Compute shadow prices (dual variables) for bottleneck analysis.
        
        Returns:
            Dict[Tuple[str, str], float]: Edge -> shadow price
        """
        # Placeholder: would extract dual variables from LP solution
        logger.warning("Shadow price computation requires solved LP, returning zeros")
        return {edge: 0.0 for edge in self.edges}


class Visualization:
    """
    Visualization and dashboard components.
    
    Provides:
    - State dashboard
    - Pareto frontier plots
    - Constraint violation maps
    - Bottleneck analysis
    
    Can use Formatter from utils for enhanced rich formatting.
    """
    
    _formatter: Optional[Any] = None
    
    @classmethod
    def _get_formatter(cls) -> Optional[Any]:
        """Get Formatter instance from utils if available."""
        if cls._formatter is None:
            try:
                from utils.formatter import Formatter
                cls._formatter = Formatter(md=True)
            except ImportError:
                cls._formatter = None
        return cls._formatter
    
    @staticmethod
    def state_dashboard(x_t: StateVector) -> str:
        """
        Generate text dashboard for state vector.
        
        Args:
            x_t: State vector
            
        Returns:
            str: Formatted dashboard string
        """
        dashboard = f"""
=== CRCA-SD State Dashboard ===
Population (P):        {x_t.P:,.0f}
Labor Force (L):       {x_t.L:,.0f}
Unemployment (U):      {x_t.U:.1%}
Wage (W):              {x_t.W:.2f}
Stability (S):         {x_t.S:.1%}
Literacy:              {x_t.literacy:.1%}
Education Capacity:    {x_t.Ecap:,.0f}
Healthcare Capacity:   {x_t.Hcap:,.0f}
Capital Stock (K):     {x_t.K:,.0f}
Infrastructure (I):    {x_t.I:.1%}
Transport Capacity:   {x_t.Tcap:,.0f}
Energy Stock:          {x_t.E_stock:,.0f}
Food Stock:            {x_t.F_stock:,.0f}
Materials Stock:       {x_t.M_stock:,.0f}
Ecological Damage (C): {x_t.C:,.0f}
Output (Y):            {x_t.Y:,.0f}
===============================
"""
        return dashboard
    
    @staticmethod
    def pareto_visualizer(
        policies: List[ControlVector],
        objectives: np.ndarray,
        objective_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate text visualization of Pareto frontier.
        
        Args:
            policies: List of policies
            objectives: Objective matrix
            objective_names: Names of objectives (default: J_U, J_ℓ, etc.)
            
        Returns:
            str: Formatted visualization string
        """
        if objective_names is None:
            objective_names = ["J_U", "J_ℓ", "J_Y", "J_ineq", "J_C", "J_risk"]
        
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)
        
        viz = "=== Pareto Frontier ===\n"
        viz += f"Policies: {len(policies)}\n"
        viz += f"Objectives: {', '.join(objective_names)}\n\n"
        
        for i, (policy, obj_vec) in enumerate(zip(policies, objectives)):
            viz += f"Policy {i+1}:\n"
            for name, value in zip(objective_names, obj_vec):
                viz += f"  {name}: {value:.4f}\n"
            viz += "\n"
        
        return viz
    
    @staticmethod
    def constraint_map(
        violations: List[str],
        x_t: StateVector
    ) -> str:
        """
        Generate constraint violation map.
        
        Args:
            violations: List of violation messages
            x_t: State vector
            
        Returns:
            str: Formatted violation map
        """
        map_str = "=== Constraint Violation Map ===\n"
        if not violations:
            map_str += "✓ All constraints satisfied\n"
        else:
            map_str += f"✗ {len(violations)} violations:\n"
            for violation in violations:
                map_str += f"  - {violation}\n"
        return map_str
    
    @staticmethod
    def realtime_dashboard(
        x_t: StateVector,
        execution_status: Optional[Dict[str, Any]] = None,
        violations: Optional[List[str]] = None,
        pending_approvals: Optional[List[Dict[str, Any]]] = None,
        system_health: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate real-time dashboard with execution status and monitoring.
        
        Args:
            x_t: Current state vector
            execution_status: Policy execution status
            violations: Constraint violations
            pending_approvals: Pending approval requests
            system_health: System health status
            
        Returns:
            str: Formatted real-time dashboard
        """
        dashboard = "=== CRCA-SD Real-Time Dashboard ===\n"
        dashboard += f"Last Update: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # State section
        dashboard += "--- Current State ---\n"
        dashboard += Visualization.state_dashboard(x_t)
        dashboard += "\n"
        
        # Execution status
        if execution_status:
            dashboard += "--- Execution Status ---\n"
            status = execution_status.get("status", "unknown")
            dashboard += f"Status: {status}\n"
            if "execution_id" in execution_status:
                dashboard += f"Execution ID: {execution_status['execution_id']}\n"
            if "requires_approval" in execution_status:
                dashboard += f"Requires Approval: {execution_status['requires_approval']}\n"
            dashboard += "\n"
        
        # Violations
        if violations:
            dashboard += "--- Constraint Violations ---\n"
            dashboard += Visualization.constraint_map(violations, x_t)
            dashboard += "\n"
        
        # Pending approvals
        if pending_approvals:
            dashboard += "--- Pending Approvals ---\n"
            dashboard += f"Count: {len(pending_approvals)}\n"
            for approval in pending_approvals[:5]:  # Show first 5
                dashboard += f"  - {approval.get('approval_id', 'unknown')}: {approval.get('reason', 'N/A')}\n"
            dashboard += "\n"
        
        # System health
        if system_health:
            dashboard += "--- System Health ---\n"
            dashboard += f"Status: {system_health.get('status', 'unknown')}\n"
            dashboard += f"Violations (24h): {system_health.get('n_violations_24h', 0)}\n"
            dashboard += "\n"
        
        dashboard += "===============================\n"
        
        return dashboard
    
    @staticmethod
    def execution_history_visualization(
        execution_history: List[Dict[str, Any]],
        n_recent: int = 10
    ) -> str:
        """
        Visualize execution history.
        
        Args:
            execution_history: List of execution records
            n_recent: Number of recent executions to show
            
        Returns:
            str: Formatted visualization
        """
        viz = "=== Execution History ===\n"
        
        recent = execution_history[-n_recent:] if len(execution_history) > n_recent else execution_history
        
        for exec_record in recent:
            exec_id = exec_record.get("execution_id", "unknown")
            status = exec_record.get("status", "unknown")
            timestamp = exec_record.get("timestamp", 0)
            time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M') if timestamp > 0 else "N/A"
            
            viz += f"{time_str} | {exec_id[:8]}... | {status}\n"
        
        return viz
    
    @staticmethod
    def policy_comparison_visualization(
        policies: List[ControlVector],
        policy_names: Optional[List[str]] = None
    ) -> str:
        """
        Visualize comparison of multiple policies.
        
        Args:
            policies: List of policies to compare
            policy_names: Optional names for policies
            
        Returns:
            str: Formatted comparison
        """
        if policy_names is None:
            policy_names = [f"Policy {i+1}" for i in range(len(policies))]
        
        viz = "=== Policy Comparison ===\n"
        
        # Get all budget categories
        all_categories = set()
        for policy in policies:
            all_categories.update(policy.budget_shares.keys())
        
        # Header
        viz += "Category".ljust(20)
        for name in policy_names:
            viz += name[:15].ljust(18)
        viz += "\n"
        viz += "-" * (20 + 18 * len(policy_names)) + "\n"
        
        # Rows
        for cat in sorted(all_categories):
            viz += cat[:19].ljust(20)
            for policy in policies:
                value = policy.budget_shares.get(cat, 0.0)
                viz += f"{value:.1%}".ljust(18)
            viz += "\n"
        
        return viz


# Config: Using CorporateSwarm CorporateConfigModel pattern (lines 75-158)
if PYDANTIC_AVAILABLE:
    class CRCA_SD_ConfigModel(BaseModel):
        """Configuration model for CRCA-SD (Pydantic)."""
        
        # State bounds
        U_max: float = Field(default=0.2, ge=0.0, le=1.0)
        S_min: float = Field(default=0.3, ge=0.0, le=1.0)
        c_min: float = Field(default=0.02, ge=0.0)
        
        # Dynamics parameters
        delta_K: float = Field(default=0.05, ge=0.0, le=1.0)
        delta_I: float = Field(default=0.02, ge=0.0, le=1.0)
        alpha: float = Field(default=0.3, ge=0.0, le=1.0)
        kappa_K: float = Field(default=0.8, ge=0.0)
        kappa_I: float = Field(default=0.7, ge=0.0)
        kappa_literacy: float = Field(default=0.1, ge=0.0)
        
        # MPC parameters
        horizon: int = Field(default=10, ge=1, le=50)
        n_scenarios: int = Field(default=10, ge=1, le=100)
        objective_weights: List[float] = Field(default_factory=lambda: [1.0/6] * 6)
        
        # Stability parameters
        max_budget_change: float = Field(default=0.2, ge=0.0, le=1.0)
        investment_smoothing: float = Field(default=0.7, ge=0.0, le=1.0)
        
        # Board parameters
        n_boards: int = Field(default=4, ge=1, le=10)
        arbitration_method: str = Field(default="weighted_vote")
        
        # Logistics parameters
        default_transport_capacity: float = Field(default=10000.0, ge=0.0)
        
        # Risk parameters
        cvar_alpha: float = Field(default=0.05, ge=0.0, le=1.0)
else:
    @dataclass
    class CRCA_SD_ConfigModel:
        """Configuration model for CRCA-SD (dataclass fallback)."""
        U_max: float = 0.2
        S_min: float = 0.3
        c_min: float = 0.02
        delta_K: float = 0.05
        delta_I: float = 0.02
        alpha: float = 0.3
        kappa_K: float = 0.8
        kappa_I: float = 0.7
        kappa_literacy: float = 0.1
        horizon: int = 10
        n_scenarios: int = 10
        objective_weights: List[float] = field(default_factory=lambda: [1.0/6] * 6)
        max_budget_change: float = 0.2
        investment_smoothing: float = 0.7
        n_boards: int = 4
        arbitration_method: str = "weighted_vote"
        default_transport_capacity: float = 10000.0
        cvar_alpha: float = 0.05


@dataclass
class CRCA_SD_Config:
    """Configuration manager (pattern from CorporateSwarm lines 160-227)."""
    
    config_file_path: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    config: CRCA_SD_ConfigModel = field(init=False)
    
    def __post_init__(self) -> None:
        """Load configuration."""
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration with priority: explicit data > file > defaults."""
        try:
            self.config = CRCA_SD_ConfigModel()
            if self.config_file_path and os.path.exists(self.config_file_path):
                self._load_from_file()
            if self.config_data:
                self._load_from_dict(self.config_data)
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            raise ValueError(f"Configuration loading failed: {e}") from e
    
    def _load_from_file(self) -> None:
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("YAML not available, cannot load config file")
            return
        
        try:
            with open(self.config_file_path, "r") as f:
                self._load_from_dict(yaml.safe_load(f))
                logger.info(f"Loaded config from: {self.config_file_path}")
        except Exception as e:
            logger.warning(f"File loading failed {self.config_file_path}: {e}")
            raise ValueError(f"Configuration file loading failed: {e}") from e
    
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                try:
                    setattr(self.config, key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Config {key} failed: {e}")
                    raise ValueError(f"Invalid configuration value for {key}: {e}") from e
    
    def get_config(self) -> CRCA_SD_ConfigModel:
        """Get configuration model."""
        return self.config


# Global config cache (pattern from CorporateSwarm lines 233-239)
_crca_sd_config: Optional[CRCA_SD_Config] = None

@lru_cache(maxsize=1)
def get_crca_sd_config(config_file_path: Optional[str] = None) -> CRCA_SD_Config:
    """Get global CRCA-SD configuration instance."""
    global _crca_sd_config
    if _crca_sd_config is None:
        _crca_sd_config = CRCA_SD_Config(config_file_path=config_file_path)
    return _crca_sd_config


class MetricsCollector:
    """
    Metrics collector for feasibility, stability, robustness.
    
    Tracks:
    - Feasibility: scenario feasibility rate, minimum slack
    - Stability: variance of U_t, Y_t, C_t, oscillation frequency, control effort
    - Robustness: worst-case performance, CVaR at different α
    - Bottleneck causality: which constraints bind most often
    """
    
    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics_history: List[Dict[str, Any]] = []
    
    def compute_all_metrics(
        self,
        trajectories: List[List[StateVector]],
        policies: List[ControlVector]
    ) -> Dict[str, Any]:
        """
        Compute all metrics from trajectories and policies.
        
        Args:
            trajectories: List of state trajectories
            policies: List of control policies
            
        Returns:
            Dict[str, Any]: Metrics dictionary
        """
        metrics = {}
        
        # Feasibility metrics
        n_feasible = sum(1 for traj in trajectories if len(traj) > 1)
        metrics["feasibility_rate"] = n_feasible / len(trajectories) if trajectories else 0.0
        
        # Stability metrics (variance)
        if trajectories:
            U_values = [x.U for traj in trajectories for x in traj[1:]]
            Y_values = [x.Y for traj in trajectories for x in traj[1:]]
            C_values = [x.C for traj in trajectories for x in traj[1:]]
            
            metrics["U_variance"] = float(np.var(U_values)) if U_values else 0.0
            metrics["Y_variance"] = float(np.var(Y_values)) if Y_values else 0.0
            metrics["C_variance"] = float(np.var(C_values)) if C_values else 0.0
        
        # Control effort (sum of changes)
        if len(policies) > 1:
            effort = 0.0
            for i in range(1, len(policies)):
                prev_shares = policies[i-1].budget_shares
                curr_shares = policies[i].budget_shares
                change = sum(abs(curr_shares.get(k, 0) - prev_shares.get(k, 0)) for k in set(list(prev_shares.keys()) + list(curr_shares.keys())))
                effort += change
            metrics["control_effort"] = effort
        else:
            metrics["control_effort"] = 0.0
        
        # Robustness (worst-case)
        if trajectories:
            final_states = [traj[-1] for traj in trajectories if traj]
            worst_U = max((x.U for x in final_states), default=0.0)
            worst_S = min((x.S for x in final_states), default=1.0)
            metrics["worst_case_U"] = float(worst_U)
            metrics["worst_case_S"] = float(worst_S)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def generate_report(self) -> str:
        """
        Generate formatted metrics report.
        
        Returns:
            str: Formatted report string
        """
        if not self.metrics_history:
            return "No metrics collected yet."
        
        latest = self.metrics_history[-1]
        report = "=== CRCA-SD Metrics Report ===\n"
        report += f"Feasibility Rate: {latest.get('feasibility_rate', 0.0):.1%}\n"
        report += f"Unemployment Variance: {latest.get('U_variance', 0.0):.4f}\n"
        report += f"Output Variance: {latest.get('Y_variance', 0.0):.2f}\n"
        report += f"Control Effort: {latest.get('control_effort', 0.0):.4f}\n"
        report += f"Worst Case Unemployment: {latest.get('worst_case_U', 0.0):.1%}\n"
        report += f"Worst Case Stability: {latest.get('worst_case_S', 0.0):.1%}\n"
        report += "============================\n"
        
        return report

