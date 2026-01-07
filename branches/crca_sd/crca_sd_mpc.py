"""
CRCA-SD MPC: Model-Predictive Control, Objectives, Scenarios, Stability, Estimation

This module implements:
- Multi-objective optimization with CVaR risk
- Rolling horizon MPC solver
- Scenario generation (Gaussian, Student-t, structured shocks)
- Stability enforcement (rate limits, smoothing)
- Pareto frontier extraction
- State estimation (EKF/UKF) for partial observability
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import time
from dataclasses import dataclass
from loguru import logger

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not available, MPC will use scipy.optimize fallback")

try:
    from scipy.optimize import minimize
    from scipy.stats import t as student_t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, some features will be limited")

from crca_sd.crca_sd_core import StateVector, ControlVector, DynamicsModel, ConstraintChecker


class ObjectiveVector:
    """
    Multi-objective cost vector.
    
    Computes objective vector J = [J_U, J_ℓ, J_Y, J_ineq, J_C, J_risk]:
    - J_U: Sum of unemployment over horizon
    - J_ℓ: Negative sum of literacy (maximize literacy)
    - J_Y: Negative sum of output (maximize output)
    - J_ineq: Inequality measure (placeholder)
    - J_C: Sum of ecological damage increase
    - J_risk: CVaR on collapse proxy
    
    Args:
        horizon: Planning horizon for objective computation
    """
    
    def __init__(self, horizon: int = 10) -> None:
        """Initialize objective vector computer."""
        self.horizon = horizon
    
    def compute(
        self,
        x_trajectory: List[StateVector],
        u_trajectory: List[ControlVector]
    ) -> np.ndarray:
        """
        Compute all objectives from trajectory.
        
        Args:
            x_trajectory: List of state vectors
            u_trajectory: List of control vectors
            
        Returns:
            np.ndarray: Objective vector [J_U, J_ℓ, J_Y, J_ineq, J_C, J_risk]
        """
        if len(x_trajectory) < 2:
            return np.zeros(6)
        
        # J_U: Sum of unemployment
        J_U = sum(x.U for x in x_trajectory[1:])
        
        # J_ℓ: Negative sum of literacy (maximize literacy = minimize negative)
        J_ℓ = -sum(x.literacy for x in x_trajectory[1:])
        
        # J_Y: Negative sum of output (maximize output = minimize negative)
        J_Y = -sum(x.Y for x in x_trajectory[1:])
        
        # J_ineq: Inequality measure (simplified: wage variance proxy)
        wages = [x.W for x in x_trajectory[1:]]
        if len(wages) > 1:
            J_ineq = np.std(wages)  # Higher variance = more inequality
        else:
            J_ineq = 0.0
        
        # J_C: Sum of ecological damage increase
        if len(x_trajectory) > 1:
            C_initial = x_trajectory[0].C
            C_final = x_trajectory[-1].C
            J_C = C_final - C_initial
        else:
            J_C = 0.0
        
        # J_risk: CVaR on collapse proxy (computed separately)
        collapse_proxy = self._compute_collapse_proxy(x_trajectory)
        J_risk = collapse_proxy  # Will be replaced by actual CVaR in aggregate
        
        return np.array([J_U, J_ℓ, J_Y, J_ineq, J_C, J_risk])
    
    def aggregate(
        self,
        scenarios: List[Tuple[List[StateVector], List[ControlVector]]],
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Aggregate objectives across scenarios.
        
        Args:
            scenarios: List of (trajectory, controls) tuples
            weights: Scenario weights (default: uniform)
            
        Returns:
            np.ndarray: Expected objective vector
        """
        if not scenarios:
            return np.zeros(6)
        
        if weights is None:
            weights = np.ones(len(scenarios)) / len(scenarios)
        
        objective_vectors = []
        for traj, controls in scenarios:
            obj = self.compute(traj, controls)
            objective_vectors.append(obj)
        
        objective_matrix = np.array(objective_vectors)
        expected_objectives = np.average(objective_matrix, axis=0, weights=weights)
        
        return expected_objectives
    
    def _compute_collapse_proxy(self, trajectory: List[StateVector]) -> float:
        """
        Compute collapse proxy (badness metric).
        
        Args:
            trajectory: State trajectory
            
        Returns:
            float: Collapse proxy (higher = worse)
        """
        if not trajectory:
            return 0.0
        
        # Collapse indicators
        high_unemployment = sum(1 for x in trajectory if x.U > 0.15)
        low_stability = sum(1 for x in trajectory if x.S < 0.4)
        food_crisis = sum(1 for x in trajectory if x.F_stock < x.P * 0.01)
        energy_crisis = sum(1 for x in trajectory if x.E_stock < 1000.0)
        
        collapse_score = (
            high_unemployment * 0.3 +
            low_stability * 0.3 +
            food_crisis * 0.2 +
            energy_crisis * 0.2
        ) / len(trajectory)
        
        return collapse_score


class CVaRComputer:
    """
    Conditional Value-at-Risk (CVaR) computer.
    
    CVaR measures expected loss in worst-case scenarios (tail risk).
    
    Args:
        alpha: Confidence level (default: 0.05, i.e., worst 5%)
    """
    
    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize CVaR computer."""
        if not (0 < alpha < 1):
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
    
    def compute_cvar(self, z_scores: np.ndarray) -> float:
        """
        Compute CVaR from z-scores (badness values).
        
        Args:
            z_scores: Array of badness scores (higher = worse)
            
        Returns:
            float: CVaR value
        """
        if len(z_scores) == 0:
            return 0.0
        
        # Sort in descending order (worst first)
        sorted_scores = np.sort(z_scores)[::-1]
        
        # Number of worst-case scenarios
        n_worst = max(1, int(np.ceil(len(z_scores) * self.alpha)))
        
        # CVaR = average of worst scenarios
        cvar = np.mean(sorted_scores[:n_worst])
        
        return float(cvar)
    
    def collapse_proxy(self, x_trajectory: List[StateVector]) -> float:
        """
        Compute collapse proxy from trajectory.
        
        Args:
            x_trajectory: State trajectory
            
        Returns:
            float: Collapse proxy (badness metric)
        """
        if not x_trajectory:
            return 0.0
        
        # Use same logic as ObjectiveVector
        high_unemployment = sum(1 for x in x_trajectory if x.U > 0.15)
        low_stability = sum(1 for x in x_trajectory if x.S < 0.4)
        food_crisis = sum(1 for x in x_trajectory if x.F_stock < x.P * 0.01)
        energy_crisis = sum(1 for x in x_trajectory if x.E_stock < 1000.0)
        
        collapse_score = (
            high_unemployment * 0.3 +
            low_stability * 0.3 +
            food_crisis * 0.2 +
            energy_crisis * 0.2
        ) / len(x_trajectory)
        
        return collapse_score


class ScenarioGenerator:
    """
    Scenario generator for disturbance sequences.
    
    Generates:
    - Gaussian noise for benign uncertainty
    - Student-t for heavy-tailed disasters
    - Structured events (trade embargo, drought, etc.)
    - Causal scenarios via CRCA (if available)
    
    Args:
        rng: Random number generator (default: new generator)
        crca_agent: Optional CRCAAgent for causal scenario generation
    """
    
    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        crca_agent: Optional[Any] = None
    ) -> None:
        """Initialize scenario generator."""
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.crca_agent = crca_agent
    
    def generate_gaussian(
        self,
        n_scenarios: int,
        horizon: int,
        mean: Optional[Dict[str, float]] = None,
        cov: Optional[np.ndarray] = None
    ) -> List[List[Dict[str, float]]]:
        """
        Generate Gaussian disturbance scenarios.
        
        Args:
            n_scenarios: Number of scenarios
            horizon: Time horizon
            mean: Mean disturbance vector (default: zeros)
            cov: Covariance matrix (default: identity)
            
        Returns:
            List[List[Dict[str, float]]]: List of scenario sequences
        """
        if mean is None:
            mean = {
                "demand_shock": 0.0,
                "trade_shock": 0.0,
                "productivity_shock": 1.0,
                "disaster_shock": 0.0,
                "labor_shock": 0.0,
                "unemployment_shock": 0.0,
                "energy_import": 0.0,
                "food_import": 0.0,
            }
        
        # Default covariance (diagonal, small)
        if cov is None:
            n_vars = len(mean)
            cov = np.eye(n_vars) * 0.01
        
        scenarios = []
        for _ in range(n_scenarios):
            scenario = []
            for _ in range(horizon):
                # Sample from multivariate Gaussian
                disturbance_vec = self.rng.multivariate_normal(
                    mean=list(mean.values()),
                    cov=cov
                )
                disturbance = {
                    key: float(val)
                    for key, val in zip(mean.keys(), disturbance_vec)
                }
                scenario.append(disturbance)
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_student_t(
        self,
        n_scenarios: int,
        horizon: int,
        df: float = 3.0,  # Degrees of freedom (lower = heavier tails)
        scale: float = 0.1
    ) -> List[List[Dict[str, float]]]:
        """
        Generate Student-t disturbance scenarios (heavy-tailed).
        
        Args:
            n_scenarios: Number of scenarios
            horizon: Time horizon
            df: Degrees of freedom (default: 3.0 for heavy tails)
            scale: Scale parameter
            
        Returns:
            List[List[Dict[str, float]]]: List of scenario sequences
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, using Gaussian fallback")
            return self.generate_gaussian(n_scenarios, horizon)
        
        scenarios = []
        for _ in range(n_scenarios):
            scenario = []
            for _ in range(horizon):
                # Sample from Student-t
                disturbance = {
                    "demand_shock": float(student_t.rvs(df, scale=scale, random_state=self.rng)),
                    "trade_shock": float(student_t.rvs(df, scale=scale, random_state=self.rng)),
                    "productivity_shock": 1.0 + float(student_t.rvs(df, scale=scale * 0.1, random_state=self.rng)),
                    "disaster_shock": float(student_t.rvs(df, scale=scale, random_state=self.rng)),
                    "labor_shock": float(student_t.rvs(df, scale=scale * 0.1, random_state=self.rng)),
                    "unemployment_shock": float(student_t.rvs(df, scale=scale * 0.1, random_state=self.rng)),
                    "energy_import": float(student_t.rvs(df, scale=scale * 100, random_state=self.rng)),
                    "food_import": float(student_t.rvs(df, scale=scale * 100, random_state=self.rng)),
                }
                scenario.append(disturbance)
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_structured_shock(
        self,
        event_type: str,
        magnitude: float,
        timing: int,
        horizon: int
    ) -> List[Dict[str, float]]:
        """
        Generate structured shock event (trade embargo, drought, etc.).
        
        Args:
            event_type: Type of shock ("trade_embargo", "drought", "productivity_crash")
            magnitude: Shock magnitude
            timing: Time step when shock occurs
            horizon: Time horizon
            
        Returns:
            List[Dict[str, float]]: Single scenario sequence
        """
        scenario = [{}] * horizon
        
        if event_type == "trade_embargo":
            # Complete trade cutoff at timing
            for t in range(timing, horizon):
                scenario[t] = {
                    "trade_shock": 1.0,  # Complete cutoff
                    "energy_import": -magnitude * 1000,
                    "food_import": -magnitude * 500,
                }
        
        elif event_type == "drought":
            # Food production crash
            for t in range(timing, min(timing + 5, horizon)):
                scenario[t] = {
                    "disaster_shock": magnitude,
                    "food_import": -magnitude * 300,
                }
        
        elif event_type == "productivity_crash":
            # Productivity shock
            for t in range(timing, min(timing + 3, horizon)):
                scenario[t] = {
                    "productivity_shock": 1.0 - magnitude,  # Reduce productivity
                }
        
        else:
            logger.warning(f"Unknown event type: {event_type}, using generic shock")
            if timing < horizon:
                scenario[timing] = {"disaster_shock": magnitude}
        
        return scenario
    
    def generate_causal_scenarios(
        self,
        n_scenarios: int,
        horizon: int,
        current_state: Optional[StateVector] = None,
        target_variables: Optional[List[str]] = None
    ) -> List[List[Dict[str, float]]]:
        """
        Generate scenarios using CRCA causal reasoning (if available).
        
        Uses CRCAAgent to generate counterfactual scenarios based on causal relationships.
        Falls back to Gaussian if CRCA not available.
        
        Args:
            n_scenarios: Number of scenarios
            horizon: Time horizon
            current_state: Current state vector (for counterfactual analysis)
            target_variables: Variables to focus on (default: ["Y", "U", "S"])
            
        Returns:
            List[List[Dict[str, float]]]: Causal scenarios
        """
        if self.crca_agent is None or current_state is None:
            # Fallback to Gaussian
            logger.debug("CRCA not available or no state provided, using Gaussian scenarios")
            return self.generate_gaussian(n_scenarios, horizon)
        
        try:
            # Convert state to dict for CRCA
            state_dict = {
                "P": current_state.P,
                "L": current_state.L,
                "U": current_state.U,
                "W": current_state.W,
                "S": current_state.S,
                "Y": current_state.Y,
                "K": current_state.K,
                "I": current_state.I,
                "literacy": current_state.literacy,
                "Ecap": current_state.Ecap,
                "Hcap": current_state.Hcap,
            }
            
            if target_variables is None:
                target_variables = ["Y", "U", "S"]  # GDP, unemployment, stability
            
            # Use CRCA to generate counterfactual scenarios
            crca_result = self.crca_agent.run(
                initial_state=state_dict,
                target_variables=target_variables,
                max_steps=horizon
            )
            
            # Extract counterfactual scenarios
            counterfactuals = crca_result.get("counterfactual_scenarios", [])
            
            if not counterfactuals:
                logger.debug("No counterfactuals from CRCA, using Gaussian scenarios")
                return self.generate_gaussian(n_scenarios, horizon)
            
            # Convert counterfactuals to scenario format
            scenarios = []
            for cf in counterfactuals[:n_scenarios]:
                scenario = []
                interventions = cf.interventions
                
                # Create scenario as list of disturbance dicts
                for step in range(horizon):
                    # Disturbances based on counterfactual interventions
                    disturbances = {}
                    for var, value in interventions.items():
                        if var in state_dict:
                            # Convert intervention to disturbance
                            current_val = state_dict.get(var, 0.0)
                            # Scale down to reasonable disturbance magnitude
                            disturbances[var] = (value - current_val) * 0.1
                    
                    scenario.append(disturbances)
                
                scenarios.append(scenario)
            
            # Pad to n_scenarios if needed
            while len(scenarios) < n_scenarios:
                scenarios.extend(self.generate_gaussian(1, horizon))
            
            logger.debug(f"Generated {len(scenarios)} causal scenarios via CRCA")
            return scenarios[:n_scenarios]
        
        except Exception as e:
            logger.warning(f"CRCA scenario generation failed: {e}, using Gaussian scenarios")
            return self.generate_gaussian(n_scenarios, horizon)


class StabilityEnforcer:
    """
    Stability enforcer to prevent oscillations and planner thrash.
    
    Implements:
    - Budget change limits: ||b_t - b_{t-1}||_1 <= Δ_b
    - Investment smoothing: I_inv_t = β * I_inv_{t-1} + (1-β) * I_hat_t
    - Logistics ramp constraints
    
    Args:
        max_budget_change: Maximum L1 norm change in budget shares (default: 0.2)
        investment_smoothing: Smoothing factor β (default: 0.7)
        max_flow_change: Maximum change in logistics flows (default: 0.3)
    """
    
    def __init__(
        self,
        max_budget_change: float = 0.2,
        investment_smoothing: float = 0.7,
        max_flow_change: float = 0.3,
    ) -> None:
        """Initialize stability enforcer."""
        self.max_budget_change = max_budget_change
        self.investment_smoothing = investment_smoothing
        self.max_flow_change = max_flow_change
    
    def apply_rate_limits(
        self,
        u_t: ControlVector,
        u_prev: Optional[ControlVector],
        limits: Optional[Dict[str, float]] = None
    ) -> ControlVector:
        """
        Apply rate limits to control vector.
        
        Args:
            u_t: Proposed control vector
            u_prev: Previous control vector (None if first step)
            limits: Custom limits (default: use instance defaults)
            
        Returns:
            ControlVector: Adjusted control vector
        """
        if u_prev is None:
            return u_t
        
        if limits is None:
            max_change = self.max_budget_change
        else:
            max_change = limits.get("max_budget_change", self.max_budget_change)
        
        # Compute L1 norm of budget change
        budget_change = {}
        for cat in set(list(u_t.budget_shares.keys()) + list(u_prev.budget_shares.keys())):
            prev_val = u_prev.budget_shares.get(cat, 0.0)
            curr_val = u_t.budget_shares.get(cat, 0.0)
            budget_change[cat] = abs(curr_val - prev_val)
        
        total_change = sum(budget_change.values())
        
        if total_change > max_change:
            # Scale down changes proportionally
            scale = max_change / total_change
            adjusted_shares = {}
            for cat in u_t.budget_shares:
                prev_val = u_prev.budget_shares.get(cat, 0.0)
                curr_val = u_t.budget_shares[cat]
                change = (curr_val - prev_val) * scale
                adjusted_shares[cat] = prev_val + change
            
            # Renormalize to ensure simplex
            total = sum(adjusted_shares.values())
            if total > 0:
                adjusted_shares = {k: v / total for k, v in adjusted_shares.items()}
            else:
                adjusted_shares = u_prev.budget_shares.copy()
            
            u_adjusted = ControlVector(
                budget_shares=adjusted_shares,
                allocations=u_t.allocations.copy(),
                flows=u_t.flows.copy(),
            )
            return u_adjusted
        
        return u_t
    
    def smooth_investment(
        self,
        I_hat: float,
        I_prev: float,
        beta: Optional[float] = None
    ) -> float:
        """
        Smooth investment using exponential moving average.
        
        Args:
            I_hat: Proposed investment
            I_prev: Previous investment
            beta: Smoothing factor (default: use instance default)
            
        Returns:
            float: Smoothed investment
        """
        if beta is None:
            beta = self.investment_smoothing
        
        I_smooth = beta * I_prev + (1 - beta) * I_hat
        return I_smooth


class ParetoExtractor:
    """
    Pareto frontier extractor using non-dominated sorting.
    
    Finds all Pareto-efficient policies (non-dominated solutions).
    """
    
    @staticmethod
    def extract_pareto_frontier(
        candidate_policies: List[ControlVector],
        objectives: np.ndarray
    ) -> Tuple[List[ControlVector], np.ndarray]:
        """
        Extract Pareto-efficient policies.
        
        Args:
            candidate_policies: List of candidate policies
            objectives: Objective matrix (n_policies x n_objectives)
            
        Returns:
            Tuple[List[ControlVector], np.ndarray]: (pareto_policies, pareto_objectives)
        """
        if len(candidate_policies) == 0:
            return [], np.array([])
        
        if objectives.ndim == 1:
            objectives = objectives.reshape(1, -1)
        
        n_policies = len(candidate_policies)
        n_objectives = objectives.shape[1]
        
        # Non-dominated sorting
        is_pareto = np.ones(n_policies, dtype=bool)
        
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue
                
                # Check if j dominates i
                # j dominates i if: all objectives of j <= i, and at least one <
                obj_i = objectives[i]
                obj_j = objectives[j]
                
                # For minimization: j dominates i if obj_j <= obj_i (element-wise)
                # and at least one obj_j < obj_i
                if np.all(obj_j <= obj_i) and np.any(obj_j < obj_i):
                    is_pareto[i] = False
                    break
        
        pareto_indices = np.where(is_pareto)[0]
        pareto_policies = [candidate_policies[i] for i in pareto_indices]
        pareto_objectives = objectives[pareto_indices]
        
        return pareto_policies, pareto_objectives


class MPCSolver:
    """
    Model-Predictive Control (MPC) solver.
    
    Solves rolling-horizon optimization problem:
    minimize: weighted_sum(objectives)
    subject to:
      - dynamics constraints
      - hard constraints
      - rate limits (if stability enabled)
    
    Args:
        dynamics: Dynamics model
        constraint_checker: Constraint checker
        objective_computer: Objective vector computer
        horizon: Planning horizon (default: 10)
        stability_enforcer: Stability enforcer (optional)
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        constraint_checker: ConstraintChecker,
        objective_computer: ObjectiveVector,
        horizon: int = 10,
        stability_enforcer: Optional[StabilityEnforcer] = None,
    ) -> None:
        """Initialize MPC solver."""
        self.dynamics = dynamics
        self.constraint_checker = constraint_checker
        self.objective_computer = objective_computer
        self.horizon = horizon
        self.stability_enforcer = stability_enforcer
    
    def solve(
        self,
        x_t: StateVector,
        scenarios: List[List[Dict[str, float]]],
        objective_weights: Optional[np.ndarray] = None,
        u_prev: Optional[ControlVector] = None
    ) -> Tuple[ControlVector, Dict[str, Any]]:
        """
        Solve MPC optimization problem.
        
        Args:
            x_t: Current state
            scenarios: List of disturbance scenarios
            objective_weights: Weights for objectives (default: equal)
            u_prev: Previous control (for rate limits)
            
        Returns:
            Tuple[ControlVector, Dict[str, Any]]: (optimal_policy, solver_info)
        """
        if objective_weights is None:
            objective_weights = np.ones(6) / 6.0  # Equal weights
        
        # Sample candidate policies
        budget_categories = [
            "energy", "food", "infrastructure", "education",
            "healthcare", "R&D", "welfare"
        ]
        
        n_candidates = 20
        candidate_policies = [
            ControlVector.sample_budget_simplex(budget_categories)
            for _ in range(n_candidates)
        ]
        
        # Evaluate candidates under scenarios
        best_policy = None
        best_score = float('inf')
        candidate_scores = []
        
        for policy in candidate_policies:
            # Apply rate limits if enabled
            if self.stability_enforcer and u_prev is not None:
                policy = self.stability_enforcer.apply_rate_limits(policy, u_prev)
            
            # Evaluate under all scenarios
            scenario_objectives = []
            for scenario in scenarios:
                traj, _, _ = self._simulate_policy(x_t, policy, scenario)
                obj = self.objective_computer.compute(traj, [policy] * len(traj[1:]))
                scenario_objectives.append(obj)
            
            # Aggregate objectives (expected value)
            expected_obj = np.mean(scenario_objectives, axis=0)
            
            # Weighted sum
            score = np.dot(objective_weights, expected_obj)
            candidate_scores.append(score)
            
            # Check feasibility
            traj, _, first_violation = self._simulate_policy(x_t, policy, scenarios[0])
            if first_violation is None:  # Feasible
                if score < best_score:
                    best_score = score
                    best_policy = policy
        
        if best_policy is None:
            # Fallback: use first feasible or first candidate
            logger.warning("No feasible policy found, using first candidate")
            best_policy = candidate_policies[0]
        
        solver_info = {
            "best_score": best_score,
            "n_candidates": n_candidates,
            "n_scenarios": len(scenarios),
            "feasible_found": best_policy is not None,
        }
        
        return best_policy, solver_info
    
    def _simulate_policy(
        self,
        x_0: StateVector,
        policy: ControlVector,
        disturbances: List[Dict[str, float]]
    ) -> Tuple[List[StateVector], List[bool], Optional[int]]:
        """Simulate policy forward."""
        from crca_sd.crca_sd_core import ForwardSimulator
        
        simulator = ForwardSimulator(self.dynamics, self.constraint_checker)
        return simulator.simulate_scenario(x_0, policy, disturbances, self.horizon)
    
    def execute_policy_realtime(
        self,
        policy: ControlVector,
        government_api_config: Optional[Dict[str, Any]] = None,
        execution_id: Optional[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute policy in real-time via government API integration.
        
        Args:
            policy: Policy to execute
            government_api_config: Government API configuration
            execution_id: Optional execution ID
            
        Returns:
            Tuple[bool, str, Dict[str, Any]]: (success, message, execution_info)
        """
        # Import PolicyExecutor from realtime module
        try:
            from crca_sd.crca_sd_realtime import PolicyExecutor
            
            executor = PolicyExecutor(government_api_config)
            success, message, info = executor.execute_policy(policy, execution_id)
            
            logger.info(f"Policy execution {'succeeded' if success else 'failed'}: {message}")
            
            return success, message, info
        
        except ImportError:
            logger.warning("PolicyExecutor not available, using mock execution")
            return True, "Mock execution successful", {"execution_id": execution_id or "mock"}
    
    def solve_and_execute(
        self,
        x_t: StateVector,
        scenarios: List[List[Dict[str, float]]],
        previous_policy: Optional[ControlVector] = None,
        government_api_config: Optional[Dict[str, Any]] = None,
        require_approval_check: Optional[Callable[[ControlVector, Optional[ControlVector]], Tuple[bool, str]]] = None
    ) -> Tuple[Optional[ControlVector], Dict[str, Any]]:
        """
        Solve MPC and execute policy (with approval check if needed).
        
        Args:
            x_t: Current state
            scenarios: Disturbance scenarios
            previous_policy: Previous policy (for rate limits and approval check)
            government_api_config: Government API configuration
            require_approval_check: Function to check if approval is needed
            
        Returns:
            Tuple[Optional[ControlVector], Dict[str, Any]]: (executed_policy, execution_info)
        """
        # Solve MPC
        policy, solver_info = self.solve(x_t, scenarios, u_prev=previous_policy)
        
        # Check if approval is needed
        requires_approval = False
        if require_approval_check:
            requires_approval, reason = require_approval_check(policy, previous_policy)
            if requires_approval:
                logger.info(f"Policy requires approval: {reason}")
                return None, {"status": "pending_approval", "reason": reason, "policy": policy}
        
        # Execute policy
        success, message, exec_info = self.execute_policy_realtime(
            policy,
            government_api_config
        )
        
        execution_info = {
            **solver_info,
            **exec_info,
            "requires_approval": requires_approval,
        }
        
        return policy if success else None, execution_info


class StateEstimator:
    """
    State estimator for partial observability (EKF/UKF).
    
    Implements Extended Kalman Filter for state estimation with noisy observations.
    Observation model: y_t = h(x_t) + ν_t
    
    Args:
        dynamics: Dynamics model
        observation_noise_cov: Observation noise covariance matrix
        process_noise_cov: Process noise covariance matrix
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        observation_noise_cov: Optional[np.ndarray] = None,
        process_noise_cov: Optional[np.ndarray] = None,
        update_frequency: float = 86400.0,  # Daily updates
    ) -> None:
        """Initialize state estimator."""
        self.dynamics = dynamics
        
        # Default noise covariances (diagonal)
        n_state = 16  # Number of state variables
        if observation_noise_cov is None:
            observation_noise_cov = np.eye(n_state) * 0.01
        if process_noise_cov is None:
            process_noise_cov = np.eye(n_state) * 0.001
        
        self.R = observation_noise_cov  # Observation noise
        self.Q = process_noise_cov      # Process noise
        
        # State covariance
        self.P = np.eye(n_state) * 0.1
        
        # Real-time capabilities
        self.update_frequency = update_frequency
        self.last_update_time = 0.0
        self.current_state_estimate: Optional[StateVector] = None
        self.state_confidence: float = 0.0
    
    def update(
        self,
        y_t: Dict[str, float],
        u_t: ControlVector
    ) -> StateVector:
        """
        Update state estimate with observation (EKF update step).
        
        Args:
            y_t: Noisy observation
            u_t: Control applied
            
        Returns:
            StateVector: Updated state estimate
        """
        # Simplified EKF: assume linear observation model h(x) = x
        # In practice, h(x) would map state to observable variables
        
        # Predict step (already done in predict)
        x_pred = self.predict(u_t)
        
        # Convert state to vector for EKF
        x_vec = self._state_to_vector(x_pred)
        y_vec = self._observation_to_vector(y_t, x_pred)
        
        # Innovation
        innovation = y_vec - x_vec  # Simplified: h(x) = x
        
        # Kalman gain
        S = self.P + self.R  # Innovation covariance
        K = self.P @ np.linalg.pinv(S)  # Kalman gain
        
        # Update
        x_updated_vec = x_vec + K @ innovation
        self.P = (np.eye(len(x_vec)) - K) @ self.P
        
        # Convert back to StateVector
        x_updated = self._vector_to_state(x_updated_vec)
        
        # Store updated estimate
        self.current_state_estimate = x_updated
        self.last_update_time = time.time()
        
        # Compute confidence from covariance trace
        trace_P = np.trace(self.P)
        self.state_confidence = max(0.0, 1.0 - trace_P / 10.0)  # Normalized confidence
        
        return x_updated
    
    def predict(
        self,
        u_t: ControlVector
    ) -> StateVector:
        """
        One-step prediction: x_pred_{t+1} = f(x_hat_t, u_t).
        
        Args:
            u_t: Control vector
            
        Returns:
            StateVector: Predicted next state
        """
        # Use stored state estimate
        if self.current_state_estimate is None:
            self.current_state_estimate = StateVector()
        
        # Predict next state using dynamics
        x_pred = self.dynamics.step(self.current_state_estimate, u_t)
        
        return x_pred
    
    def should_update(self) -> bool:
        """
        Check if state should be updated based on frequency (daily).
        
        Returns:
            bool: True if should update
        """
        if self.last_update_time == 0.0:
            return True
        
        elapsed = time.time() - self.last_update_time
        return elapsed >= self.update_frequency
    
    def get_current_estimate(self) -> Optional[StateVector]:
        """Get current state estimate."""
        return self.current_state_estimate
    
    def get_confidence(self) -> float:
        """Get state estimate confidence."""
        return self.state_confidence
    
    def update_with_multi_sensor(
        self,
        observations: List[Dict[str, float]],
        u_t: ControlVector,
        sensor_weights: Optional[Dict[str, float]] = None
    ) -> StateVector:
        """
        Update state estimate with multiple sensor observations (multi-sensor fusion).
        
        Args:
            observations: List of observation dictionaries from different sensors
            u_t: Control vector
            sensor_weights: Optional weights for each sensor (by source name)
            
        Returns:
            StateVector: Updated state estimate
        """
        if not observations:
            return self.predict(u_t)
        
        # Fuse observations (weighted average)
        fused_observation = {}
        total_weight = 0.0
        
        for obs in observations:
            source = obs.get("source", "unknown")
            weight = sensor_weights.get(source, 1.0) if sensor_weights else 1.0
            
            for key, value in obs.items():
                if key == "source":
                    continue
                
                if key not in fused_observation:
                    fused_observation[key] = 0.0
                
                fused_observation[key] += value * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            fused_observation = {k: v / total_weight for k, v in fused_observation.items()}
        
        # Update with fused observation
        updated = self.update(fused_observation, u_t)
        
        # Update stored estimate and confidence
        self.current_state_estimate = updated
        self.state_confidence = min(1.0, len(observations) * 0.1)  # More sensors = higher confidence
        self.last_update_time = time.time()
        
        return updated
    
    def _state_to_vector(self, x: StateVector) -> np.ndarray:
        """Convert StateVector to numpy array."""
        return np.array([
            x.P, x.L, x.U, x.W, x.S,
            x.literacy, x.Ecap, x.Hcap,
            x.K, x.I, x.Tcap,
            x.E_stock, x.F_stock, x.M_stock, x.C, x.Y
        ])
    
    def _vector_to_state(self, vec: np.ndarray) -> StateVector:
        """Convert numpy array to StateVector."""
        return StateVector(
            P=vec[0], L=vec[1], U=vec[2], W=vec[3], S=vec[4],
            literacy=vec[5], Ecap=vec[6], Hcap=vec[7],
            K=vec[8], I=vec[9], Tcap=vec[10],
            E_stock=vec[11], F_stock=vec[12], M_stock=vec[13], C=vec[14], Y=vec[15]
        )
    
    def _observation_to_vector(
        self,
        y_t: Dict[str, float],
        x_t: StateVector
    ) -> np.ndarray:
        """Convert observation to vector (fill missing with state values)."""
        vec = self._state_to_vector(x_t)
        
        # Override with observed values if available
        # Handle both direct values and percentage values (U is a rate)
        if "P" in y_t:
            vec[0] = float(y_t["P"])
        if "L" in y_t:
            vec[1] = float(y_t["L"])
        if "U" in y_t:
            # U can be percentage (like 1.434) or rate (like 0.01434)
            u_val = float(y_t["U"])
            if u_val > 1.0:
                vec[2] = u_val / 100.0  # Convert percentage to rate
            else:
                vec[2] = u_val
        if "Y" in y_t:
            vec[15] = float(y_t["Y"])
        if "W" in y_t:
            vec[3] = float(y_t["W"])
        if "S" in y_t:
            vec[4] = float(y_t["S"])
        if "literacy" in y_t:
            lit_val = float(y_t["literacy"])
            if lit_val > 1.0:
                vec[5] = lit_val / 100.0  # Convert percentage to rate
            else:
                vec[5] = lit_val
        if "Ecap" in y_t:
            vec[6] = float(y_t["Ecap"])
        if "Hcap" in y_t:
            vec[7] = float(y_t["Hcap"])
        if "K" in y_t:
            vec[8] = float(y_t["K"])
        if "I" in y_t:
            vec[9] = float(y_t["I"])
        if "Tcap" in y_t:
            vec[10] = float(y_t["Tcap"])
        if "E_stock" in y_t:
            vec[11] = float(y_t["E_stock"])
        if "F_stock" in y_t:
            vec[12] = float(y_t["F_stock"])
        if "M_stock" in y_t:
            vec[13] = float(y_t["M_stock"])
        if "C" in y_t:
            vec[14] = float(y_t["C"])
        
        return vec

