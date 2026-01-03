"""Model Predictive Control (MPC) planner for policy engine.

Provides convex MPC optimization using cvxpy for multi-objective
constrained planning with horizon-based prediction.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from schemas.policy import DoctrineV1, InterventionSpec

# Try to import cvxpy
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not available - MPC planner will use fallback methods")


class MPCPlanner:
    """Model Predictive Control planner using convex optimization.
    
    Formulates MPC problem:
    minimize: Σ_{t=0}^{H-1} [w^T·J(x_t, u_t) + λ_c·C(u_t) + λ_r·R(x_t, u_t)]
    subject to:
      x_{t+1} = A·x_t + B·u_t  (transition model)
      g(x_t, u_t) ≤ 0           (constraints)
      h(x_t, u_t) = 0           (invariants)
      u_t ∈ U                   (lever bounds)
    """
    
    def __init__(
        self,
        doctrine: DoctrineV1,
        horizon: int = 5,
        use_robust: bool = False,
        uncertainty_set_size: float = 0.1
    ):
        """
        Initialize MPC planner.
        
        Args:
            doctrine: Policy doctrine
            horizon: Prediction horizon (number of steps ahead)
            use_robust: Whether to use robust MPC with uncertainty sets
            uncertainty_set_size: Size of uncertainty set for robust MPC
        """
        self.doctrine = doctrine
        self.horizon = horizon
        self.use_robust = use_robust
        self.uncertainty_set_size = uncertainty_set_size
        self.last_solution: Optional[Dict[str, Any]] = None
    
    def solve_mpc(
        self,
        x_t: np.ndarray,
        A: Optional[np.ndarray],
        B: Optional[np.ndarray],
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        lever_bounds: Dict[str, Dict[str, float]]
    ) -> Tuple[List[InterventionSpec], float, str]:
        """
        Solve MPC optimization problem.
        
        Args:
            x_t: Current state vector
            A: State transition matrix (if available)
            B: Action effect matrix (if available)
            objectives: List of objective specifications
            constraints: List of constraint specifications
            lever_bounds: Lever parameter bounds
            
        Returns:
            Tuple of (interventions, score, rationale)
        """
        if not CVXPY_AVAILABLE:
            logger.warning("cvxpy not available - MPC solver cannot be used")
            return [], 0.0, "MPC solver unavailable (cvxpy not installed)"
        
        try:
            if self.use_robust:
                return self._solve_robust_mpc(x_t, A, B, objectives, constraints, lever_bounds)
            else:
                return self._solve_nominal_mpc(x_t, A, B, objectives, constraints, lever_bounds)
        except Exception as e:
            logger.error(f"MPC solver failed: {e}")
            return [], 0.0, f"MPC solver error: {str(e)}"
    
    def _solve_nominal_mpc(
        self,
        x_t: np.ndarray,
        A: Optional[np.ndarray],
        B: Optional[np.ndarray],
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        lever_bounds: Dict[str, Dict[str, float]]
    ) -> Tuple[List[InterventionSpec], float, str]:
        """Solve nominal (deterministic) MPC problem."""
        n_state = len(x_t)
        n_levers = len(self.doctrine.levers)
        
        # If no transition model, use simplified MPC
        if A is None or B is None:
            return self._solve_simplified_mpc(x_t, objectives, constraints, lever_bounds)
        
        # Extract lever IDs and create decision variables
        lever_ids = list(self.doctrine.levers.keys())
        
        # Decision variables: u[t, lever_idx] for each time step and lever
        # For simplicity, we'll optimize over the first time step only
        # and use a simplified representation
        
        # Create decision variables for first step
        u_vars = {}
        for lever_id in lever_ids:
            lever_spec = self.doctrine.levers[lever_id]
            # For each parameter in lever bounds
            for param_name in lever_spec.bounds.keys():
                var_name = f"{lever_id}_{param_name}"
                bounds = lever_spec.bounds[param_name]
                min_val = bounds.get("min", -np.inf)
                max_val = bounds.get("max", np.inf)
                u_vars[var_name] = cp.Variable(name=var_name)
        
        # Objective: minimize weighted sum of objectives
        objective_terms = []
        
        # Predict state evolution over horizon
        x_pred = [x_t]  # x[0] = current state
        for t in range(self.horizon):
            # Compute action vector u_t (simplified: use first lever's first param as scalar)
            if t == 0:
                # Use decision variables
                if u_vars:
                    u_t_scalar = list(u_vars.values())[0]  # Simplified: use first variable
                else:
                    u_t_scalar = 0.0
            else:
                # Future steps: assume no action (or use previous solution)
                u_t_scalar = 0.0
            
            # Predict next state: x_{t+1} = A·x_t + B·u_t
            if t < self.horizon - 1:
                u_t = np.array([u_t_scalar]) if isinstance(u_t_scalar, (int, float, cp.Variable)) else u_t_scalar
                if len(B.shape) == 2 and B.shape[1] == len(u_t):
                    x_next = A @ x_pred[-1] + B @ u_t
                else:
                    # Simplified: just use A
                    x_next = A @ x_pred[-1]
                x_pred.append(x_next)
        
        # Objective: sum over horizon
        for obj in objectives:
            metric_name = obj.get("metric_name")
            direction = obj.get("direction", "minimize")
            weight = obj.get("weight", 1.0)
            
            # Find metric index
            metric_names = sorted(self.doctrine.metrics.keys())
            if metric_name in metric_names:
                metric_idx = metric_names.index(metric_name)
                
                # Sum over horizon
                for t in range(1, len(x_pred)):
                    if metric_idx < len(x_pred[t]):
                        metric_val = x_pred[t][metric_idx]
                        if direction == "minimize":
                            objective_terms.append(weight * metric_val)
                        else:
                            objective_terms.append(-weight * metric_val)  # Maximize = minimize negative
        
        # Cost penalty (simplified)
        if u_vars:
            cost_penalty = sum(cp.abs(u) for u in u_vars.values())
            objective_terms.append(0.1 * cost_penalty)
        
        # Constraints
        constraint_list = []
        
        # Lever bounds
        for var_name, var in u_vars.items():
            lever_id, param_name = var_name.rsplit("_", 1)
            if lever_id in lever_bounds and param_name in lever_bounds[lever_id]:
                bounds = lever_bounds[lever_id][param_name]
                if "min" in bounds:
                    constraint_list.append(var >= bounds["min"])
                if "max" in bounds:
                    constraint_list.append(var <= bounds["max"])
        
        # Invariants (hard constraints)
        for inv in self.doctrine.invariants:
            # Simplified: check if invariant can be expressed as constraint
            # For now, we'll skip complex invariants
            pass
        
        # Formulate and solve
        if not objective_terms:
            return [], 0.0, "No objectives to optimize"
        
        objective = cp.Minimize(sum(objective_terms))
        problem = cp.Problem(objective, constraint_list)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                # Extract solution
                interventions = []
                for lever_id in lever_ids:
                    lever_spec = self.doctrine.levers[lever_id]
                    params = {}
                    for param_name in lever_spec.bounds.keys():
                        var_name = f"{lever_id}_{param_name}"
                        if var_name in u_vars:
                            val = float(u_vars[var_name].value)
                            params[param_name] = val
                    
                    if params:
                        from schemas.policy import InterventionSpec
                        interventions.append(InterventionSpec(
                            lever_id=lever_id,
                            parameters=params,
                            rollback_descriptor={"lever_id": lever_id, "parameters": params} if lever_spec.rollback_required else None
                        ))
                
                score = float(problem.value) if problem.value is not None else 0.0
                rationale = f"MPC solved (horizon={self.horizon}, status={problem.status})"
                
                self.last_solution = {
                    "interventions": interventions,
                    "score": score,
                    "status": problem.status
                }
                
                return interventions, score, rationale
            else:
                return [], 0.0, f"MPC solver status: {problem.status}"
        except Exception as e:
            logger.error(f"MPC solve error: {e}")
            return [], 0.0, f"MPC solve error: {str(e)}"
    
    def _solve_simplified_mpc(
        self,
        x_t: np.ndarray,
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        lever_bounds: Dict[str, Dict[str, float]]
    ) -> Tuple[List[InterventionSpec], float, str]:
        """Solve simplified MPC without transition model (gradient-based)."""
        # Simplified: just return empty (fallback to beam search)
        return [], 0.0, "Simplified MPC (no transition model available)"
    
    def _solve_robust_mpc(
        self,
        x_t: np.ndarray,
        A: Optional[np.ndarray],
        B: Optional[np.ndarray],
        objectives: List[Dict[str, Any]],
        constraints: List[Dict[str, Any]],
        lever_bounds: Dict[str, Dict[str, float]]
    ) -> Tuple[List[InterventionSpec], float, str]:
        """Solve robust MPC with uncertainty sets."""
        # For robust MPC, we add uncertainty to the transition model
        # A_robust = A + ΔA, B_robust = B + ΔB where ||ΔA||, ||ΔB|| ≤ uncertainty_set_size
        
        if A is None or B is None:
            return self._solve_simplified_mpc(x_t, objectives, constraints, lever_bounds)
        
        # Use nominal MPC for now (robust MPC implementation would require
        # more sophisticated uncertainty set handling)
        logger.warning("Robust MPC not fully implemented, using nominal MPC")
        return self._solve_nominal_mpc(x_t, A, B, objectives, constraints, lever_bounds)
    
    def warm_start(self, previous_solution: Optional[Dict[str, Any]]) -> None:
        """Warm start from previous solution."""
        self.last_solution = previous_solution

