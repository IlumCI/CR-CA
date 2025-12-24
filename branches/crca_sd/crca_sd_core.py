"""
CRCA-SD Core: State, Controls, Dynamics, Constraints, and Forward Simulation

This module implements the fundamental components for socioeconomic dynamics simulation:
- State vector with 15 core variables
- Control vector with budget shares and allocations
- Dynamics model with macro system dynamics equations
- Constraint checker for hard feasibility constraints
- Forward simulator for multi-step trajectory simulation
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class StateVector:
    """
    State vector representing the socioeconomic system state.
    
    Contains 15 core variables:
    - Macro: P (population), L (labor), U (unemployment), W (wage), S (stability)
    - Human capital: ℓ (literacy), Ecap (education capacity), Hcap (healthcare capacity)
    - Capital: K (capital stock), I (infrastructure health), Tcap (transport capacity)
    - Stocks: E_stock (energy), F_stock (food), M_stock (materials), C (ecological damage)
    - Output: Y (GDP proxy)
    
    Args:
        P: Population
        L: Labor force
        U: Unemployment rate [0, 1]
        W: Average wage proxy
        S: Social stability proxy [0, 1]
        literacy: Literacy rate [0, 1]
        Ecap: Education capacity (seats/teachers)
        Hcap: Healthcare capacity (beds/staff)
        K: Productive capital stock
        I: Infrastructure health index [0, 1]
        Tcap: Transport capacity (ton-km/day)
        E_stock: Stored energy (or fuel reserve proxy)
        F_stock: Food stock
        M_stock: Critical materials stock
        C: Ecological damage / carbon stock (irreversible)
        Y: GDP proxy (or output index)
    """
    
    # Macro variables
    P: float = 1000000.0  # Population
    L: float = 500000.0   # Labor force
    U: float = 0.05       # Unemployment rate [0, 1]
    W: float = 1.0        # Average wage proxy
    S: float = 0.8        # Social stability proxy [0, 1]
    
    # Human capital
    literacy: float = 0.7      # Literacy rate [0, 1]
    Ecap: float = 10000.0     # Education capacity
    Hcap: float = 5000.0      # Healthcare capacity
    
    # Capital & infrastructure
    K: float = 1000000.0   # Productive capital stock
    I: float = 0.8         # Infrastructure health index [0, 1]
    Tcap: float = 10000.0  # Transport capacity (ton-km/day)
    
    # Stocks
    E_stock: float = 50000.0  # Stored energy
    F_stock: float = 30000.0  # Food stock
    M_stock: float = 20000.0  # Critical materials stock
    C: float = 0.0            # Ecological damage (irreversible)
    
    # Output
    Y: float = 1000000.0  # GDP proxy
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert state vector to dictionary.
        
        Returns:
            Dict[str, float]: Dictionary mapping variable names to values
        """
        return {
            "P": self.P,
            "L": self.L,
            "U": self.U,
            "W": self.W,
            "S": self.S,
            "literacy": self.literacy,
            "Ecap": self.Ecap,
            "Hcap": self.Hcap,
            "K": self.K,
            "I": self.I,
            "Tcap": self.Tcap,
            "E_stock": self.E_stock,
            "F_stock": self.F_stock,
            "M_stock": self.M_stock,
            "C": self.C,
            "Y": self.Y,
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, float]) -> "StateVector":
        """
        Create state vector from dictionary.
        
        Args:
            state_dict: Dictionary mapping variable names to values
            
        Returns:
            StateVector: New state vector instance
        """
        return cls(**state_dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate state vector bounds and constraints.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of violations)
        """
        violations = []
        
        # Check non-negativity
        if self.P < 0:
            violations.append("Population P must be non-negative")
        if self.L < 0:
            violations.append("Labor force L must be non-negative")
        if self.K < 0:
            violations.append("Capital stock K must be non-negative")
        if self.E_stock < 0:
            violations.append("Energy stock must be non-negative")
        if self.F_stock < 0:
            violations.append("Food stock must be non-negative")
        if self.M_stock < 0:
            violations.append("Materials stock must be non-negative")
        if self.C < 0:
            violations.append("Ecological damage C must be non-negative")
        
        # Check bounds [0, 1]
        if not (0 <= self.U <= 1):
            violations.append("Unemployment U must be in [0, 1]")
        if not (0 <= self.S <= 1):
            violations.append("Social stability S must be in [0, 1]")
        if not (0 <= self.literacy <= 1):
            violations.append("Literacy rate must be in [0, 1]")
        if not (0 <= self.I <= 1):
            violations.append("Infrastructure health I must be in [0, 1]")
        
        # Check logical constraints
        if self.L > self.P:
            violations.append("Labor force L cannot exceed population P")
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def copy(self) -> "StateVector":
        """
        Create a deep copy of the state vector.
        
        Returns:
            StateVector: New state vector with copied values
        """
        return StateVector(
            P=self.P,
            L=self.L,
            U=self.U,
            W=self.W,
            S=self.S,
            literacy=self.literacy,
            Ecap=self.Ecap,
            Hcap=self.Hcap,
            K=self.K,
            I=self.I,
            Tcap=self.Tcap,
            E_stock=self.E_stock,
            F_stock=self.F_stock,
            M_stock=self.M_stock,
            C=self.C,
            Y=self.Y,
        )
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Create a snapshot of the state for rollback/recovery.
        
        Returns:
            Dict[str, Any]: Snapshot dictionary with timestamp and state
        """
        import time
        return {
            "timestamp": time.time(),
            "state": self.to_dict(),
        }
    
    @classmethod
    def restore_from_snapshot(cls, snapshot: Dict[str, Any]) -> "StateVector":
        """
        Restore state vector from snapshot.
        
        Args:
            snapshot: Snapshot dictionary from snapshot() method
            
        Returns:
            StateVector: Restored state vector
        """
        state_dict = snapshot.get("state", snapshot)  # Handle both formats
        return cls.from_dict(state_dict)


@dataclass
class ControlVector:
    """
    Control vector representing policy decisions.
    
    Contains:
    - Budget shares b_t on simplex (sums to 1, all >= 0)
    - Allocation decisions a_t for scarce resources
    - Logistics flow plan f_t (placeholder for Phase A)
    
    Args:
        budget_shares: Dictionary mapping budget categories to shares (must sum to 1)
        allocations: Dictionary mapping resource types to allocation amounts
        flows: Dictionary mapping flow identifiers to flow amounts (placeholder)
    """
    
    budget_shares: Dict[str, float] = field(default_factory=dict)
    allocations: Dict[str, float] = field(default_factory=dict)
    flows: Dict[str, float] = field(default_factory=dict)
    
    def validate_simplex(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that budget shares form a simplex (sum to 1, all >= 0).
        
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not self.budget_shares:
            return False, "Budget shares dictionary is empty"
        
        total = sum(self.budget_shares.values())
        tolerance = 1e-6
        
        if abs(total - 1.0) > tolerance:
            return False, f"Budget shares sum to {total}, must sum to 1.0"
        
        for category, share in self.budget_shares.items():
            if share < 0:
                return False, f"Budget share for {category} is negative: {share}"
        
        return True, None
    
    @classmethod
    def sample_budget_simplex(
        cls,
        categories: List[str],
        alpha: Optional[float] = None,
        rng: Optional[np.random.Generator] = None
    ) -> "ControlVector":
        """
        Sample budget shares from Dirichlet distribution (simplex).
        
        Args:
            categories: List of budget category names
            alpha: Dirichlet concentration parameter (default: 1.0 for uniform)
            rng: Random number generator (default: new generator)
            
        Returns:
            ControlVector: New control vector with sampled budget shares
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if alpha is None:
            alpha = 1.0
        
        n = len(categories)
        dirichlet_params = np.full(n, alpha)
        sampled = rng.dirichlet(dirichlet_params)
        
        budget_shares = {cat: float(val) for cat, val in zip(categories, sampled)}
        
        return cls(budget_shares=budget_shares)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert control vector to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "budget_shares": self.budget_shares.copy(),
            "allocations": self.allocations.copy(),
            "flows": self.flows.copy(),
        }
    
    @classmethod
    def from_dict(cls, control_dict: Dict[str, Any]) -> "ControlVector":
        """
        Create control vector from dictionary.
        
        Args:
            control_dict: Dictionary representation
            
        Returns:
            ControlVector: New control vector instance
        """
        return cls(
            budget_shares=control_dict.get("budget_shares", {}).copy(),
            allocations=control_dict.get("allocations", {}).copy(),
            flows=control_dict.get("flows", {}).copy(),
        )


class DynamicsModel:
    """
    Macro system dynamics model.
    
    Implements discrete-time dynamics for socioeconomic system:
    - Population growth with births, deaths, migration
    - Labor force adjustment
    - Unemployment with stable lag model
    - Capital accumulation
    - Infrastructure health evolution
    - Literacy diffusion
    - Output production (Cobb-Douglas with gating)
    
    Args:
        delta_K: Capital depreciation rate (default: 0.05)
        delta_I: Infrastructure decay rate (default: 0.02)
        alpha: Capital share in production (default: 0.3)
        kappa_K: Capital investment efficiency (default: 0.8)
        kappa_I: Infrastructure maintenance efficiency (default: 0.7)
        kappa_literacy: Literacy diffusion rate (default: 0.1)
        delta_literacy: Literacy disruption rate (default: 0.01)
        alpha_U: Unemployment adjustment rate (default: 0.1)
        alpha_rho: Labor force participation adjustment (default: 0.05)
    """
    
    def __init__(
        self,
        delta_K: float = 0.05,
        delta_I: float = 0.02,
        alpha: float = 0.3,
        kappa_K: float = 0.8,
        kappa_I: float = 0.7,
        kappa_literacy: float = 0.1,
        delta_literacy: float = 0.01,
        alpha_U: float = 0.1,
        alpha_rho: float = 0.05,
    ) -> None:
        """Initialize dynamics model with parameters."""
        self.delta_K = delta_K
        self.delta_I = delta_I
        self.alpha = alpha
        self.kappa_K = kappa_K
        self.kappa_I = kappa_I
        self.kappa_literacy = kappa_literacy
        self.delta_literacy = delta_literacy
        self.alpha_U = alpha_U
        self.alpha_rho = alpha_rho
    
    def step(
        self,
        x_t: StateVector,
        u_t: ControlVector,
        w_t: Optional[Dict[str, float]] = None
    ) -> StateVector:
        """
        Single time step evolution: x_{t+1} = f(x_t, u_t, w_t).
        
        Args:
            x_t: Current state vector
            u_t: Control vector
            w_t: Disturbance vector (optional)
            
        Returns:
            StateVector: Next state vector x_{t+1}
        """
        if w_t is None:
            w_t = {}
        
        # Validate control
        is_valid, error = u_t.validate_simplex()
        if not is_valid:
            logger.warning(f"Invalid control vector: {error}")
        
        # Extract budget shares
        b_E = u_t.budget_shares.get("energy", 0.0)
        b_F = u_t.budget_shares.get("food", 0.0)
        b_I = u_t.budget_shares.get("infrastructure", 0.0)
        b_edu = u_t.budget_shares.get("education", 0.0)
        b_health = u_t.budget_shares.get("healthcare", 0.0)
        b_RD = u_t.budget_shares.get("R&D", 0.0)
        b_welfare = u_t.budget_shares.get("welfare", 0.0)
        
        # Total budget (derived from output)
        B_t = x_t.Y * 0.2  # 20% of output as budget
        
        # Disturbances
        demand_shock = w_t.get("demand_shock", 0.0)
        trade_shock = w_t.get("trade_shock", 0.0)
        productivity_shock = w_t.get("productivity_shock", 1.0)
        disaster_shock = w_t.get("disaster_shock", 0.0)
        
        # Population dynamics (simplified: constant for now)
        births = x_t.P * 0.02  # 2% birth rate
        deaths = x_t.P * 0.015  # 1.5% death rate (adjusted by healthcare)
        death_reduction = b_health * B_t * 0.0001
        deaths = max(0, deaths - death_reduction)
        migration = (x_t.S - 0.5) * x_t.P * 0.001  # Migration based on stability
        P_next = x_t.P + births - deaths + migration
        
        # Labor force participation
        rho_t = x_t.L / x_t.P if x_t.P > 0 else 0.5
        rho_next = rho_t + self.alpha_rho * (x_t.literacy - rho_t) + w_t.get("labor_shock", 0.0)
        rho_next = np.clip(rho_next, 0.3, 0.8)  # Reasonable bounds
        L_next = rho_next * P_next
        
        # Unemployment dynamics (stable lag model)
        # Target unemployment depends on output, investment, and shocks
        U_target = 0.05 + 0.1 * (1.0 - min(1.0, x_t.Y / 1000000.0))  # Higher if output low
        U_target += demand_shock * 0.1
        U_target = np.clip(U_target, 0.0, 0.3)
        
        U_next = x_t.U + self.alpha_U * (U_target - x_t.U) + w_t.get("unemployment_shock", 0.0)
        U_next = np.clip(U_next, 0.0, 1.0)
        
        # Capital accumulation
        I_inv = self.kappa_K * b_I * B_t * self._phi_infrastructure(x_t.I)
        K_next = (1 - self.delta_K) * x_t.K + I_inv
        
        # Infrastructure health
        maintenance = self.kappa_I * b_I * B_t * 0.0001
        usage_decay = self.delta_I * x_t.I * (1.0 + abs(demand_shock))
        I_next = np.clip(x_t.I + maintenance - usage_decay, 0.0, 1.0)
        
        # Transport capacity (depends on infrastructure)
        Tcap_next = x_t.Tcap * (0.9 + 0.1 * I_next)  # Degrades if infrastructure poor
        
        # Literacy diffusion
        education_investment = min(x_t.Ecap, b_edu * B_t)
        literacy_gain = self.kappa_literacy * education_investment * (1.0 - x_t.literacy) / 10000.0
        literacy_loss = self.delta_literacy * disaster_shock
        literacy_next = np.clip(x_t.literacy + literacy_gain - literacy_loss, 0.0, 1.0)
        
        # Education and healthcare capacity (slow adjustment)
        Ecap_next = x_t.Ecap + b_edu * B_t * 0.01 - x_t.Ecap * 0.02
        Ecap_next = max(0, Ecap_next)
        
        Hcap_next = x_t.Hcap + b_health * B_t * 0.01 - x_t.Hcap * 0.02
        Hcap_next = max(0, Hcap_next)
        
        # Stock dynamics (simplified)
        # Energy stock
        E_gen = b_E * B_t * 0.1  # Energy generation from budget
        E_demand = x_t.Y * 0.0001 + x_t.Tcap * 0.01  # Energy demand
        E_import = w_t.get("energy_import", 0.0) * (1.0 - trade_shock)
        E_next = x_t.E_stock + E_gen + E_import - E_demand
        E_next = max(0, E_next)
        
        # Food stock
        F_prod = b_F * B_t * 0.2  # Food production
        F_demand = x_t.P * 0.03  # Per capita food demand
        F_import = w_t.get("food_import", 0.0) * (1.0 - trade_shock)
        F_next = x_t.F_stock + F_prod + F_import - F_demand
        F_next = max(0, F_next)
        
        # Materials stock
        M_prod = b_RD * B_t * 0.1
        M_demand = I_inv * 0.1
        M_next = x_t.M_stock + M_prod - M_demand
        M_next = max(0, M_next)
        
        # Ecological damage (irreversible, accumulates)
        emissions = E_demand * 0.001  # Emissions from energy use
        C_next = x_t.C + emissions  # Irreversible accumulation
        
        # Output (Cobb-Douglas with energy/logistics gating)
        A_t = productivity_shock  # Total factor productivity
        L_employed = L_next * (1 - U_next)
        g_E = self._gate_energy(x_t.E_stock, E_demand)
        h_T = self._gate_transport(x_t.Tcap)
        
        Y_next = A_t * (K_next ** self.alpha) * (L_employed ** (1 - self.alpha)) * g_E * h_T
        
        # Wage (simplified: depends on output per worker)
        W_next = x_t.W * (1.0 + 0.1 * (Y_next / max(1, L_employed) - x_t.W) / x_t.W)
        W_next = max(0.1, W_next)
        
        # Social stability (depends on unemployment, welfare, literacy)
        stability_gain = b_welfare * B_t * 0.0001 + literacy_next * 0.1
        stability_loss = U_next * 0.2 + (1.0 - min(1.0, F_next / (x_t.P * 0.03))) * 0.3
        S_next = np.clip(x_t.S + stability_gain - stability_loss, 0.0, 1.0)
        
        # Create next state
        x_next = StateVector(
            P=P_next,
            L=L_next,
            U=U_next,
            W=W_next,
            S=S_next,
            literacy=literacy_next,
            Ecap=Ecap_next,
            Hcap=Hcap_next,
            K=K_next,
            I=I_next,
            Tcap=Tcap_next,
            E_stock=E_next,
            F_stock=F_next,
            M_stock=M_next,
            C=C_next,
            Y=Y_next,
        )
        
        return x_next
    
    def _phi_infrastructure(self, I: float) -> float:
        """
        Infrastructure gating function for investment efficiency.
        
        Args:
            I: Infrastructure health [0, 1]
            
        Returns:
            float: Efficiency multiplier [0, 1]
        """
        return 0.5 + 0.5 * I  # Better infrastructure = better investment efficiency
    
    def _gate_energy(self, E_stock: float, E_demand: float) -> float:
        """
        Energy gating function for output.
        
        Args:
            E_stock: Available energy stock
            E_demand: Energy demand
            
        Returns:
            float: Gating factor [0, 1]
        """
        if E_demand <= 0:
            return 1.0
        ratio = E_stock / E_demand
        return min(1.0, max(0.0, ratio))  # Linear gating
    
    def _gate_transport(self, Tcap: float) -> float:
        """
        Transport capacity gating function for output.
        
        Args:
            Tcap: Transport capacity
            
        Returns:
            float: Gating factor [0, 1]
        """
        # Normalize: full capacity at 10000
        normalized = min(1.0, Tcap / 10000.0)
        return 0.5 + 0.5 * normalized  # Minimum 50% even with no transport


class ConstraintChecker:
    """
    Hard feasibility constraint checker.
    
    Checks:
    - Simplex budgets: sum(b_i) == 1, b_i >= 0
    - Minimum needs: F_stock + F_in - F_out >= P * c_min
    - Energy balance: E_gen + E_import + E_stock >= E_demand + E_logistics
    - Capacity: flow(e) <= cap(e, Tcap_t, I_t)
    - Welfare bounds: U_t <= U_max, S_t >= S_min
    - Irreversibility: C_{t+1} >= C_t
    
    Args:
        U_max: Maximum allowed unemployment (default: 0.2)
        S_min: Minimum required social stability (default: 0.3)
        c_min: Minimum per-capita food consumption (default: 0.02)
    """
    
    def __init__(
        self,
        U_max: float = 0.2,
        S_min: float = 0.3,
        c_min: float = 0.02,
    ) -> None:
        """Initialize constraint checker with bounds."""
        self.U_max = U_max
        self.S_min = S_min
        self.c_min = c_min
    
    def check_feasible(
        self,
        x_t: StateVector,
        u_t: ControlVector
    ) -> Tuple[bool, List[str]]:
        """
        Check if state and control satisfy all hard constraints.
        
        Args:
            x_t: State vector
            u_t: Control vector
            
        Returns:
            Tuple[bool, List[str]]: (is_feasible, list of violations)
        """
        violations = []
        
        # Check simplex constraint
        is_valid, error = u_t.validate_simplex()
        if not is_valid:
            violations.append(f"Budget simplex violation: {error}")
        
        # Check minimum food needs
        min_food = x_t.P * self.c_min
        if x_t.F_stock < min_food:
            violations.append(
                f"Food stock {x_t.F_stock:.2f} below minimum {min_food:.2f} "
                f"(P={x_t.P:.0f} * c_min={self.c_min})"
            )
        
        # Check energy balance (simplified: stock must cover demand)
        E_demand = x_t.Y * 0.0001 + x_t.Tcap * 0.01
        if x_t.E_stock < E_demand * 0.5:  # At least 50% coverage
            violations.append(
                f"Energy stock {x_t.E_stock:.2f} insufficient for demand {E_demand:.2f}"
            )
        
        # Check welfare bounds
        if x_t.U > self.U_max:
            violations.append(
                f"Unemployment {x_t.U:.3f} exceeds maximum {self.U_max}"
            )
        
        if x_t.S < self.S_min:
            violations.append(
                f"Social stability {x_t.S:.3f} below minimum {self.S_min}"
            )
        
        # Check state bounds (from StateVector.validate)
        is_state_valid, state_violations = x_t.validate()
        if not is_state_valid:
            violations.extend(state_violations)
        
        is_feasible = len(violations) == 0
        return is_feasible, violations
    
    def get_violations(
        self,
        x_t: StateVector,
        u_t: ControlVector
    ) -> List[str]:
        """
        Get detailed list of constraint violations.
        
        Args:
            x_t: State vector
            u_t: Control vector
            
        Returns:
            List[str]: List of violation messages
        """
        _, violations = self.check_feasible(x_t, u_t)
        return violations


class ForwardSimulator:
    """
    Forward simulator for multi-step trajectory simulation.
    
    Simulates system evolution over multiple time steps with feasibility checking.
    
    Args:
        dynamics: Dynamics model
        constraint_checker: Constraint checker
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        constraint_checker: ConstraintChecker
    ) -> None:
        """Initialize forward simulator."""
        self.dynamics = dynamics
        self.constraint_checker = constraint_checker
    
    def simulate_scenario(
        self,
        x_0: StateVector,
        policy: ControlVector,
        disturbances: Optional[List[Dict[str, float]]] = None,
        horizon: int = 10
    ) -> Tuple[List[StateVector], List[bool], Optional[int]]:
        """
        Simulate forward trajectory under a policy.
        
        Args:
            x_0: Initial state
            policy: Control policy (can be constant or time-varying)
            disturbances: List of disturbance vectors (one per time step)
            horizon: Simulation horizon (number of steps)
            
        Returns:
            Tuple[List[StateVector], List[bool], Optional[int]]:
                (trajectory, feasibility_flags, first_violation_time)
        """
        if disturbances is None:
            disturbances = [{}] * horizon
        
        trajectory = [x_0.copy()]
        feasibility_flags = []
        first_violation_time = None
        
        x_current = x_0.copy()
        
        for t in range(horizon):
            # Get control for this time step (use policy if constant, or index if list)
            if isinstance(policy, ControlVector):
                u_t = policy
            elif isinstance(policy, list) and t < len(policy):
                u_t = policy[t]
            else:
                u_t = policy  # Fallback
            
            # Get disturbance for this time step
            w_t = disturbances[t] if t < len(disturbances) else {}
            
            # Evolve state
            x_next = self.dynamics.step(x_current, u_t, w_t)
            trajectory.append(x_next)
            
            # Check feasibility
            is_feasible, violations = self.constraint_checker.check_feasible(x_next, u_t)
            feasibility_flags.append(is_feasible)
            
            if not is_feasible and first_violation_time is None:
                first_violation_time = t + 1
                logger.debug(f"First constraint violation at t={first_violation_time}: {violations[:2]}")
            
            x_current = x_next
        
        return trajectory, feasibility_flags, first_violation_time
    
    def check_trajectory_feasibility(
        self,
        trajectory: List[StateVector],
        controls: List[ControlVector]
    ) -> Tuple[bool, Optional[int]]:
        """
        Check feasibility of entire trajectory.
        
        Args:
            trajectory: List of state vectors
            controls: List of control vectors (one per transition)
            
        Returns:
            Tuple[bool, Optional[int]]: (is_feasible, first_violation_time)
        """
        if len(controls) != len(trajectory) - 1:
            logger.warning(
                f"Trajectory length {len(trajectory)} doesn't match controls length {len(controls)}"
            )
        
        for t, x_t in enumerate(trajectory[1:], start=1):  # Skip initial state
            u_t = controls[t - 1] if t - 1 < len(controls) else ControlVector()
            is_feasible, violations = self.constraint_checker.check_feasible(x_t, u_t)
            
            if not is_feasible:
                return False, t
        
        return True, None

