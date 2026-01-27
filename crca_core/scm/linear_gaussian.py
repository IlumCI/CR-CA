"""Linear-Gaussian Structural Causal Model (SCM).

Implements counterfactual reasoning via abduction–action–prediction:
- Abduction: infer exogenous noise U from a factual observation of all endogenous variables.
- Action: apply do-interventions by replacing structural equations for intervened variables.
- Prediction: propagate values in topological order using the same inferred noise U.

This implementation is intentionally conservative:
- v0.1 assumes a fully observed system and (by default) independent noises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import networkx as nx  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("networkx is required for LinearGaussianSCM") from e

from crca_core.models.spec import SCMSpec, StructuralEquationSpec


@dataclass(frozen=True)
class LinearGaussianSCM:
    """A linear SCM with additive Gaussian noise for each endogenous variable."""

    variables: Tuple[str, ...]
    parents: Dict[str, Tuple[str, ...]]
    coefficients: Dict[Tuple[str, str], float]  # (parent, child) -> beta
    intercepts: Dict[str, float]
    # Noise is represented per-variable; v0.1 assumes diagonal covariance by default.
    noise_cov: Optional[np.ndarray] = None

    @classmethod
    def from_spec(cls, spec: SCMSpec) -> "LinearGaussianSCM":
        if spec.scm_type != "linear_gaussian":
            raise ValueError(f"Unsupported scm_type: {spec.scm_type}")

        variables: List[str] = []
        parents: Dict[str, Tuple[str, ...]] = {}
        coefficients: Dict[Tuple[str, str], float] = {}
        intercepts: Dict[str, float] = {}

        for eq in spec.equations:
            if eq.form != "linear_gaussian":
                raise ValueError(f"Unsupported equation form: {eq.form}")
            v = eq.variable
            if v in variables:
                raise ValueError(f"Duplicate equation for variable: {v}")
            variables.append(v)
            parents[v] = tuple(eq.parents)
            intercepts[v] = float(eq.intercept)
            for p, beta in eq.coefficients.items():
                coefficients[(p, v)] = float(beta)

        noise_cov = None
        if spec.noise_cov is not None:
            noise_cov = np.array(spec.noise_cov, dtype=float)
            if noise_cov.shape[0] != noise_cov.shape[1]:
                raise ValueError("noise_cov must be square")
            if noise_cov.shape[0] != len(variables):
                raise ValueError("noise_cov dimension must match number of equations/variables")

        model = cls(
            variables=tuple(variables),
            parents=parents,
            coefficients=coefficients,
            intercepts=intercepts,
            noise_cov=noise_cov,
        )
        # Validate acyclicity/topological order.
        _ = model.topological_order()
        return model

    def topological_order(self) -> List[str]:
        g = nx.DiGraph()
        for v in self.variables:
            g.add_node(v)
        for child, ps in self.parents.items():
            for p in ps:
                g.add_edge(p, child)
        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("SCM graph must be a DAG")
        return list(nx.topological_sort(g))

    def _matrix_form(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (A, b, Sigma_u) for A V = b + U."""
        n = len(self.variables)
        index = {v: i for i, v in enumerate(self.variables)}
        M = np.zeros((n, n), dtype=float)
        b = np.zeros((n,), dtype=float)
        for v, intercept in self.intercepts.items():
            b[index[v]] = float(intercept)
        for (p, v), beta in self.coefficients.items():
            i = index[v]
            j = index[p]
            M[i, j] = float(beta)
        A = np.eye(n) - M
        if self.noise_cov is None:
            Sigma_u = np.eye(n)
        else:
            Sigma_u = self.noise_cov
        return A, b, Sigma_u

    def abduce_noise_conditional(self, factual: Mapping[str, float]) -> Dict[str, float]:
        """Conditional Gaussian abduction for partial observations."""
        A, b, Sigma_u = self._matrix_form()
        n = len(self.variables)
        index = {v: i for i, v in enumerate(self.variables)}
        obs_idx = [index[v] for v in factual.keys() if v in index]
        miss_idx = [i for i in range(n) if i not in obs_idx]
        v_o = np.array([float(factual[self.variables[i]]) for i in obs_idx], dtype=float)

        # Compute mean and covariance of V
        A_inv = np.linalg.inv(A)
        mu_v = A_inv @ b
        Sigma_v = A_inv @ Sigma_u @ A_inv.T

        if not miss_idx:
            v_full = np.zeros(n, dtype=float)
            for i in range(n):
                v_full[i] = float(factual[self.variables[i]])
        else:
            Sigma_oo = Sigma_v[np.ix_(obs_idx, obs_idx)]
            Sigma_mo = Sigma_v[np.ix_(miss_idx, obs_idx)]
            mu_o = mu_v[obs_idx]
            mu_m = mu_v[miss_idx]
            # Conditional mean of V_m given V_o
            v_m = mu_m + Sigma_mo @ np.linalg.inv(Sigma_oo) @ (v_o - mu_o)
            v_full = np.zeros(n, dtype=float)
            for i, idx in enumerate(obs_idx):
                v_full[idx] = v_o[i]
            for i, idx in enumerate(miss_idx):
                v_full[idx] = v_m[i]

        u_mean = A @ v_full - b
        return {self.variables[i]: float(u_mean[i]) for i in range(n)}

    def abduce_noise(self, factual: Mapping[str, float], *, allow_partial: bool = False) -> Dict[str, float]:
        """Infer per-variable noise U from a factual state.

        If `allow_partial` is False, all endogenous variables must be observed.
        If True, noise is inferred only for observed variables whose parents are observed.
        """
        order = self.topological_order()
        u: Dict[str, float] = {}
        x: Dict[str, float] = {k: float(v) for k, v in factual.items()}

        missing = [v for v in order if v not in x]
        if missing and not allow_partial:
            raise ValueError(f"Factual observation missing variables: {missing}")
        if missing and allow_partial and self.noise_cov is not None:
            return self.abduce_noise_conditional(factual)

        for v in order:
            if v not in x:
                continue
            pred = self.intercepts.get(v, 0.0)
            for p in self.parents.get(v, ()):
                if p not in x:
                    if allow_partial:
                        pred = None
                        break
                    raise ValueError(f"Factual observation missing parent '{p}' for '{v}'")
                beta = self.coefficients.get((p, v), 0.0)
                pred += beta * x[p]
            if pred is None:
                continue
            u[v] = float(x[v] - pred)
        return u

    def predict(self, noise: Mapping[str, float], interventions: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        """Forward simulate endogenous variables under interventions using fixed noise."""
        interventions = interventions or {}
        order = self.topological_order()
        x: Dict[str, float] = {}

        for v in order:
            if v in interventions:
                x[v] = float(interventions[v])
                continue
            pred = self.intercepts.get(v, 0.0)
            for p in self.parents.get(v, ()):
                beta = self.coefficients.get((p, v), 0.0)
                pred += beta * x[p]
            pred += float(noise.get(v, 0.0))
            x[v] = float(pred)
        return x

    def counterfactual(self, factual: Mapping[str, float], interventions: Mapping[str, float]) -> Dict[str, float]:
        """Compute a counterfactual state via abduction–action–prediction."""
        u = self.abduce_noise(factual)
        return self.predict(u, interventions=interventions)

