"""Synthetic SCM generators with ground truth.

These are used for correctness testing and benchmarking. They are not intended
to be "realistic" by default; they are intended to be diagnosable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from crca_core.models.spec import (
    CausalGraphSpec,
    EdgeSpec,
    NodeSpec,
    NoiseSpec,
    SCMSpec,
    StructuralEquationSpec,
)


def generate_linear_gaussian_chain(
    *,
    n_vars: int = 3,
    beta: float = 1.0,
    intercept: float = 0.0,
    noise_std: float = 1.0,
    seed: int = 0,
) -> Tuple[SCMSpec, Dict[str, float], Dict[str, float]]:
    """Generate a simple chain SCM: X0 -> X1 -> ... -> X{n-1}.

    Returns:
        (scm_spec, factual_sample, noise_sample)
    """
    if n_vars < 2:
        raise ValueError("n_vars must be >= 2")

    rng = np.random.default_rng(seed)
    vars_: List[str] = [f"X{i}" for i in range(n_vars)]

    equations: List[StructuralEquationSpec] = []
    noise: Dict[str, float] = {}
    factual: Dict[str, float] = {}

    for i, v in enumerate(vars_):
        u = float(rng.normal(0.0, noise_std))
        noise[v] = u
        if i == 0:
            equations.append(
                StructuralEquationSpec(
                    variable=v,
                    parents=[],
                    coefficients={},
                    intercept=intercept,
                    noise=NoiseSpec(distribution="gaussian", params={"mean": 0.0, "std": noise_std}),
                )
            )
            factual[v] = intercept + u
        else:
            p = vars_[i - 1]
            equations.append(
                StructuralEquationSpec(
                    variable=v,
                    parents=[p],
                    coefficients={p: beta},
                    intercept=intercept,
                    noise=NoiseSpec(distribution="gaussian", params={"mean": 0.0, "std": noise_std}),
                )
            )
            factual[v] = intercept + beta * factual[p] + u

    spec = SCMSpec(scm_type="linear_gaussian", equations=equations)
    return spec, factual, noise


def generate_latent_confounder_graph() -> CausalGraphSpec:
    """Return a simple graph with latent confounding between X and Y."""
    return CausalGraphSpec(
        nodes=[
            NodeSpec(name="X", observed=True),
            NodeSpec(name="Y", observed=True),
        ],
        edges=[],
        latent_confounders=["X<->Y"],
    )


def generate_lagged_timeseries(
    *,
    n_steps: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Generate a simple two-variable lagged SCM time series."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    for t in range(1, n_steps):
        x[t] = 0.7 * x[t - 1] + rng.normal(0, 1)
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 1] + rng.normal(0, 1)
    data = np.vstack([x, y]).T
    return data, ["X", "Y"]

