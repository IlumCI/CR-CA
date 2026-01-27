"""Synthetic benchmark generators (ground-truth) for crca_core."""

from crca_core.benchmarks.synthetic_scm import (
    generate_latent_confounder_graph,
    generate_lagged_timeseries,
    generate_linear_gaussian_chain,
)

__all__ = [
    "generate_linear_gaussian_chain",
    "generate_latent_confounder_graph",
    "generate_lagged_timeseries",
]

