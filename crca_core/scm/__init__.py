"""SCM implementations for `crca_core`.

Counterfactuals require an explicit SCM. v0.1 implements linear-Gaussian SCMs.
"""

from crca_core.scm.linear_gaussian import LinearGaussianSCM

__all__ = ["LinearGaussianSCM"]

