"""Causal discovery (hypothesis generation) modules.

Discovery outputs are hypotheses under explicit assumptions; they are not truth.
"""

from crca_core.discovery.tabular import TabularDiscoveryConfig, discover_tabular

__all__ = ["TabularDiscoveryConfig", "discover_tabular"]

