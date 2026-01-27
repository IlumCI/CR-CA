"""Run `crca_core` synthetic benchmarks and emit structured JSON.

This is a technical harness (not marketing). It should be runnable in CI/CD or
locally and produce machine-parseable outputs with provenance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from crca_core.benchmarks.synthetic_scm import (
    generate_latent_confounder_graph,
    generate_lagged_timeseries,
    generate_linear_gaussian_chain,
)
from crca_core.discovery import TabularDiscoveryConfig, discover_tabular
from crca_core.identify import identify_effect
from crca_core.core.lifecycle import lock_spec
from crca_core.models.spec import CausalGraphSpec, DraftSpec, EdgeSpec, NodeSpec, RoleSpec
from crca_core.timeseries import PCMCIConfig, discover_timeseries_pcmci
from crca_core.models.provenance import ProvenanceManifest
from crca_core.scm import LinearGaussianSCM
from utils.canonical import stable_hash


def run_linear_gaussian_chain() -> Dict[str, Any]:
    spec, factual, noise = generate_linear_gaussian_chain(n_vars=4, beta=0.9, seed=1)
    scm = LinearGaussianSCM.from_spec(spec)

    abduced = scm.abduce_noise(factual)
    max_abs_err = max(abs(abduced[k] - noise[k]) for k in noise.keys())

    cf = scm.counterfactual(factual, interventions={"X0": factual["X0"] + 1.0})

    prov = ProvenanceManifest.minimal(
        spec_hash=stable_hash(
            {
                "benchmark": "linear_gaussian_chain",
                "generator": {"n_vars": 4, "beta": 0.9, "seed": 1},
                "scm_type": "linear_gaussian",
            }
        ),
        algorithm_config={"benchmark": "linear_gaussian_chain"},
        random_seeds={"numpy": 1},
    )

    return {
        "result_type": "BenchmarkResult",
        "benchmark": "linear_gaussian_chain",
        "provenance": prov.model_dump(),
        "metrics": {"abduction_max_abs_error": float(max_abs_err)},
        "artifacts": {
            "factual": factual,
            "counterfactual": cf,
            "notes": [
                "This benchmark checks abduction correctness (noise recovery) under full observability.",
                "Counterfactual uses abduction–action–prediction with fixed exogenous noise.",
            ],
        },
    }


def run_identification_benchmarks() -> Dict[str, Any]:
    # Identifiable (simple chain)
    draft = DraftSpec(
        graph=CausalGraphSpec(
            nodes=[NodeSpec(name="X"), NodeSpec(name="Y")],
            edges=[EdgeSpec(source="X", target="Y")],
        ),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    ident_simple = identify_effect(locked_spec=locked, treatment="X", outcome="Y")

    # Latent confounder (non-identifiable in conservative ID)
    latent_graph = generate_latent_confounder_graph()
    draft_latent = DraftSpec(
        graph=latent_graph,
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked_latent = lock_spec(draft_latent, approvals=["human"])
    ident_latent = identify_effect(locked_spec=locked_latent, treatment="X", outcome="Y")

    def _dump(obj: Any) -> Dict[str, Any]:
        return obj.model_dump() if hasattr(obj, "model_dump") else {"value": str(obj)}

    return {
        "result_type": "BenchmarkResult",
        "benchmark": "identification",
        "provenance": ProvenanceManifest.minimal(
            spec_hash=stable_hash({"benchmark": "identification"})
        ).model_dump(),
        "metrics": {},
        "artifacts": {
            "identifiable_case": _dump(ident_simple),
            "latent_confounder_case": _dump(ident_latent),
        },
    }


def run_discovery_benchmarks() -> Dict[str, Any]:
    # Tabular discovery
    import pandas as pd
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y})
    tabular = discover_tabular(df, TabularDiscoveryConfig(algorithm="pc", alpha=0.05))

    # Time-series discovery
    ts, cols = generate_lagged_timeseries(n_steps=200, seed=1)
    ts_df = pd.DataFrame(ts, columns=cols)
    ts_res = discover_timeseries_pcmci(
        ts_df, PCMCIConfig(max_lag=3, alpha=0.05, assume_sorted=True)
    )

    def _dump(obj: Any) -> Dict[str, Any]:
        return obj.model_dump() if hasattr(obj, "model_dump") else {"value": str(obj)}

    return {
        "result_type": "BenchmarkResult",
        "benchmark": "discovery",
        "provenance": ProvenanceManifest.minimal(
            spec_hash=stable_hash({"benchmark": "discovery"})
        ).model_dump(),
        "metrics": {},
        "artifacts": {
            "tabular": _dump(tabular),
            "timeseries": _dump(ts_res),
        },
    }


def main() -> None:
    results = {
        "benchmarks": [
            run_linear_gaussian_chain(),
            run_identification_benchmarks(),
            run_discovery_benchmarks(),
        ]
    }

    out_dir = REPO_ROOT / "benchmark_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "crca_core_benchmarks.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(str(out_path))


if __name__ == "__main__":
    main()

