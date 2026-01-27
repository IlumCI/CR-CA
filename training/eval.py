"""Mixed evaluation harness for LRM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from crca_core.benchmarks.synthetic_scm import generate_latent_confounder_graph
from crca_core.identify import identify_effect
from crca_core.core.lifecycle import lock_spec
from crca_core.models.spec import CausalGraphSpec, DraftSpec, EdgeSpec, NodeSpec, RoleSpec
from crca_reasoning.types import LRMPlanResult


@dataclass
class EvalConfig:
    output_path: str = "eval_results/lrm_eval.json"


def eval_react_metrics(plans: List[LRMPlanResult]) -> Dict[str, float]:
    if not plans:
        return {
            "cycle_convergence": 0.0,
            "refusal_rate": 0.0,
            "refusal_structured_rate": 0.0,
            "tool_call_coverage": 0.0,
        }
    cycles = [len(p.cycle_traces) for p in plans]
    refusal_count = sum(len(p.refusals) for p in plans)
    refusal_structured = sum(
        1
        for p in plans
        for r in p.refusals
        if r.reason_codes and r.message
    )
    action_count = 0
    observed_actions = 0
    for plan in plans:
        for cycle in plan.cycle_traces:
            action_count += len(cycle.actions)
            observed_actions += sum(
                1
                for act in cycle.actions
                if any(obs.tool_name == act.tool_name for obs in cycle.observations)
            )
    tool_call_coverage = (observed_actions / float(action_count)) if action_count else 0.0
    return {
        "cycle_convergence": sum(1 for c in cycles if c == 1) / float(len(cycles)),
        "refusal_rate": refusal_count / float(len(plans)),
        "refusal_structured_rate": refusal_structured / float(max(1, refusal_count)),
        "tool_call_coverage": tool_call_coverage,
    }


def eval_causal_identification() -> Dict[str, str]:
    # Identifiable chain
    draft = DraftSpec(
        graph=CausalGraphSpec(
            nodes=[NodeSpec(name="X"), NodeSpec(name="Y")],
            edges=[EdgeSpec(source="X", target="Y")],
        ),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    ident_chain = identify_effect(locked_spec=locked, treatment="X", outcome="Y")

    # Latent confounding case
    latent_graph = generate_latent_confounder_graph()
    draft_latent = DraftSpec(
        graph=latent_graph,
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked_latent = lock_spec(draft_latent, approvals=["human"])
    ident_latent = identify_effect(locked_spec=locked_latent, treatment="X", outcome="Y")

    return {
        "ident_chain": ident_chain.result_type,
        "latent_case": ident_latent.result_type,
    }


def run_eval(plans: List[LRMPlanResult], cfg: EvalConfig) -> None:
    results = {
        "react_metrics": eval_react_metrics(plans),
        "causal_identification": eval_causal_identification(),
    }
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

