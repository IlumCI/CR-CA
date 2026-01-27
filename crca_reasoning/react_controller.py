"""Multi-cycle ReAct controller with strict tool gating and end-of-cycle critique."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from crca_core.models.refusal import RefusalResult
from crca_core.models.result import AnyResult
from crca_reasoning.critique import critique_cycle
from crca_reasoning.memory import StructuredMemory
from crca_reasoning.types import (
    LRMPlanResult,
    RationaleTrace,
    ReActAction,
    ReActCycleTrace,
    ReActObservation,
)
from crca_reasoning.tool_router import ToolRouter


ReasonerFn = Callable[[StructuredMemory], tuple[str, List[ReActAction], str]]


@dataclass
class ReActController:
    max_cycles: int = 3
    tool_router: ToolRouter = field(default_factory=ToolRouter)
    memory: StructuredMemory = field(default_factory=StructuredMemory)

    def run(self, reasoner: ReasonerFn) -> LRMPlanResult:
        cycle_traces: List[ReActCycleTrace] = []
        core_results: List[AnyResult] = []
        refusals: List[RefusalResult] = []
        rationale = RationaleTrace()

        for _ in range(self.max_cycles):
            reasoning, actions, rationale_step = reasoner(self.memory)
            cycle = ReActCycleTrace(reasoning=reasoning, actions=actions, observations=[])
            if rationale_step:
                rationale.steps.append(rationale_step)

            for action in actions:
                result = self.tool_router.call_tool(
                    tool_name=action.tool_name, payload=action.payload
                )
                if isinstance(result, RefusalResult):
                    obs = ReActObservation(
                        tool_name=action.tool_name, refusal=result
                    )
                    refusals.append(result)
                else:
                    obs = ReActObservation(
                        tool_name=action.tool_name, result=result
                    )
                    core_results.append(result)
                cycle.observations.append(obs)

            critique = critique_cycle(cycle)
            cycle.critique = critique.model_dump()
            cycle_traces.append(cycle)

            if critique.should_revise:
                continue
            break

        # Write-on-finalize memory policy
        for res in core_results:
            try:
                self.memory.stage_write(res.model_dump())
            except Exception:
                continue
        self.memory.finalize()

        return LRMPlanResult(
            cycle_traces=cycle_traces,
            rationale_trace=rationale,
            core_results=core_results,
            refusals=refusals,
            finalized=True,
        )
