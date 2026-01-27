import os

import pandas as pd

from crca_core.core.lifecycle import lock_spec
from crca_core.models.spec import CausalGraphSpec, DraftSpec, EdgeSpec, NodeSpec, RoleSpec
from crca_core.models.refusal import RefusalResult
from crca_llm.orchestrator import LLMOrchestrator


class FakeClient:
    def __init__(self, content: str):
        self._content = content

    def chat_completion(self, **kwargs) -> str:
        return self._content


def test_orchestrator_refuses_without_api_key() -> None:
    old = os.environ.pop("OPENAI_API_KEY", None)
    try:
        orch = LLMOrchestrator()
        res = orch.run(user_text="Test", observed_columns=["X", "Y"])
        assert res.refusals
        assert not res.draft_bundle.drafts
        assert isinstance(res.refusals[0], RefusalResult)
    finally:
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


def test_orchestrator_never_emits_locked_spec() -> None:
    payload = {
        "drafts": [
            {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
                "treatments": ["X"],
                "outcomes": ["Y"],
                "columns": ["X", "Y"],
            }
        ],
        "review_checklist": ["Confirm time ordering"],
    }
    orch = LLMOrchestrator(client=FakeClient(content=str(payload).replace("'", '"')))
    res = orch.run(user_text="Test", observed_columns=["X", "Y"])
    assert res.draft_bundle.drafts
    assert all(d.status.value == "draft" for d in res.draft_bundle.drafts)


def test_orchestrator_refuses_estimate_without_identification() -> None:
    payload = {
        "drafts": [
            {
                "nodes": ["X", "Y"],
                "edges": [["X", "Y"]],
                "treatments": ["X"],
                "outcomes": ["Y"],
                "columns": ["X", "Y"],
            }
        ],
        "review_checklist": [],
    }
    orch = LLMOrchestrator(client=FakeClient(content=str(payload).replace("'", '"')))

    draft = DraftSpec(
        graph=CausalGraphSpec(nodes=[NodeSpec(name="X"), NodeSpec(name="Y")], edges=[EdgeSpec(source="X", target="Y")]),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    df = pd.DataFrame({"X": [1, 2, 3], "Y": [2, 3, 4]})

    res = orch.run(
        user_text="Test",
        observed_columns=["X", "Y"],
        locked_spec=locked,
        data=df,
        actions=["estimate"],
    )
    assert res.refusals
