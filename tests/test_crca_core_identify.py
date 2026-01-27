from crca_core.identify import identify_effect
from crca_core.core.lifecycle import lock_spec
from crca_core.models.result import IdentificationResult
from crca_core.models.spec import CausalGraphSpec, DraftSpec, EdgeSpec, NodeSpec, RoleSpec


def test_identify_backdoor_empty_set_for_simple_chain() -> None:
    draft = DraftSpec(
        graph=CausalGraphSpec(
            nodes=[NodeSpec(name="X"), NodeSpec(name="Y")],
            edges=[EdgeSpec(source="X", target="Y")],
        ),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    res = identify_effect(locked_spec=locked, treatment="X", outcome="Y")
    assert isinstance(res, IdentificationResult)
    assert res.method == "backdoor"
