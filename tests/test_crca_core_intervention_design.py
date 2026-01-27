from crca_core.core.intervention_design import FeasibilityConstraints, TargetQuery, design_intervention
from crca_core.core.lifecycle import lock_spec
from crca_core.models.spec import DraftSpec, CausalGraphSpec, EdgeSpec, NodeSpec, RoleSpec


def test_design_intervention_randomize_when_manipulable() -> None:
    draft = DraftSpec(
        graph=CausalGraphSpec(nodes=[NodeSpec(name="X"), NodeSpec(name="Y")], edges=[EdgeSpec(source="X", target="Y")]),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    res = design_intervention(
        locked_spec=locked,
        target_query=TargetQuery(query_type="identify_effect", treatment="X", outcome="Y"),
        constraints=FeasibilityConstraints(manipulable_variables=["X"]),
    )
    assert res.result_type == "InterventionDesign"
    assert any(d["design_type"] == "randomize_treatment" for d in res.designs)


def test_design_intervention_measure_parents_of_treatment() -> None:
    draft = DraftSpec(
        graph=CausalGraphSpec(
            nodes=[NodeSpec(name="Z"), NodeSpec(name="X"), NodeSpec(name="Y")],
            edges=[EdgeSpec(source="Z", target="X"), EdgeSpec(source="X", target="Y")],
        ),
        roles=RoleSpec(treatments=["X"], outcomes=["Y"]),
    )
    locked = lock_spec(draft, approvals=["human"])
    res = design_intervention(
        locked_spec=locked,
        target_query=TargetQuery(query_type="identify_effect", treatment="X", outcome="Y"),
        constraints=FeasibilityConstraints(manipulable_variables=[]),
    )
    assert any(d["design_type"] == "measure_confounder_candidates" for d in res.designs)

