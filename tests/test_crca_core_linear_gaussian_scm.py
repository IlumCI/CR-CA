import numpy as np

from crca_core.scm import LinearGaussianSCM
from crca_core.models.spec import NoiseSpec, SCMSpec, StructuralEquationSpec


def test_linear_gaussian_scm_counterfactual_simple_chain() -> None:
    # X := Ux
    # Y := 2*X + Uy
    spec = SCMSpec(
        scm_type="linear_gaussian",
        equations=[
            StructuralEquationSpec(
                variable="X",
                parents=[],
                coefficients={},
                intercept=0.0,
                noise=NoiseSpec(distribution="gaussian", params={"mean": 0.0, "std": 1.0}),
            ),
            StructuralEquationSpec(
                variable="Y",
                parents=["X"],
                coefficients={"X": 2.0},
                intercept=0.0,
                noise=NoiseSpec(distribution="gaussian", params={"mean": 0.0, "std": 1.0}),
            ),
        ],
    )
    scm = LinearGaussianSCM.from_spec(spec)

    factual = {"X": 1.0, "Y": 3.0}  # implies Uy = 1
    cf = scm.counterfactual(factual, interventions={"X": 2.0})
    assert cf["X"] == 2.0
    assert cf["Y"] == 5.0


def test_linear_gaussian_scm_abduction_action_prediction_matches_manual() -> None:
    rng = np.random.default_rng(0)
    # Chain: A -> B -> C
    beta_ab = 0.7
    beta_bc = -1.3
    spec = SCMSpec(
        scm_type="linear_gaussian",
        equations=[
            StructuralEquationSpec(variable="A", parents=[], coefficients={}, intercept=0.2),
            StructuralEquationSpec(variable="B", parents=["A"], coefficients={"A": beta_ab}, intercept=-0.1),
            StructuralEquationSpec(variable="C", parents=["B"], coefficients={"B": beta_bc}, intercept=0.0),
        ],
    )
    scm = LinearGaussianSCM.from_spec(spec)

    # Sample one factual realization
    uA, uB, uC = rng.normal(0, 1, size=3)
    A = 0.2 + uA
    B = -0.1 + beta_ab * A + uB
    C = 0.0 + beta_bc * B + uC
    factual = {"A": float(A), "B": float(B), "C": float(C)}

    # Counterfactual intervention on A
    A_do = float(A + 1.0)
    cf = scm.counterfactual(factual, interventions={"A": A_do})

    # Manual AAP with same u's
    B_do = -0.1 + beta_ab * A_do + uB
    C_do = 0.0 + beta_bc * B_do + uC

    assert np.isclose(cf["B"], B_do)
    assert np.isclose(cf["C"], C_do)

