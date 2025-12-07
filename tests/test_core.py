import os
import importlib.util


def load_crca_module():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    # CRCA.py lives inside the ceca_lite package directory
    # In this repository layout the implementation file is `CRCA.py` under CR-CA
    target = os.path.join(repo_root, "CR-CA", "CRCA.py")
    spec = importlib.util.spec_from_file_location("crca_module", target)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_crca_agent_basic_flow():
    mod = load_crca_module()
    CRCAAgent = getattr(mod, "CRCAAgent")

    agent = CRCAAgent(variables=["x", "y"], causal_edges=[("x", "y")])

    # Set sensible standardization stats to avoid std=0 edge cases
    agent.set_standardization_stats("x", mean=1.0, std=2.0)
    agent.set_standardization_stats("y", mean=10.0, std=5.0)

    factual = {"x": 2.0, "y": 12.0}
    interventions = {}

    # Basic prediction (no cache)
    agent.enable_cache(False)
    out = agent._predict_outcomes_cached(factual, interventions)
    assert isinstance(out, dict)
    assert "x" in out and "y" in out

    # Enable cache and check caching pathway
    agent.enable_cache(True)
    agent.clear_cache()
    out1 = agent._predict_outcomes_cached(factual, interventions)
    out2 = agent._predict_outcomes_cached(factual, interventions)
    assert out1 == out2

    # Graph utilities
    edges = agent.get_edges()
    assert ("x", "y") in edges
    assert agent.is_dag()


