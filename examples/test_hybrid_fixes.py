"""
Test script to verify hybrid agent fixes for variable cleaning and state usage.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.hybrid.hybrid_agent import HybridAgent
import json

def test_extraction():
    """Test variable extraction and cleaning."""
    agent = HybridAgent()
    task = "If product price is 20000 & demand is 61% buy, 39% sell, what is the expected price of the product in 7 days?"
    
    print("=" * 70)
    print("Testing Variable Extraction and Cleaning")
    print("=" * 70)
    
    result = agent.extract_causal_variables(task)
    
    print(f"\nVariables extracted: {result['variables']}")
    print(f"Edges extracted: {result['edges']}")
    print(f"Values extracted: {result['metadata']['variables_with_values']}")
    
    # Check for invalid variables
    invalid_vars = ['if', 'sell', 'buy', 'days', 'day']
    found_invalid = [v for v in result['variables'] if any(inv in v.lower() for inv in invalid_vars)]
    if found_invalid:
        print(f"\nWARNING: Found potentially invalid variables: {found_invalid}")
    else:
        print("\n✓ No invalid variables found")
    
    return result

def test_full_analysis():
    """Test full analysis with factual state."""
    agent = HybridAgent()
    task = "If product price is 20000 & demand is 61% buy, 39% sell, what is the expected price of the product in 7 days?"
    
    print("\n" + "=" * 70)
    print("Testing Full Analysis with Factual State")
    print("=" * 70)
    
    result_dict = agent.orchestrator.reason_hybrid(task)
    
    print(f"\nVariables in analysis: {result_dict['analysis']['variables']}")
    print(f"Number of relationships: {len(result_dict['analysis']['relationships'])}")
    print(f"Factual state: {result_dict['analysis'].get('factual_state', {})}")
    
    # Check if factual state has non-zero values
    factual_state = result_dict['analysis'].get('factual_state', {})
    non_zero = {k: v for k, v in factual_state.items() if v != 0.0}
    if non_zero:
        print(f"\n✓ Factual state has extracted values: {non_zero}")
    else:
        print("\nWARNING: Factual state has no extracted values (all zeros)")
    
    # Check counterfactuals
    if result_dict.get('counterfactuals'):
        cf = result_dict['counterfactuals'][0]
        outcomes = cf.get('expected_outcomes', {})
        non_zero_outcomes = {k: v for k, v in outcomes.items() if v != 0.0}
        if non_zero_outcomes:
            print(f"✓ Counterfactuals have non-zero predictions: {len(non_zero_outcomes)} variables")
        else:
            print("WARNING: Counterfactuals show all zeros")
    
    return result_dict

if __name__ == "__main__":
    test_extraction()
    test_full_analysis()
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)
