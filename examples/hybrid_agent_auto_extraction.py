"""
Example demonstrating automatic variable and edge extraction from natural language.

The HybridAgent can automatically extract causal variables and relationships
from natural language text without any LLM dependency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from architecture.hybrid.hybrid_agent import HybridAgent


def main():
    """Demonstrate auto-extraction capabilities."""
    
    # Create agent
    agent = HybridAgent()
    
    print("=" * 70)
    print("Hybrid Agent - Automatic Variable and Edge Extraction")
    print("=" * 70)
    print()
    
    # Example 1: Simple causal relationship
    print("Example 1: Simple causal relationship")
    print("-" * 70)
    task1 = "product price depends on demand and supply"
    result1 = agent.extract_causal_variables(task1)
    print(f"Input: {task1}")
    print(f"Variables extracted: {result1['variables']}")
    print(f"Edges extracted: {result1['edges']}")
    print(f"Relationships found: {result1['metadata']['total_relationships']}")
    print()
    
    # Example 2: Multiple relationships
    print("Example 2: Multiple causal relationships")
    print("-" * 70)
    task2 = "customer satisfaction is influenced by product quality, and product quality depends on manufacturing process"
    result2 = agent.extract_causal_variables(task2)
    print(f"Input: {task2}")
    print(f"Variables extracted: {result2['variables']}")
    print(f"Edges extracted: {result2['edges']}")
    print(f"Relationships found: {result2['metadata']['total_relationships']}")
    print()
    
    # Example 3: Complex sentence
    print("Example 3: Complex sentence with multiple relationships")
    print("-" * 70)
    task3 = "sales revenue increases when marketing spend rises, and marketing spend is controlled by budget allocation"
    result3 = agent.extract_causal_variables(task3)
    print(f"Input: {task3}")
    print(f"Variables extracted: {result3['variables']}")
    print(f"Edges extracted: {result3['edges']}")
    print(f"Relationships found: {result3['metadata']['total_relationships']}")
    print()
    
    # Example 4: Full analysis
    print("Example 4: Full causal analysis")
    print("-" * 70)
    task4 = "employee productivity affects company revenue, and employee productivity is determined by training quality and work environment"
    result4 = agent.run(task4)
    print(f"Input: {task4}")
    print(f"\nAnalysis Result:\n{result4}")
    print()
    
    # Example 5: Arrow notation
    print("Example 5: Arrow notation")
    print("-" * 70)
    task5 = "temperature -> pressure -> volume"
    result5 = agent.extract_causal_variables(task5)
    print(f"Input: {task5}")
    print(f"Variables extracted: {result5['variables']}")
    print(f"Edges extracted: {result5['edges']}")
    print()
    
    print("=" * 70)
    print("Auto-extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
