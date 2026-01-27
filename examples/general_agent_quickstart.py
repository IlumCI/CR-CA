"""
Quick Start Guide: Spawning GeneralAgent for Immediate Use

This file demonstrates the simplest ways to create and use the GeneralAgent.
"""

from branches.general_agent import GeneralAgent

# ============================================================================
# METHOD 1: Simplest - Default Settings
# ============================================================================

def quick_start_minimal():
    """Minimal setup - just create and use."""
    agent = GeneralAgent()
    response = agent.run("Hello! What can you do?")
    print(response)
    return agent


# ============================================================================
# METHOD 2: With Custom Model
# ============================================================================

def quick_start_custom_model():
    """Create with custom model."""
    agent = GeneralAgent(
        model_name="gpt-4o",  # or "gpt-4o-mini", "claude-3-5-sonnet", etc.
    )
    response = agent.run("Analyze the causal factors affecting product pricing.")
    print(response)
    return agent


# ============================================================================
# METHOD 3: With Personality
# ============================================================================

def quick_start_with_personality():
    """Create with personality."""
    agent = GeneralAgent(
        personality="friendly",  # or "neutral", "technical"
    )
    response = agent.run("Explain quantum computing in simple terms.")
    print(response)
    return agent


# ============================================================================
# METHOD 4: Production-Ready Setup
# ============================================================================

def quick_start_production():
    """Production-ready setup with all features enabled."""
    agent = GeneralAgent(
        agent_name="production-agent",
        model_name="gpt-4o-mini",
        temperature=0.4,
        personality="neutral",
        enable_agent_routing="auto",
        enable_code_execution=True,
        enable_multimodal=True,
        enable_streaming=True,
        enable_persistence=True,
        enable_causal_reasoning=True,
        enable_file_operations=True,
        persistence_path="./conversations/",
        rate_limit_rpm=60,
        rate_limit_rph=1000,
    )
    response = agent.run("Help me understand the causal relationships in my business model.")
    print(response)
    return agent


# ============================================================================
# METHOD 5: Using Helper Function
# ============================================================================

def quick_start_helper():
    """Using the get_general_agent helper function."""
    from branches.general_agent import get_general_agent
    
    agent = get_general_agent(
        model_name="gpt-4o-mini",
        personality="neutral",
    )
    
    if agent:
        response = agent.run("What is causal reasoning?")
        print(response)
        return agent
    else:
        print("Failed to create agent")


# ============================================================================
# METHOD 6: Async Usage
# ============================================================================

async def quick_start_async():
    """Async usage for concurrent operations."""
    agent = GeneralAgent()
    
    # Run async
    response = await agent.run_async("Analyze this task using causal reasoning.")
    print(response)
    
    # Batch processing
    tasks = [
        "What factors affect sales?",
        "Explain supply and demand relationships.",
        "How does marketing impact revenue?",
    ]
    results = await agent.run_batch_async(tasks)
    for task, result in zip(tasks, results):
        print(f"\nTask: {task}\nResult: {result}\n")
    
    return agent


# ============================================================================
# METHOD 7: With Causal Reasoning Focus
# ============================================================================

def quick_start_causal_focus():
    """Create agent focused on causal reasoning tasks."""
    agent = GeneralAgent(
        model_name="gpt-4o",
        personality="technical",
        enable_causal_reasoning=True,
        enable_agent_routing="auto",  # Will route to CRCAAgent if needed
    )
    
    # The agent will automatically use causal reasoning tools
    response = agent.run(
        "I want to understand what factors influence customer satisfaction. "
        "Extract the causal variables and relationships."
    )
    print(response)
    return agent


# ============================================================================
# METHOD 8: Minimal with Environment Variables
# ============================================================================

def quick_start_env_vars():
    """Create agent using environment variables for API keys."""
    import os
    
    # Set your API keys in environment (or use .env file)
    # os.environ["OPENAI_API_KEY"] = "your-key-here"
    # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
    
    agent = GeneralAgent(
        model_name="gpt-4o-mini",
    )
    response = agent.run("Hello!")
    print(response)
    return agent


# ============================================================================
# MAIN: Run Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GeneralAgent Quick Start Examples")
    print("=" * 70)
    
    # Uncomment the example you want to run:
    
    # Example 1: Minimal
    # quick_start_minimal()
    
    # Example 2: Custom Model
    # quick_start_custom_model()
    
    # Example 3: With Personality
    # quick_start_with_personality()
    
    # Example 4: Production Setup
    # quick_start_production()
    
    # Example 5: Helper Function
    # quick_start_helper()
    
    # Example 6: Async (requires asyncio.run())
    # import asyncio
    # asyncio.run(quick_start_async())
    
    # Example 7: Causal Reasoning Focus
    # quick_start_causal_focus()
    
    # Example 8: Environment Variables
    # quick_start_env_vars()
    
    print("\n" + "=" * 70)
    print("Choose an example and uncomment it to run!")
    print("=" * 70)
