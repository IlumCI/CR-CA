"""
Example usage of GeneralAgent.

Demonstrates:
- Basic conversation
- Tool usage
- Agent routing (auto-discovered)
- Code execution
- Multimodal inputs
- Async operations
- Batch processing
- Custom personality configuration
"""

import asyncio
from loguru import logger

try:
    from branches.general_agent import GeneralAgent, get_personality, create_custom_personality
    GENERAL_AGENT_AVAILABLE = True
except ImportError:
    logger.error("GeneralAgent not available. Please ensure all dependencies are installed.")
    GENERAL_AGENT_AVAILABLE = False


def example_basic_conversation():
    """Example: Basic conversation with GeneralAgent."""
    print("\n=== Example 1: Basic Conversation ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    # Create agent with default settings
    agent = GeneralAgent(
        agent_name="example-agent",
        model_name="gpt-4o-mini",
        personality="neutral",
    )
    
    # Simple conversation
    response = agent.run("Hello! Can you explain what you can do?")
    print(f"Agent: {response}\n")
    
    # Continue conversation
    response = agent.run("What's the weather like?")
    print(f"Agent: {response}\n")


def example_custom_personality():
    """Example: Using custom personality."""
    print("\n=== Example 2: Custom Personality ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    # Create agent with friendly personality
    agent = GeneralAgent(
        agent_name="friendly-agent",
        personality="friendly",
    )
    
    response = agent.run("Tell me a joke!")
    print(f"Friendly Agent: {response}\n")
    
    # Create agent with technical personality
    agent = GeneralAgent(
        agent_name="technical-agent",
        personality="technical",
    )
    
    response = agent.run("Explain how neural networks work")
    print(f"Technical Agent: {response}\n")


def example_async_operations():
    """Example: Async operations."""
    print("\n=== Example 3: Async Operations ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    async def run_async_example():
        agent = GeneralAgent(
            agent_name="async-agent",
        )
        
        # Run async
        response = await agent.run_async("What is 2+2?")
        print(f"Async Response: {response}\n")
        
        # Run multiple async tasks
        tasks = [
            "What is Python?",
            "What is machine learning?",
            "What is AI?",
        ]
        
        responses = await asyncio.gather(*[agent.run_async(task) for task in tasks])
        for task, response in zip(tasks, responses):
            print(f"Task: {task}")
            print(f"Response: {response}\n")
    
    asyncio.run(run_async_example())


def example_batch_processing():
    """Example: Batch processing."""
    print("\n=== Example 4: Batch Processing ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    agent = GeneralAgent(
        agent_name="batch-agent",
    )
    
    # Process batch of tasks
    tasks = [
        "Explain quantum computing",
        "What is blockchain?",
        "Describe machine learning",
    ]
    
    results, stats = agent.run_batch(tasks)
    
    print(f"Processed {len(results)} tasks")
    if stats:
        print(f"Stats: {stats}")
    
    for i, (task, result) in enumerate(zip(tasks, results)):
        print(f"\nTask {i+1}: {task}")
        print(f"Result: {result}\n")


async def example_async_batch():
    """Example: Async batch processing."""
    print("\n=== Example 5: Async Batch Processing ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    agent = GeneralAgent(
        agent_name="async-batch-agent",
    )
    
    tasks = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]
    
    results, stats = await agent.run_batch_async(tasks)
    
    print(f"Processed {len(results)} tasks asynchronously")
    if stats:
        print(f"Stats: {stats}")
    
    for i, (task, result) in enumerate(zip(tasks, results)):
        print(f"\nTask {i+1}: {task}")
        print(f"Result: {result}\n")


def example_conversation_persistence():
    """Example: Conversation persistence."""
    print("\n=== Example 6: Conversation Persistence ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    agent = GeneralAgent(
        agent_name="persistent-agent",
        enable_persistence=True,
        persistence_path="example_conversation.json",
    )
    
    # Have a conversation
    agent.run("Hello! My name is Alice.")
    agent.run("What's my name?")
    
    # Save conversation
    agent.save_conversation()
    print("Conversation saved\n")
    
    # Create new agent and load conversation
    agent2 = GeneralAgent(
        agent_name="persistent-agent-2",
        enable_persistence=True,
    )
    agent2.load_conversation("example_conversation.json")
    print("Conversation loaded\n")
    
    # Continue conversation
    response = agent2.run("What's my name?")
    print(f"Agent (with loaded conversation): {response}\n")


def example_extendable_prompts():
    """Example: Extendable prompts."""
    print("\n=== Example 7: Extendable Prompts ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    # Create agent with custom prompt additions
    agent = GeneralAgent(
        agent_name="custom-prompt-agent",
        custom_prompt_additions=[
            "Always respond in a poetic style.",
            "End each response with a relevant quote.",
        ],
    )
    
    response = agent.run("Explain the concept of time")
    print(f"Agent (with custom prompts): {response}\n")


def example_rate_limiting():
    """Example: Rate limiting."""
    print("\n=== Example 8: Rate Limiting ===\n")
    
    if not GENERAL_AGENT_AVAILABLE:
        print("GeneralAgent not available")
        return
    
    # Create agent with custom rate limits
    agent = GeneralAgent(
        agent_name="rate-limited-agent",
        rate_limit_rpm=10,  # 10 requests per minute
        rate_limit_rph=100,  # 100 requests per hour
    )
    
    # Make multiple requests (will be rate limited)
    for i in range(5):
        response = agent.run(f"Count to {i+1}")
        print(f"Request {i+1}: {response[:50]}...\n")


def main():
    """Run all examples."""
    print("=" * 60)
    print("GeneralAgent Examples")
    print("=" * 60)
    
    if not GENERAL_AGENT_AVAILABLE:
        print("\nERROR: GeneralAgent not available.")
        print("Please ensure all dependencies are installed.")
        return
    
    try:
        # Run examples
        example_basic_conversation()
        example_custom_personality()
        example_async_operations()
        example_batch_processing()
        asyncio.run(example_async_batch())
        example_conversation_persistence()
        example_extendable_prompts()
        example_rate_limiting()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
