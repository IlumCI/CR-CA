"""
General Agent Module

A general-purpose conversational agent (GPT-style) that can handle diverse tasks,
use tools, and optionally access other specialized CRCA agents via AOP/router integration.
"""

from typing import Optional

__version__ = "0.1.0"

# Lazy imports
try:
    from branches.general_agent.general_agent import GeneralAgent, GeneralAgentConfig
    GENERAL_AGENT_AVAILABLE = True
except ImportError as e:
    GeneralAgent = None
    GeneralAgentConfig = None
    GENERAL_AGENT_AVAILABLE = False
    _import_error = e

try:
    from branches.general_agent.personality import (
        Personality,
        get_personality,
        create_custom_personality,
        list_personalities,
        PERSONALITIES,
    )
    PERSONALITY_AVAILABLE = True
except ImportError:
    Personality = None
    get_personality = None
    create_custom_personality = None
    list_personalities = None
    PERSONALITIES = {}
    PERSONALITY_AVAILABLE = False

__all__ = [
    "GeneralAgent",
    "GeneralAgentConfig",
    "Personality",
    "get_personality",
    "create_custom_personality",
    "list_personalities",
    "PERSONALITIES",
    "get_general_agent",
    "create_agent",
    "GENERAL_AGENT_AVAILABLE",
    "PERSONALITY_AVAILABLE",
    "__version__",
]


def get_general_agent(**kwargs) -> Optional["GeneralAgent"]:
    """
    Get GeneralAgent instance (simple factory function).
    
    Simplest usage:
        agent = get_general_agent()
        agent = get_general_agent(model_name="gpt-4o")
        agent = get_general_agent(personality="friendly")
    
    Args:
        **kwargs: Arguments to pass to GeneralAgent constructor
        
    Returns:
        GeneralAgent instance or None if not available
    """
    if not GENERAL_AGENT_AVAILABLE:
        if '_import_error' in globals():
            raise ImportError(f"GeneralAgent not available: {_import_error}")
        return None
    
    try:
        return GeneralAgent(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create GeneralAgent: {e}") from e


def create_agent(
    model_name: Optional[str] = None,
    personality: Optional[str] = None,
    **kwargs
) -> Optional["GeneralAgent"]:
    """
    Ultra-simple agent creation function.
    
    Usage:
        agent = create_agent()
        agent = create_agent("gpt-4o")
        agent = create_agent("gpt-4o", "friendly")
    
    Args:
        model_name: LLM model name (optional)
        personality: Personality name (optional)
        **kwargs: Additional parameters
        
    Returns:
        GeneralAgent instance
    """
    return get_general_agent(model_name=model_name, personality=personality, **kwargs)
