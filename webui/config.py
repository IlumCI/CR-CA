"""
Configuration registry for agents and tools.

This module provides an extensible registry system for adding new agents
and tools to the web UI without modifying core application code.
"""

from typing import Dict, Any, Optional, Type
from loguru import logger

# Agent registry - add new agents here
AGENTS: Dict[str, Dict[str, Any]] = {}

# Tool registry - add new tools here
TOOLS: Dict[str, Dict[str, Any]] = {}

# Lazy imports to handle optional dependencies
try:
    from branches.general_agent import GeneralAgent
    GENERAL_AGENT_AVAILABLE = True
except ImportError as e:
    GeneralAgent = None
    GENERAL_AGENT_AVAILABLE = False
    logger.warning(f"GeneralAgent not available: {e}")

try:
    from architecture.hybrid.hybrid_agent import HybridAgent
    HYBRID_AGENT_AVAILABLE = True
except ImportError as e:
    HybridAgent = None
    HYBRID_AGENT_AVAILABLE = False
    logger.warning(f"HybridAgent not available: {e}")


def _register_agents() -> None:
    """Register available agents in the AGENTS dictionary."""
    global AGENTS
    
    # Register GeneralAgent if available
    if GENERAL_AGENT_AVAILABLE and GeneralAgent is not None:
        AGENTS["general"] = {
            "name": "General Agent",
            "class": GeneralAgent,
            "description": "LLM-powered general-purpose agent with causal reasoning capabilities",
            "requires_llm": True,
            "default": True,  # This is the default agent
        }
        logger.info("Registered GeneralAgent")
    
    # Register HybridAgent if available
    if HYBRID_AGENT_AVAILABLE and HybridAgent is not None:
        AGENTS["hybrid"] = {
            "name": "Hybrid Agent",
            "class": HybridAgent,
            "description": "Symbolic-statistical agent (no LLM required)",
            "requires_llm": False,
            "default": False,
        }
        logger.info("Registered HybridAgent")


def get_default_agent_key() -> Optional[str]:
    """
    Get the key of the default agent.
    
    Returns:
        Agent key string or None if no agents available
    """
    for key, agent_info in AGENTS.items():
        if agent_info.get("default", False):
            return key
    
    # If no default marked, return first available agent
    if AGENTS:
        return list(AGENTS.keys())[0]
    
    return None


def get_agent_info(key: str) -> Optional[Dict[str, Any]]:
    """
    Get agent information by key.
    
    Args:
        key: Agent registry key
        
    Returns:
        Agent info dictionary or None if not found
    """
    return AGENTS.get(key)


def list_agents() -> Dict[str, Dict[str, Any]]:
    """
    List all registered agents.
    
    Returns:
        Dictionary of agent keys to agent info
    """
    return AGENTS.copy()


# Initialize agent registry on module import
_register_agents()
