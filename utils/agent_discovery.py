"""
Agent discovery utilities.

Provides functionality for:
- Auto-discovery of AOP instances
- Auto-discovery of router instances
- Agent listing and metadata
- Route-first routing helpers
"""

import inspect
import sys
from typing import Any, Dict, List, Optional, Union
from loguru import logger

# Try to import AOP and Router
try:
    from utils.aop import AOP
    AOP_AVAILABLE = True
except ImportError:
    AOP = None
    AOP_AVAILABLE = False
    logger.debug("AOP not available for agent discovery")

try:
    from utils.router import SwarmRouter
    ROUTER_AVAILABLE = True
except ImportError:
    SwarmRouter = None
    ROUTER_AVAILABLE = False
    logger.debug("Router not available for agent discovery")


def discover_aop_instances() -> List[Any]:
    """Auto-discover AOP instances in the current runtime.
    
    Searches for AOP instances in:
    - Global namespace
    - Module-level variables
    - Active objects
    
    Returns:
        List of discovered AOP instances
    """
    if not AOP_AVAILABLE:
        return []
    
    instances = []
    
    # Search in global namespace
    try:
        frame = sys._getframe(1)
        while frame:
            for name, obj in frame.f_globals.items():
                if isinstance(obj, AOP):
                    instances.append(obj)
            frame = frame.f_back
    except Exception as e:
        logger.debug(f"Error discovering AOP instances from frames: {e}")
    
    # Search in module-level variables (limited approach)
    # This is a best-effort discovery
    try:
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, AOP):
                if obj not in instances:
                    instances.append(obj)
    except Exception as e:
        logger.debug(f"Error discovering AOP instances from GC: {e}")
    
    logger.debug(f"Discovered {len(instances)} AOP instance(s)")
    return instances


def discover_router_instances() -> List[Any]:
    """Auto-discover router instances in the current runtime.
    
    Searches for router instances in:
    - Global namespace
    - Module-level variables
    - Active objects
    
    Returns:
        List of discovered router instances
    """
    if not ROUTER_AVAILABLE:
        return []
    
    instances = []
    
    # Search in global namespace
    try:
        frame = sys._getframe(1)
        while frame:
            for name, obj in frame.f_globals.items():
                if isinstance(obj, SwarmRouter):
                    instances.append(obj)
            frame = frame.f_back
    except Exception as e:
        logger.debug(f"Error discovering router instances from frames: {e}")
    
    # Search in module-level variables
    try:
        import gc
        for obj in gc.get_objects():
            if isinstance(obj, SwarmRouter):
                if obj not in instances:
                    instances.append(obj)
    except Exception as e:
        logger.debug(f"Error discovering router instances from GC: {e}")
    
    logger.debug(f"Discovered {len(instances)} router instance(s)")
    return instances


def get_agents_from_aop(aop_instance: Any) -> Dict[str, Any]:
    """Get list of agents from an AOP instance.
    
    Args:
        aop_instance: AOP instance to query
        
    Returns:
        Dictionary mapping agent names to agent metadata
    """
    if not AOP_AVAILABLE or not isinstance(aop_instance, AOP):
        return {}
    
    agents = {}
    
    try:
        if hasattr(aop_instance, 'agents'):
            for agent_name, agent in aop_instance.agents.items():
                agents[agent_name] = {
                    "name": agent_name,
                    "type": type(agent).__name__,
                    "description": getattr(agent, 'agent_description', 'No description'),
                    "available": True,
                }
        
        if hasattr(aop_instance, 'tool_configs'):
            for tool_name, config in aop_instance.tool_configs.items():
                if tool_name not in agents:
                    agents[tool_name] = {
                        "name": tool_name,
                        "type": "tool",
                        "description": getattr(config, 'tool_description', 'No description'),
                        "available": True,
                    }
    except Exception as e:
        logger.error(f"Error getting agents from AOP instance: {e}")
    
    return agents


def get_agents_from_router(router_instance: Any) -> Dict[str, Any]:
    """Get list of agents from a router instance.
    
    Args:
        router_instance: Router instance to query
        
    Returns:
        Dictionary mapping agent names to agent metadata
    """
    if not ROUTER_AVAILABLE or not isinstance(router_instance, SwarmRouter):
        return {}
    
    agents = {}
    
    try:
        if hasattr(router_instance, 'agents'):
            for i, agent in enumerate(router_instance.agents):
                agent_name = getattr(agent, 'agent_name', f"agent_{i}")
                agents[agent_name] = {
                    "name": agent_name,
                    "type": type(agent).__name__,
                    "description": getattr(agent, 'agent_description', 'No description'),
                    "available": True,
                }
    except Exception as e:
        logger.error(f"Error getting agents from router instance: {e}")
    
    return agents


def discover_all_agents(
    aop_instances: Optional[List[Any]] = None,
    router_instances: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Discover all available agents from AOP and router instances.
    
    Args:
        aop_instances: Optional list of AOP instances (auto-discovered if None)
        router_instances: Optional list of router instances (auto-discovered if None)
        
    Returns:
        Dictionary mapping agent names to agent metadata
    """
    all_agents = {}
    
    # Discover AOP instances if not provided
    if aop_instances is None:
        aop_instances = discover_aop_instances()
    
    # Discover router instances if not provided
    if router_instances is None:
        router_instances = discover_router_instances()
    
    # Get agents from AOP instances
    for aop_instance in aop_instances:
        agents = get_agents_from_aop(aop_instance)
        all_agents.update(agents)
    
    # Get agents from router instances
    for router_instance in router_instances:
        agents = get_agents_from_router(router_instance)
        all_agents.update(agents)
    
    logger.info(f"Discovered {len(all_agents)} total agent(s)")
    return all_agents


def find_best_agent_for_task(
    task: str,
    available_agents: Dict[str, Any],
    aop_instances: Optional[List[Any]] = None,
    router_instances: Optional[List[Any]] = None,
) -> Optional[tuple[str, Any, str]]:
    """Find the best agent for a given task (route-first strategy).
    
    Args:
        task: Task description
        available_agents: Dictionary of available agents
        aop_instances: Optional list of AOP instances
        router_instances: Optional list of router instances
        
    Returns:
        Tuple of (agent_name, agent_instance, source) or None if no suitable agent found
        source is either 'aop' or 'router'
    """
    if not available_agents:
        return None
    
    # Simple keyword-based matching (can be enhanced with LLM-based routing)
    task_lower = task.lower()
    
    # Check for specialized agents first (route-first strategy)
    for agent_name, agent_info in available_agents.items():
        description = agent_info.get("description", "").lower()
        agent_type = agent_info.get("type", "").lower()
        
        # Simple matching logic
        if any(keyword in description or keyword in agent_type for keyword in task_lower.split()):
            # Try to get the actual agent instance
            if aop_instances:
                for aop in aop_instances:
                    if hasattr(aop, 'agents') and agent_name in aop.agents:
                        return (agent_name, aop.agents[agent_name], "aop")
            
            if router_instances:
                for router in router_instances:
                    if hasattr(router, 'agents'):
                        for agent in router.agents:
                            if getattr(agent, 'agent_name', None) == agent_name:
                                return (agent_name, agent, "router")
    
    # If no match found, return None (fallback to direct handling)
    return None


def route_to_agent(
    agent_name: str,
    task: str,
    aop_instances: Optional[List[Any]] = None,
    router_instances: Optional[List[Any]] = None,
) -> Optional[Any]:
    """Route a task to a specific agent.
    
    Args:
        agent_name: Name of the agent to route to
        task: Task to execute
        aop_instances: Optional list of AOP instances
        router_instances: Optional list of router instances
        
    Returns:
        Agent response or None if agent not found
    """
    # Discover instances if not provided
    if aop_instances is None:
        aop_instances = discover_aop_instances()
    
    if router_instances is None:
        router_instances = discover_router_instances()
    
    # Try AOP instances first
    for aop in aop_instances:
        if hasattr(aop, 'agents') and agent_name in aop.agents:
            agent = aop.agents[agent_name]
            if hasattr(agent, 'run'):
                return agent.run(task)
    
    # Try router instances
    for router in router_instances:
        if hasattr(router, 'agents'):
            for agent in router.agents:
                if getattr(agent, 'agent_name', None) == agent_name:
                    if hasattr(agent, 'run'):
                        return agent.run(task)
    
    logger.warning(f"Agent '{agent_name}' not found for routing")
    return None
