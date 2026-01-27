"""
Dynamic tool discovery utilities.

Provides functionality for:
- Tool registry and scanning
- Tool metadata extraction
- Tool schema generation
"""

import inspect
from typing import Any, Callable, Dict, List, Optional
from loguru import logger

try:
    from swarms.tools.tool import convert_function_to_openai_function_schema
    TOOL_CONVERSION_AVAILABLE = True
except ImportError:
    TOOL_CONVERSION_AVAILABLE = False
    logger.debug("Tool conversion utilities not available")


class ToolRegistry:
    """Registry for managing and discovering tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        logger.debug("Initialized ToolRegistry")
    
    def register_tool(
        self,
        tool: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a tool in the registry.
        
        Args:
            tool: Callable tool function
            name: Optional tool name (uses function name if None)
            description: Optional tool description
            metadata: Optional additional metadata
            
        Returns:
            Registered tool name
        """
        tool_name = name or tool.__name__
        
        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' already registered, overwriting")
        
        self._tools[tool_name] = tool
        
        # Extract metadata
        tool_metadata = {
            "name": tool_name,
            "description": description or tool.__doc__ or "No description",
            "function": tool,
            "signature": str(inspect.signature(tool)),
        }
        
        if metadata:
            tool_metadata.update(metadata)
        
        self._tool_metadata[tool_name] = tool_metadata
        
        logger.debug(f"Registered tool: {tool_name}")
        return tool_name
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool from the registry.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            if name in self._tool_metadata:
                del self._tool_metadata[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool function or None if not found
        """
        return self._tools.get(name)
    
    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool metadata by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata dictionary or None if not found
        """
        return self._tool_metadata.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all registered tools.
        
        Returns:
            Dictionary mapping tool names to tool functions
        """
        return self._tools.copy()
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all tools.
        
        Returns:
            Dictionary mapping tool names to metadata
        """
        return self._tool_metadata.copy()


# Global tool registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry instance.
    
    Returns:
        Global ToolRegistry instance
    """
    return _global_registry


def discover_tools_in_module(module: Any) -> List[Callable]:
    """Discover all callable tools in a module.
    
    Args:
        module: Module to scan for tools
        
    Returns:
        List of discovered tool functions
    """
    tools = []
    
    try:
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and not name.startswith("_"):
                # Check if it looks like a tool (has docstring, takes parameters, etc.)
                if inspect.isfunction(obj) or inspect.ismethod(obj):
                    tools.append(obj)
    except Exception as e:
        logger.error(f"Error discovering tools in module: {e}")
    
    return tools


def discover_tools_in_object(obj: Any) -> List[Callable]:
    """Discover all callable tools in an object.
    
    Args:
        obj: Object to scan for tools
        
    Returns:
        List of discovered tool methods
    """
    tools = []
    
    try:
        for name in dir(obj):
            attr = getattr(obj, name)
            if callable(attr) and not name.startswith("_"):
                if inspect.ismethod(attr) or (inspect.isfunction(attr) and hasattr(obj, name)):
                    tools.append(attr)
    except Exception as e:
        logger.error(f"Error discovering tools in object: {e}")
    
    return tools


def extract_tool_schema(tool: Callable) -> Optional[Dict[str, Any]]:
    """Extract OpenAI function schema from a tool.
    
    Args:
        tool: Tool function to extract schema from
        
    Returns:
        OpenAI function schema dictionary or None if extraction fails
    """
    if not TOOL_CONVERSION_AVAILABLE:
        # Fallback: create basic schema from function signature
        try:
            sig = inspect.signature(tool)
            params = {}
            for param_name, param in sig.parameters.items():
                param_type = "string"
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == float:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                
                params[param_name] = {
                    "type": param_type,
                    "description": param_name,
                }
            
            return {
                "name": tool.__name__,
                "description": tool.__doc__ or "No description",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": [
                        name for name, param in sig.parameters.items()
                        if param.default == inspect.Parameter.empty
                    ],
                },
            }
        except Exception as e:
            logger.error(f"Error extracting tool schema: {e}")
            return None
    
    try:
        schema = convert_function_to_openai_function_schema(tool)
        return schema
    except Exception as e:
        logger.error(f"Error converting tool to schema: {e}")
        return None


def generate_tool_schemas(tools: List[Callable]) -> List[Dict[str, Any]]:
    """Generate OpenAI function schemas for a list of tools.
    
    Args:
        tools: List of tool functions
        
    Returns:
        List of OpenAI function schema dictionaries
    """
    schemas = []
    
    for tool in tools:
        schema = extract_tool_schema(tool)
        if schema:
            schemas.append(schema)
    
    return schemas


def discover_and_register_tools(
    source: Any,
    registry: Optional[ToolRegistry] = None,
) -> List[str]:
    """Discover tools from a source and register them.
    
    Args:
        source: Module, object, or list of tools to discover from
        registry: Optional tool registry (uses global if None)
        
    Returns:
        List of registered tool names
    """
    if registry is None:
        registry = get_global_registry()
    
    tools = []
    
    # Handle different source types
    if inspect.ismodule(source):
        tools = discover_tools_in_module(source)
    elif isinstance(source, list):
        tools = [t for t in source if callable(t)]
    elif hasattr(source, "__dict__"):
        tools = discover_tools_in_object(source)
    elif callable(source):
        tools = [source]
    
    registered = []
    for tool in tools:
        try:
            name = registry.register_tool(tool)
            registered.append(name)
        except Exception as e:
            logger.error(f"Error registering tool {tool}: {e}")
    
    logger.info(f"Registered {len(registered)} tool(s)")
    return registered
