"""
General-purpose conversational agent.

A jack-of-all-trades agent capable of handling diverse tasks through conversation,
tool usage, agent routing, code execution, and multimodal processing.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger

from templates.base_specialized_agent import BaseSpecializedAgent

# Import utilities
try:
    from utils.rate_limiter import RateLimiter, RateLimitConfig
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    logger.debug("Rate limiter not available")

try:
    from utils.agent_discovery import (
        discover_all_agents,
        find_best_agent_for_task,
        route_to_agent,
        discover_aop_instances,
        discover_router_instances,
    )
    AGENT_DISCOVERY_AVAILABLE = True
except ImportError:
    AGENT_DISCOVERY_AVAILABLE = False
    logger.debug("Agent discovery not available")

try:
    # Tool discovery utilities available for future use
    from utils.tool_discovery import (
        get_global_registry,
        discover_and_register_tools,
        generate_tool_schemas,
    )
    TOOL_DISCOVERY_AVAILABLE = True
except ImportError:
    TOOL_DISCOVERY_AVAILABLE = False
    logger.debug("Tool discovery not available")

try:
    from utils.batch_processor import BatchProcessor
    BATCH_PROCESSOR_AVAILABLE = True
except ImportError:
    BATCH_PROCESSOR_AVAILABLE = False
    logger.debug("Batch processor not available")

try:
    from utils.conversation import Conversation
    CONVERSATION_AVAILABLE = True
except ImportError:
    CONVERSATION_AVAILABLE = False
    logger.debug("Conversation not available")

try:
    from utils.formatter import Formatter
    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False
    logger.debug("Formatter not available")

# Import agent-specific modules
try:
    from branches.general_agent.personality import get_personality, Personality
    from branches.general_agent.utils.prompt_builder import PromptBuilder
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False
    logger.debug("Personality system not available")

try:
    from prompts.general_agent import DEFAULT_GENERAL_AGENT_PROMPT
except ImportError:
    DEFAULT_GENERAL_AGENT_PROMPT = "You are a general-purpose AI assistant."

# Image annotation imports (optional - graceful fallback if not available)
try:
    from image_annotation.annotation_engine import ImageAnnotationEngine
    IMAGE_ANNOTATION_AVAILABLE = True
except ImportError:
    IMAGE_ANNOTATION_AVAILABLE = False
    logger.debug("Image annotation engine not available")
except Exception as e:
    IMAGE_ANNOTATION_AVAILABLE = False
    logger.warning(f"Image annotation engine import failed: {e}")

# Global singleton for image annotation engine (lazy-loaded)
_image_annotation_engine: Optional[Any] = None

# Try to import CRCAAgent for causal reasoning integration
try:
    from CRCA import CRCAAgent
    CRCA_AVAILABLE = True
except ImportError:
    CRCA_AVAILABLE = False
    CRCAAgent = None
    logger.debug("CRCAAgent not available for causal reasoning integration")

# Try to import file operations
try:
    from tools.file_operations import IntelligentFileManager, FileOperationsRegistry
    FILE_OPERATIONS_AVAILABLE = True
except ImportError:
    FILE_OPERATIONS_AVAILABLE = False
    IntelligentFileManager = None
    FileOperationsRegistry = None
    logger.debug("File operations not available")


@dataclass
class GeneralAgentConfig:
    """Configuration for GeneralAgent.
    
    Attributes:
        personality: Personality name or Personality instance
        enable_agent_routing: Enable agent routing (default: auto)
        enable_code_execution: Enable code interpreter (default: True)
        enable_multimodal: Enable multimodal support (default: True)
        enable_streaming: Enable streaming (default: True)
        enable_persistence: Enable conversation persistence (default: True)
        enable_causal_reasoning: Enable causal reasoning tools (default: True)
        enable_file_operations: Enable file operations tools (default: True)
        persistence_path: Path for conversation storage
        rate_limit_rpm: Rate limit requests per minute
        rate_limit_rph: Rate limit requests per hour
        custom_prompt_additions: Extendable prompt additions
    """
    personality: Union[str, Any, None] = "neutral"  # Any to handle Personality type if available
    enable_agent_routing: Union[bool, str] = "auto"
    enable_code_execution: bool = True
    enable_multimodal: bool = True
    enable_streaming: bool = True
    enable_persistence: bool = True
    enable_causal_reasoning: bool = True
    enable_file_operations: bool = True
    persistence_path: Optional[str] = None
    rate_limit_rpm: int = 60
    rate_limit_rph: int = 1000
    custom_prompt_additions: List[str] = field(default_factory=list)
    
    @classmethod
    def auto(cls, **overrides) -> "GeneralAgentConfig":
        """Create config with smart auto-detection and defaults.
        
        Args:
            **overrides: Any config values to override
            
        Returns:
            GeneralAgentConfig with smart defaults
        """
        import os
        from pathlib import Path
        
        # Auto-detect persistence path
        persistence_path = overrides.get("persistence_path")
        if persistence_path is None:
            default_path = Path.home() / ".crca" / "conversations"
            default_path.mkdir(parents=True, exist_ok=True)
            persistence_path = str(default_path)
        
        # Auto-detect model from env or use default
        # (Model selection handled in GeneralAgent.__init__)
        
        # Create config with smart defaults
        config = cls(
            persistence_path=persistence_path,
            **{k: v for k, v in overrides.items() if k != "persistence_path"}
        )
        
        return config


class GeneralAgent(BaseSpecializedAgent):
    """
    Pure hardened CR-CA Agent - a production-ready general-purpose agent
    that embodies the full power of the CR-CA (Causal Reasoning with Counterfactual Analysis) framework.
    
    This is NOT a generic general-purpose agent. It is a specialized CR-CA agent
    whose specialization IS being useful across diverse domains while maintaining
    the core CR-CA philosophy: causal reasoning, counterfactual analysis, and
    structured decision-making.
    
    Core CR-CA Identity:
    - Causal reasoning first: Understands cause-and-effect relationships
    - Counterfactual thinking: Explores "what-if" scenarios systematically
    - Structured analysis: Uses causal graphs and variable extraction
    - Evidence-based decisions: Grounds recommendations in causal analysis
    - Multi-domain applicability: Applies CR-CA principles across all domains
    
    Hardened Production Features:
    - Robust error handling with retry and fallback mechanisms
    - Rate limiting and resource management
    - Conversation persistence and state management
    - Comprehensive logging and monitoring
    - Graceful degradation when dependencies unavailable
    - Async/sync operations with proper resource cleanup
    - Batch processing for efficiency
    
    Full CR-CA Codebase Integration:
    - Causal reasoning tools (extract_causal_variables, generate_causal_analysis)
    - Meta-reasoning (scenario-level informativeness analysis, task-level strategic planning)
    - Image annotation (full ImageAnnotationEngine with geometric analysis)
    - File operations (read/write/list with intelligent management)
    - Agent discovery and routing (AOP/router integration for specialized agents)
    - Tool discovery (dynamic tool registry and schema generation)
    - Multi-step reasoning (always enabled for complex causal chains)
    - Code execution (for data analysis and causal modeling)
    - Multimodal processing (images, text, structured data)
    
    Routing Strategy:
    - Can route to specialized CR-CA agents (CRCAAgent, CRCA-SD, CRCA-CG, CRCA-Q)
    - Route-first approach: Check for specialized agents before direct handling
    - Falls back to direct CR-CA tool usage when appropriate
    - Maintains CR-CA methodology even when routing
    
    This agent represents the "useful" specialization - applying CR-CA's
    causal reasoning and counterfactual analysis capabilities to any domain
    or task, making it the go-to agent when you need CR-CA power without
    domain-specific constraints.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        personality: Optional[Union[str, Any]] = None,
        agent_name: Optional[str] = None,
        config: Optional[GeneralAgentConfig] = None,
        # Legacy parameters (for backwards compatibility)
        max_loops: Optional[Union[int, str]] = None,
        agent_description: Optional[str] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_agent_routing: Optional[Union[bool, str]] = None,
        enable_code_execution: Optional[bool] = None,
        enable_multimodal: Optional[bool] = None,
        enable_streaming: Optional[bool] = None,
        enable_persistence: Optional[bool] = None,
        enable_causal_reasoning: Optional[bool] = None,
        enable_file_operations: Optional[bool] = None,
        persistence_path: Optional[str] = None,
        rate_limit_rpm: Optional[int] = None,
        rate_limit_rph: Optional[int] = None,
        custom_prompt_additions: Optional[List[str]] = None,
        aop_instance: Optional[Any] = None,
        router_instance: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize GeneralAgent with smart auto-configuration.
        
        Simple usage (recommended):
            agent = GeneralAgent()  # Uses all smart defaults
            agent = GeneralAgent(model_name="gpt-4o")  # Just change model
            agent = GeneralAgent(personality="friendly")  # Just change personality
        
        Advanced usage:
            agent = GeneralAgent(config=GeneralAgentConfig.auto(...))
        
        Args:
            model_name: LLM model (auto-detected from env or defaults to gpt-4o-mini)
            personality: Personality name (default: neutral)
            agent_name: Unique identifier (auto-generated if None)
            config: Pre-configured GeneralAgentConfig (uses auto() if None)
            **kwargs: Legacy parameters for backwards compatibility
        """
        # Auto-detect model from environment or use default
        if model_name is None:
            import os
            model_name = os.getenv("CRCA_MODEL_NAME", "gpt-4o-mini")
        
        # Use provided config or create auto-config
        if config is None:
            # Merge legacy parameters into config overrides
            config_overrides = {}
            if personality is not None:
                config_overrides["personality"] = personality
            if enable_agent_routing is not None:
                config_overrides["enable_agent_routing"] = enable_agent_routing
            if enable_code_execution is not None:
                config_overrides["enable_code_execution"] = enable_code_execution
            if enable_multimodal is not None:
                config_overrides["enable_multimodal"] = enable_multimodal
            if enable_streaming is not None:
                config_overrides["enable_streaming"] = enable_streaming
            if enable_persistence is not None:
                config_overrides["enable_persistence"] = enable_persistence
            if enable_causal_reasoning is not None:
                config_overrides["enable_causal_reasoning"] = enable_causal_reasoning
            if enable_file_operations is not None:
                config_overrides["enable_file_operations"] = enable_file_operations
            if persistence_path is not None:
                config_overrides["persistence_path"] = persistence_path
            if rate_limit_rpm is not None:
                config_overrides["rate_limit_rpm"] = rate_limit_rpm
            if rate_limit_rph is not None:
                config_overrides["rate_limit_rph"] = rate_limit_rph
            if custom_prompt_additions is not None:
                config_overrides["custom_prompt_additions"] = custom_prompt_additions
            
            config = GeneralAgentConfig.auto(**config_overrides)
        
        # Store configuration
        self.config = config
        
        # Auto-generate agent name if not provided
        if agent_name is None:
            import uuid
            agent_name = f"crca-agent-{uuid.uuid4().hex[:8]}"
        
        # Set defaults for other parameters
        if max_loops is None:
            max_loops = 3
        if agent_description is None:
            agent_description = description or "Pure hardened CR-CA Agent - useful across all domains"
        if temperature is None:
            temperature = 0.4
        
        # Build system prompt with personality and extensions
        if system_prompt is None:
            system_prompt = self._build_system_prompt()
        
        # Enable meta-reasoning (planning) for strategic task approach
        # This allows the agent to reason about its reasoning process
        kwargs.setdefault("plan_enabled", True)
        kwargs.setdefault("planning", True)
        
        # Initialize base agent with auto-configured settings
        super().__init__(
            max_loops=max_loops,
            agent_name=agent_name,
            agent_description=agent_description,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            code_interpreter=self.config.enable_code_execution,
            multi_modal=self.config.enable_multimodal,
            streaming_on=self.config.enable_streaming,
            **kwargs,
        )
        
        # Store instances for later use
        self._aop_instance = aop_instance
        self._router_instance = router_instance
        
        logger.info(f"Initialized GeneralAgent: {agent_name} (model: {model_name}, personality: {self.config.personality})")
    
    @classmethod
    def create(
        cls,
        model_name: Optional[str] = None,
        personality: Optional[str] = None,
        **kwargs
    ) -> "GeneralAgent":
        """Factory method for easy agent creation.
        
        Simplest usage:
            agent = GeneralAgent.create()
            agent = GeneralAgent.create(model_name="gpt-4o")
            agent = GeneralAgent.create(personality="friendly")
        
        Args:
            model_name: LLM model name
            personality: Personality name
            **kwargs: Additional parameters
            
        Returns:
            Configured GeneralAgent instance
        """
        return cls(model_name=model_name, personality=personality, **kwargs)
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with personality and extensions.
        
        Returns:
            Complete system prompt string
        """
        builder = PromptBuilder(DEFAULT_GENERAL_AGENT_PROMPT) if PERSONALITY_AVAILABLE else None
        
        if builder:
            # Add personality
            if self.config.personality:
                try:
                    if isinstance(self.config.personality, str):
                        personality = get_personality(self.config.personality) if PERSONALITY_AVAILABLE else None
                    else:
                        personality = self.config.personality
                    
                    if personality and hasattr(personality, "get_prompt_addition"):
                        builder.add_personality(personality.get_prompt_addition())
                except Exception as e:
                    logger.warning(f"Error adding personality: {e}")
            
            # Add custom additions
            for addition in self.config.custom_prompt_additions:
                builder.add_custom(addition)
            
            return builder.build()
        
        # Fallback if PromptBuilder not available
        prompt = DEFAULT_GENERAL_AGENT_PROMPT
        if self.config.custom_prompt_additions:
            prompt += "\n\n" + "\n".join(self.config.custom_prompt_additions)
        return prompt
    
    def _get_domain_schema(self) -> Optional[Dict[str, Any]]:
        """Return tool schemas for agent discovery and dynamic tools.
        
        Returns:
            Dictionary containing tool schemas or None
        """
        # BaseSpecializedAgent wraps the schema in a list, so we return None
        # and handle tools in _domain_specific_setup by setting tools_list_dictionary directly
        return None
    
    def _get_agent_discovery_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for agent discovery tools.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        # discover_agents tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "discover_agents",
                "description": "Discover all available agents from AOP and router instances",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })
        
        # route_to_agent tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "route_to_agent",
                "description": "Route a task to a specific agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "Name of the agent to route to",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task to execute",
                        },
                    },
                    "required": ["agent_name", "task"],
                },
            },
        })
        
        return schemas
    
    def _get_tool_discovery_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for tool discovery.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        # discover_tools tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "discover_tools",
                "description": "Discover available tools dynamically",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        })
        
        return schemas
    
    def _get_image_annotation_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for image annotation tools.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        # annotate_image tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "annotate_image",
                "description": "Annotate an image with geometric primitives, semantic labels, and measurements. Automatically detects image type, tunes parameters, and extracts primitives (lines, circles, contours). Returns overlay image, formal report, and JSON data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file, URL, or description of image location"
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["overlay", "json", "report", "all"],
                            "default": "all",
                            "description": "Output format: 'overlay' (numpy array), 'json' (structured data), 'report' (text), 'all' (AnnotationResult)"
                        },
                        "frame_id": {
                            "type": "integer",
                            "description": "Optional frame ID for temporal tracking in video sequences"
                        }
                    },
                    "required": ["image_path"]
                }
            }
        })
        
        # query_image tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "query_image",
                "description": "Answer a specific query about an image using natural language. Performs annotation first, then analyzes the results to answer questions like 'find the largest building', 'measure dimensions', 'count objects', etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to image file, URL, or description of image location"
                        },
                        "query": {
                            "type": "string",
                            "description": "Natural language query about the image (e.g., 'find the largest building and measure its dimensions', 'count how many circles are in the image', 'identify all lines and measure their lengths')"
                        },
                        "frame_id": {
                            "type": "integer",
                            "description": "Optional frame ID for temporal tracking"
                        }
                    },
                    "required": ["image_path", "query"]
                }
            }
        })
        
        return schemas
    
    def _get_causal_reasoning_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for causal reasoning tools.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        # extract_causal_variables tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "extract_causal_variables",
                "description": "Extract and propose causal variables, relationships, and counterfactual scenarios needed for causal analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "required_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Core variables that must be included for causal analysis"
                        },
                        "optional_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Additional variables that may be useful but not essential"
                        },
                        "causal_edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2
                            },
                            "description": "Causal relationships as [source, target] pairs"
                        },
                        "counterfactual_variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Variables to explore in counterfactual scenarios"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why these variables and relationships are needed"
                        }
                    },
                    "required": ["required_variables", "causal_edges", "reasoning"]
                }
            }
        })
        
        # generate_causal_analysis tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "generate_causal_analysis",
                "description": "Generates structured causal reasoning and counterfactual analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "causal_analysis": {
                            "type": "string",
                            "description": "Analysis of causal relationships and mechanisms"
                        },
                        "intervention_planning": {
                            "type": "string", 
                            "description": "Planned interventions to test causal hypotheses"
                        },
                        "counterfactual_scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "scenario_name": {"type": "string"},
                                    "interventions": {"type": "object"},
                                    "expected_outcomes": {"type": "object"},
                                    "reasoning": {"type": "string"}
                                }
                            },
                            "description": "Multiple counterfactual scenarios to explore"
                        },
                        "causal_strength_assessment": {
                            "type": "string",
                            "description": "Assessment of causal relationship strengths and confounders"
                        },
                        "optimal_solution": {
                            "type": "string",
                            "description": "Recommended optimal solution based on causal analysis"
                        }
                    },
                    "required": [
                        "causal_analysis",
                        "intervention_planning", 
                        "counterfactual_scenarios",
                        "causal_strength_assessment",
                        "optimal_solution"
                    ]
                }
            }
        })
        
        return schemas
    
    def _get_file_operations_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for file operations tools.
        
        Returns:
            List of tool schema dictionaries
        """
        schemas = []
        
        # write_file tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates parent directories if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file"
                        },
                        "encoding": {
                            "type": "string",
                            "default": "utf-8",
                            "description": "File encoding (default: utf-8)"
                        }
                    },
                    "required": ["filepath", "content"]
                }
            }
        })
        
        # read_file tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read content from a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "encoding": {
                            "type": "string",
                            "default": "utf-8",
                            "description": "File encoding (default: utf-8)"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        })
        
        # list_directory tool
        schemas.append({
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories in a path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list"
                        },
                        "recursive": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to list recursively"
                        }
                    },
                    "required": ["directory_path"]
                }
            }
        })
        
        return schemas
    
    @staticmethod
    def _get_image_annotation_engine() -> Optional[Any]:
        """Get or create singleton image annotation engine instance.
        
        Returns:
            ImageAnnotationEngine instance or None if not available
        """
        global _image_annotation_engine
        if not IMAGE_ANNOTATION_AVAILABLE:
            return None
        if _image_annotation_engine is None:
            try:
                _image_annotation_engine = ImageAnnotationEngine()
                logger.info("Image annotation engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize image annotation engine: {e}")
                return None
        return _image_annotation_engine
    
    def _build_domain_prompt(self, task: str) -> str:
        """Build domain-specific prompt for the given task.
        
        Args:
            task: The task description
            
        Returns:
            Formatted prompt string
        """
        # Include full conversation history if available
        context = ""
        if hasattr(self, "conversation") and CONVERSATION_AVAILABLE:
            try:
                history = self.conversation.conversation_history
                if history:
                    # Get last few messages for context
                    recent = history[-5:] if len(history) > 5 else history
                    context = "\n\n## Recent Conversation Context\n\n"
                    for msg in recent:
                        role = msg.get("role", "Unknown")
                        content = msg.get("content", "")
                        context += f"{role}: {content}\n\n"
            except Exception as e:
                logger.debug(f"Error building conversation context: {e}")
        
        prompt = f"{task}\n{context}"
        return prompt
    
    def _domain_specific_setup(self) -> None:
        """Set up domain-specific attributes and integrations."""
        # Set up tools list
        tools_list = []
        
        if AGENT_DISCOVERY_AVAILABLE and self.config.enable_agent_routing:
            tools_list.extend(self._get_agent_discovery_schemas())
        
        if TOOL_DISCOVERY_AVAILABLE:
            tools_list.extend(self._get_tool_discovery_schemas())
        
        # Add causal reasoning tools if enabled and available
        if self.config.enable_causal_reasoning and CRCA_AVAILABLE:
            tools_list.extend(self._get_causal_reasoning_schemas())
            # Initialize CRCAAgent instance for causal reasoning (lazy)
            if not hasattr(self, '_crca_agent') or self._crca_agent is None:
                try:
                    self._crca_agent = CRCAAgent(
                        agent_name=f"{self.agent_name}-crca",
                        model_name=self.model_name,
                        use_crca_tools=False,  # We're providing tools directly
                        use_image_annotation=False,  # Handled separately
                    )
                    logger.debug("CRCAAgent instance created for causal reasoning")
                except Exception as e:
                    logger.warning(f"Failed to create CRCAAgent instance: {e}")
                    self._crca_agent = None
            
            # Initialize tools list if needed
            if not hasattr(self, 'tools') or self.tools is None:
                self.tools = []
            
            # Add causal reasoning tool handlers
            if self._crca_agent is not None:
                def extract_causal_variables(
                    required_variables: List[str],
                    causal_edges: List[List[str]],
                    reasoning: str,
                    optional_variables: Optional[List[str]] = None,
                    counterfactual_variables: Optional[List[str]] = None,
                ) -> Dict[str, Any]:
                    """Tool handler for extract_causal_variables."""
                    try:
                        return self._crca_agent._extract_causal_variables_handler(
                            required_variables=required_variables,
                            causal_edges=causal_edges,
                            reasoning=reasoning,
                            optional_variables=optional_variables,
                            counterfactual_variables=counterfactual_variables,
                        )
                    except Exception as e:
                        logger.error(f"Error in extract_causal_variables tool: {e}")
                        return {"error": str(e)}
                
                def generate_causal_analysis(
                    causal_analysis: str,
                    intervention_planning: str,
                    counterfactual_scenarios: List[Dict[str, Any]],
                    causal_strength_assessment: str,
                    optimal_solution: str,
                ) -> Dict[str, Any]:
                    """Tool handler for generate_causal_analysis."""
                    try:
                        return self._crca_agent._generate_causal_analysis_handler(
                            causal_analysis=causal_analysis,
                            intervention_planning=intervention_planning,
                            counterfactual_scenarios=counterfactual_scenarios,
                            causal_strength_assessment=causal_strength_assessment,
                            optimal_solution=optimal_solution,
                        )
                    except Exception as e:
                        logger.error(f"Error in generate_causal_analysis tool: {e}")
                        return {"error": str(e)}
                
                self.add_tool(extract_causal_variables)
                self.add_tool(generate_causal_analysis)
                logger.info("Causal reasoning tools added to GeneralAgent")
        
        # Add file operations tools if enabled and available
        if self.config.enable_file_operations and FILE_OPERATIONS_AVAILABLE:
            tools_list.extend(self._get_file_operations_schemas())
            
            # Initialize file manager
            if not hasattr(self, '_file_manager') or self._file_manager is None:
                try:
                    self._file_manager = IntelligentFileManager()
                    logger.debug("File manager initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize file manager: {e}")
                    self._file_manager = None
            
            # Initialize tools list if needed
            if not hasattr(self, 'tools') or self.tools is None:
                self.tools = []
            
            # Add file operations tool handlers
            if self._file_manager is not None:
                def write_file(
                    filepath: str,
                    content: str,
                    encoding: str = "utf-8"
                ) -> Dict[str, Any]:
                    """Tool handler for write_file."""
                    try:
                        return self._file_manager.writer.write_file(
                            filepath=filepath,
                            content=content,
                            encoding=encoding
                        )
                    except Exception as e:
                        logger.error(f"Error in write_file tool: {e}")
                        return {"success": False, "error": str(e)}
                
                def read_file(
                    filepath: str,
                    encoding: str = "utf-8"
                ) -> Dict[str, Any]:
                    """Tool handler for read_file."""
                    try:
                        from pathlib import Path
                        path = Path(filepath)
                        if not path.exists():
                            return {"success": False, "error": f"File not found: {filepath}"}
                        
                        with open(path, 'r', encoding=encoding) as f:
                            content = f.read()
                        
                        return {
                            "success": True,
                            "filepath": str(path),
                            "content": content,
                            "size": len(content.encode(encoding))
                        }
                    except Exception as e:
                        logger.error(f"Error in read_file tool: {e}")
                        return {"success": False, "error": str(e)}
                
                def list_directory(
                    directory_path: str,
                    recursive: bool = False
                ) -> Dict[str, Any]:
                    """Tool handler for list_directory."""
                    try:
                        from pathlib import Path
                        path = Path(directory_path)
                        if not path.exists():
                            return {"success": False, "error": f"Directory not found: {directory_path}"}
                        if not path.is_dir():
                            return {"success": False, "error": f"Path is not a directory: {directory_path}"}
                        
                        if recursive:
                            files = [str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()]
                            dirs = [str(p.relative_to(path)) for p in path.rglob('*') if p.is_dir()]
                        else:
                            files = [p.name for p in path.iterdir() if p.is_file()]
                            dirs = [p.name for p in path.iterdir() if p.is_dir()]
                        
                        return {
                            "success": True,
                            "directory": str(path),
                            "files": files,
                            "directories": dirs,
                            "total_files": len(files),
                            "total_directories": len(dirs)
                        }
                    except Exception as e:
                        logger.error(f"Error in list_directory tool: {e}")
                        return {"success": False, "error": str(e)}
                
                self.add_tool(write_file)
                self.add_tool(read_file)
                self.add_tool(list_directory)
                logger.info("File operations tools added to GeneralAgent")
        
        # Add image annotation tools if multimodal is enabled and available
        if self.config.enable_multimodal and IMAGE_ANNOTATION_AVAILABLE:
            tools_list.extend(self._get_image_annotation_schemas())
            # Initialize image annotation engine (lazy singleton)
            self._image_annotation_engine = self._get_image_annotation_engine()
            
            # Initialize tools list if needed
            if not hasattr(self, 'tools') or self.tools is None:
                self.tools = []
            
            # Add tool handlers only if engine is available
            if self._image_annotation_engine is not None:
                # Add annotate_image handler
                def annotate_image(
                    image_path: str,
                    output_format: str = "all",
                    frame_id: Optional[int] = None
                ) -> Dict[str, Any]:
                    """Tool handler for annotate_image."""
                    engine = GeneralAgent._get_image_annotation_engine()
                    if engine is None:
                        return {"error": "Image annotation engine not available"}
                    try:
                        result = engine.annotate(image_path, frame_id=frame_id, output=output_format)
                        if output_format == "overlay":
                            return {"overlay_image": "numpy array returned", "shape": str(result.shape) if hasattr(result, 'shape') else "unknown"}
                        elif output_format == "json":
                            return result
                        elif output_format == "report":
                            return {"report": result}
                        else:  # all
                            return {
                                "entities": len(result.annotation_graph.entities),
                                "labels": len(result.annotation_graph.labels),
                                "contradictions": len(result.annotation_graph.contradictions),
                                "processing_time": result.processing_time,
                                "formal_report": result.formal_report[:500] + "..." if len(result.formal_report) > 500 else result.formal_report,
                                "json_summary": {k: str(v)[:200] for k, v in list(result.json_output.items())[:5]}
                            }
                    except Exception as e:
                        logger.error(f"Error in annotate_image tool: {e}")
                        return {"error": str(e)}
                
                # Add query_image handler
                def query_image(
                    image_path: str,
                    query: str,
                    frame_id: Optional[int] = None
                ) -> Dict[str, Any]:
                    """Tool handler for query_image."""
                    engine = GeneralAgent._get_image_annotation_engine()
                    if engine is None:
                        return {"error": "Image annotation engine not available"}
                    try:
                        result = engine.query(image_path, query, frame_id=frame_id)
                        return {
                            "answer": result["answer"],
                            "entities_found": len(result["entities"]),
                            "measurements": result["measurements"],
                            "confidence": result["confidence"],
                            "reasoning": result["reasoning"][:500] + "..." if len(result["reasoning"]) > 500 else result["reasoning"]
                        }
                    except Exception as e:
                        logger.error(f"Error in query_image tool: {e}")
                        return {"error": str(e)}
                
                # Add tools to agent
                self.add_tool(annotate_image)
                self.add_tool(query_image)
                
                # Re-initialize tool_struct after adding tools (similar to CRCAAgent)
                if hasattr(self, 'setup_tools'):
                    try:
                        self.tool_struct = self.setup_tools()
                    except Exception as e:
                        logger.debug(f"Error setting up tools: {e}")
                
                logger.info("Image annotation tools added to GeneralAgent")
        
        # Re-initialize tool_struct after adding all tools
        if hasattr(self, 'setup_tools') and self.tools:
            try:
                self.tool_struct = self.setup_tools()
            except Exception as e:
                logger.debug(f"Error setting up tools: {e}")
        
        if tools_list:
            # Update tools_list_dictionary (BaseSpecializedAgent initializes it as a list)
            if hasattr(self, "tools_list_dictionary"):
                if self.tools_list_dictionary:
                    # Extend existing tools
                    self.tools_list_dictionary.extend(tools_list)
                else:
                    # Set new tools list
                    self.tools_list_dictionary = tools_list
            else:
                # Initialize if not present
                self.tools_list_dictionary = tools_list
            
            # Update short_memory with tools if available
            if hasattr(self, "short_memory") and self.short_memory:
                try:
                    self.short_memory.add(
                        role=self.agent_name,
                        content=tools_list,
                    )
                except Exception as e:
                    logger.debug(f"Error adding tools to short_memory: {e}")
        
        # Set up rate limiting
        if RATE_LIMITER_AVAILABLE:
            config = RateLimitConfig(
                requests_per_minute=self.config.rate_limit_rpm,
                requests_per_hour=self.config.rate_limit_rph,
            )
            self.rate_limiter = RateLimiter(config)
        else:
            self.rate_limiter = None
        
        # Set up conversation persistence
        if CONVERSATION_AVAILABLE and self.config.enable_persistence:
            try:
                self.conversation = Conversation(
                    name=self.agent_name,
                    autosave=True,
                    save_filepath=self.config.persistence_path,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize conversation persistence: {e}")
                self.conversation = None
        else:
            self.conversation = None
        
        # Set up formatter
        if FORMATTER_AVAILABLE:
            self.formatter = Formatter(md=True)
        else:
            self.formatter = None
        
        # Auto-discover AOP/router if needed
        if self.config.enable_agent_routing == "auto" or self.config.enable_agent_routing is True:
            if AGENT_DISCOVERY_AVAILABLE:
                if self._aop_instance is None:
                    aop_instances = discover_aop_instances()
                    if aop_instances:
                        self._aop_instance = aop_instances[0]
                        logger.debug(f"Auto-discovered AOP instance: {self._aop_instance}")
                
                if self._router_instance is None:
                    router_instances = discover_router_instances()
                    if router_instances:
                        self._router_instance = router_instances[0]
                        logger.debug(f"Auto-discovered router instance: {self._router_instance}")
        
        # Set up batch processor
        if BATCH_PROCESSOR_AVAILABLE:
            self.batch_processor = BatchProcessor(
                max_workers=4,
                rate_limiter=self.rate_limiter,
            )
        else:
            self.batch_processor = None
        
        logger.debug("Domain-specific setup complete")
    
    def _comprehensive_error_handler(
        self,
        task: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """Comprehensive error handling with retry, fallback, and user guidance.
        
        Args:
            task: Task to execute
            max_retries: Maximum number of retries
            retry_delay: Initial retry delay (exponential backoff)
            
        Returns:
            Response string
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Apply rate limiting
                if self.rate_limiter:
                    user_id = getattr(self, "_user_id", "default")
                    is_allowed, error_msg = self.rate_limiter.check_rate_limit(user_id)
                    if not is_allowed:
                        # Wait if rate limited
                        self.rate_limiter.wait_if_rate_limited(user_id, max_wait=60.0)
                
                # Try to route to specialized agent first (route-first strategy)
                if AGENT_DISCOVERY_AVAILABLE and self.config.enable_agent_routing:
                    try:
                        available_agents = discover_all_agents(
                            aop_instances=[self._aop_instance] if self._aop_instance else None,
                            router_instances=[self._router_instance] if self._router_instance else None,
                        )
                        
                        if available_agents:
                            best_agent = find_best_agent_for_task(
                                task,
                                available_agents,
                                aop_instances=[self._aop_instance] if self._aop_instance else None,
                                router_instances=[self._router_instance] if self._router_instance else None,
                            )
                            
                            if best_agent:
                                agent_name, agent_instance, source = best_agent
                                logger.info(f"Routing task to specialized agent: {agent_name} (source: {source})")
                                result = route_to_agent(
                                    agent_name,
                                    task,
                                    aop_instances=[self._aop_instance] if self._aop_instance else None,
                                    router_instances=[self._router_instance] if self._router_instance else None,
                                )
                                if result:
                                    return str(result)
                    except Exception as e:
                        logger.debug(f"Agent routing failed, falling back to direct handling: {e}")
                
                # Direct handling (pass through img, imgs, streaming_callback if available)
                run_kwargs = {}
                if hasattr(self, "_current_img") and self._current_img:
                    run_kwargs["img"] = self._current_img
                if hasattr(self, "_current_imgs") and self._current_imgs:
                    run_kwargs["imgs"] = self._current_imgs
                if hasattr(self, "_current_streaming_callback") and self._current_streaming_callback:
                    run_kwargs["streaming_callback"] = self._current_streaming_callback
                if hasattr(self, "_current_kwargs") and self._current_kwargs:
                    run_kwargs.update(self._current_kwargs)
                
                response = super().run(task, **run_kwargs)
                
                # Save to conversation if persistence enabled
                if self.conversation:
                    try:
                        self.conversation.add("User", task)
                        self.conversation.add(self.agent_name, response)
                    except Exception as e:
                        logger.debug(f"Error saving to conversation: {e}")
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, try fallback
                    try:
                        # Fallback: simpler approach
                        fallback_prompt = f"Please provide a helpful response to: {task}"
                        response = super().run(fallback_prompt)
                        return f"[Fallback Response] {response}"
                    except Exception as fallback_error:
                        # Ask user for guidance
                        error_msg = f"Failed after {max_retries} retries. Last error: {str(last_error)}. Fallback also failed: {str(fallback_error)}"
                        return f"[Error] {error_msg}. Please provide more details or try rephrasing your request."
        
        return f"[Error] Failed to process task after {max_retries} attempts: {str(last_error)}"
    
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Any:
        """Run the agent with comprehensive error handling.
        
        Args:
            task: Task to execute
            img: Optional image input
            imgs: Optional list of images
            streaming_callback: Optional streaming callback
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        if task is None:
            task = ""
        
        # Store img/imgs for use in error handler if needed
        self._current_img = img
        self._current_imgs = imgs
        self._current_streaming_callback = streaming_callback
        self._current_kwargs = kwargs
        
        # Use comprehensive error handler
        return self._comprehensive_error_handler(str(task))
    
    async def run_async(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        """Run the agent asynchronously.
        
        Args:
            task: Task to execute
            img: Optional image input
            imgs: Optional list of images
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        if task is None:
            task = ""
        
        # Run in executor to avoid blocking
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return await loop.run_in_executor(None, self.run, task, img, imgs, **kwargs)
    
    def run_batch(
        self,
        tasks: List[str],
        task_ids: Optional[List[str]] = None,
        user_id: str = "default",
    ) -> Tuple[List[Any], Any]:
        """Process a batch of tasks.
        
        Args:
            tasks: List of tasks to process
            task_ids: Optional list of task identifiers
            user_id: User identifier for rate limiting
            
        Returns:
            Tuple of (results, stats)
        """
        if not BATCH_PROCESSOR_AVAILABLE or not self.batch_processor:
            # Fallback: process sequentially
            results = [self.run(task) for task in tasks]
            return results, None
        
        return self.batch_processor.process_batch(
            tasks=tasks,
            task_fn=self.run,
            task_ids=task_ids,
            user_id=user_id,
        )
    
    async def run_batch_async(
        self,
        tasks: List[str],
        task_ids: Optional[List[str]] = None,
        user_id: str = "default",
    ) -> Tuple[List[Any], Any]:
        """Process a batch of tasks asynchronously.
        
        Args:
            tasks: List of tasks to process
            task_ids: Optional list of task identifiers
            user_id: User identifier for rate limiting
            
        Returns:
            Tuple of (results, stats)
        """
        if not BATCH_PROCESSOR_AVAILABLE or not self.batch_processor:
            # Fallback: process concurrently
            results = await asyncio.gather(*[self.run_async(task) for task in tasks])
            return results, None
        
        return await self.batch_processor.process_batch_async(
            tasks=tasks,
            task_fn=self.run_async,
            task_ids=task_ids,
            user_id=user_id,
        )
    
    def save_conversation(self, filepath: Optional[str] = None) -> None:
        """Save conversation to file.
        
        Args:
            filepath: Optional filepath (uses default if None)
        """
        if not self.conversation:
            logger.warning("Conversation persistence not enabled")
            return
        
        try:
            if filepath:
                self.conversation.save_filepath = filepath
            
            if hasattr(self.conversation, "save_with_metadata"):
                self.conversation.save_with_metadata(force=True)
            else:
                self.conversation.export(force=True)
            
            logger.info(f"Conversation saved to {self.conversation.save_filepath}")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
    
    def load_conversation(self, filepath: str) -> None:
        """Load conversation from file.
        
        Args:
            filepath: Filepath to load from
        """
        if not CONVERSATION_AVAILABLE:
            logger.warning("Conversation utilities not available")
            return
        
        try:
            if not self.conversation:
                self.conversation = Conversation(
                    name=self.agent_name,
                    load_filepath=filepath,
                )
            else:
                self.conversation.load(filepath)
            
            logger.info(f"Conversation loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
