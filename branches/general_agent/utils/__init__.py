"""
General Agent utilities module.

Provides utility functions for the general agent.
"""

from typing import Optional

try:
    from .prompt_builder import PromptBuilder
    PROMPT_BUILDER_AVAILABLE = True
except ImportError:
    PromptBuilder = None
    PROMPT_BUILDER_AVAILABLE = False

__all__ = [
    "PromptBuilder",
    "PROMPT_BUILDER_AVAILABLE",
]
