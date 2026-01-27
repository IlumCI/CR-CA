"""
Prompt builder utility for extendable prompts.

Provides template system for prompt construction with support for adding custom sections.
"""

from typing import Any, Dict, List, Optional
from loguru import logger


class PromptBuilder:
    """Builder for constructing extendable prompts.
    
    Provides functionality for:
    - Building prompts from templates
    - Adding custom sections
    - Personality integration
    - Section management
    """
    
    def __init__(self, base_prompt: Optional[str] = None):
        """Initialize prompt builder.
        
        Args:
            base_prompt: Optional base prompt to start with
        """
        self.base_prompt = base_prompt or ""
        self.sections: List[Dict[str, Any]] = []
        self.custom_additions: List[str] = []
        logger.debug("Initialized PromptBuilder")
    
    def add_section(
        self,
        title: str,
        content: str,
        priority: int = 0,
    ) -> "PromptBuilder":
        """Add a section to the prompt.
        
        Args:
            title: Section title
            content: Section content
            priority: Section priority (higher = earlier in prompt)
            
        Returns:
            Self for chaining
        """
        self.sections.append({
            "title": title,
            "content": content,
            "priority": priority,
        })
        
        # Sort by priority (higher first)
        self.sections.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.debug(f"Added section: {title} (priority: {priority})")
        return self
    
    def add_custom(self, addition: str) -> "PromptBuilder":
        """Add custom prompt addition.
        
        Args:
            addition: Custom prompt text to add
            
        Returns:
            Self for chaining
        """
        self.custom_additions.append(addition)
        logger.debug("Added custom prompt addition")
        return self
    
    def add_personality(self, personality_prompt: str) -> "PromptBuilder":
        """Add personality-specific prompt.
        
        Args:
            personality_prompt: Personality prompt text
            
        Returns:
            Self for chaining
        """
        self.add_section("Personality", personality_prompt, priority=10)
        return self
    
    def remove_section(self, title: str) -> "PromptBuilder":
        """Remove a section by title.
        
        Args:
            title: Section title to remove
            
        Returns:
            Self for chaining
        """
        self.sections = [s for s in self.sections if s["title"] != title]
        logger.debug(f"Removed section: {title}")
        return self
    
    def clear_sections(self) -> "PromptBuilder":
        """Clear all sections.
        
        Returns:
            Self for chaining
        """
        self.sections.clear()
        self.custom_additions.clear()
        logger.debug("Cleared all sections")
        return self
    
    def build(self, include_base: bool = True) -> str:
        """Build the final prompt.
        
        Args:
            include_base: Whether to include base prompt
            
        Returns:
            Complete prompt string
        """
        parts = []
        
        # Add base prompt
        if include_base and self.base_prompt:
            parts.append(self.base_prompt)
        
        # Add sections
        for section in self.sections:
            title = section["title"]
            content = section["content"]
            parts.append(f"\n## {title}\n\n{content}")
        
        # Add custom additions
        for addition in self.custom_additions:
            parts.append(f"\n{addition}")
        
        result = "\n".join(parts).strip()
        logger.debug(f"Built prompt with {len(self.sections)} sections and {len(self.custom_additions)} custom additions")
        return result
    
    def extend(self, other: "PromptBuilder") -> "PromptBuilder":
        """Extend this builder with another builder's sections.
        
        Args:
            other: Another PromptBuilder to extend with
            
        Returns:
            Self for chaining
        """
        # Add sections from other builder
        for section in other.sections:
            self.add_section(
                section["title"],
                section["content"],
                section["priority"],
            )
        
        # Add custom additions
        self.custom_additions.extend(other.custom_additions)
        
        logger.debug(f"Extended with {len(other.sections)} sections from other builder")
        return self
    
    def copy(self) -> "PromptBuilder":
        """Create a copy of this builder.
        
        Returns:
            New PromptBuilder instance with same content
        """
        new_builder = PromptBuilder(self.base_prompt)
        new_builder.sections = [s.copy() for s in self.sections]
        new_builder.custom_additions = self.custom_additions.copy()
        return new_builder
