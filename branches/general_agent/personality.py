"""
Personality system for configurable agent personality.

Provides predefined personalities and support for custom personalities.
"""

from typing import Dict, Optional
from loguru import logger


class Personality:
    """Personality configuration for agents.
    
    Attributes:
        name: Personality name
        description: Personality description
        tone: Tone of communication
        style: Communication style
        prompt_template: Template for personality-specific prompts
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        tone: str,
        style: str,
        prompt_template: Optional[str] = None,
    ):
        """Initialize personality.
        
        Args:
            name: Personality name
            description: Personality description
            tone: Tone of communication
            style: Communication style
            prompt_template: Optional prompt template
        """
        self.name = name
        self.description = description
        self.tone = tone
        self.style = style
        self.prompt_template = prompt_template or self._default_template()
    
    def _default_template(self) -> str:
        """Generate default prompt template.
        
        Returns:
            Default prompt template string
        """
        return f"""You communicate with a {self.tone} tone and {self.style} style.

{self.description}

Remember to:
- Use {self.tone} language
- Maintain a {self.style} communication style
- Be helpful and informative
"""
    
    def get_prompt_addition(self) -> str:
        """Get prompt addition for this personality.
        
        Returns:
            Prompt addition string
        """
        return self.prompt_template


# Predefined personalities
PERSONALITIES: Dict[str, Personality] = {
    "neutral": Personality(
        name="neutral",
        description="Professional and neutral communication style, similar to ChatGPT.",
        tone="professional",
        style="neutral",
        prompt_template="""You are a helpful AI assistant with a professional and neutral communication style.

Your communication should be:
- Clear and concise
- Professional but approachable
- Factual and informative
- Balanced in tone
""",
    ),
    "friendly": Personality(
        name="friendly",
        description="Warm and friendly communication style.",
        tone="warm",
        style="friendly",
        prompt_template="""You are a helpful AI assistant with a warm and friendly communication style.

Your communication should be:
- Warm and welcoming
- Conversational and approachable
- Encouraging and supportive
- Personable and engaging
""",
    ),
    "technical": Personality(
        name="technical",
        description="Precise and technical communication style.",
        tone="precise",
        style="technical",
        prompt_template="""You are a helpful AI assistant with a precise and technical communication style.

Your communication should be:
- Precise and accurate
- Technically detailed
- Structured and organized
- Focused on facts and data
""",
    ),
}


def get_personality(name: str) -> Optional[Personality]:
    """Get a predefined personality by name.
    
    Args:
        name: Personality name
        
    Returns:
        Personality instance or None if not found
    """
    return PERSONALITIES.get(name.lower())


def create_custom_personality(
    name: str,
    description: str,
    tone: str,
    style: str,
    prompt_template: Optional[str] = None,
) -> Personality:
    """Create a custom personality.
    
    Args:
        name: Personality name
        description: Personality description
        tone: Tone of communication
        style: Communication style
        prompt_template: Optional custom prompt template
        
    Returns:
        Custom Personality instance
    """
    personality = Personality(
        name=name,
        description=description,
        tone=tone,
        style=style,
        prompt_template=prompt_template,
    )
    
    # Optionally register it
    PERSONALITIES[name.lower()] = personality
    logger.debug(f"Created custom personality: {name}")
    
    return personality


def list_personalities() -> list[str]:
    """List all available personality names.
    
    Returns:
        List of personality names
    """
    return list(PERSONALITIES.keys())
