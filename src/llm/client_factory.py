"""LLM Client Factory."""

import logging
import os
from pathlib import Path
from typing import Optional

# Load .env file if it exists
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from src.llm.mock_client import MockLLMClient

logger = logging.getLogger(__name__)

HAS_OPENAI = False
HAS_ANTHROPIC = False

try:
    from src.llm.openai_client import OpenAIClient
    HAS_OPENAI = True
except ImportError:
    pass

try:
    from src.llm.anthropic_client import AnthropicClient
    HAS_ANTHROPIC = True
except ImportError:
    pass


def is_llm_available(name: str) -> bool:
    """Check if an LLM client is available."""
    if name == "mock":
        return True
    if name in ("openai", "gpt"):
        return HAS_OPENAI
    if name in ("anthropic", "claude"):
        return HAS_ANTHROPIC
    return False


def create_llm_client(
    name: str = "mock",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
):
    """Create an LLM client by name."""
    if name == "mock":
        return MockLLMClient()

    if name in ("openai", "gpt"):
        if not HAS_OPENAI:
            logger.warning("OpenAI package not installed, using mock")
            return MockLLMClient()

        key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        mdl = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        return OpenAIClient(
            model=mdl,
            api_key=key,
            base_url=base_url,
            **kwargs,
        )

    if name in ("anthropic", "claude"):
        if not HAS_ANTHROPIC:
            logger.warning("Anthropic package not installed, using mock")
            return MockLLMClient()

        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        mdl = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        return AnthropicClient(
            model=mdl,
            api_key=key,
            **kwargs,
        )

    logger.warning(f"Unknown LLM client '{name}', using mock")
    return MockLLMClient()
