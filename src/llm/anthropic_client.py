"""Anthropic API client for real LLM hint generation.

Requires the `anthropic` package. Install with:
    pip install anthropic

This is a placeholder implementation. Full integration requires:
1. Installing the anthropic package
2. Setting the ANTHROPIC_API_KEY environment variable
3. Implementing the Anthropic Messages API call
"""

from __future__ import annotations

import logging

from src.domain.schema import VRPTWInstance, RouteSolution
from src.llm.hint_schema import LLMHints

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Claude API client for LLM-guided VRPTW solving.

    This is a stub implementation. Full support requires installing the
    anthropic package and providing an API key.

    Usage:
        client = AnthropicClient(
            model="claude-sonnet-4-20250514",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        hints = client.query(instance, incumbent)
    """

    _DEFAULT_MODEL = "claude-sonnet-4-20250514"
    _DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        **kwargs,
    ):
        """Initialize the Anthropic client.

        Args:
            model: Claude model name. Defaults to claude-sonnet-4.
            max_tokens: Max response tokens. Defaults to 1024.
            timeout: Request timeout in seconds.
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        """
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        self._model = model or self._DEFAULT_MODEL
        self._max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS
        self._timeout = timeout

        logger.warning(
            "AnthropicClient is a stub. "
            "Install anthropic package and implement the API call to use with real Claude models."
        )

    def query(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
        **kwargs,
    ) -> LLMHints:
        """Generate VRPTW hints using Anthropic Claude.

        Currently returns empty hints as a placeholder.
        """
        logger.warning(
            f"AnthropicClient.query called with model={self._model} "
            "but is not fully implemented. Returning empty hints."
        )
        from src.llm.hint_schema import empty_hints
        return empty_hints()
