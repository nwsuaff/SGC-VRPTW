"""OpenAI API client for real LLM hint generation.

Requires the `openai` package. Install with:
    pip install openai
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.domain.schema import VRPTWInstance, RouteSolution
from src.llm.hint_schema import LLMHints, empty_hints
from src.llm.prompt_builder import build_llm_prompt
from src.llm.parser import parse_llm_output

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError
    from openai.types.chat import ChatCompletion

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore
    APIError = None  # type: ignore
    RateLimitError = None  # type: ignore
    APITimeoutError = None  # type: ignore
    ChatCompletion = None  # type: ignore


class OpenAIClient:
    """Real OpenAI API client for LLM-guided VRPTW solving.

    Wraps the OpenAI API with proper error handling, retry logic,
    token usage tracking, and cost estimation.

    Usage:
        client = OpenAIClient(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=1024,
            api_key=os.environ["OPENAI_API_KEY"],
        )
        hints = client.query(instance, incumbent)
    """

    _DEFAULT_MODEL = "gpt-4o-mini"
    _DEFAULT_TEMPERATURE = 0.0
    _DEFAULT_MAX_TOKENS = 1024
    _DEFAULT_TIMEOUT = 60.0
    _DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        seed: int | None = None,
    ):
        """Initialize the OpenAI client.

        Args:
            model: OpenAI model name. Defaults to gpt-4o-mini.
            temperature: Sampling temperature. Defaults to 0.0 (deterministic).
            max_tokens: Max response tokens. Defaults to 1024.
            timeout: Request timeout in seconds. Defaults to 60.
            max_retries: Max retry attempts on rate limit / timeout. Defaults to 3.
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
            base_url: Optional base URL for proxy endpoints.
            seed: Optional seed for reproducibility (passed to API).
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        self._model = model or self._DEFAULT_MODEL
        self._temperature = temperature if temperature is not None else self._DEFAULT_TEMPERATURE
        self._max_tokens = max_tokens or self._DEFAULT_MAX_TOKENS
        self._timeout = timeout or self._DEFAULT_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else self._DEFAULT_MAX_RETRIES
        self._seed = seed

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

        # Runtime stats
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._query_count = 0

    @property
    def model(self) -> str:
        return self._model

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def total_prompt_tokens(self) -> int:
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self._total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    def query(
        self,
        instance: VRPTWInstance,
        incumbent: RouteSolution | None = None,
        **kwargs: Any,
    ) -> LLMHints:
        """Generate VRPTW hints by querying the OpenAI API.

        Args:
            instance: The VRPTW instance.
            incumbent: Current incumbent solution (optional).
            **kwargs: Additional arguments (ignored).

        Returns:
            LLMHints object, or empty hints on failure.
        """
        prompt = build_llm_prompt(instance, incumbent)
        start_time = time.perf_counter()
        raw_output = self._call_api(prompt)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"[OpenAI] Query {self._query_count}: model={self._model}, "
            f"latency={latency_ms:.0f}ms, "
            f"prompt_tokens={self._total_prompt_tokens}, "
            f"cost=${self._total_cost_usd:.4f}"
        )

        return parse_llm_output(raw_output)

    def _call_api(self, prompt: str) -> str:
        """Call the OpenAI API with the given prompt.

        Returns the raw response text. Raises on unrecoverable errors.
        """
        self._query_count += 1

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a logistics optimization expert. You generate structured "
                    "JSON hints for a vehicle routing solver. Your output must be "
                    "valid JSON only, without any additional text or markdown fences."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        if self._seed is not None:
            kwargs["seed"] = self._seed

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"[OpenAI] API call failed: {e}")
            return ""

        # Handle string responses from OpenAI-compatible endpoints.
        if isinstance(response, str):
            return response

        # Handle dict responses (raw JSON from some APIs)
        if isinstance(response, dict):
            try:
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception:
                return str(response)

        # Handle proper ChatCompletion objects
        try:
            usage = response.usage
            if usage:
                prompt_toks = usage.prompt_tokens or 0
                completion_toks = usage.completion_tokens or 0
                self._total_prompt_tokens += prompt_toks
                self._total_completion_tokens += completion_toks
                self._total_cost_usd += self._estimate_cost(prompt_toks, completion_toks)
        except Exception:
            pass

        text = response.choices[0].message.content or ""
        return text

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on token usage.

        Pricing (approximate, as of 2025):
        - GPT-4o-mini input: $0.15 / 1M tokens
        - GPT-4o-mini output: $0.60 / 1M tokens
        - GPT-4o input: $2.50 / 1M tokens
        - GPT-4o output: $10.00 / 1M tokens
        """
        model = self._model.lower()
        if "gpt-4o-mini" in model:
            input_cost = prompt_tokens * 0.15 / 1_000_000
            output_cost = completion_tokens * 0.60 / 1_000_000
        elif "gpt-4o" in model:
            input_cost = prompt_tokens * 2.50 / 1_000_000
            output_cost = completion_tokens * 10.00 / 1_000_000
        elif "gpt-3.5" in model:
            input_cost = prompt_tokens * 0.50 / 1_000_000
            output_cost = completion_tokens * 1.50 / 1_000_000
        else:
            input_cost = prompt_tokens * 0.50 / 1_000_000
            output_cost = completion_tokens * 1.50 / 1_000_000

        return input_cost + output_cost

    def reset_stats(self) -> None:
        """Reset token and cost counters."""
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0
        self._query_count = 0
