"""Event Logger - Structured event logging for LLM solver loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.event_schema import Phase, LLMLoopEvent

logger = logging.getLogger(__name__)


class EventLogger:
    """Logs structured events during the LLM solver loop.
    
    Logs to both Python logging and a JSONL file for post-hoc analysis.
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        experiment_id: str | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.experiment_id = experiment_id or f"event_{int(time.time())}"
        self._jsonl_path: Path | None = None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._jsonl_path = self.output_dir / f"{self.experiment_id}.jsonl"

    def log_phase_start(
        self,
        phase: "Phase",
        **kwargs,
    ) -> "LLMLoopEvent":
        """Log the start of a phase."""
        from src.domain.event_schema import LLMLoopEvent
        
        event = LLMLoopEvent(
            timestamp=time.perf_counter(),
            phase=phase,
            **kwargs,
        )
        self._write_event(event)
        logger.debug(f"[Event] Phase started: {phase.value}")
        return event

    def log_phase_end(
        self,
        phase: "Phase",
        phase_runtime_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Log the end of a phase."""
        from src.domain.event_schema import LLMLoopEvent
        
        event = LLMLoopEvent(
            timestamp=time.perf_counter(),
            phase=phase,
            phase_runtime_ms=phase_runtime_ms,
            **kwargs,
        )
        self._write_event(event)

    def log_llm_query(
        self,
        instance_name: str,
        llm_provider: str,
        llm_model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        llm_latency_ms: float | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Log LLM query details."""
        from src.domain.event_schema import LLMLoopEvent, Phase
        
        event = LLMLoopEvent(
            timestamp=time.perf_counter(),
            phase=Phase.LLM_QUERY,
            instance_name=instance_name,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_latency_ms=llm_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
        self._write_event(event)

    def _write_event(self, event: "LLMLoopEvent") -> None:
        """Write an event to the JSONL file."""
        if self._jsonl_path:
            with open(self._jsonl_path, "a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")


def load_events_from_jsonl(path: str | Path) -> tuple[list, list["LLMLoopEvent"]]:
    """Load events from a JSONL event log file.

    Args:
        path: Path to the .jsonl file produced by EventLogger.

    Returns:
        Tuple of (metadata_list, events_list). The metadata list is empty
        for compatibility; the events list contains LLMLoopEvent objects.
    """
    import json
    from src.domain.event_schema import LLMLoopEvent

    events: list[LLMLoopEvent] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(LLMLoopEvent(**d))

    return [], events
