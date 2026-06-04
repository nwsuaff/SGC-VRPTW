"""Event schema for structured LLM loop logging.

This schema captures every phase transition and decision point in the LLM-guided
solver loop, enabling systematic failure mode analysis and reproducibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class Phase(str, Enum):
    """Phases of the LLM-guided solver loop."""

    INITIAL_SOLVE = "initial_solve"
    PROMPT_BUILD = "prompt_build"
    LLM_QUERY = "llm_query"
    PARSE = "parse"
    HINT_TRANSFORM = "hint_transform"
    WARMSTART_APPLY = "warmstart_apply"
    FINAL_SOLVE = "final_solve"
    VALIDATION = "validation"
    SAVE = "save"


class Improvement(str, Enum):
    """Outcome of the LLM loop comparison against incumbent."""

    IMPROVED = "improved"
    NO_CHANGE = "no_change"
    DEGRADED = "degraded"
    INFEASIBLE = "infeasible"


class FailureMode(str, Enum):
    """Categories of failure in the LLM loop."""

    NONE = "none"
    LLM_TIMEOUT = "llm_timeout"
    LLM_ERROR = "llm_error"
    PARSE_ERROR = "parse_error"
    NO_HINTS = "no_hints"
    WARMSTART_INFEASIBLE = "warmstart_infeasible"
    WARMSTART_REJECTED = "warmstart_rejected"
    SOLVER_TIMEOUT = "solver_timeout"
    SOLVER_ERROR = "solver_error"
    VALIDATION_FAILED = "validation_failed"


class WarmstartType(str, Enum):
    """Type of warm-start produced by hint transformation.

    Note: ORDER_ONLY is deprecated because OR-Tools does not support insertion
    order constraints. The biased_order field is no longer generated.
    """

    NONE = "none"
    ROUTES_ONLY = "routes_only"
    # ORDER_ONLY = "order_only"  # DEPRECATED: biased_order not supported by OR-Tools
    # MIXED = "mixed"  # DEPRECATED: same reason
    REPAIR_ONLY = "repair_only"


@dataclass
class LLMLoopEvent:
    """Structured event record for a single phase in the LLM solver loop.

    Each experiment run produces a list of these events, one per phase,
    plus a final summary event.
    """

    timestamp: float = field(default_factory=time.perf_counter)
    phase: Phase = Phase.INITIAL_SOLVE

    # Experiment identity
    experiment_id: str = ""
    instance_name: str = ""
    family: str | None = None
    size: int = 0
    seed: int = 42

    # LLM layer
    llm_provider: str = "mock"
    llm_model: str | None = None
    llm_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None

    # Hint layer
    hints_received: bool = False
    hint_type_counts: dict[str, int] = field(default_factory=dict)
    hint_rejected_count: int = 0

    # Warm-start layer
    warmstart_type: WarmstartType = WarmstartType.NONE
    warmstart_accepted: bool = False
    warmstart_rejected_reason: str | None = None

    # Solution comparison
    initial_vehicles: int = 0
    initial_distance: float = 0.0
    initial_feasible: bool = False
    final_vehicles: int = 0
    final_distance: float = 0.0
    final_feasible: bool = False
    improvement: Improvement = Improvement.NO_CHANGE

    # Runtime breakdown
    phase_runtime_ms: float | None = None
    total_runtime_sec: float = 0.0
    llm_time_sec: float = 0.0
    parse_time_sec: float = 0.0
    solver_time_sec: float = 0.0
    checker_time_sec: float = 0.0

    # Failure tracking
    failure_mode: FailureMode = FailureMode.NONE
    failure_detail: str | None = None

    # Per-route detail (for debugging specific instances)
    routes: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, handling enums as strings."""
        d = asdict(self)
        d["phase"] = self.phase.value
        d["improvement"] = self.improvement.value
        d["failure_mode"] = self.failure_mode.value
        d["warmstart_type"] = self.warmstart_type.value
        return d

    def to_json_line(self) -> str:
        """Return a JSON line string for appending to a .jsonl file."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class ExperimentMetadata:
    """Static metadata for an entire experiment run."""

    experiment_run_id: str
    git_commit: str | None
    python_version: str
    timestamp: str
    seed: int
    short_budget: int
    final_budget: int
    llm_provider: str
    llm_model: str | None
    instance_count: int
    method_count: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)
