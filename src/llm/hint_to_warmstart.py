"""Hint to Warm-Start Transformer.

Transforms LLM-generated hints into solver-compatible warm-start candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance, RouteSolution
    from src.llm.hint_schema import LLMHints
    from src.domain.event_schema import WarmstartType


@dataclass
class WarmStartCandidate:
    """A warm-start candidate for the VRPTW solver.

    Attributes:
        warmstart_type: Type of warm-start (routes, repair, etc.)
        initial_routes: List of routes as lists of customer IDs.
        avoid_pairs: Pairs of customers to avoid in same route.
        biased_order: Preferred visit order (advisory only).
        extra_seconds: Additional time to grant solver.
    """
    warmstart_type: "WarmstartType" = None
    initial_routes: list[list[int]] = field(default_factory=list)
    avoid_pairs: list[list[int]] = field(default_factory=list)
    biased_order: list[int] = field(default_factory=list)
    extra_seconds: Optional[float] = None

    def __post_init__(self):
        if self.warmstart_type is None:
            from src.domain.event_schema import WarmstartType
            self.warmstart_type = WarmstartType.NONE


def hints_to_warmstart(
    instance: "VRPTWInstance",
    hints: "LLMHints",
    incumbent: "RouteSolution",
) -> WarmStartCandidate:
    """Transform LLM hints into a warm-start candidate.

    Args:
        instance: The VRPTW instance.
        hints: LLM-generated hints.
        incumbent: Current incumbent solution.

    Returns:
        WarmStartCandidate ready for solver consumption.
    """
    from src.domain.event_schema import WarmstartType

    # Extract routes from incumbent
    routes = []
    if incumbent and incumbent.routes:
        # routes is list[list[int]], not list of objects
        routes = [list(route) for route in incumbent.routes]
    
    # Extract avoid_pairs from hints
    avoid_pairs = hints.avoid_pairs if hints.avoid_pairs else []

    # Determine warm-start type
    if routes:
        warmstart_type = WarmstartType.ROUTES_ONLY
    else:
        warmstart_type = WarmstartType.NONE

    # Extra solver time from solver_control hint
    extra_seconds = None
    if hints.solver_control and hints.solver_control.extra_seconds:
        extra_seconds = hints.solver_control.extra_seconds

    return WarmStartCandidate(
        warmstart_type=warmstart_type,
        initial_routes=routes,
        avoid_pairs=avoid_pairs,
        biased_order=list(hints.priority_customers) if hints.priority_customers else [],
        extra_seconds=extra_seconds,
    )
