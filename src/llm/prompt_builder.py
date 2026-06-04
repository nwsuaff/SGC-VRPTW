"""Enhanced prompt builder: constructs prompts with diagnostics, history, and constraints.

This module implements the enhanced prompt building for the verifier-gated
LLM guidance protocol. It includes:
- Instance summary
- Solver diagnostics (from diagnostic_encoder)
- Hint history (from hint_history)
- Hard constraints
- Enhanced JSON schema
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.schema import VRPTWInstance, RouteSolution
    from src.llm.hint_schema import LLMHints


_TIGHT_CUSTOMER_LIMIT = 10
_MAX_ROUTES_SHOWN = 5
_MAX_HISTORY_ROUNDS = 3


def build_llm_prompt(
    instance: "VRPTWInstance",
    incumbent: "RouteSolution | None" = None,
    hints: "LLMHints | None" = None,
    extra_instructions: str | None = None,
    diagnostics: "FullDiagnostics | None" = None,
    history_summary: str | None = None,
) -> str:
    """Build a comprehensive prompt for the LLM.

    The prompt includes:
    - Instance summary (size, capacity, coordinate ranges)
    - Solver diagnostics (route-level, customer-level, edge-level)
    - Incumbent route summary with diagnostic insights
    - Hint history (recent accepted/rejected hints)
    - Hard constraints that MUST be respected
    - JSON schema matching LLMHints model exactly
    - Optional extra instructions

    Note: The `hints` parameter is reserved for future use (hint refinement).

    Args:
        instance: The VRPTW instance being solved.
        incumbent: Current incumbent solution (may be None for first call).
        hints: Previous LLM hints (reserved for future use, currently unused).
        extra_instructions: Optional domain-specific instructions.
        diagnostics: Pre-computed diagnostics from diagnostic_encoder.
        history_summary: Summary text from hint_history manager.

    Returns:
        A fully-formed prompt string.
    """
    parts = []

    # 1. System prompt
    parts.append(_build_system_prompt())

    # 2. Instance summary
    parts.append(_build_instance_summary(instance))

    # 3. Solver diagnostics (if available)
    if diagnostics:
        parts.append(diagnostics.to_text_summary())

    # 4. Hard constraints
    parts.append(_build_hard_constraints(instance))

    # 5. Current solution summary
    parts.append(_build_incumbent_summary(incumbent))

    # 6. Hint history
    if history_summary:
        parts.append(history_summary)

    # 7. Enhanced JSON schema
    parts.append(_build_enhanced_json_schema())

    # 8. Format instruction
    parts.append(_build_format_instruction())

    # 9. Extra instructions
    if extra_instructions:
        parts.append(f"\n## Additional Instructions\n{extra_instructions}")

    return "\n\n".join(parts)


def _build_system_prompt() -> str:
    """Build the system prompt defining the LLM's role."""
    return """## SYSTEM PROMPT: VRPTW Solver Guidance

You are an expert in Vehicle Routing Problems with Time Windows (VRPTW).
Your role is to provide STRUCTURED, VERIFIABLE guidance hints to a VRPTW solver.

### YOUR TASK
Analyze the current solution state and generate typed hints that will help
the solver find a better feasible solution. You do NOT generate routes directly;
instead, you provide strategic guidance.

### OUTPUT FORMAT
You MUST output ONLY valid JSON matching the schema below.
Do NOT include any text outside the JSON object.

### QUALITY GUIDELINES
1. Focus on customers with tight time windows - they are the hardest to serve
2. Look for routes with low utilization - they may be candidates for consolidation
3. Identify long-distance edges that could be improved with local search
4. Preserve good subroutes that are working well
5. Separate conflicting customers that hurt solution quality

### HINT TYPES
- **priority_customers**: List of customer IDs to serve early (sorted by due time)
- **locked_subroutes**: Preserved route fragments from the incumbent solution
- **avoid_pairs**: Customer pairs that should NOT be on the same route
- **suggested_moves**: Local search moves (relocate/swap/reverse/split)
- **solver_control**: Runtime parameter suggestions (extra_seconds/intensify/diversify)
"""


def _build_instance_summary(instance: "VRPTWInstance") -> str:
    """Summarize key instance characteristics."""
    n = instance.size
    capacity = instance.vehicle_capacity
    depot_x = instance.x_coords.get(instance.depot_id, 0)
    depot_y = instance.y_coords.get(instance.depot_id, 0)

    x_vals = list(instance.x_coords.values())
    y_vals = list(instance.y_coords.values())
    total_demand = sum(instance.demand.get(cid, 0) for cid in instance.customer_ids)

    lines = [
        "## Instance Summary",
        f"- **Name**: {instance.name}",
        f"- **Source**: {instance.source}",
        f"- **Family**: {instance.family or 'unknown'}",
        f"- **Customers**: {n}",
        f"- **Vehicle capacity**: {capacity}",
        f"- **Depot location**: ({depot_x:.1f}, {depot_y:.1f})",
        f"- **Service area**: X=[{min(x_vals):.1f}, {max(x_vals):.1f}], Y=[{min(y_vals):.1f}, {max(y_vals):.1f}]",
        f"- **Total demand**: {total_demand}",
        f"- **Avg demand per customer**: {total_demand / n:.2f}",
        f"- **Demand/Capacity ratio**: {total_demand / (capacity * 100):.2f}%",
    ]

    # Time window statistics
    windows = []
    for cid in instance.customer_ids:
        ready = instance.ready_time.get(cid, 0)
        due = instance.due_time.get(cid, 999999)
        windows.append(due - ready)

    if windows:
        lines.append(f"- **Time window stats**: min={min(windows):.1f}, avg={sum(windows)/len(windows):.1f}, max={max(windows):.1f}")

    return "\n".join(lines)


def _build_hard_constraints(instance: "VRPTWInstance") -> str:
    """Define hard constraints that MUST be respected."""
    capacity = instance.vehicle_capacity

    return f"""## HARD CONSTRAINTS (ALL HINTS MUST RESPECT THESE)

The VRPTW has the following hard constraints:

1. **Vehicle Capacity**: No route may exceed {capacity} total demand
2. **Time Windows**: Each customer must be served within [ready_time, due_time]
3. **Service Duration**: Each customer requires {instance.service_time} time units
4. **Single Depot**: All routes start and end at depot {instance.depot_id}
5. **No Duplicates**: Each customer appears exactly once in the solution
6. **Valid Customer IDs**: All referenced customers must exist in the instance

### CONSEQUENCES OF VIOLATIONS
- Routes exceeding capacity will be rejected by the verifier
- Customers outside time windows make the solution infeasible
- Invalid customer IDs will cause parsing errors

### ADVISORY: TIGHT WINDOWS
Customers with small time windows (window size < 30) are hardest to serve.
Prioritize good positioning for these customers."""


def _build_incumbent_summary(incumbent: "RouteSolution | None") -> str:
    """Summarize the incumbent solution with diagnostic insights."""
    lines = ["## Current Solution State"]

    if incumbent is None or incumbent.is_empty():
        lines.append("**Status**: No incumbent solution available (cold start).")
        lines.append("**Recommendation**: Focus on building an initial feasible solution.")
        lines.append("  - Prioritize customers with tight time windows")
        lines.append("  - Group spatially close customers together")
        lines.append("  - Avoid overloading any single route")
        return "\n".join(lines)

    lines.append(f"- **Vehicles used**: {incumbent.vehicles_used}")
    lines.append(f"- **Total distance**: {incumbent.total_distance:.2f}")
    lines.append(f"- **Total duration**: {incumbent.total_duration:.2f}")
    lines.append(f"- **Feasibility**: {'✅ Feasible' if incumbent.feasible else '❌ Infeasible'}")

    if incumbent.feasible:
        lines.append(f"- **Late violations**: {incumbent.late_violations}")
        lines.append(f"- **Capacity violations**: {incumbent.capacity_violations}")

    lines.append(f"- **Runtime so far**: {incumbent.runtime_sec:.2f}s")

    # Route details
    if incumbent.routes:
        lines.append("\n### Route Details")
        for i, route in enumerate(incumbent.routes[:_MAX_ROUTES_SHOWN]):
            load = sum(0 for _ in route)  # Simplified
            lines.append(f"  Route {i}: {route[:10]}{'...' if len(route) > 10 else ''} ({len(route)} customers)")

        if len(incumbent.routes) > _MAX_ROUTES_SHOWN:
            lines.append(f"  ... and {len(incumbent.routes) - _MAX_ROUTES_SHOWN} more routes")

    # Diagnostic insights
    lines.append("\n### Solution Insights")
    if incumbent.vehicles_used > 60:
        lines.append("- **Opportunity**: High vehicle count - consider consolidation")
    elif incumbent.vehicles_used < 50:
        lines.append("- **Observation**: Efficient vehicle usage")

    if incumbent.feasible:
        lines.append("- **Status**: Solution is feasible, focus on optimization")
    else:
        lines.append("- **Priority**: Fix infeasibility before optimizing")

    return "\n".join(lines)


def _build_enhanced_json_schema() -> str:
    """Return the JSON schema matching LLMHints model.

    The schema must match LLMHints (src/llm/hint_schema.py) exactly.
    The parser (src/llm/parser.py) handles conversion from LLM's richer output.
    """
    schema = {
        "priority_customers": "list[int] - customer IDs to prioritize in insertion order",
        "locked_subroutes": "list[list[int]] - subroutes to preserve intact (each is a list of customer IDs)",
        "avoid_pairs": "list[list[int]] - pairs of customers to keep on different routes (each is [a, b])",
        "suggested_moves": [
            {
                # relocate:
                "type": "relocate",
                "customer": "int - customer ID to relocate",
                "target_route": "int - target route index (0-based)",
                "target_position": "int|null - position in target route (null = append)"
            },
            {
                # swap:
                "type": "swap",
                "a": "int - first customer ID",
                "b": "int - second customer ID"
            },
            {
                # reverse:
                "type": "reverse",
                "route_index": "int - route index to reverse",
                "start_customer": "int - first customer of segment",
                "end_customer": "int - last customer of segment"
            },
            {
                # split_route:
                "type": "split_route",
                "route_index": "int - route index to split",
                "after_customer": "int - customer after which to split"
            }
        ],
        "solver_control": {
            "extra_seconds": "int|null - additional solver time",
            "intensify": "bool - focus on exploitation",
            "diversify": "bool - focus on exploration"
        },
        "rationale": "string|null - your reasoning for these hints"
    }

    lines = [
        "## Required JSON Output Schema",
        "",
        "Return ONLY valid JSON. No markdown, no explanation outside the JSON.",
        "IMPORTANT: The fields below are the EXACT format the parser expects.",
        "",
        "```json",
        json.dumps(schema, indent=2),
        "```",
        "",
        "### Field Descriptions:",
        "- **priority_customers**: Simple list of customer IDs. Example: [17, 23, 42]",
        "- **locked_subroutes**: List of customer ID lists. Example: [[1, 2, 3], [5, 6]]",
        "- **avoid_pairs**: List of customer ID pairs. Example: [[1, 10], [2, 20]]",
        "- **suggested_moves**: Each move has type-specific fields only (see examples below)",
        "- **solver_control**: Runtime parameter suggestions (all fields optional)",
    ]

    return "\n".join(lines)


def _build_format_instruction() -> str:
    """Final format instruction."""
    return """## FORMAT REQUIREMENTS

1. Output ONLY a valid JSON object - no markdown, no text before or after
2. All arrays default to empty [] if no data
3. All objects default to {} if no data
4. Use null for optional string fields when not applicable
5. All customer IDs must be positive integers
6. Routes are 0-indexed (first route is route 0)
7. The parser will extract only the required fields from each move type

## RESPONSE EXAMPLE

```json
{
  "priority_customers": [17, 23, 42],
  "locked_subroutes": [[1, 2, 3], [5, 6]],
  "avoid_pairs": [[10, 20], [30, 40]],
  "suggested_moves": [
    {"type": "relocate", "customer": 15, "target_route": 0},
    {"type": "swap", "a": 23, "b": 42},
    {"type": "reverse", "route_index": 0, "start_customer": 5, "end_customer": 10}
  ],
  "solver_control": {"extra_seconds": 30, "intensify": true, "diversify": false},
  "rationale": "Prioritized tight-TW customers, preserved good subroutes, swapped distant pairs"
}
```"""


def build_quick_prompt(
    instance: "VRPTWInstance",
    incumbent: "RouteSolution | None" = None,
) -> str:
    """Build a quick prompt without history or diagnostics (for fast iteration).

    Use this for rapid prototyping or when you don't have diagnostic data yet.
    """
    parts = []

    parts.append("## VRPTW Solver Guidance (Quick Mode)")

    parts.append(_build_instance_summary(instance))
    parts.append(_build_hard_constraints(instance))
    parts.append(_build_incumbent_summary(incumbent))
    parts.append(_build_enhanced_json_schema())
    parts.append(_build_format_instruction())

    return "\n\n".join(parts)
