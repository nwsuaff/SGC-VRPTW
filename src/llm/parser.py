"""LLM output parser: parses raw text and validates against the hint schema."""

from __future__ import annotations

import json
import logging

from src.llm.hint_schema import LLMHints, empty_hints

logger = logging.getLogger(__name__)


class LLMParseError(Exception):
    """Raised when LLM output cannot be parsed."""

    def __init__(self, raw_output: str, cause: Exception):
        self.raw_output = raw_output
        self.cause = cause
        super().__init__(f"Failed to parse LLM output: {cause}")


def parse_llm_output(raw_output: str) -> LLMHints:
    """Parse raw LLM text output into an LLMHints object.

    The parser:
    1. Strips markdown code fences if present.
    2. Extracts the JSON object.
    3. Validates against the Pydantic schema.
    4. Returns an empty-hints object on any failure.

    Args:
        raw_output: The raw text returned by the LLM.

    Returns:
        LLMHints, or an empty-hints object if parsing/validation fails.
    """
    if not raw_output or not raw_output.strip():
        logger.warning("Empty LLM output received")
        return empty_hints()

    # Check for HTML responses (API error pages)
    if raw_output.strip().startswith("<!DOCTYPE") or raw_output.strip().startswith("<html"):
        logger.error(f"Received HTML error page instead of JSON. This usually means the API key is invalid or authentication failed.")
        return empty_hints()

    cleaned = _strip_markdown_fences(raw_output)
    cleaned = _strip_any_pre_postamble(cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error: {e}. Raw output: {raw_output[:200]}")
        return empty_hints()

    try:
        return LLMHints.model_validate(data)
    except Exception as e:
        logger.warning(f"Schema validation error: {e}. Attempting partial parse.")
        return _try_partial_parse(data)


def _strip_markdown_fences(text: str) -> str:
    """Remove triple-backtick code fences from text.

    If the text contains no backtick fences, returns the text as-is.
    Otherwise, strips the opening and closing ```fence lines
    but preserves the content between them.
    """
    # Fast path: if no backticks at all, treat as plain JSON
    if "```" not in text:
        return text.strip()

    lines = text.strip().splitlines()
    in_fence = False
    result_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            # Toggle state, then skip the fence marker line itself
            in_fence = not in_fence
            continue
        if in_fence:
            # Only collect lines while inside the fence
            result_lines.append(line)

    return "\n".join(result_lines).strip()


def _strip_any_pre_postamble(text: str) -> str:
    """Remove any non-JSON text before or after the JSON object."""
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace == -1 or last_brace == -1:
        return text

    return text[first_brace : last_brace + 1]


def _try_partial_parse(data: dict) -> LLMHints:
    """Attempt to extract valid hints from malformed data.

    First normalizes the LLM output format to match the schema, then validates.
    """
    # Normalize priority_customers: LLM may return dicts with customer_id, 
    # but we need simple list[int]
    priority_customers = data.get("priority_customers", [])
    if not isinstance(priority_customers, list):
        priority_customers = []
    else:
        # Handle list of dicts: extract customer_id from each dict
        if priority_customers and isinstance(priority_customers[0], dict):
            priority_customers = [
                item["customer_id"] 
                for item in priority_customers 
                if isinstance(item, dict) and "customer_id" in item
            ]

    # Normalize locked_subroutes: LLM may return dicts with customers list,
    # but we need list[list[int]]
    locked_subroutes = data.get("locked_subroutes", [])
    if not isinstance(locked_subroutes, list):
        locked_subroutes = []
    else:
        # Handle list of dicts: extract customers list from each dict
        if locked_subroutes and isinstance(locked_subroutes[0], dict):
            locked_subroutes = [
                item["customers"] 
                for item in locked_subroutes 
                if isinstance(item, dict) and "customers" in item
                and isinstance(item["customers"], list)
            ]
        # Ensure inner elements are lists of ints
        locked_subroutes = [
            [c for c in subroute if isinstance(c, int)] 
            for subroute in locked_subroutes 
            if isinstance(subroute, list)
        ]

    # Normalize avoid_pairs: same as locked_subroutes
    avoid_pairs = data.get("avoid_pairs", [])
    if not isinstance(avoid_pairs, list):
        avoid_pairs = []
    else:
        # Handle list of dicts: extract customer_a/customer_b from each dict
        if avoid_pairs and isinstance(avoid_pairs[0], dict):
            avoid_pairs = [
                [item.get("customer_a"), item.get("customer_b")]
                for item in avoid_pairs
                if isinstance(item, dict) 
                and "customer_a" in item 
                and "customer_b" in item
            ]
            # Filter out pairs with None values
            avoid_pairs = [pair for pair in avoid_pairs if None not in pair]
        # Ensure inner elements are lists of ints
        avoid_pairs = [
            [c for c in pair if isinstance(c, int)]
            for pair in avoid_pairs
            if isinstance(pair, list) and len(pair) >= 2
        ]

    suggested_moves = data.get("suggested_moves", [])
    if not isinstance(suggested_moves, list):
        suggested_moves = []
    else:
        # Normalize suggested_moves: LLM may return dicts with all fields mixed,
        # but each move type only allows specific fields
        normalized_moves = []
        for move in suggested_moves:
            if not isinstance(move, dict):
                continue
            
            move_type = move.get("type")
            if move_type == "relocate":
                # RelocateMove requires customer and target_route to be int
                customer = move.get("customer")
                target_route = move.get("target_route")
                if isinstance(customer, int) and isinstance(target_route, int):
                    normalized_moves.append({
                        "type": "relocate",
                        "customer": customer,
                        "target_route": target_route,
                        "target_position": move.get("target_position"),
                    })
            elif move_type == "swap":
                # SwapMove requires a and b to be int
                a = move.get("a")
                b = move.get("b")
                if isinstance(a, int) and isinstance(b, int):
                    normalized_moves.append({
                        "type": "swap",
                        "a": a,
                        "b": b,
                    })
            elif move_type == "reverse":
                # ReverseMove requires route_index, start_customer, end_customer to be int
                route_index = move.get("route_index")
                start = move.get("start_customer")
                end = move.get("end_customer")
                if isinstance(route_index, int) and isinstance(start, int) and isinstance(end, int):
                    normalized_moves.append({
                        "type": "reverse",
                        "route_index": route_index,
                        "start_customer": start,
                        "end_customer": end,
                    })
            elif move_type == "split_route":
                # SplitRouteMove requires route_index and after_customer to be int
                route_index = move.get("route_index")
                after = move.get("after_customer")
                if isinstance(route_index, int) and isinstance(after, int):
                    normalized_moves.append({
                        "type": "split_route",
                        "route_index": route_index,
                        "after_customer": after,
                    })
        suggested_moves = normalized_moves

    solver_control = data.get("solver_control", {})
    if not isinstance(solver_control, dict):
        solver_control = {}

    rationale = data.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = None

    return LLMHints(
        priority_customers=priority_customers,
        locked_subroutes=locked_subroutes,
        avoid_pairs=avoid_pairs,
        suggested_moves=suggested_moves,
        solver_control=solver_control if solver_control else None,
        rationale=rationale,
    )
