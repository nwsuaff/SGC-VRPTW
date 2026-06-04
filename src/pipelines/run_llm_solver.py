"""LLM-guided solver pipeline: the core research pipeline.

The loop:
1. Load instance
2. Run PyVRP with a short budget to get an incumbent
3. Validate and summarize incumbent
4. Build prompt
5. Query MockLLMClient
6. Parse JSON hints
7. Transform hints into warm-start / ordering / repair suggestions
8. Run PyVRP again with warm-start (or fallback)
9. Validate final solution
10. Save all outputs
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Type

from src.domain.schema import VRPTWInstance, RouteSolution, ExperimentResult
from src.domain.metrics import lexicographic_compare
from src.solvers.pyvrp_solver import solve_pyvrp
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.llm.client_factory import create_llm_client
from src.llm.prompt_builder import build_llm_prompt
from src.llm.hint_to_warmstart import hints_to_warmstart
from src.domain.event_schema import FailureMode, Improvement, Phase, WarmstartType
from src.llm.parser import parse_llm_output

if TYPE_CHECKING:
    from src.llm.hint_schema import LLMHints
    from src.utils.event_logger import EventLogger
    from src.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)

# Default LLM client used when none is passed in
_DEFAULT_LLM_NAME = "mock"
_DEFAULT_LLM_MODEL = "gpt-4o-mini"


def _dump_model(obj):
    """Return a JSON-serializable representation for Pydantic or dataclass objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if is_dataclass(obj):
        return asdict(obj)
    return obj


def run_llm_solver(
    instance: VRPTWInstance,
    llm_client=None,
    short_budget: int = 1,
    final_budget: int = 30,
    seed: int = 42,
    output_dir: str | Path | None = None,
    event_logger: "EventLogger | None" = None,
) -> tuple[RouteSolution, RouteSolution | None, "LLMHints", ExperimentResult]:
    """Run the full LLM-guided solver pipeline on an instance.

    Args:
        instance: The VRPTW instance to solve.
        llm_client: LLM client (defaults to MockLLMClient).
        short_budget: Time limit in seconds for initial PyVRP solve.
        final_budget: Time limit in seconds for final PyVRP solve.
        seed: Random seed for reproducibility.
        output_dir: Directory to save results.

    Returns:
        Tuple of (incumbent_solution, final_solution, llm_hints, experiment_result).
        final_solution may be None if the loop fails.
    """
    logger.info(f"[LLM Loop] Starting on {instance.name}")
    start_time = time.perf_counter()

    if llm_client is None:
        llm_client = create_llm_client(name=_DEFAULT_LLM_NAME, model=_DEFAULT_LLM_MODEL)
        logger.info(f"[LLM Loop] Created default client: {type(llm_client).__name__} ({llm_client.model})")

    experiment_id = f"{instance.name}_llm_loop_{int(time.time())}"

    if event_logger:
        event_logger.log_phase_start(
            Phase.INITIAL_SOLVE,
            instance_name=instance.name,
            family=getattr(instance, "family", None),
            size=getattr(instance, "size", 0),
            seed=seed,
            experiment_id=experiment_id,
        )

    incumbent, initial_result = _phase1_initial_solve(
        instance, short_budget, seed
    )
    logger.info(
        f"[LLM Loop] Initial solve: vehicles={incumbent.vehicles_used}, "
        f"distance={incumbent.total_distance:.2f}, feasible={incumbent.feasible}"
    )

    prompt = _phase2_build_prompt(instance, incumbent)
    logger.debug(f"[LLM Loop] Built prompt ({len(prompt)} chars)")

    hints = _phase3_query_llm(llm_client, instance, incumbent)
    logger.info(
        f"[LLM Loop] Hints: {len(hints.priority_customers)} priority, "
        f"{len(hints.locked_subroutes)} locked, "
        f"{len(hints.suggested_moves)} moves"
    )

    warmstart = _phase4_transform_hints(instance, hints, incumbent)
    extra_time = warmstart.extra_seconds or 0
    adjusted_budget = final_budget + extra_time

    final_solution = _phase5_final_solve(
        instance,
        adjusted_budget,
        seed,
        warmstart,
        incumbent,
    )
    logger.info(
        f"[LLM Loop] Final solve: vehicles={final_solution.vehicles_used}, "
        f"distance={final_solution.total_distance:.2f}, feasible={final_solution.feasible}"
    )

    comparison = "improved" if lexicographic_compare(final_solution, incumbent) < 0 else "no_improvement"

    experiment_result = _phase6_save(
        instance,
        incumbent,
        final_solution,
        hints,
        comparison,
        start_time,
        output_dir,
        seed,
    )

    return incumbent, final_solution, hints, experiment_result


def _phase1_initial_solve(
    instance: VRPTWInstance,
    short_budget: int,
    seed: int,
) -> tuple[RouteSolution, ExperimentResult]:
    """Phase 1: Run PyVRP with a short time budget to get an incumbent."""
    solution, meta = solve_pyvrp(instance, time_limit=short_budget, seed=seed)
    validation = check_feasibility(instance, solution)

    result = ExperimentResult(
        experiment_id=f"{instance.name}_llm_initial_{int(time.time())}",
        instance_name=instance.name,
        solver="pyvrp",
        vehicles_used=solution.vehicles_used,
        total_distance=solution.total_distance,
        total_duration=solution.total_duration,
        feasible=validation.feasible,
        late_violations=validation.late_violations,
        capacity_violations=validation.capacity_violations,
        runtime_sec=solution.runtime_sec,
        first_feasible_sec=solution.first_feasible_sec,
        metadata={"phase": "initial", "time_limit": short_budget, "seed": seed},
    )

    return solution, result


def _phase2_build_prompt(
    instance: VRPTWInstance,
    incumbent: RouteSolution,
) -> str:
    """Phase 2: Build the LLM prompt."""
    return build_llm_prompt(instance, incumbent)


def _phase3_query_llm(
    llm_client: LLMClientType,
    instance: VRPTWInstance,
    incumbent: RouteSolution,
) -> "LLMHints":
    """Phase 3: Query the LLM and parse its response."""
    try:
        response = llm_client.query(instance, incumbent)
        return response
    except Exception as e:
        logger.warning(f"[LLM Loop] LLM query failed: {e}. Using empty hints.")
        from src.llm.hint_schema import empty_hints
        return empty_hints()


def _phase4_transform_hints(
    instance: VRPTWInstance,
    hints: "LLMHints",
    incumbent: RouteSolution,
) -> WarmStartCandidate:
    """Phase 4: Transform LLM hints into a warm-start candidate."""
    return hints_to_warmstart(instance, hints, incumbent)


def _phase5_final_solve(
    instance: VRPTWInstance,
    final_budget: int,
    seed: int,
    warmstart: WarmStartCandidate,
    incumbent: RouteSolution,
) -> RouteSolution:
    """Phase 5: Run OR-Tools with warm-start, or fall back to incumbent.

    Uses OR-Tools because it supports warm-start injection via SetFirstSolutionHint.
    OR-Tools does NOT support insertion order (biased_order) or strict avoid_pairs
    constraints. These hints are logged for debugging but do not affect solver behavior.

    Supported features:
    - initial_routes: Warm-start fragments (fully supported)
    - avoid_pairs: Advisory only (logged, not enforced)

    Unsupported features (debug only):
    - priority_customers: Converted to biased_order but ignored by OR-Tools
    """
    # Priority 1: warm-start routes (OR-Tools supports this!)
    if warmstart.initial_routes:
        logger.info(f"[LLM Loop] Applying warm-start with {len(warmstart.initial_routes)} routes")

        solution, meta = solve_ortools(
            instance,
            time_limit=final_budget,
            warm_start=warmstart.initial_routes,
            avoid_pairs=warmstart.avoid_pairs if warmstart.avoid_pairs else None,
        )
        warm_start_applied = meta.get("warm_start_applied", False)
        logger.info(f"[LLM Loop] Warm-start applied: {warm_start_applied}")

        validation = check_feasibility(instance, solution)
        if validation.feasible:
            return solution
        logger.warning(
            f"[LLM Loop] Warm-start solution infeasible. "
            f"Using incumbent as fallback."
        )
        return incumbent

    # Priority 2: nothing available, use incumbent
    logger.info("[LLM Loop] No warm-start routes available. Using incumbent.")
    return incumbent


def _phase6_save(
    instance: VRPTWInstance,
    incumbent: RouteSolution,
    final_solution: RouteSolution,
    hints: "LLMHints",
    comparison: str,
    start_time: float,
    output_dir: Path | None,
    seed: int,
) -> ExperimentResult:
    """Phase 6: Validate, build ExperimentResult, and save outputs."""
    runtime = time.perf_counter() - start_time
    validation = check_feasibility(instance, final_solution)

    experiment_id = f"{instance.name}_llm_loop_{int(time.time())}"

    result = ExperimentResult(
        experiment_id=experiment_id,
        instance_name=instance.name,
        solver="llm_loop",
        vehicles_used=final_solution.vehicles_used,
        total_distance=final_solution.total_distance,
        total_duration=final_solution.total_duration,
        feasible=validation.feasible,
        late_violations=validation.late_violations,
        capacity_violations=validation.capacity_violations,
        runtime_sec=runtime,
        first_feasible_sec=incumbent.first_feasible_sec,
        metadata={
            "initial_vehicles": incumbent.vehicles_used,
            "initial_distance": incumbent.total_distance,
            "comparison": comparison,
            "llm_hints": _dump_model(hints),
            "seed": seed,
        },
    )

    if output_dir:
        _save_llm_outputs(Path(output_dir), experiment_id, instance, incumbent, final_solution, hints, result)

    return result


def _save_llm_outputs(
    output_dir: Path,
    experiment_id: str,
    instance: VRPTWInstance,
    incumbent: RouteSolution,
    final_solution: RouteSolution,
    hints: "LLMHints",
    result: ExperimentResult,
) -> None:
    """Save all LLM loop outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = output_dir / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "instance.json", "w", encoding="utf-8") as f:
        json.dump(_dump_model(instance), f, indent=2)

    with open(exp_dir / "initial_solution.json", "w", encoding="utf-8") as f:
        json.dump(_dump_model(incumbent), f, indent=2)

    with open(exp_dir / "llm_hints.json", "w", encoding="utf-8") as f:
        json.dump(_dump_model(hints), f, indent=2)

    with open(exp_dir / "final_solution.json", "w", encoding="utf-8") as f:
        json.dump(_dump_model(final_solution), f, indent=2)

    with open(exp_dir / "experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(_dump_model(result), f, indent=2)

    _append_llm_summary_csv(output_dir / "llm_summary.csv", result)

    logger.info(f"[LLM Loop] Results saved to {exp_dir}")


def _append_llm_summary_csv(csv_path: Path, result: ExperimentResult) -> None:
    """Append a row to the LLM loop summary CSV."""
    import csv

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id", "instance_name", "vehicles_used", "total_distance",
                "feasible", "initial_vehicles", "initial_distance", "comparison", "runtime_sec",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_id": result.experiment_id,
            "instance_name": result.instance_name,
            "vehicles_used": result.vehicles_used,
            "total_distance": f"{result.total_distance:.2f}",
            "feasible": result.feasible,
            "initial_vehicles": result.metadata.get("initial_vehicles", ""),
            "initial_distance": f"{result.metadata.get('initial_distance', ''):.2f}",
            "comparison": result.metadata.get("comparison", ""),
            "runtime_sec": f"{result.runtime_sec:.3f}",
        })
