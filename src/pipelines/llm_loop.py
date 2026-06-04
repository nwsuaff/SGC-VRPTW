"""LLM Loop - Standalone LLM-guided solver loop pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.domain.schema import VRPTWInstance, RouteSolution, ExperimentResult
from src.domain.metrics import lexicographic_compare
from src.solvers.feasibility_checker import check_feasibility
from src.llm.client_factory import create_llm_client
from src.llm.prompt_builder import build_llm_prompt
from src.llm.hint_to_warmstart import hints_to_warmstart
from src.llm.diagnostic_encoder import compute_diagnostics
from src.llm.hint_verifier import HintVerifier, verify_hints
from src.llm.hint_history import HintHistory, HintHistoryManager
from src.domain.event_schema import Phase

logger = logging.getLogger(__name__)


def run_llm_loop(
    instance: VRPTWInstance,
    llm_client: Any = None,
    short_budget: int = 1,
    final_budget: int = 30,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[RouteSolution, RouteSolution | None, Any, ExperimentResult]:
    """Run the LLM-guided solver loop on a single instance.
    
    This is a simplified version of run_llm_solver without event logging.
    
    Args:
        instance: The VRPTW instance to solve.
        llm_client: LLM client (defaults to MockLLMClient).
        short_budget: Time limit in seconds for initial solve.
        final_budget: Time limit in seconds for final solve.
        seed: Random seed.
        output_dir: Directory to save results.
    
    Returns:
        Tuple of (incumbent_solution, final_solution, llm_hints, experiment_result).
    """
    from src.solvers.pyvrp_solver import solve_pyvrp
    from src.solvers.ortools_solver import solve_ortools
    
    logger.info(f"[LLM Loop] Starting on {instance.name}")
    start_time = time.perf_counter()
    
    if llm_client is None:
        llm_client = create_llm_client("mock")
    
    # Phase 1: Initial solve with PyVRP
    incumbent, _ = solve_pyvrp(instance, time_limit=short_budget, seed=seed)
    logger.info(
        f"[LLM Loop] Initial: vehicles={incumbent.vehicles_used}, "
        f"distance={incumbent.total_distance:.2f}"
    )
    
    # Phase 2: Compute diagnostics and build prompt
    diagnostics = compute_diagnostics(instance, incumbent)
    prompt = build_llm_prompt(instance, incumbent)
    
    # Phase 3: Query LLM
    try:
        hints = llm_client.query(instance, incumbent)
    except Exception as e:
        logger.warning(f"[LLM Loop] LLM query failed: {e}")
        from src.llm.hint_schema import empty_hints
        hints = empty_hints()
    
    # Phase 4: Transform hints to warm-start
    warmstart = hints_to_warmstart(instance, hints, incumbent)
    
    # Phase 5: Final solve
    if warmstart.initial_routes:
        logger.info(f"[LLM Loop] Applying warm-start with {len(warmstart.initial_routes)} routes")
        final, meta = solve_ortools(
            instance,
            time_limit=final_budget,
            warm_start=warmstart.initial_routes,
            avoid_pairs=warmstart.avoid_pairs if warmstart.avoid_pairs else None,
        )
    else:
        logger.info("[LLM Loop] No warm-start, using incumbent")
        final = incumbent
    
    runtime = time.perf_counter() - start_time
    validation = check_feasibility(instance, final)
    
    result = ExperimentResult(
        experiment_id=f"{instance.name}_llm_loop_{int(time.time())}",
        instance_name=instance.name,
        solver="llm_loop",
        vehicles_used=final.vehicles_used,
        total_distance=final.total_distance,
        total_duration=final.total_duration,
        feasible=validation.feasible,
        late_violations=validation.late_violations,
        capacity_violations=validation.capacity_violations,
        runtime_sec=runtime,
        first_feasible_sec=incumbent.first_feasible_sec,
        metadata={
            "initial_vehicles": incumbent.vehicles_used,
            "initial_distance": incumbent.total_distance,
            "comparison": "improved" if lexicographic_compare(final, incumbent) < 0 else "no_improvement",
            "seed": seed,
        },
    )
    
    if output_dir:
        _save_outputs(Path(output_dir), instance, incumbent, final, hints, result)
    
    return incumbent, final, hints, result


def _save_outputs(
    output_dir: Path,
    instance: VRPTWInstance,
    incumbent: RouteSolution,
    final: RouteSolution,
    hints: Any,
    result: ExperimentResult,
) -> None:
    """Save LLM loop outputs to disk."""
    import json
    
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = output_dir / result.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "instance.json", "w", encoding="utf-8") as f:
        json.dump(instance.model_dump(), f, indent=2)
    with open(exp_dir / "initial_solution.json", "w", encoding="utf-8") as f:
        json.dump(incumbent.model_dump(), f, indent=2)
    with open(exp_dir / "final_solution.json", "w", encoding="utf-8") as f:
        json.dump(final.model_dump(), f, indent=2)
    with open(exp_dir / "experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    
    logger.info(f"[LLM Loop] Results saved to {exp_dir}")
