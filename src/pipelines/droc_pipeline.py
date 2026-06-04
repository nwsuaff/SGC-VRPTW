"""SGC-VRPTW code generation pipeline.

This module implements the solver-grounded generative-control workflow for
LLM-guided VRPTW solving through code generation.

Workflow:
1. Load instance and prepare solve parameters
2. Generate solver code via LLM with template + RAG
3. Execute code and validate result
4. Self-debug on errors (up to max_iterations)
5. Return best solution found
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.domain.schema import VRPTWInstance, RouteSolution, ExperimentResult
from src.domain.metrics import lexicographic_compare
from src.solvers.feasibility_checker import check_feasibility
from src.llm.code_generator import CodeGenerator, CodeGenResult
from src.llm.code_executor import ExecutionResult

if TYPE_CHECKING:
    from src.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def run_droc_solver(
    instance: VRPTWInstance,
    llm_client: Any,
    max_iterations: int = 4,
    time_limit: float = 60.0,
    seed: int = 42,
    output_dir: str | Path | None = None,
    enable_self_debug: bool = True,
    enable_rag: bool = True,
    optimal: float | None = None,
    droc_timeout: float | None = None,
    incumbent: RouteSolution | None = None,
) -> tuple[RouteSolution, CodeGenResult, ExperimentResult]:
    """Run the SGC-VRPTW code-generation pipeline.

    Args:
        instance: The VRPTW instance to solve.
        llm_client: LLM client, for example OpenAIClient or MockLLMClient.
        max_iterations: Maximum self-debug iterations.
        time_limit: Maximum total time in seconds.
        seed: Random seed for reproducibility.
        output_dir: Directory to save generated code.
        enable_self_debug: Whether to enable self-debugging.
        enable_rag: Whether to use RAG (not implemented yet).
        optimal: Known optimal value for validation.
        droc_timeout: Hard timeout for the entire SGC run (including all retries).
        incumbent: Warm-start solution (e.g. from baseline solver) injected into the
            generation prompt to guide the LLM toward a known-good solution space.
            The incumbent is also used as the initial search target.

    Returns:
        Tuple of (final_solution, code_gen_result, experiment_result).
    """
    logger.info(f"[SGC] Starting on {instance.name}")
    start_time = time.perf_counter()

    # Per-iteration budget: use the outer droc_timeout as the reference.
    # llm_timeout is per-LLM-call budget; exec_timeout is per-execution budget.
    # The code generator also tracks total elapsed time and stops early if needed.
    effective_timeout = droc_timeout if droc_timeout is not None else time_limit
    # Increased exec_timeout: 90s → 150s. For 800-customer instances, the first attempt
    # was timing out 95% of the time. 150s gives OR-Tools enough room to find an initial
    # feasible solution before the self-debug loop kicks in.
    # exec_timeout: give OR-Tools enough time to find a feasible solution on large instances.
    # 150s was too tight for 800-customer instances — raise to 250s.
    # For small instances (effective_timeout <= 400s) the 0.25 fraction is still respected.
    llm_timeout = min(effective_timeout * 0.5, 180.0)
    exec_timeout = min(effective_timeout * 0.25, 250.0)
    exec_timeout = max(exec_timeout, 90.0)  # at least 90s for large instances
    outer_timeout = effective_timeout * 0.95  # leave 5% buffer for overhead

    # Initialize code generator with timeout parameters
    generator = CodeGenerator(
        llm_client=llm_client,
        max_iterations=max_iterations,
        enable_self_debug=enable_self_debug,
        enable_rag=enable_rag,
        objective="lexicographic",
        exec_timeout=exec_timeout,
        llm_timeout=llm_timeout,
        outer_timeout=outer_timeout,
    )

    # Generate and execute code
    # Warm-start: pass the baseline incumbent solution to guide the LLM.
    # The incumbent gives the LLM a known-good reference point (vehicle count,
    # distance) so it targets a realistic solution space from the first attempt.
    gen_result = generator.generate(
        instance=instance,
        incumbent=incumbent,
        optimal=optimal,
    )

    runtime = time.perf_counter() - start_time

    # Guard: if we somehow exceeded time_limit, log it
    if runtime > time_limit:
        logger.warning(
            f"[SGC] Runtime {runtime:.1f}s exceeded limit {time_limit:.1f}s for {instance.name}"
        )

    # Convert execution result to RouteSolution
    if gen_result.success and gen_result.execution_result:
        solution = _execution_to_solution(
            gen_result.execution_result,
            instance,
            runtime,
        )
    else:
        # Return empty solution on failure
        solution = RouteSolution(
            routes=[],
            vehicles_used=0,
            total_distance=0.0,
            total_duration=0.0,
            feasible=False,
            runtime_sec=runtime,
        )

    # Validate solution
    validation = check_feasibility(instance, solution)

    # Build experiment result
    result = ExperimentResult(
        experiment_id=f"{instance.name}_droc_{int(time.time())}",
        instance_name=instance.name,
        solver="droc",
        vehicles_used=solution.vehicles_used,
        total_distance=solution.total_distance,
        total_duration=solution.total_duration,
        feasible=validation.feasible,
        late_violations=validation.late_violations,
        capacity_violations=validation.capacity_violations,
        runtime_sec=runtime,
        metadata={
            "code_success": gen_result.success,
            "code_iterations": gen_result.iterations,
            "llm_latency_ms": gen_result.llm_latency_ms,
            "error_message": gen_result.error_message,
        },
    )

    # Save output if requested
    if output_dir:
        _save_outputs(Path(output_dir), instance, gen_result, solution, result)

    logger.info(
        f"[SGC] Complete: vehicles={solution.vehicles_used}, "
        f"distance={solution.total_distance:.2f}, feasible={validation.feasible}, "
        f"runtime={runtime:.1f}s"
    )

    return solution, gen_result, result


def _execution_to_solution(
    exec_result: ExecutionResult,
    instance: VRPTWInstance,
    runtime: float,
) -> RouteSolution:
    """Convert execution result to RouteSolution."""
    routes = exec_result.routes or []
    total_distance = exec_result.total_distance or 0.0

    return RouteSolution(
        routes=routes,
        vehicles_used=exec_result.vehicles_used or len(routes),
        total_distance=total_distance,
        total_duration=0.0,  # Not computed in basic execution
        feasible=True,  # Assumed feasible if execution succeeded
        runtime_sec=runtime,
    )


def _save_outputs(
    output_dir: Path,
    instance: VRPTWInstance,
    gen_result: CodeGenResult,
    solution: RouteSolution,
    result: ExperimentResult,
) -> None:
    """Save SGC outputs to disk."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = output_dir / result.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save instance
    with open(exp_dir / "instance.json", "w", encoding="utf-8") as f:
        json.dump(instance.model_dump(), f, indent=2)

    # Save generated code
    if gen_result.code:
        with open(exp_dir / "generated_code.py", "w", encoding="utf-8") as f:
            if gen_result.imports:
                f.write(gen_result.imports + "\n\n")
            f.write(gen_result.code)

    # Save solution
    with open(exp_dir / "solution.json", "w", encoding="utf-8") as f:
        json.dump(solution.model_dump(), f, indent=2)

    # Save experiment result
    with open(exp_dir / "experiment_result.json", "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)

    logger.info(f"[SGC] Results saved to {exp_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation support: compare SGC vs original hint-based approach
# ─────────────────────────────────────────────────────────────────────────────


def run_ablation_comparison(
    instances: list[VRPTWInstance],
    llm_client: Any,
    seeds: list[int] | None = None,
    max_iterations: int = 4,
    time_limit: float = 60.0,
    output_dir: str | Path = "results/droc_comparison",
) -> dict[str, Any]:
    """Compare SGC-style vs hint-based approaches.

    Args:
        instances: List of VRPTW instances.
        llm_client: LLM client.
        seeds: Random seeds for each instance.
        max_iterations: Max self-debug iterations.
        time_limit: Time limit per run.
        output_dir: Output directory.

    Returns:
        Dictionary with comparison results.
    """
    from src.pipelines.run_llm_solver import run_llm_solver

    seeds = seeds or [42]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "droc": [],
        "hints": [],
        "comparison": [],
    }

    for i, instance in enumerate(instances):
        seed = seeds[i % len(seeds)]

        # Run SGC
        logger.info(f"[Ablation] SGC on {instance.name} (seed={seed})")
        droc_solution, droc_result, _ = run_droc_solver(
            instance=instance,
            llm_client=llm_client,
            max_iterations=max_iterations,
            time_limit=time_limit,
            seed=seed,
        )

        # Run hint-based
        logger.info(f"[Ablation] Hints on {instance.name} (seed={seed})")
        _, hint_solution, _, _ = run_llm_solver(
            instance=instance,
            llm_client=llm_client,
            short_budget=1,
            final_budget=time_limit,
            seed=seed,
        )

        # Compare
        droc_validation = check_feasibility(instance, droc_solution)
        hint_validation = check_feasibility(instance, hint_solution)

        comparison = {
            "instance": instance.name,
            "seed": seed,
            "droc_vehicles": droc_solution.vehicles_used,
            "droc_distance": droc_solution.total_distance,
            "droc_feasible": droc_validation.feasible,
            "hints_vehicles": hint_solution.vehicles_used,
            "hints_distance": hint_solution.total_distance,
            "hints_feasible": hint_validation.feasible,
        }

        results["droc"].append(droc_solution)
        results["hints"].append(hint_solution)
        results["comparison"].append(comparison)

    # Print summary
    _print_ablation_summary(results["comparison"])

    return results


def _print_ablation_summary(comparisons: list[dict]) -> None:
    """Print ablation comparison summary."""
    if not comparisons:
        return

    logger.info("=" * 60)
    logger.info("SGC vs Hints Ablation Summary")
    logger.info("=" * 60)

    droc_feas = sum(1 for c in comparisons if c["droc_feasible"])
    hint_feas = sum(1 for c in comparisons if c["hints_feasible"])
    n = len(comparisons)

    logger.info(f"Feasibility Rate: SGC={droc_feas/n*100:.1f}%, Hints={hint_feas/n*100:.1f}%")
    logger.info(f"Avg Vehicles:    SGC={sum(c['droc_vehicles'] for c in comparisons)/n:.2f}, Hints={sum(c['hints_vehicles'] for c in comparisons)/n:.2f}")
    logger.info(f"Avg Distance:    SGC={sum(c['droc_distance'] for c in comparisons)/n:.1f}, Hints={sum(c['hints_distance'] for c in comparisons)/n:.1f}")
    logger.info("=" * 60)
