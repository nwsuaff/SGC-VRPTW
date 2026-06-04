#!/usr/bin/env python
"""Compare LLM-guided solver vs standalone solver on a single instance.

Usage:
    python compare_llm_vs_solver.py [--instance PATH] [--llm mock|openai] [--budget SECONDS]
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_instance(path: str | Path) -> "VRPTWInstance":
    """Load VRPTW instance from JSON file."""
    from src.domain.schema import VRPTWInstance
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return VRPTWInstance(**data)


def run_baseline_solver(
    instance: "VRPTWInstance",
    time_limit: float,
    seed: int = 42,
) -> tuple["RouteSolution", float]:
    """Run standalone solver (PyVRP or OR-Tools fallback)."""
    from src.solvers.pyvrp_solver import solve_pyvrp
    from src.solvers.ortools_solver import solve_ortools
    
    start = time.perf_counter()
    
    try:
        solution, _ = solve_pyvrp(instance, time_limit=time_limit, seed=seed)
        elapsed = time.perf_counter() - start
        logger.info(f"[Baseline] PyVRP solved in {elapsed:.2f}s")
        return solution, elapsed
    except Exception as e:
        logger.warning(f"[Baseline] PyVRP failed: {e}, trying OR-Tools")
        try:
            solution, _ = solve_ortools(instance, time_limit=time_limit)
            elapsed = time.perf_counter() - start
            logger.info(f"[Baseline] OR-Tools solved in {elapsed:.2f}s")
            return solution, elapsed
        except Exception as e2:
            logger.error(f"[Baseline] Both solvers failed: {e2}")
            raise


def run_llm_solver(
    instance: "VRPTWInstance",
    llm_client_type: str,
    time_limit: float,
    seed: int = 42,
) -> tuple["RouteSolution", float]:
    """Run LLM-guided solver pipeline."""
    from src.llm.mock_client import MockLLMClient
    from src.llm.client_factory import create_llm_client
    from src.pipelines.run_llm_solver import run_llm_solver as llm_loop
    
    start = time.perf_counter()
    
    if llm_client_type == "mock":
        llm_client = MockLLMClient(seed=seed)
    elif llm_client_type == "openai":
        llm_client = create_llm_client("openai")
    else:
        raise ValueError(f"Unknown LLM client type: {llm_client_type}")
    
    incumbent, final_solution, hints, result = llm_loop(
        instance=instance,
        llm_client=llm_client,
        short_budget=time_limit / 2,
        final_budget=time_limit / 2,
        seed=seed,
    )
    
    elapsed = time.perf_counter() - start
    logger.info(f"[LLM+Solver] Completed in {elapsed:.2f}s")
    
    return final_solution, elapsed


def main():
    parser = argparse.ArgumentParser(description="Compare LLM+solver vs solver")
    parser.add_argument(
        "--instance", "-i",
        type=str,
        default="data/toy/vrptw_tiny.json",
        help="Path to instance JSON file",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help="LLM client type",
    )
    parser.add_argument(
        "--budget", "-b",
        type=float,
        default=10.0,
        help="Time budget in seconds per solver",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    
    args = parser.parse_args()
    
    # Load instance
    instance_path = Path(args.instance)
    if not instance_path.exists():
        print(f"Error: Instance file not found: {instance_path}")
        return 1
    
    instance = load_instance(instance_path)
    print(f"\n{'='*60}")
    print(f"Instance: {instance.name}")
    print(f"Customers: {instance.size}")
    print(f"Vehicle capacity: {instance.vehicle_capacity}")
    print(f"{'='*60}\n")
    
    # Run baseline solver
    print(f"[1/2] Running standalone solver (budget: {args.budget}s)...")
    try:
        baseline_solution, baseline_time = run_baseline_solver(
            instance, args.budget, args.seed
        )
    except Exception as e:
        print(f"Error: Baseline solver failed: {e}")
        return 1
    
    # Run LLM-guided solver
    print(f"\n[2/2] Running LLM-guided solver (budget: {args.budget}s, LLM: {args.llm})...")
    try:
        llm_solution, llm_time = run_llm_solver(
            instance, args.llm, args.budget, args.seed
        )
    except Exception as e:
        print(f"Error: LLM solver failed: {e}")
        return 1
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Baseline':<15} {'LLM+Solver':<15} {'Improvement':<15}")
    print(f"{'-'*60}")
    print(f"{'Vehicles used':<20} {baseline_solution.vehicles_used:<15} {llm_solution.vehicles_used:<15} {baseline_solution.vehicles_used - llm_solution.vehicles_used:<15}")
    print(f"{'Total distance':<20} {baseline_solution.total_distance:<15.2f} {llm_solution.total_distance:<15.2f} {baseline_solution.total_distance - llm_solution.total_distance:<15.2f}")
    print(f"{'Feasible':<20} {str(baseline_solution.feasible):<15} {str(llm_solution.feasible):<15} {'-'*15}")
    print(f"{'Runtime (s)':<20} {baseline_time:<15.2f} {llm_time:<15.2f} {'-'*15}")
    print(f"{'-'*60}")
    
    # Determine winner
    if llm_solution.feasible and baseline_solution.feasible:
        if llm_solution.vehicles_used < baseline_solution.vehicles_used:
            winner = "LLM+Solver (fewer vehicles)"
        elif llm_solution.total_distance < baseline_solution.total_distance:
            winner = "LLM+Solver (shorter distance)"
        else:
            winner = "No improvement"
    elif llm_solution.feasible and not baseline_solution.feasible:
        winner = "LLM+Solver (found feasible, baseline did not)"
    elif not llm_solution.feasible and baseline_solution.feasible:
        winner = "Baseline (LLM solver returned infeasible)"
    else:
        winner = "Both infeasible"
    
    print(f"\nWinner: {winner}")
    print(f"{'='*60}\n")
    
    # Save results
    results = {
        "instance": instance.name,
        "seed": args.seed,
        "budget": args.budget,
        "llm_client": args.llm,
        "baseline": {
            "vehicles_used": baseline_solution.vehicles_used,
            "total_distance": baseline_solution.total_distance,
            "feasible": baseline_solution.feasible,
            "runtime_sec": baseline_time,
        },
        "llm_solver": {
            "vehicles_used": llm_solution.vehicles_used,
            "total_distance": llm_solution.total_distance,
            "feasible": llm_solution.feasible,
            "runtime_sec": llm_time,
        },
        "winner": winner,
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
