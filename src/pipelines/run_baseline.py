"""Baseline solver pipeline: load instance, run solver, validate, save."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

from src.domain.schema import VRPTWInstance, RouteSolution, ExperimentResult
from src.domain.metrics import compute_gap
from src.baselines.greedy_insertion import solve_greedy
from src.solvers.ortools_solver import solve_ortools
from src.solvers.pyvrp_solver import solve_pyvrp
from src.solvers.feasibility_checker import check_feasibility

logger = logging.getLogger(__name__)

SolverName = Literal["greedy", "ortools", "pyvrp"]


def run_baseline(
    instance: VRPTWInstance,
    solver: SolverName,
    time_limit: int = 5,
    output_dir: str | Path | None = None,
    seed: int = 42,
    bks_path: str | Path | None = None,
) -> tuple[RouteSolution, ExperimentResult]:
    """Run a single baseline solver on an instance.

    Args:
        instance: The VRPTW instance to solve.
        solver: Which solver to use ('greedy', 'ortools', 'pyvrp').
        time_limit: Time limit in seconds for OR-Tools and PyVRP.
        output_dir: Directory to save results. If None, no files are written.
        seed: Random seed (for PyVRP).
        bks_path: Optional path to a BKS CSV file for gap computation.

    Returns:
        Tuple of (RouteSolution, ExperimentResult).
    """
    logger.info(f"Running {solver} on {instance.name}")
    start_time = time.perf_counter()

    if solver == "greedy":
        solution = solve_greedy(instance)
    elif solver == "ortools":
        solution, _ = solve_ortools(instance, time_limit=time_limit)
    elif solver == "pyvrp":
        solution, _ = solve_pyvrp(instance, time_limit=time_limit, seed=seed)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    validation = check_feasibility(instance, solution)

    if not validation.feasible:
        logger.warning(
            f"Solution for {instance.name} is infeasible: "
            f"capacity_violations={validation.capacity_violations}, "
            f"late_violations={validation.late_violations}"
        )

    bks_vehicles, bks_distance = _load_bks(instance.name, bks_path)
    gap_result = compute_gap(solution, bks_distance) if bks_distance else None

    experiment_id = f"{instance.name}_{solver}_{int(time.time())}"
    result = ExperimentResult(
        experiment_id=experiment_id,
        instance_name=instance.name,
        solver=solver,
        vehicles_used=solution.vehicles_used,
        total_distance=solution.total_distance,
        total_duration=solution.total_duration,
        feasible=validation.feasible,
        late_violations=validation.late_violations,
        capacity_violations=validation.capacity_violations,
        runtime_sec=solution.runtime_sec,
        first_feasible_sec=solution.first_feasible_sec,
        gap_to_bks=gap_result if gap_result is not None else None,
        lexicographic_gap=gap_result if gap_result is not None else None,
        metadata={
            **solution.metadata,
            "validation_feasible": validation.feasible,
            "bks_vehicles": bks_vehicles,
            "bks_distance": bks_distance,
        },
    )

    if output_dir:
        _save_results(Path(output_dir), instance, solution, result)

    return solution, result


def _load_bks(instance_name: str, bks_path: str | Path | None) -> tuple[int | None, float | None]:
    """Load best-known solution for an instance from a CSV file."""
    if bks_path is None:
        return None, None

    try:
        import pandas as pd
        df = pd.read_csv(bks_path)
        row = df[df["instance_name"] == instance_name]
        if not row.empty:
            bks_v = int(row.iloc[0]["best_vehicles"]) if "best_vehicles" in row.columns else None
            bks_d = float(row.iloc[0]["best_distance"]) if "best_distance" in row.columns else None
            return bks_v, bks_d
    except Exception as e:
        logger.debug(f"Could not load BKS: {e}")

    return None, None


def _save_results(
    output_dir: Path,
    instance: VRPTWInstance,
    solution: RouteSolution,
    result: ExperimentResult,
) -> None:
    """Save solution and experiment result to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = output_dir / result.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    instance_path = exp_dir / "instance.json"
    with open(instance_path, "w", encoding="utf-8") as f:
        json.dump(instance.model_dump(), f, indent=2)

    solution_path = exp_dir / "solution.json"
    with open(solution_path, "w", encoding="utf-8") as f:
        json.dump(solution.model_dump(), f, indent=2)

    result_path = exp_dir / "result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)

    summary_csv = output_dir / "summary.csv"
    _append_summary_csv(summary_csv, result)
    logger.info(f"Results saved to {exp_dir}")


def _append_summary_csv(csv_path: Path, result: ExperimentResult) -> None:
    """Append a single row to the summary CSV."""
    import csv

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_id", "instance_name", "solver", "vehicles_used",
                "total_distance", "feasible", "runtime_sec", "gap_to_bks",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_id": result.experiment_id,
            "instance_name": result.instance_name,
            "solver": result.solver,
            "vehicles_used": result.vehicles_used,
            "total_distance": f"{result.total_distance:.2f}",
            "feasible": result.feasible,
            "runtime_sec": f"{result.runtime_sec:.3f}",
            "gap_to_bks": f"{result.gap_to_bks:.2f}" if result.gap_to_bks is not None else "",
        })
