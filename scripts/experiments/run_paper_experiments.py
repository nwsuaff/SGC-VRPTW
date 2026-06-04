"""
完整的论文实验脚本 - 使用真实 LLM API

支持:
1. RQ1: Solomon 基准对比
2. RQ2: 规模泛化 (Size Shift)
3. RQ3: 时间窗口泛化 (Family Shift)
4. RQ4: Few-shot 迁移实验 & 消融分析

Usage:
    python run_paper_experiments.py --all
    python run_paper_experiments.py --rq1
    python run_paper_experiments.py --rq4
"""

import argparse
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.domain.schema import VRPTWInstance
from src.domain.metrics import lexicographic_compare
from src.domain.event_schema import FailureMode
from src.solvers.feasibility_checker import check_feasibility
from src.solvers.pyvrp_solver import solve_pyvrp
from src.solvers.ortools_solver import solve_ortools
from src.llm.client_factory import create_llm_client
from src.llm.hint_to_warmstart import hints_to_warmstart
from src.llm.hint_schema import LLMHints, SolverControl


@dataclass
class ExperimentConfig:
    llm_provider: str = "openai"
    llm_model: str = "gpt-5.4"
    short_budget: int = 3
    final_budget: int = 5
    baseline_time: int = 5
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    max_workers: int = 2
    output_dir: Path = Path("results")
    track_llm_cost: bool = True


def load_solomon_instance(name: str) -> VRPTWInstance:
    from src.data.solomon_loader import load_solomon_instance as _load
    path = Path(f"data/VRPTW/Solomon/{name}.vrp")
    inst = _load(str(path))
    inst.name = name
    return inst


def load_homberger_instance(name: str, size: int) -> VRPTWInstance:
    from src.data.homberger_loader import load_homberger_instance as _load
    folder = f"GH{size}"
    path = Path(f"data/VRPTW/{folder}/{name}.vrp")
    inst = _load(str(path))
    inst.name = name
    return inst


def get_instance_family(name: str) -> str:
    import re
    if name.startswith("C1") or name.startswith("C2"):
        return "C" + name[1]
    elif name.startswith("R1") or name.startswith("R2"):
        return "R" + name[1]
    elif name.startswith("RC1") or name.startswith("RC2"):
        return "RC" + name[1]
    else:
        m = re.match(r"^([A-Z]+[12])", name)
        if m:
            return m.group(1)
        return "Unknown"


def _normalize_priority(priority: list) -> list[int]:
    """Normalize priority_customers to list[int] format."""
    if not priority:
        return []
    if priority and isinstance(priority[0], dict):
        return [item["customer_id"] for item in priority if isinstance(item, dict) and "customer_id" in item]
    return [c for c in priority if isinstance(c, int)]


def _normalize_locked(locked: list) -> list[list[int]]:
    """Normalize locked_subroutes to list[list[int]] format."""
    if not locked:
        return []
    if locked and isinstance(locked[0], dict):
        return [item["customers"] for item in locked if isinstance(item, dict) and "customers" in item]
    return [subroute for subroute in locked if isinstance(subroute, list)]


def run_baseline_solver(
    instance: VRPTWInstance,
    time_limit: int = 5,
    seed: int = 42,
    method: Literal["pyvrp", "ortools"] = "ortools",
) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance": instance.name,
        "family": get_instance_family(instance.name),
        "seed": seed,
        "method": method,
    }
    try:
        if method == "pyvrp":
            sol, meta = solve_pyvrp(instance, time_limit=time_limit, seed=seed)
        else:
            sol, meta = solve_ortools(instance, time_limit=time_limit)
        validation = check_feasibility(instance, sol)
        result["vehicles_used"] = sol.vehicles_used
        result["total_distance"] = round(sol.total_distance, 2)
        result["feasible"] = validation.feasible
        result["runtime"] = time.perf_counter() - t0
    except Exception as e:
        logger.error(f"Baseline {method} failed: {e}")
        result["vehicles_used"] = 999
        result["total_distance"] = 999999
        result["feasible"] = False
        result["failure_mode"] = str(e)
    return result


def run_llm_pipeline(
    instance: VRPTWInstance,
    llm_client,
    short_budget: int = 3,
    final_budget: int = 5,
    seed: int = 42,
) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance": instance.name,
        "family": get_instance_family(instance.name),
        "seed": seed,
        "llm_latency_ms": None,
        "warmstart_type": "none",
        "improvement": False,
        "failure_mode": FailureMode.NONE.value,
    }

    try:
        incumbent, _ = solve_pyvrp(instance, time_limit=short_budget, seed=seed)
        incumbent_validation = check_feasibility(instance, incumbent)
        result["initial_vehicles"] = incumbent.vehicles_used
        result["initial_distance"] = round(incumbent.total_distance, 2)
        result["initial_feasible"] = incumbent_validation.feasible
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        result["failure_mode"] = FailureMode.SOLVER_ERROR.value
        return result

    phase2_t0 = time.perf_counter()
    try:
        hints = llm_client.query(instance, incumbent)
        result["llm_latency_ms"] = (time.perf_counter() - phase2_t0) * 1000
        result["n_priority"] = len(hints.priority_customers) if hints.priority_customers else 0
        result["n_locked"] = len(hints.locked_subroutes) if hints.locked_subroutes else 0
        result["n_avoid"] = len(hints.avoid_pairs) if hints.avoid_pairs else 0
        result["n_moves"] = len(hints.suggested_moves) if hints.suggested_moves else 0
    except Exception as e:
        logger.error(f"Phase 2 LLM query failed: {e}")
        result["failure_mode"] = FailureMode.LLM_ERROR.value
        return result

    warmstart = hints_to_warmstart(instance, hints, incumbent)
    result["warmstart_type"] = "routes" if warmstart.initial_routes else "none"

    try:
        if warmstart.initial_routes:
            final, meta = solve_ortools(
                instance,
                time_limit=final_budget,
                warm_start=warmstart.initial_routes,
                avoid_pairs=warmstart.avoid_pairs if warmstart.avoid_pairs else None,
            )
        else:
            final, meta = solve_ortools(instance, time_limit=final_budget)

        final_validation = check_feasibility(instance, final)
        result["vehicles_used"] = final.vehicles_used
        result["total_distance"] = round(final.total_distance, 2)
        result["feasible"] = final_validation.feasible
        result["warm_start_applied"] = meta.get("warm_start_applied", False)
    except Exception as e:
        logger.error(f"Phase 4 OR-Tools failed: {e}")
        result["failure_mode"] = FailureMode.SOLVER_ERROR.value
        return result

    result["total_runtime"] = time.perf_counter() - t0

    if result["feasible"] and incumbent_validation.feasible:
        cmp = lexicographic_compare(final, incumbent)
        result["improvement"] = cmp < 0
    elif result["feasible"] and not incumbent_validation.feasible:
        result["improvement"] = True
    else:
        result["improvement"] = False

    return result


def run_ablation_pipeline(
    instance: VRPTWInstance,
    llm_client,
    short_budget: int = 3,
    final_budget: int = 30,
    seed: int = 42,
    hint_type: str = "all",
    debug: bool = False,
) -> dict:
    t0 = time.perf_counter()
    result = {
        "instance": instance.name,
        "family": get_instance_family(instance.name),
        "seed": seed,
        "hint_type": hint_type,
        "llm_latency_ms": None,
        "warmstart_type": "none",
        "improvement": False,
        "failure_mode": FailureMode.NONE.value,
        "n_priority": 0,
        "n_locked": 0,
        "n_avoid": 0,
        "n_moves": 0,
    }

    try:
        incumbent, _ = solve_pyvrp(instance, time_limit=short_budget, seed=seed)
        incumbent_validation = check_feasibility(instance, incumbent)
        result["initial_vehicles"] = incumbent.vehicles_used
        result["initial_distance"] = round(incumbent.total_distance, 2)
        result["initial_feasible"] = incumbent_validation.feasible
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}")
        result["failure_mode"] = FailureMode.SOLVER_ERROR.value
        return result

    hints = None
    if hint_type != "none":
        phase2_t0 = time.perf_counter()
        try:
            hints = llm_client.query(instance, incumbent)
            result["llm_latency_ms"] = (time.perf_counter() - phase2_t0) * 1000

            # Extract counts from original hints (before filtering)
            orig_priority = hints.priority_customers or []
            orig_locked = hints.locked_subroutes or []
            orig_avoid = hints.avoid_pairs or []
            orig_moves = hints.suggested_moves or []

            # Normalize to simple formats (LLM may return dicts)
            priority_list = _normalize_priority(orig_priority)
            locked_list = _normalize_locked(orig_locked)

            result["n_priority"] = len(priority_list)
            result["n_locked"] = len(locked_list)
            result["n_avoid"] = len(orig_avoid)
            result["n_moves"] = len(orig_moves)

            # Filter hints based on ablation type
            if hint_type == "priority":
                hints = LLMHints(
                    priority_customers=priority_list,
                    locked_subroutes=[],
                    avoid_pairs=[],
                    suggested_moves=[],
                )
            elif hint_type == "locked":
                hints = LLMHints(
                    priority_customers=[],
                    locked_subroutes=locked_list,
                    avoid_pairs=[],
                    suggested_moves=[],
                )
                if debug:
                    logger.info(f"[DEBUG] locked_list: {locked_list}")
                    logger.info(f"[DEBUG] incumbent.routes: {incumbent.routes}")
                    logger.info(f"[DEBUG] instance.customer_ids: {instance.customer_ids}")
            elif hint_type == "moves":
                hints = LLMHints(
                    priority_customers=[],
                    locked_subroutes=[],
                    avoid_pairs=[],
                    suggested_moves=orig_moves,
                )
            # hint_type == "all" keeps original hints unchanged
        except Exception as e:
            logger.error(f"Phase 2 LLM query failed: {e}")
            result["failure_mode"] = FailureMode.LLM_ERROR.value
            return result

    warmstart = hints_to_warmstart(instance, hints, incumbent) if hints else None
    result["warmstart_type"] = "routes" if (warmstart and warmstart.initial_routes) else "none"

    try:
        if warmstart and warmstart.initial_routes:
            final, meta = solve_ortools(
                instance,
                time_limit=final_budget,
                warm_start=warmstart.initial_routes,
                avoid_pairs=warmstart.avoid_pairs if warmstart.avoid_pairs else None,
            )
        else:
            final, meta = solve_ortools(instance, time_limit=final_budget)

        final_validation = check_feasibility(instance, final)
        result["vehicles_used"] = final.vehicles_used
        result["total_distance"] = round(final.total_distance, 2)
        result["feasible"] = final_validation.feasible
        result["warm_start_applied"] = meta.get("warm_start_applied", False)
    except Exception as e:
        logger.error(f"Phase 4 OR-Tools failed: {e}")
        result["failure_mode"] = FailureMode.SOLVER_ERROR.value
        return result

    result["total_runtime"] = time.perf_counter() - t0

    if result["feasible"] and incumbent_validation.feasible:
        cmp = lexicographic_compare(final, incumbent)
        result["improvement"] = cmp < 0
    elif result["feasible"] and not incumbent_validation.feasible:
        result["improvement"] = True
    else:
        result["improvement"] = False

    return result


def run_rq1_solomon_benchmark(config: ExperimentConfig) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("RQ1: Solomon Benchmark Comparison")
    logger.info("=" * 60)

    output_dir = config.output_dir / "rq1_solomon"
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = ["C101", "C102", "C103", "C201", "C202", "C203",
                 "R101", "R102", "R103", "R201", "R202", "R203",
                 "RC101", "RC102", "RC103", "RC201", "RC202", "RC203"]

    rows = []
    total_runs = len(instances) * (1 + len(config.seeds))

    llm_client = create_llm_client(config.llm_provider, config.llm_model)

    run_idx = 0
    for inst_name in instances:
        inst = load_solomon_instance(inst_name)
        for seed in config.seeds:
            run_idx += 1
            logger.info(f"[{run_idx}/{total_runs}] {inst_name} seed={seed} - baseline")
            baseline = run_baseline_solver(inst, config.baseline_time, seed, "ortools")
            baseline["variant"] = "baseline"
            rows.append(baseline)

            logger.info(f"[{run_idx}/{total_runs}] {inst_name} seed={seed} - LLM")
            llm_result = run_llm_pipeline(inst, llm_client, config.short_budget, config.final_budget, seed)
            llm_result["variant"] = "llm_guided"
            rows.append(llm_result)

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"rq1_solomon_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    logger.info("\n" + "=" * 70)
    logger.info("RQ1 SOLOMON BENCHMARK SUMMARY")
    logger.info("=" * 70)

    summary = df.groupby(["variant", "family"]).agg({
        "vehicles_used": "mean",
        "total_distance": "mean",
        "feasible": "mean",
    }).round(2)
    logger.info("\n" + summary.to_string())

    return df


def run_rq4_ablation(config: ExperimentConfig) -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("RQ4: Ablation Analysis")
    logger.info("=" * 60)

    output_dir = config.output_dir / "rq4_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    test_instances = [
        load_solomon_instance(name)
        for name in ["C201", "C202", "R201", "R202", "RC201", "RC202"]
    ]

    variants = [
        ("zero_shot", "none"),
        ("priority_only", "priority"),
        ("locked_only", "locked"),
        ("moves_only", "moves"),
        ("full_hints", "all"),
    ]

    rows = []
    total_runs = len(test_instances) * len(variants) * len(config.seeds)

    llm_client = create_llm_client(config.llm_provider, config.llm_model)

    run_idx = 0
    for variant_name, hint_type in variants:
        for inst in test_instances:
            for seed in config.seeds:
                run_idx += 1
                logger.info(f"[{run_idx}/{total_runs}] {inst.name} variant={variant_name} seed={seed}")

                result = run_ablation_pipeline(
                    inst, llm_client,
                    config.short_budget, config.final_budget, seed,
                    hint_type=hint_type
                )
                result["variant"] = variant_name
                rows.append(result)

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"rq4_ablation_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    logger.info("\n" + "=" * 70)
    logger.info("RQ4 ABLATION SUMMARY")
    logger.info("=" * 70)

    summary = df.groupby("variant").agg({
        "vehicles_used": ["mean", "std"],
        "total_distance": ["mean", "std"],
        "feasible": "mean",
        "improvement": "mean",
        "llm_latency_ms": "mean",
    }).round(3)
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["feasible_rate"] = (summary["feasible_mean"] * 100).round(1)
    print("\n" + summary.to_string())

    logger.info("\n--- Pairwise Comparison (vs zero_shot) ---")
    if "zero_shot" in summary.index:
        base = summary.loc["zero_shot"]
        for v in ["priority_only", "locked_only", "moves_only", "full_hints"]:
            if v in summary.index:
                curr = summary.loc[v]
                logger.info(f"\n{v} vs zero_shot:")
                logger.info(f"  Vehicles: {curr['vehicles_used_mean']:.2f} vs {base['vehicles_used_mean']:.2f} ({curr['vehicles_used_mean']-base['vehicles_used_mean']:+.2f})")
                logger.info(f"  Distance: {curr['total_distance_mean']:.2f} vs {base['total_distance_mean']:.2f} ({curr['total_distance_mean']-base['total_distance_mean']:+.2f})")
                logger.info(f"  Feasible: {curr['feasible_rate']:.1f}% vs {base['feasible_rate']:.1f}% ({curr['feasible_rate']-base['feasible_rate']:+.1f}%)")

    logger.info("\n" + "=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(description="Run paper experiments")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--rq1", action="store_true", help="Run RQ1")
    parser.add_argument("--rq4", action="store_true", help="Run RQ4")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--output-dir", default="results", help="Output directory")

    args = parser.parse_args()

    config = ExperimentConfig(
        seeds=[int(s) for s in args.seeds.split(",")],
        output_dir=Path(args.output_dir),
    )

    logger.info(f"Configuration: {config}")

    if args.all or args.rq1:
        run_rq1_solomon_benchmark(config)

    if args.all or args.rq4:
        run_rq4_ablation(config)

    logger.info("\nAll experiments complete!")
    logger.info(f"Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
