"""Comparison Pipeline: DRoC vs Baseline Solvers.

This module provides comparison functionality between DRoC (LLM-generated code)
and baseline solvers (OR-Tools, PyVRP, Gurobi).
"""

from __future__ import annotations

import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading
from logging.handlers import QueueHandler

import pandas as pd

from src.domain.schema import VRPTWInstance, RouteSolution, ValidationReport
from src.solvers.ortools_solver import solve_ortools
from src.solvers.pyvrp_solver import solve_pyvrp
from src.solvers.feasibility_checker import check_feasibility
from src.pipelines.droc_pipeline import run_droc_solver

logger = logging.getLogger(__name__)


# ── Thread-safe logging via QueueHandler ───────────────────────────────────────
# When multiple worker threads write to the console concurrently, log lines get
# fragmented. We funnel all logging through a single background listener thread.
#
# Key design decisions:
# - Bounded queue (maxsize=1000): if the listener falls behind, worker threads
#   will block on put() after 1000 pending messages. We add a timeout so they
#   never deadlock indefinitely.
# - emit() failures are silently dropped: we never want logging to block the
#   worker threads. If the console is locked, we skip the record.
# - Listener is daemon=True so it won't block process exit.

_listener_queue: queue.Queue = queue.Queue(maxsize=1000)
_listener_stop = threading.Event()


def _configure_queue_logging() -> None:
    """Install QueueHandler on the root logger so worker threads route through the listener."""
    root = logging.getLogger()
    h = QueueHandler(_listener_queue)
    h.setLevel(logging.DEBUG)
    root.addHandler(h)
    for name in logging.Logger.manager.loggerDict:
        lg = logging.getLogger(name)
        lg.propagate = False


def _start_log_listener() -> None:
    """Start the background thread that drains the queue and prints sequentially."""
    def listener():
        dropped = 0
        while not _listener_stop.is_set():
            try:
                record: logging.LogRecord = _listener_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            root = logging.getLogger()
            for handler in root.handlers:
                try:
                    handler.emit(record)
                    handler.flush()
                except Exception:
                    # Never let logging failures block the worker threads
                    dropped += 1
                    if dropped % 100 == 1:
                        print(f"[LogListener] {dropped} log records dropped (console busy)", flush=True)
                    break

    _listener_stop.clear()
    t = threading.Thread(target=listener, daemon=True, name="LogListener")
    t.start()


def _stop_log_listener() -> None:
    """Stop the log listener and drain remaining records."""
    _listener_stop.set()
    while True:
        try:
            _listener_queue.get_nowait()
        except queue.Empty:
            break


@dataclass
class ComparisonResult:
    """Result of a single comparison run."""
    instance_name: str
    instance_size: int
    
    # DRoC results
    droc_vehicles: Optional[int] = None
    droc_distance: Optional[float] = None
    droc_feasible: Optional[bool] = None
    droc_runtime_sec: Optional[float] = None
    droc_iterations: Optional[int] = None
    droc_success: bool = False
    droc_error: Optional[str] = None
    
    # Baseline results
    baseline_vehicles: Optional[int] = None
    baseline_distance: Optional[float] = None
    baseline_feasible: Optional[bool] = None
    baseline_runtime_sec: Optional[float] = None
    baseline_method: str = ""
    
    # Comparison metrics
    vehicle_improvement: Optional[float] = None  # percentage
    distance_improvement: Optional[float] = None  # percentage
    droc_wins: Optional[bool] = None


def run_baseline_solver(
    instance: VRPTWInstance,
    solver: str = "ortools",
    time_limit: int = 300,
    seed: int = 42,
) -> tuple[RouteSolution, float]:
    """Run a baseline solver on the instance.
    
    Args:
        instance: VRPTW instance to solve.
        solver: Solver name ("ortools", "pyvrp", "gurobi").
        time_limit: Time limit in seconds.
        seed: Random seed.
    
    Returns:
        Tuple of (solution, runtime_sec).
    """
    start_time = time.time()
    
    if solver == "ortools":
        solution, _ = solve_ortools(
            instance,
            time_limit=time_limit,
        )
    elif solver == "pyvrp":
        solution, _ = solve_pyvrp(
            instance,
            time_limit=time_limit,
        )
    else:
        raise ValueError(f"Unknown solver: {solver}")
    
    runtime = time.time() - start_time
    return solution, runtime


def run_comparison(
    instance_path: str,
    llm_client,
    solver: str = "ortools",
    max_iterations: int = 4,
    time_limit: int = 300,
    seed: int = 42,
    droc_timeout: float = 600.0,
    baseline_timeout: float = 300.0,
    output_dir: str | None = None,
) -> ComparisonResult:
    """Compare DRoC vs baseline solver on a single instance.

    Args:
        instance_path: Path to VRPTW instance file.
        llm_client: LLM client for DRoC.
        solver: Baseline solver name.
        max_iterations: Max DRoC self-debug iterations.
        time_limit: Soft time limit for solvers.
        seed: Random seed.
        droc_timeout: Hard timeout for the entire DRoC run (including all retries).
        baseline_timeout: Hard timeout for baseline solver.

    Returns:
        ComparisonResult with both results.
    """
    from src.utils.io import read_instance

    instance = read_instance(instance_path)

    result = ComparisonResult(
        instance_name=instance.name,
        instance_size=instance.size,
        baseline_method=solver,
    )

    # Run baseline solver with hard timeout
    logger.info(f"[{instance.name}] Running {solver} baseline...")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_baseline_solver,
                instance, solver, time_limit, seed
            )
            baseline_solution, baseline_runtime = future.result(timeout=baseline_timeout)
        result.baseline_vehicles = baseline_solution.vehicles_used
        result.baseline_distance = baseline_solution.total_distance
        result.baseline_feasible = check_feasibility(instance, baseline_solution).feasible
        result.baseline_runtime_sec = baseline_runtime
        logger.info(
            f"[{instance.name}] {solver}: "
            f"vehicles={result.baseline_vehicles}, "
            f"distance={result.baseline_distance:.1f}, "
            f"feasible={result.baseline_feasible}"
        )
    except FuturesTimeoutError:
        logger.error(
            f"[{instance.name}] {solver} baseline timed out after {baseline_timeout:.0f}s"
        )
        result.baseline_feasible = False
        result.baseline_runtime_sec = baseline_timeout
    except Exception as e:
        logger.error(f"[{instance.name}] {solver} failed: {e}")
        result.baseline_feasible = False
        result.baseline_runtime_sec = baseline_timeout

    # Run DRoC with a hard per-instance timeout
    droc_retry = 0
    droc_max_retries = 2
    droc_api_error = None

    while droc_retry <= droc_max_retries:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_droc_solver,
                    instance,
                    llm_client=llm_client,
                    max_iterations=max_iterations,
                    time_limit=time_limit,
                    seed=seed,
                    droc_timeout=droc_timeout,
                    output_dir=output_dir,
                    incumbent=baseline_solution,
                )
                droc_solution, gen_result, exp_result = future.result(timeout=droc_timeout + 60)
            result.droc_vehicles = droc_solution.vehicles_used
            result.droc_distance = droc_solution.total_distance
            droc_validation = check_feasibility(instance, droc_solution)
            result.droc_feasible = droc_validation.feasible
            if not droc_validation.feasible:
                logger.warning(
                    f"[{instance.name}] DRoC INFEASIBLE: "
                    f"late_violations={droc_validation.late_violations}, "
                    f"capacity_violations={droc_validation.capacity_violations}"
                )
            result.droc_runtime_sec = exp_result.runtime_sec
            result.droc_iterations = gen_result.iterations
            result.droc_success = True
            logger.info(
                f"[{instance.name}] DRoC: "
                f"vehicles={result.droc_vehicles}, "
                f"distance={result.droc_distance:.1f}, "
                f"feasible={result.droc_feasible}, "
                f"iterations={result.droc_iterations}"
            )
            break
        except FuturesTimeoutError:
            droc_retry += 1
            droc_api_error = f"[{instance.name}] DRoC timed out after {droc_timeout:.0f}s"
            logger.error(
                f"[{instance.name}] DRoC hard-timeout exceeded "
                f"(limit={droc_timeout:.0f}s). Giving up on this instance."
            )
            result.droc_success = False
            result.droc_error = droc_api_error
            break  # timeouts don't benefit from retry
        except Exception as e:
            droc_retry += 1
            droc_api_error = str(e)
            if droc_retry <= droc_max_retries:
                import time as _time
                wait = 5 * droc_retry
                logger.warning(
                    f"[{instance.name}] DRoC attempt {droc_retry} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                _time.sleep(wait)
            else:
                logger.error(f"[{instance.name}] DRoC failed after {droc_retry} attempts: {e}")
                result.droc_success = False
                result.droc_error = droc_api_error

    # Calculate improvements
    if result.droc_success and result.baseline_feasible:
        if result.baseline_vehicles and result.baseline_vehicles > 0:
            result.vehicle_improvement = (
                (result.baseline_vehicles - result.droc_vehicles)
                / result.baseline_vehicles * 100
            )
        if result.baseline_distance and result.baseline_distance > 0:
            result.distance_improvement = (
                (result.baseline_distance - result.droc_distance)
                / result.baseline_distance * 100
            )
        result.droc_wins = (
            result.droc_vehicles <= result.baseline_vehicles
            if result.droc_vehicles and result.baseline_vehicles else False
        )

    return result


def run_batch_comparison(
    data_dir: str,
    llm_client,
    solver: str = "ortools",
    instances: str = "gh800",
    max_iterations: int = 4,
    time_limit: int = 300,
    seeds: list[int] = None,
    output_dir: str = "results/comparison",
    max_instances: int = None,
    per_instance_timeout: float = 900.0,
    resume: bool = True,
) -> dict:
    """Run comparison on multiple instances, one by one, with incremental save.

    Each instance runs baseline first, then DRoC. Results are saved immediately
    after each instance completes. If the process is interrupted and re-run with
    resume=True (default), already-completed instances are skipped.

    Args:
        data_dir: Directory containing VRPTW instances.
        llm_client: LLM client for DRoC.
        solver: Baseline solver name.
        instances: Instance filter ("gh200", "gh400", "gh800", "gh1000", or "all").
        max_iterations: Max DRoC self-debug iterations.
        time_limit: Soft time limit passed to solvers (DRoC uses this to budget
            per-call time, but per_instance_timeout is the hard ceiling).
        seeds: List of random seeds.
        output_dir: Directory to save results.
        max_instances: Maximum number of instances to process.
        per_instance_timeout: Hard timeout per (instance, seed) pair in seconds.
            If an instance runs longer than this, it is recorded as timed-out
            and the next instance starts. Default 900s = 15 minutes.
        resume: If True (default), skip instances already present in the
            results CSV. Use False to re-run everything from scratch.

    Returns:
        Dictionary with results, DataFrame, and summary statistics.
    """
    import json

    if seeds is None:
        seeds = [42]

    # Collect and filter instance files
    data_path = Path(data_dir)
    # Use recursive glob to find .vrp files in subdirectories (e.g. GH200/, GH400/, etc.)
    all_vrp_files = list(data_path.glob("**/*.vrp"))
    # Exclude Solomon-style small instances (C1_10_, R2_10_, etc.) which are in the root
    all_vrp_files = [f for f in all_vrp_files if "_10_" not in f.stem]

    def get_size(filename: str) -> int:
        import re
        match = re.search(r"_(\d+)_", filename)
        if match:
            return int(match.group(1)) * 100
        return 0

    if instances == "all":
        # Include all GH families: GH200 (_2_), GH400 (_4_), GH600 (_6_), GH800 (_8_)
        all_sizes = {200, 400, 600, 800}
        filtered_files = sorted([f for f in all_vrp_files if get_size(f.stem) in all_sizes])
    else:
        target_size = int(instances.replace("gh", "").replace("gh", ""))
        filtered_files = sorted([
            f for f in all_vrp_files if get_size(f.name) == target_size
        ])

    if max_instances:
        filtered_files = filtered_files[:max_instances]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "comparison_results.csv"

    # ── Resume: load existing results ───────────────────────────────────────────
    completed_keys: set[str] = set()
    if resume and csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            for _, row in existing_df.iterrows():
                completed_keys.add(f"{row['instance']}_s{row['seed']}")
            logger.info(
                f"[Incremental] Resuming: found {len(completed_keys)} completed runs "
                f"in {csv_path}, skipping them."
            )
        except Exception as e:
            logger.warning(f"[Incremental] Could not read existing CSV ({e}), starting fresh.")
            completed_keys.clear()

    # ── Per-instance run with hard timeout ──────────────────────────────────────
    all_results: list[dict] = []
    if completed_keys:
        try:
            all_results = pd.read_csv(csv_path).to_dict("records")
        except Exception:
            all_results = []

    total = len(filtered_files) * len(seeds)
    remaining = total - len(completed_keys)
    done = len(completed_keys)

    logger.info(
        f"[Incremental] {len(filtered_files)} instances x {len(seeds)} seeds "
        f"= {total} runs total. {done} already done, {remaining} remaining."
    )

    for f in filtered_files:
        for seed in seeds:
            key = f"{f.stem}_s{seed}"
            if key in completed_keys:
                continue

            logger.info(f"[Incremental] [{done + 1}/{total}] Starting {key} "
                        f"(timeout={per_instance_timeout:.0f}s)")

            t0 = time.perf_counter()
            result = None
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        run_comparison,
                        instance_path=str(f),
                        llm_client=llm_client,
                        solver=solver,
                        max_iterations=max_iterations,
                        time_limit=time_limit,
                        seed=seed,
                        droc_timeout=per_instance_timeout,
                        baseline_timeout=per_instance_timeout,
                        output_dir=output_dir,
                    )
                    result = future.result(timeout=per_instance_timeout + 30)
            except FuturesTimeoutError:
                elapsed = time.perf_counter() - t0
                logger.error(
                    f"[Incremental] {key} hard-timeout after {elapsed:.0f}s "
                    f"(>{per_instance_timeout:.0f}s), skipping."
                )
                result = ComparisonResult(
                    instance_name=f.stem,
                    instance_size=get_size(f.name),
                    baseline_method=solver,
                    droc_success=False,
                    droc_error=f"Hard timeout after {elapsed:.0f}s",
                    droc_runtime_sec=elapsed,
                )
            except Exception as e:
                elapsed = time.perf_counter() - t0
                logger.error(f"[Incremental] {key} failed after {elapsed:.1f}s: {e}")
                result = ComparisonResult(
                    instance_name=f.stem,
                    instance_size=get_size(f.name),
                    baseline_method=solver,
                    droc_success=False,
                    droc_error=str(e),
                    droc_runtime_sec=elapsed,
                )

            if result is not None:
                row = _comparison_result_to_row(result, seed)
                all_results.append(row)
                done += 1

                # Save incrementally after each instance
                df = pd.DataFrame(all_results)
                df.to_csv(csv_path, index=False)

                # Log progress
                droc_ok = row["droc_feasible"] if row["droc_feasible"] is not None else False
                base_ok = row["baseline_feasible"] if row["baseline_feasible"] is not None else False
                logger.info(
                    f"[Incremental] [{done}/{total}] {key} done "
                    f"({time.perf_counter() - t0:.0f}s): "
                    f"droc={'OK' if droc_ok else 'FAIL'}, "
                    f"baseline={'OK' if base_ok else 'FAIL'}"
                )

    # ── Final summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)

    summary = _generate_summary(df, solver)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / "summary.csv", index=False)

    # Save failed instances separately
    failed = df[df["droc_success"] == False]
    if not failed.empty:
        failed_path = output_path / "failed_instances.csv"
        failed.to_csv(failed_path, index=False)
        logger.info(f"[Incremental] {len(failed)} failed runs saved to {failed_path}")

    success_cnt = len(df) - len(failed)
    logger.info(
        f"[Incremental] Done! {success_cnt}/{len(df)} succeeded. "
        f"Results: {csv_path}"
    )

    return {
        "results": all_results,
        "df": df,
        "summary": summary,
    }


def _comparison_result_to_row(r: ComparisonResult, seed: int) -> dict:
    """Convert a ComparisonResult to a dict row for the results DataFrame."""
    return {
        "instance": r.instance_name,
        "size": r.instance_size,
        "seed": seed,
        "droc_vehicles": r.droc_vehicles,
        "droc_distance": r.droc_distance,
        "droc_feasible": r.droc_feasible,
        "droc_runtime": r.droc_runtime_sec,
        "droc_iterations": r.droc_iterations,
        "droc_success": r.droc_success,
        "droc_error": r.droc_error,
        "baseline_vehicles": r.baseline_vehicles,
        "baseline_distance": r.baseline_distance,
        "baseline_feasible": r.baseline_feasible,
        "baseline_runtime": r.baseline_runtime_sec,
        "vehicle_improvement": r.vehicle_improvement,
        "distance_improvement": r.distance_improvement,
        "droc_wins": r.droc_wins,
    }


@dataclass
class InstanceStatus:
    """Runtime status of a single instance in the scheduler."""
    instance_name: str
    instance_size: int
    status: str = "pending"  # pending | running | success | failed | timeout
    droc_vehicles: Optional[int] = None
    droc_distance: Optional[float] = None
    droc_feasible: Optional[bool] = None
    droc_runtime_sec: Optional[float] = None
    droc_iterations: Optional[int] = None
    droc_error: Optional[str] = None
    droc_success: bool = False
    baseline_vehicles: Optional[int] = None
    baseline_distance: Optional[float] = None
    baseline_feasible: Optional[bool] = None
    baseline_runtime_sec: Optional[float] = None
    elapsed_sec: float = 0.0


def run_batch_comparison_queued(
    data_dir: str,
    llm_client,
    solver: str = "ortools",
    instances: str = "gh800",
    max_iterations: int = 4,
    time_limit: int = 300,
    seeds: list[int] = None,
    output_dir: str = "results/comparison",
    max_instances: int = None,
    max_workers: int = 4,
    droc_timeout: float = 600.0,
    baseline_timeout: float = 300.0,
) -> dict:
    """Run comparison on multiple instances with parallel execution and per-instance timeout.

    Instances are submitted to a thread pool concurrently. Results are collected
    as soon as each instance completes (via as_completed), so slow/hanging instances
    do not block faster ones. Per-instance hard timeouts ensure no single instance
    can stall the entire batch.

    Failed (timeout or error) instances are recorded and their details are saved to
    failed_instances.json alongside the results CSV.

    Args:
        data_dir: Directory containing VRPTW instances.
        llm_client: LLM client for DRoC.
        solver: Baseline solver name.
        instances: Instance filter ("gh200", "gh400", "gh800", "gh1000", or "all").
        max_iterations: Max DRoC self-debug iterations.
        time_limit: Soft time limit per instance (used by solver / DRoC internals).
        seeds: List of random seeds.
        output_dir: Directory to save results.
        max_instances: Maximum number of instances to process.
        max_workers: Max parallel workers (default 4).
        droc_timeout: Hard timeout for DRoC per instance (default 180s).
        baseline_timeout: Hard timeout for baseline solver per instance (default 300s).

    Returns:
        Dictionary with results, status map, and summary statistics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.utils.io import read_instance

    # Install thread-safe logging: all worker threads route through a single
    # background listener that owns stdout, eliminating line interleaving.
    _configure_queue_logging()
    _start_log_listener()

    if seeds is None:
        seeds = [42]

    # ── 1. Collect instance files ──────────────────────────────────────────────
    data_path = Path(data_dir)
    all_vrp_files = list(data_path.glob("*.vrp"))

    def get_size(filename: str) -> int:
        import re
        match = re.search(r"_(\d+)_", filename)
        if match:
            return int(match.group(1)) * 100
        return 0

    if instances == "all":
        filtered_files = sorted(all_vrp_files)
    else:
        target_size = int(instances.replace("gh", "").replace("gh", ""))
        filtered_files = sorted([
            f for f in all_vrp_files if get_size(f.name) == target_size
        ])

    if max_instances:
        filtered_files = filtered_files[:max_instances]

    total = len(filtered_files)
    logger.info(f"[Scheduler] Found {total} instances matching '{instances}', "
                f"max_workers={max_workers}")

    # ── 2. Pre-scan: load all instances (fast, no solving) ───────────────────
    all_instance_tasks: list[tuple[Path, int, str]] = []  # (path, seed, label)
    for f in filtered_files:
        for seed in seeds:
            all_instance_tasks.append((f, seed, f"{f.stem}_s{seed}"))

    # Build status map
    status_map: dict[str, InstanceStatus] = {}
    for f, seed, label in all_instance_tasks:
        try:
            inst = read_instance(str(f))
            status_map[label] = InstanceStatus(
                instance_name=inst.name,
                instance_size=inst.size,
            )
        except Exception as e:
            status_map[label] = InstanceStatus(
                instance_name=f.stem,
                instance_size=get_size(f.name),
            )

    # ── 3. Prepare shared output state ───────────────────────────────────────
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[ComparisonResult] = []
    results_lock = threading.Lock()
    completed_count = 0

    def _save_partial():
        """Thread-safe partial save of results so far."""
        with results_lock:
            df = pd.DataFrame([
                _result_to_row(r) for r in results
            ])
            df.to_csv(output_path / "comparison_results.csv", index=False)

    def _save_failed():
        """Save failed instances with error details."""
        failed = [s for s in status_map.values() if s.status in ("failed", "timeout")]
        failed_data = [
            {
                "instance": s.instance_name,
                "size": s.instance_size,
                "status": s.status,
                "error": s.droc_error,
                "elapsed_sec": s.elapsed_sec,
                "droc_success": s.droc_success,
                "baseline_vehicles": s.baseline_vehicles,
                "baseline_feasible": s.baseline_feasible,
            }
            for s in failed
        ]
        import json
        with open(output_path / "failed_instances.json", "w", encoding="utf-8") as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[Scheduler] Saved {len(failed)} failed instance(s) to "
                    f"{output_path / 'failed_instances.json'}")

    def _result_to_row(r: ComparisonResult) -> dict:
        return {
            "instance": r.instance_name,
            "size": r.instance_size,
            "seed": getattr(r, "_seed", 0),
            "droc_vehicles": r.droc_vehicles,
            "droc_distance": r.droc_distance,
            "droc_feasible": r.droc_feasible,
            "droc_runtime": r.droc_runtime_sec,
            "droc_iterations": r.droc_iterations,
            "baseline_vehicles": r.baseline_vehicles,
            "baseline_distance": r.baseline_distance,
            "baseline_feasible": r.baseline_feasible,
            "baseline_runtime": r.baseline_runtime_sec,
            "vehicle_improvement": r.vehicle_improvement,
            "distance_improvement": r.distance_improvement,
            "droc_wins": r.droc_wins,
            "droc_success": r.droc_success,
        }

    # ── 4. Worker function ────────────────────────────────────────────────────
    def _run_single(instance_path: str, seed: int, label: str) -> tuple[str, ComparisonResult, float]:
        """Run one (instance, seed) comparison. Returns (label, result, elapsed)."""
        t0 = time.perf_counter()
        try:
            result = run_comparison(
                instance_path=instance_path,
                llm_client=llm_client,
                solver=solver,
                max_iterations=max_iterations,
                time_limit=time_limit,
                seed=seed,
                droc_timeout=droc_timeout,
                baseline_timeout=baseline_timeout,
            )
            result._seed = seed  # type: ignore
            return label, result, time.perf_counter() - t0
        except Exception as e:
            logger.error(f"[Scheduler][{label}] Unexpected error: {e}")
            result = ComparisonResult(
                instance_name=label,
                instance_size=status_map.get(label, InstanceStatus(label, 0)).instance_size,
                droc_success=False,
                droc_error=str(e),
                baseline_method=solver,
            )
            result._seed = seed  # type: ignore
            return label, result, time.perf_counter() - t0

    # ── 5. Submit all tasks to thread pool ────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict = {}
        for f, seed, label in all_instance_tasks:
            fut = pool.submit(_run_single, str(f), seed, label)
            status_map[label].status = "running"
            futures[fut] = label

        # ── 6. Collect results as they complete ────────────────────────────────
        for fut in as_completed(futures):
            label = futures[fut]
            elapsed = 0.0
            try:
                result_label, result, elapsed = fut.result()
            except FuturesTimeoutError:
                logger.error(f"[Scheduler][{label}] Hard timeout exceeded, skipping")
                status_map[label].status = "timeout"
                status_map[label].droc_error = f"Hard timeout after {droc_timeout}s"
                status_map[label].elapsed_sec = droc_timeout
                result = ComparisonResult(
                    instance_name=label,
                    instance_size=status_map[label].instance_size,
                    droc_success=False,
                    droc_error=f"Hard timeout after {droc_timeout}s",
                    droc_runtime_sec=droc_timeout,
                    baseline_method=solver,
                )
            except Exception as e:
                logger.error(f"[Scheduler][{label}] Worker exception: {e}")
                status_map[label].status = "failed"
                status_map[label].droc_error = str(e)
                status_map[label].elapsed_sec = elapsed
                result = ComparisonResult(
                    instance_name=label,
                    instance_size=status_map[label].instance_size,
                    droc_success=False,
                    droc_error=str(e),
                    droc_runtime_sec=elapsed,
                    baseline_method=solver,
                )

            # Update status map
            status_map[label].status = "success" if result.droc_success else "failed"
            status_map[label].droc_success = result.droc_success
            status_map[label].droc_error = result.droc_error
            status_map[label].droc_vehicles = result.droc_vehicles
            status_map[label].droc_distance = result.droc_distance
            status_map[label].droc_feasible = result.droc_feasible
            status_map[label].droc_runtime_sec = result.droc_runtime_sec
            status_map[label].droc_iterations = result.droc_iterations
            status_map[label].baseline_vehicles = result.baseline_vehicles
            status_map[label].baseline_distance = result.baseline_distance
            status_map[label].baseline_feasible = result.baseline_feasible
            status_map[label].baseline_runtime_sec = result.baseline_runtime_sec
            status_map[label].elapsed_sec = elapsed

            # Append and save immediately
            with results_lock:
                results.append(result)
                completed_count += 1
                _save_partial()

            # Log progress
            success_cnt = sum(1 for s in status_map.values() if s.status == "success")
            failed_cnt = sum(1 for s in status_map.values() if s.status in ("failed", "timeout"))
            logger.info(
                f"[Scheduler] [{completed_count}/{total}] {label}: "
                f"{'OK' if result.droc_success else 'FAIL'} "
                f"(vehicles={result.droc_vehicles}, feasible={result.droc_feasible}, "
                f"elapsed={elapsed:.1f}s) | "
                f"total: {success_cnt} ok, {failed_cnt} failed"
            )

    # ── 7. Save failed instances ───────────────────────────────────────────────
    _save_failed()

    # ── 8. Final summary ─────────────────────────────────────────────────────
    df = pd.DataFrame([_result_to_row(r) for r in results])
    df.to_csv(output_path / "comparison_results.csv", index=False)

    summary = _generate_summary(df, solver)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_path / "summary.csv", index=False)

    success_cnt = sum(1 for s in status_map.values() if s.status == "success")
    failed_cnt = sum(1 for s in status_map.values() if s.status in ("failed", "timeout"))

    logger.info(
        f"[Scheduler] Complete. "
        f"{success_cnt}/{total} succeeded, {failed_cnt} failed. "
        f"Results saved to {output_path}"
    )

    _stop_log_listener()

    return {
        "results": results,
        "df": df,
        "summary": summary,
        "status_map": status_map,
    }


def _generate_summary(df: pd.DataFrame, baseline_method: str) -> dict:
    """Generate summary statistics from comparison results."""
    
    summary = {
        "method": f"DRoC vs {baseline_method}",
        "total_instances": len(df),
        "successful_runs": df["droc_success"].sum() if "droc_success" in df.columns else 0,
    }
    
    # DRoC statistics
    if "droc_vehicles" in df.columns:
        droc_valid = df[df["droc_feasible"] == True]
        if len(droc_valid) > 0:
            summary["droc_avg_vehicles"] = droc_valid["droc_vehicles"].mean()
            summary["droc_avg_distance"] = droc_valid["droc_distance"].mean()
            summary["droc_feasible_rate"] = len(droc_valid) / len(df)
        else:
            summary["droc_feasible_rate"] = 0
    
    # Baseline statistics
    if "baseline_vehicles" in df.columns:
        baseline_valid = df[df["baseline_feasible"] == True]
        if len(baseline_valid) > 0:
            summary["baseline_avg_vehicles"] = baseline_valid["baseline_vehicles"].mean()
            summary["baseline_avg_distance"] = baseline_valid["baseline_distance"].mean()
            summary["baseline_feasible_rate"] = len(baseline_valid) / len(df)
        else:
            summary["baseline_feasible_rate"] = 0
    
    # Comparison statistics
    if "vehicle_improvement" in df.columns:
        valid_imp = df[df["vehicle_improvement"].notna()]
        if len(valid_imp) > 0:
            summary["avg_vehicle_improvement"] = valid_imp["vehicle_improvement"].mean()
            summary["droc_wins_rate"] = valid_imp["droc_wins"].mean()
    
    return summary
