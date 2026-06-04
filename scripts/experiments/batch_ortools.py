"""Batch OR-Tools experiment on ORTEC static and dynamic VRPTW scenarios.

Supports:
  - Static: solve every ORTEC-SYNTH-*.txt instance
  - Dynamic: solve every ORTEC-SYNTH-*_dyn5e.json scenario (epoch by epoch)

Usage:
    # Static only, 60s per instance, resume
    python batch_ortools.py static --time-limit 60 --resume

    # Dynamic only, 30s per epoch
    python batch_ortools.py dynamic --time-limit 30 --resume

    # Both
    python batch_ortools.py all --time-limit 60 --resume

    # Small smoke test (3 instances each)
    python batch_ortools.py all --max-instances 3 --time-limit 30
"""
import argparse
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from src.utils.io import read_instance
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.domain.schema import RouteSolution


# ---------------------------------------------------------------------------
# Static experiment
# ---------------------------------------------------------------------------

@dataclass
class StaticResult:
    instance: str = ""
    size: int = 0
    solver: str = "ortools"
    time_limit: int = 0
    vehicles: int = 0
    distance: float = 0.0
    feasible: bool = False
    late_violations: int = 0
    cap_violations: int = 0
    runtime_sec: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_static_instance(args_tuple):
    """Solve one static instance. Designed for ProcessPoolExecutor."""
    (inst_path, time_limit, result_id) = args_tuple
    t0 = time.time()
    try:
        inst = read_instance(str(inst_path))
        solution, _ = solve_ortools(inst, time_limit=float(time_limit), warm_start=None)
        r = StaticResult(
            instance=inst.name,
            size=inst.size,
            solver="ortools",
            time_limit=time_limit,
            vehicles=solution.vehicles_used if solution else 0,
            distance=solution.total_distance if solution else 0.0,
            feasible=solution.feasible if solution else False,
            late_violations=solution.late_violations if solution else -1,
            cap_violations=solution.capacity_violations if solution else -1,
            runtime_sec=round(time.time() - t0, 2),
        )
        if solution:
            report = check_feasibility(inst, solution)
            r.feasible = report.feasible
        return r
    except Exception as e:
        return StaticResult(
            instance=Path(inst_path).stem,
            size=0,
            solver="ortools",
            time_limit=time_limit,
            runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


# ---------------------------------------------------------------------------
# Dynamic experiment
# ---------------------------------------------------------------------------

@dataclass
class EpochResultRow:
    scenario: str
    base: str
    epoch: int
    served_rate: float
    feasible: bool
    late_violations: int
    cap_violations: int
    runtime_sec: float
    orders_dispatched: int
    orders_pending: int
    distance: float
    error: str = ""


@dataclass
class DynamicResult:
    scenario: str = ""
    base_instance: str = ""
    num_epochs: int = 0
    solver: str = "ortools"
    time_limit: int = 0
    overall_served_rate: float = 0.0
    overall_feasible: bool = False
    total_late: int = 0
    total_cap_viol: int = 0
    total_distance: float = 0.0
    total_runtime_sec: float = 0.0
    error: str = ""
    epoch_rows: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_dynamic_scenario(args_tuple):
    """Solve one dynamic scenario. Designed for ProcessPoolExecutor."""
    (scen_path, inst_path, time_limit, result_id) = args_tuple
    t0 = time.time()
    try:
        from src.pipelines.dynamic_dispatch_pipeline import DynamicDispatchPipeline, load_scenario

        scenario = load_scenario(str(scen_path))
        base = read_instance(str(inst_path))

        pipeline = DynamicDispatchPipeline(
            solver="ortools",
            llm_client=None,
            time_limit_per_epoch=float(time_limit),
            droc_timeout=float(time_limit * 3),
            output_dir=Path("results/dynamic_batch"),
        )
        result = pipeline.run(scenario=scenario, base_instance=base)

        dr = DynamicResult(
            scenario=scenario.name,
            base_instance=base.name,
            num_epochs=result.num_epochs,
            solver="ortools",
            time_limit=time_limit,
            overall_served_rate=result.overall_served_rate,
            overall_feasible=result.overall_feasible,
            total_late=result.total_late_violations,
            total_cap_viol=result.total_capacity_violations,
            total_distance=result.total_distance,
            total_runtime_sec=round(time.time() - t0, 2),
            epoch_rows=[
                EpochResultRow(
                    scenario=scenario.name,
                    base=base.name,
                    epoch=er.epoch,
                    served_rate=er.served_rate,
                    feasible=er.feasible,
                    late_violations=er.late_violations,
                    cap_violations=er.capacity_violations,
                    runtime_sec=er.runtime_sec,
                    orders_dispatched=er.orders_dispatched,
                    orders_pending=er.orders_pending,
                    distance=er.total_distance,
                )
                for er in result.epoch_results
            ],
        )
        return dr
    except Exception as e:
        return DynamicResult(
            scenario=Path(scen_path).stem,
            base_instance=Path(inst_path).stem,
            num_epochs=0,
            solver="ortools",
            time_limit=time_limit,
            total_runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
            epoch_rows=[],
        )


# ---------------------------------------------------------------------------
# Batch runner utilities
# ---------------------------------------------------------------------------

WORKERS = 4  # concurrent threads


def load_completed_static(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["instance"] for row in csv.DictReader(f) if row.get("instance")}


def load_completed_dynamic(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row["scenario"] for row in csv.DictReader(f) if row.get("scenario")}


def write_static_row(csv_path: Path, row: StaticResult):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "instance", "size", "solver", "time_limit",
            "vehicles", "distance", "feasible", "late_violations",
            "cap_violations", "runtime_sec", "error",
        ])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": row.timestamp,
            "instance": row.instance,
            "size": row.size,
            "solver": row.solver,
            "time_limit": row.time_limit,
            "vehicles": row.vehicles,
            "distance": round(row.distance, 2),
            "feasible": row.feasible,
            "late_violations": row.late_violations,
            "cap_violations": row.cap_violations,
            "runtime_sec": row.runtime_sec,
            "error": row.error,
        })


def write_dynamic_row(csv_path: Path, row: DynamicResult):
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "timestamp", "scenario", "base_instance", "num_epochs", "solver", "time_limit",
            "overall_served_rate", "overall_feasible", "total_late",
            "total_cap_viol", "total_distance", "total_runtime_sec", "error",
        ])
        if not exists:
            w.writeheader()
        w.writerow({
            "timestamp": row.timestamp,
            "scenario": row.scenario,
            "base_instance": row.base_instance,
            "num_epochs": row.num_epochs,
            "solver": row.solver,
            "time_limit": row.time_limit,
            "overall_served_rate": f"{row.overall_served_rate:.4f}",
            "overall_feasible": row.overall_feasible,
            "total_late": row.total_late,
            "total_cap_viol": row.total_cap_viol,
            "total_distance": round(row.total_distance, 2),
            "total_runtime_sec": row.total_runtime_sec,
            "error": row.error,
        })


def write_epoch_rows(csv_path: Path, rows: list[EpochResultRow]):
    if not rows:
        return
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "scenario", "base", "epoch",
            "served_rate", "feasible", "late_violations",
            "cap_violations", "runtime_sec", "orders_dispatched",
            "orders_pending", "distance", "error",
        ])
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({
                "scenario": r.scenario,
                "base": r.base,
                "epoch": r.epoch,
                "served_rate": f"{r.served_rate:.4f}",
                "feasible": r.feasible,
                "late_violations": r.late_violations,
                "cap_violations": r.cap_violations,
                "runtime_sec": r.runtime_sec,
                "orders_dispatched": r.orders_dispatched,
                "orders_pending": r.orders_pending,
                "distance": round(r.distance, 2),
                "error": r.error,
            })


def print_summary_static(results: list[StaticResult]):
    total = len(results)
    feasible = sum(1 for r in results if r.feasible and not r.error)
    failed = sum(1 for r in results if r.error)
    times = [r.runtime_sec for r in results if not r.error]
    vehicles = [r.vehicles for r in results if r.feasible and not r.error]
    distances = [r.distance for r in results if r.feasible and not r.error]

    print(f"\n{'='*60}")
    print(f"STATIC SUMMARY  ({total} instances)")
    print(f"{'='*60}")
    print(f"  Feasible:  {feasible}/{total}  ({feasible/total*100:.1f}%)")
    print(f"  Failed:    {failed}")
    if vehicles:
        print(f"  Vehicles:  avg={sum(vehicles)/len(vehicles):.1f}  "
              f"min={min(vehicles)}  max={max(vehicles)}")
        print(f"  Distance:  avg={sum(distances)/len(distances):.0f}  "
              f"min={min(distances):.0f}  max={max(distances):.0f}")
    if times:
        print(f"  Runtime:   avg={sum(times)/len(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")

    by_size = {}
    for r in results:
        if r.error or not r.feasible:
            continue
        by_size.setdefault(r.size, []).append(r)
    if by_size:
        print()
        for size in sorted(by_size.keys()):
            grp = by_size[size]
            print(f"  [{size} nodes] {len(grp)} instances | "
                  f"avg vehicles={sum(x.vehicles for x in grp)/len(grp):.1f} | "
                  f"avg distance={sum(x.distance for x in grp)/len(grp):.0f}")


def print_summary_dynamic(results: list[DynamicResult]):
    total = len(results)
    feasible = sum(1 for r in results if r.overall_feasible and not r.error)
    served = [r.overall_served_rate for r in results if not r.error]
    times = [r.total_runtime_sec for r in results if not r.error]
    failed = sum(1 for r in results if r.error)

    print(f"\n{'='*60}")
    print(f"DYNAMIC SUMMARY  ({total} scenarios)")
    print(f"{'='*60}")
    print(f"  Overall feasible:  {feasible}/{total}  ({feasible/total*100:.1f}%)")
    print(f"  Failed:            {failed}")
    if served:
        print(f"  Avg served rate:   {sum(served)/len(served)*100:.1f}%")
        print(f"  Min served rate:  {min(served)*100:.1f}%  |  Max: {max(served)*100:.1f}%")
    if times:
        print(f"  Avg total runtime: {sum(times)/len(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")


# ---------------------------------------------------------------------------
# Main batch functions
# ---------------------------------------------------------------------------

def batch_static(
    time_limit: int = 60,
    max_instances: int | None = None,
    resume: bool = True,
    workers: int = WORKERS,
):
    static_dir = Path("data/VRPTW/ORTEC/static")
    output_dir = Path("results/static_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "static_results.csv"

    print(f"\n[STATIC] Scanning {static_dir} ...")
    txt_files = sorted(static_dir.glob("ORTEC-SYNTH-*.txt"))
    print(f"[STATIC] Found {len(txt_files)} static instances")

    completed = load_completed_static(csv_path) if resume else set()
    to_run = [f for f in txt_files if f.stem not in completed]
    if max_instances:
        to_run = to_run[:max_instances]

    total = len(to_run)
    print(f"[STATIC] {'Resume:'} {len(completed)} done, {total} to run (max={max_instances})")

    if not to_run:
        print("[STATIC] Nothing to do.")
        return

    tasks = [(str(f), time_limit, i) for i, f in enumerate(to_run)]
    done = 0
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_static_instance, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            write_static_row(csv_path, r)
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) if done else 0
            status = "OK" if not r.error else f"ERR: {r.error[:40]}"
            print(f"  [{done:3d}/{total}] {r.instance:<40s}  "
                  f"v={r.vehicles:3d}  d={r.distance:12.1f}  "
                  f"feas={r.feasible}  t={r.runtime_sec:6.1f}s  {status}  "
                  f"ETA={eta/60:.0f}m")

    print_summary_static(results)


def batch_dynamic(
    time_limit: int = 30,
    max_instances: int | None = None,
    resume: bool = True,
    workers: int = WORKERS,
):
    dynamic_dir = Path("data/VRPTW/ORTEC/dynamic")
    static_dir = Path("data/VRPTW/ORTEC/static")
    output_dir = Path("results/dynamic_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dynamic_results.csv"
    epoch_csv_path = output_dir / "dynamic_epoch_results.csv"

    print(f"\n[DYNAMIC] Scanning {dynamic_dir} ...")
    dyn_files = sorted(dynamic_dir.glob("ORTEC-SYNTH-*_dyn5e.json"))
    print(f"[DYNAMIC] Found {len(dyn_files)} dynamic scenarios")

    completed = load_completed_dynamic(csv_path) if resume else set()
    to_run = [f for f in dyn_files if f.stem.replace("_dyn5e", "") not in completed]
    if max_instances:
        to_run = to_run[:max_instances]

    total = len(to_run)
    print(f"[DYNAMIC] Resume: {len(completed)} done, {total} to run (max={max_instances})")

    if not to_run:
        print("[DYNAMIC] Nothing to do.")
        return

    # Pair with base instance
    tasks = []
    for f in to_run:
        base_stem = f.stem.replace("_dyn5e", "")
        base_path = static_dir / f"{base_stem}.txt"
        if base_path.exists():
            tasks.append((str(f), str(base_path), time_limit, 0))
        else:
            print(f"  [SKIP] No base instance for {f.name}")

    total = len(tasks)
    done = 0
    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_dynamic_scenario, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            write_dynamic_row(csv_path, r)
            write_epoch_rows(epoch_csv_path, r.epoch_rows)
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) if done else 0
            status = "OK" if not r.error else f"ERR: {r.error[:40]}"
            print(f"  [{done:3d}/{total}] {r.scenario:<50s}  "
                  f"served={r.overall_served_rate*100:5.1f}%  "
                  f"feas={r.overall_feasible}  "
                  f"t={r.total_runtime_sec:6.1f}s  {status}  "
                  f"ETA={eta/60:.0f}m")

    print_summary_dynamic(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch OR-Tools ORTEC experiments")
    parser.add_argument("mode", choices=["static", "dynamic", "all"],
                        help="static=dynamic scenarios only, dynamic=dynamic scenarios only, all=both")
    parser.add_argument("--time-limit", type=int, default=60,
                        help="Time limit per instance / per epoch (seconds, default 60)")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Max number of instances per type (default: all)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from previous run (skip completed)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Start fresh (ignore previous results)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Number of parallel workers (default {WORKERS})")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# OR-Tools Batch Experiment")
    print(f"# Mode: {args.mode}  |  Time limit: {args.time_limit}s  |  Workers: {args.workers}")
    print(f"# Resume: {args.resume}")
    print(f"{'#'*60}")

    total_start = time.time()

    if args.mode in ("static", "all"):
        batch_static(
            time_limit=args.time_limit,
            max_instances=args.max_instances,
            resume=args.resume,
            workers=args.workers,
        )

    if args.mode in ("dynamic", "all"):
        batch_dynamic(
            time_limit=args.time_limit,
            max_instances=args.max_instances,
            resume=args.resume,
            workers=args.workers,
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'#'*60}")
    print(f"# DONE  (total wall-clock: {total_elapsed/60:.1f} min)")
    print(f"{'#'*60}")
    print(f"  Static results:  results/static_batch/static_results.csv")
    print(f"  Dynamic results: results/dynamic_batch/dynamic_results.csv")
    print(f"  Epoch detail:    results/dynamic_batch/dynamic_epoch_results.csv")


if __name__ == "__main__":
    main()
