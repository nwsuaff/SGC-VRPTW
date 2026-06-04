"""DRoC vs OR-Tools 批量对比实验。
支持静态 VRPTW 和动态分批调度场景。

使用方法：
    # Mock LLM（快速验证，无需 API key）
    python run_droc_batch.py --mode all --llm mock --time-limit 60 --workers 2

    # OpenAI-compatible API
    python run_droc_batch.py --mode static --llm openai --time-limit 120 --workers 2

    # OpenAI
    python run_droc_batch.py --mode static --llm openai --model gpt-4o --time-limit 120 --workers 2

    # 断点续跑（跳过已完成的）
    python run_droc_batch.py --mode static --llm mock --resume

    # 仅动态场景
    python run_droc_batch.py --mode dynamic --llm mock --time-limit 60 --workers 2
"""
import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, ".")
from src.utils.io import read_instance
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.domain.schema import VRPTWInstance, RouteSolution
from src.domain.metrics import compute_gap
from src.llm.client_factory import create_llm_client
from src.pipelines.droc_pipeline import run_droc_solver
from src.pipelines.dynamic_dispatch_pipeline import DynamicDispatchPipeline, load_scenario


# ─────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────
WORKERS = 2          # 并发线程数
DEFAULT_MODE = "all"
DEFAULT_LLM  = "mock"
DEFAULT_TIME_LIMIT = 120  # 秒


# ─────────────────────────────────────────────────────────────────
# 结果结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class ComparisonResult:
    timestamp: str = ""
    instance: str = ""
    size: int = 0
    method: str = ""

    # OR-Tools 基线
    ortools_vehicles: int = 0
    ortools_distance: float = 0.0
    ortools_feasible: bool = False
    ortools_late: int = 0
    ortools_cap: int = 0
    ortools_runtime: float = 0.0

    # DRoC
    droc_vehicles: int = 0
    droc_distance: float = 0.0
    droc_feasible: bool = False
    droc_late: int = 0
    droc_cap: int = 0
    droc_runtime: float = 0.0
    droc_iterations: int = 0
    droc_llm_latency_ms: float = 0.0
    droc_code_success: bool = False
    droc_error: str = ""

    # 对比
    vehicle_improvement: float = 0.0
    distance_improvement: float = 0.0
    droc_wins: bool = False

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "instance": self.instance,
            "size": self.size,
            "method": self.method,

            "ortools_vehicles": self.ortools_vehicles,
            "ortools_distance": round(self.ortools_distance, 2),
            "ortools_feasible": self.ortools_feasible,
            "ortools_late_violations": self.ortools_late,
            "ortools_cap_violations": self.ortools_cap,
            "ortools_runtime_sec": round(self.ortools_runtime, 2),

            "droc_vehicles": self.droc_vehicles,
            "droc_distance": round(self.droc_distance, 2),
            "droc_feasible": self.droc_feasible,
            "droc_late_violations": self.droc_late,
            "droc_cap_violations": self.droc_cap,
            "droc_runtime_sec": round(self.droc_runtime, 2),
            "droc_iterations": self.droc_iterations,
            "droc_llm_latency_ms": round(self.droc_llm_latency_ms, 1),
            "droc_code_success": self.droc_code_success,
            "droc_error": self.droc_error,

            "vehicle_improvement_pct": f"{self.vehicle_improvement*100:.2f}",
            "distance_improvement_pct": f"{self.distance_improvement*100:.2f}",
            "droc_wins": self.droc_wins,
        }


@dataclass
class DynamicComparisonResult:
    timestamp: str = ""
    scenario: str = ""
    base_instance: str = ""
    method: str = ""

    ortools_served: float = 0.0
    ortools_feasible: bool = False
    ortools_late: int = 0
    ortools_runtime: float = 0.0
    ortools_distance: float = 0.0

    droc_served: float = 0.0
    droc_feasible: bool = False
    droc_late: int = 0
    droc_runtime: float = 0.0
    droc_distance: float = 0.0
    droc_iterations: int = 0
    droc_llm_latency_ms: float = 0.0
    droc_code_success: bool = False
    droc_error: str = ""

    droc_wins_served: bool = False
    droc_wins_feasible: bool = False

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "base_instance": self.base_instance,
            "method": self.method,

            "ortools_served_rate": f"{self.ortools_served:.4f}",
            "ortools_feasible": self.ortools_feasible,
            "ortools_late_violations": self.ortools_late,
            "ortools_runtime_sec": round(self.ortools_runtime, 2),
            "ortools_distance": round(self.ortools_distance, 2),

            "droc_served_rate": f"{self.droc_served:.4f}",
            "droc_feasible": self.droc_feasible,
            "droc_late_violations": self.droc_late,
            "droc_runtime_sec": round(self.droc_runtime, 2),
            "droc_distance": round(self.droc_distance, 2),
            "droc_iterations": self.droc_iterations,
            "droc_llm_latency_ms": round(self.droc_llm_latency_ms, 1),
            "droc_code_success": self.droc_code_success,
            "droc_error": self.droc_error,

            "droc_wins_served": self.droc_wins_served,
            "droc_wins_feasible": self.droc_wins_feasible,
        }


# ─────────────────────────────────────────────────────────────────
# CSV 工具
# ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/droc_batch")
STATIC_CSV  = OUTPUT_DIR / "comparison_results.csv"
DYN_CSV      = OUTPUT_DIR / "dynamic_comparison_results.csv"

STATIC_FIELDS = [
    "timestamp", "instance", "size", "method",
    "ortools_vehicles", "ortools_distance", "ortools_feasible",
    "ortools_late_violations", "ortools_cap_violations", "ortools_runtime_sec",
    "droc_vehicles", "droc_distance", "droc_feasible",
    "droc_late_violations", "droc_cap_violations", "droc_runtime_sec",
    "droc_iterations", "droc_llm_latency_ms", "droc_code_success", "droc_error",
    "vehicle_improvement_pct", "distance_improvement_pct", "droc_wins",
]

DYN_FIELDS = [
    "timestamp", "scenario", "base_instance", "method",
    "ortools_served_rate", "ortools_feasible", "ortools_late_violations",
    "ortools_runtime_sec", "ortools_distance",
    "droc_served_rate", "droc_feasible", "droc_late_violations",
    "droc_runtime_sec", "droc_distance",
    "droc_iterations", "droc_llm_latency_ms", "droc_code_success", "droc_error",
    "droc_wins_served", "droc_wins_feasible",
]


def load_done(csv_path: Path, key_col: str) -> set[str]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {row[key_col] for row in csv.DictReader(f) if row.get(key_col)}


def append_row(csv_path: Path, fields: list[str], row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─────────────────────────────────────────────────────────────────
# 单实例对比
# ─────────────────────────────────────────────────────────────────

def solve_static_comparison(inst_path: str, llm_client, time_limit: int) -> ComparisonResult:
    """对比 OR-Tools 和 DRoC 在单个静态实例上的表现。"""
    ts = datetime.now().isoformat()
    t0 = time.time()
    inst_name = Path(inst_path).stem

    try:
        inst = read_instance(str(inst_path))
    except Exception as e:
        return ComparisonResult(
            timestamp=ts, instance=inst_name, size=0, method="static",
            droc_error=f"load: {type(e).__name__}: {e}",
        )

    # ── OR-Tools 基线 ──────────────────────────────────────────────
    t_ortools = time.time()
    try:
        ortools_sol, _ = solve_ortools(inst, time_limit=float(time_limit), warm_start=None)
    except Exception as e:
        ortools_sol = None
    ortools_wall = time.time() - t_ortools

    if ortools_sol:
        ort_report = check_feasibility(inst, ortools_sol)
        ortools_vehicles = ortools_sol.vehicles_used
        ortools_distance = ortools_sol.total_distance
        ortools_feasible = ort_report.feasible
        ortools_late = ortools_sol.late_violations
        ortools_cap = ortools_sol.capacity_violations
    else:
        ortools_vehicles = ortools_distance = 0
        ortools_feasible = False
        ortools_late = ortools_cap = -1

    # ── DRoC ──────────────────────────────────────────────────────
    t_droc = time.time()
    droc_error = ""
    droc_code_success = False
    droc_iterations = 0
    droc_llm_latency_ms = 0.0

    try:
        droc_sol, gen_result, exp_result = run_droc_solver(
            instance=inst,
            llm_client=llm_client,
            max_iterations=4,
            time_limit=float(time_limit),
            droc_timeout=float(time_limit * 2),  # 允许双倍超时
            enable_self_debug=True,
            enable_rag=True,
        )
        droc_code_success = gen_result.success
        droc_iterations = gen_result.iterations
        droc_llm_latency_ms = gen_result.llm_latency_ms or 0.0
    except Exception as e:
        droc_sol = None
        droc_error = f"{type(e).__name__}: {e}"

    droc_wall = time.time() - t_droc

    if droc_sol:
        droc_report = check_feasibility(inst, droc_sol)
        droc_vehicles = droc_sol.vehicles_used
        droc_distance = droc_sol.total_distance
        droc_feasible = droc_report.feasible
        droc_late = droc_sol.late_violations
        droc_cap = droc_sol.capacity_violations
    else:
        droc_vehicles = droc_distance = 0
        droc_feasible = False
        droc_late = droc_cap = -1

    # ── 对比指标 ────────────────────────────────────────────────────
    if ortools_feasible and droc_feasible and ortools_vehicles > 0:
        vehicle_improvement = (ortools_vehicles - droc_vehicles) / ortools_vehicles
        dist_o = ortools_distance if ortools_distance > 0 else 1
        dist_d = droc_distance if droc_distance > 0 else 1
        distance_improvement = (dist_o - dist_d) / dist_o
        droc_wins = droc_vehicles < ortools_vehicles or (
            droc_vehicles == ortools_vehicles and droc_distance < ortools_distance
        )
    else:
        vehicle_improvement = distance_improvement = 0.0
        droc_wins = False

    return ComparisonResult(
        timestamp=ts,
        instance=inst.name,
        size=inst.size,
        method="droc_vs_ortools",
        ortools_vehicles=ortools_vehicles,
        ortools_distance=ortools_distance,
        ortools_feasible=ortools_feasible,
        ortools_late=ortools_late,
        ortools_cap=ortools_cap,
        ortools_runtime=ortools_wall,
        droc_vehicles=droc_vehicles,
        droc_distance=droc_distance,
        droc_feasible=droc_feasible,
        droc_late=droc_late,
        droc_cap=droc_cap,
        droc_runtime=droc_wall,
        droc_iterations=droc_iterations,
        droc_llm_latency_ms=droc_llm_latency_ms,
        droc_code_success=droc_code_success,
        droc_error=droc_error,
        vehicle_improvement=vehicle_improvement,
        distance_improvement=distance_improvement,
        droc_wins=droc_wins,
    )


def solve_dynamic_comparison(
    scen_path: str, base_path: str, llm_client, time_limit: int
) -> DynamicComparisonResult:
    """对比 OR-Tools 和 DRoC 在单个动态场景上的表现。"""
    ts = datetime.now().isoformat()
    t0 = time.time()
    scen_name = Path(scen_path).stem

    try:
        scenario = load_scenario(str(scen_path))
        base = read_instance(str(base_path))
    except Exception as e:
        return DynamicComparisonResult(
            timestamp=ts, scenario=scen_name, base_instance=Path(base_path).stem,
            method="dynamic", droc_error=f"load: {type(e).__name__}: {e}",
        )

    # ── OR-Tools 基线 ──────────────────────────────────────────────
    t_ortools = time.time()
    try:
        pipeline_ortools = DynamicDispatchPipeline(
            solver="ortools",
            llm_client=None,
            time_limit_per_epoch=float(time_limit),
            droc_timeout=float(time_limit * 3),
            output_dir=OUTPUT_DIR / "dynamic",
        )
        ort_result = pipeline_ortools.run(scenario=scenario, base_instance=base)
    except Exception as e:
        ort_result = None
    ortools_wall = time.time() - t_ortools

    if ort_result:
        ortools_served = ort_result.overall_served_rate
        ortools_feasible = ort_result.overall_feasible
        ortools_late = ort_result.total_late_violations
        ortools_distance = ort_result.total_distance
    else:
        ortools_served = 0.0
        ortools_feasible = False
        ortools_late = -1
        ortools_distance = 0.0

    # ── DRoC ──────────────────────────────────────────────────────
    t_droc = time.time()
    droc_error = ""
    droc_code_success = False
    droc_iterations = 0
    droc_llm_latency_ms = 0.0

    try:
        pipeline_droc = DynamicDispatchPipeline(
            solver="droc",
            llm_client=llm_client,
            time_limit_per_epoch=float(time_limit),
            droc_max_iterations=4,
            droc_timeout=float(time_limit * 3),
            enable_self_debug=True,
            enable_rag=True,
            output_dir=OUTPUT_DIR / "dynamic",
        )
        droc_result = pipeline_droc.run(scenario=scenario, base_instance=base)

        # 从 metadata 提取 LLM 信息（如果有）
        total_iters = 0
        total_llm_ms = 0.0
        total_success = True
        for er in droc_result.epoch_results:
            meta = getattr(er, "metadata", {}) or {}
            total_iters += meta.get("iterations", 0)
            total_llm_ms += meta.get("llm_latency_ms", 0.0)
            total_success = total_success and meta.get("code_success", True)
        droc_code_success = total_success
        droc_iterations = total_iters
        droc_llm_latency_ms = total_llm_ms
    except Exception as e:
        droc_result = None
        droc_error = f"{type(e).__name__}: {e}"

    droc_wall = time.time() - t_droc

    if droc_result:
        droc_served = droc_result.overall_served_rate
        droc_feasible = droc_result.overall_feasible
        droc_late = droc_result.total_late_violations
        droc_distance = droc_result.total_distance
    else:
        droc_served = 0.0
        droc_feasible = False
        droc_late = -1
        droc_distance = 0.0

    return DynamicComparisonResult(
        timestamp=ts,
        scenario=scenario.name,
        base_instance=base.name,
        method="droc_vs_ortools",
        ortools_served=ortools_served,
        ortools_feasible=ortools_feasible,
        ortools_late=ortools_late,
        ortools_runtime=ortools_wall,
        ortools_distance=ortools_distance,
        droc_served=droc_served,
        droc_feasible=droc_feasible,
        droc_late=droc_late,
        droc_runtime=droc_wall,
        droc_distance=droc_distance,
        droc_iterations=droc_iterations,
        droc_llm_latency_ms=droc_llm_latency_ms,
        droc_code_success=droc_code_success,
        droc_error=droc_error,
        droc_wins_served=droc_served > ortools_served,
        droc_wins_feasible=droc_feasible and not ortools_feasible,
    )


# ─────────────────────────────────────────────────────────────────
# 汇总打印
# ─────────────────────────────────────────────────────────────────

def print_static_summary(results: list[ComparisonResult]):
    total = len(results)
    ort_ok = [r for r in results if r.ortools_feasible]
    droc_ok = [r for r in results if r.droc_feasible]
    droc_success = [r for r in results if r.droc_code_success]
    both_ok = [r for r in results if r.ortools_feasible and r.droc_feasible]
    wins = [r for r in results if r.droc_wins]

    print(f"\n{'='*65}")
    print(f"  STATIC 对比汇总  ({total} instances)")
    print(f"{'='*65}")
    print(f"  OR-Tools 可行: {len(ort_ok)}/{total}")
    print(f"  DRoC  可行:    {len(droc_ok)}/{total}  (代码成功: {len(droc_success)})")
    print(f"  DRoC  胜出:   {len(wins)}/{len(both_ok)} (双可行时)")

    if both_ok:
        v_imp = [r.vehicle_improvement for r in both_ok]
        d_imp = [r.distance_improvement for r in both_ok]
        print(f"  车辆改进:  avg={sum(v_imp)/len(v_imp)*100:.1f}%  "
              f"min={min(v_imp)*100:.1f}%  max={max(v_imp)*100:.1f}%")
        print(f"  距离改进:  avg={sum(d_imp)/len(d_imp)*100:.1f}%  "
              f"min={min(d_imp)*100:.1f}%  max={max(d_imp)*100:.1f}%")

    if droc_ok:
        t = [r.droc_runtime for r in droc_ok]
        print(f"  DRoC 运行时间: avg={sum(t)/len(t):.1f}s")

    errs = [r for r in results if r.droc_error]
    if errs:
        print(f"\n  DRoC 错误 ({len(errs)} 个):")
        for r in errs[:3]:
            print(f"    {r.instance}: {r.droc_error[:80]}")

    by_size = {}
    for r in both_ok:
        by_size.setdefault(r.size, []).append(r)
    for sz in sorted(by_size):
        grp = by_size[sz]
        w = [r for r in grp if r.droc_wins]
        print(f"    [{sz:>4} 节点] {len(grp):>3} instances | DRoC wins {len(w)} | "
              f"avg_v_droc={sum(r.droc_vehicles for r in grp)/len(grp):.1f} | "
              f"avg_v_ortools={sum(r.ortools_vehicles for r in grp)/len(grp):.1f}")


def print_dynamic_summary(results: list[DynamicComparisonResult]):
    total = len(results)
    droc_success = [r for r in results if r.droc_code_success]
    wins_served = [r for r in results if r.droc_wins_served]
    wins_feas = [r for r in results if r.droc_wins_feasible]

    print(f"\n{'='*65}")
    print(f"  DYNAMIC 对比汇总  ({total} scenarios)")
    print(f"{'='*65}")
    print(f"  DRoC 代码成功:  {len(droc_success)}/{total}")
    print(f"  DRoC 服务率胜出: {len(wins_served)}/{total}")
    print(f"  DRoC 可行性胜出: {len(wins_feas)}/{total}")

    served = [r.droc_served for r in results if r.droc_served > 0]
    if served:
        print(f"  DRoC 服务率: avg={sum(served)/len(served)*100:.1f}%  "
              f"min={min(served)*100:.1f}%  max={max(served)*100:.1f}%")

    errs = [r for r in results if r.droc_error]
    if errs:
        print(f"\n  DRoC 错误 ({len(errs)} 个):")
        for r in errs[:3]:
            print(f"    {r.scenario}: {r.droc_error[:80]}")


# ─────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────

def batch_static(llm_client, time_limit: int, max_instances: int | None,
                 resume: bool, workers: int):
    static_dir = Path("data/VRPTW/ORTEC/static")
    print(f"\n[静态] 扫描 {static_dir} ...")
    txt_files = sorted(static_dir.glob("ORTEC-SYNTH-*.txt"))
    print(f"[静态] 共 {len(txt_files)} 个实例")

    done_set = load_done(STATIC_CSV, "instance") if resume else set()
    pending = [f for f in txt_files if f.stem not in done_set]
    if max_instances:
        pending = pending[:max_instances]
    total = len(pending)

    print(f"[静态] {'断点续跑' if resume else '从头开始'} | {total} 个待跑 | "
          f"{len(done_set)} 个已完成")

    if not total:
        print("[静态] 全部完成，无事可做")
        return

    start = time.time()
    done = 0
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(solve_static_comparison, str(f), llm_client, time_limit): f
                   for f in pending}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            append_row(STATIC_CSV, STATIC_FIELDS, r.to_row())
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.droc_error else f"ERR({r.droc_error[:30]})"
            print(f"  [{done:3d}/{total}] {r.instance:<42s}  "
                  f"O:v={r.ortools_vehicles:3d}  D:v={r.droc_vehicles:3d}  "
                  f"D_win={r.droc_wins}  t={r.droc_runtime:6.1f}s  "
                  f"{status}  ETA={eta:.0f}m")

    print_static_summary(results)


def batch_dynamic(llm_client, time_limit: int, max_instances: int | None,
                  resume: bool, workers: int):
    dynamic_dir = Path("data/VRPTW/ORTEC/dynamic")
    static_dir  = Path("data/VRPTW/ORTEC/static")

    print(f"\n[动态] 扫描 {dynamic_dir} ...")
    dyn_files = sorted(dynamic_dir.glob("ORTEC-SYNTH-*_dyn5e.json"))
    print(f"[动态] 共 {len(dyn_files)} 个场景")

    done_set = load_done(DYN_CSV, "scenario") if resume else set()
    pending = [f for f in dyn_files if f.stem not in done_set]
    if max_instances:
        pending = pending[:max_instances]
    total = len(pending)

    tasks = []
    for f in pending:
        base_stem = f.stem.replace("_dyn5e", "")
        base_path = static_dir / f"{base_stem}.txt"
        if base_path.exists():
            tasks.append((str(f), str(base_path)))
        else:
            print(f"  [SKIP] 找不到 base: {base_stem}.txt")

    total = len(tasks)
    print(f"[动态] {'断点续跑' if resume else '从头开始'} | {total} 个待跑 | "
          f"{len(done_set)} 个已完成")

    if not total:
        print("[动态] 全部完成，无事可做")
        return

    start = time.time()
    done = 0
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(solve_dynamic_comparison, p[0], p[1], llm_client, time_limit): p
                   for p in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            append_row(DYN_CSV, DYN_FIELDS, r.to_row())
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.droc_error else f"ERR({r.droc_error[:30]})"
            print(f"  [{done:3d}/{total}] {r.scenario:<50s}  "
                  f"O:srv={r.ortools_served*100:5.1f}%  D:srv={r.droc_served*100:5.1f}%  "
                  f"D_win={r.droc_wins_served}  "
                  f"t={r.droc_runtime:6.1f}s  {status}  ETA={eta:.0f}m")

    print_dynamic_summary(results)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DRoC vs OR-Tools 批量对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Mock LLM 快速验证（无需 API key）
  python run_droc_batch.py --mode all --llm mock --time-limit 60 --workers 2

  # OpenAI-compatible API
  python run_droc_batch.py --mode static --llm openai --time-limit 120 --workers 2

  # OpenAI GPT-4o
  python run_droc_batch.py --mode static --llm openai --model gpt-4o --time-limit 120 --workers 2

  # 断点续跑
  python run_droc_batch.py --mode dynamic --llm mock --resume

  # 每类各跑 5 个（快速测试）
  python run_droc_batch.py --mode all --llm mock --max-instances 5 --time-limit 30

输出文件:
  results/droc_batch/comparison_results.csv          # 静态对比结果
  results/droc_batch/dynamic_comparison_results.csv   # 动态对比结果
""",
    )
    parser.add_argument("--mode", choices=["static", "dynamic", "all"],
                        default=DEFAULT_MODE,
                        help=f"实验类型 (默认: {DEFAULT_MODE})")
    parser.add_argument("--llm", default=DEFAULT_LLM,
                        help=f"LLM type: mock/openai/anthropic (default: {DEFAULT_LLM})")
    parser.add_argument("--model", default=None,
                        help="模型名称 (如 gpt-4o, gpt-5.1)")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
                        help=f"时间限制秒数 (默认: {DEFAULT_TIME_LIMIT})")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="每类最多跑多少个 (默认: 全部)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"并发线程数 (默认: {WORKERS})")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续跑 (默认开启)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="从头开始")
    args = parser.parse_args()

    # ── 初始化 LLM 客户端 ──────────────────────────────────────────
    print(f"\n[初始化] LLM client = {args.llm}", end="")
    if args.model:
        print(f", model = {args.model}")
    else:
        print()

    try:
        llm_client = create_llm_client(
            name=args.llm,
            model=args.model,
        )
        print(f"[初始化] LLM client 创建成功: {type(llm_client).__name__}")
    except Exception as e:
        print(f"[错误] LLM client 初始化失败: {e}")
        print("提示: 使用 --llm mock 进行无 API key 测试")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"#  DRoC vs OR-Tools 对比实验")
    print(f"#  模式: {args.mode}  |  LLM: {args.llm}  |  时间限制: {args.time_limit}s")
    print(f"#  并发: {args.workers}  |  断点续跑: {args.resume}  |  最大实例: {args.max_instances or '全部'}")
    print(f"{'#'*65}")

    t_start = time.time()

    if args.mode in ("static", "all"):
        batch_static(llm_client, args.time_limit, args.max_instances, args.resume, args.workers)

    if args.mode in ("dynamic", "all"):
        batch_dynamic(llm_client, args.time_limit, args.max_instances, args.resume, args.workers)

    elapsed = time.time() - t_start
    print(f"\n{'#'*65}")
    print(f"#  完成  (总耗时: {elapsed/60:.1f} 分钟)")
    print(f"#  静态结果: {STATIC_CSV}")
    print(f"#  动态结果: {DYN_CSV}")
    print(f"{'#'*65}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
