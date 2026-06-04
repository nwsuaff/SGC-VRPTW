"""
DRoC 大规模对比实验（对应 run_batch.py 的 OR-Tools 结果）

使用方法：
    python run_batch_droc.py --help

    # 快速验证（3个实例，mock LLM，不花 API 费用）
    python run_batch_droc.py --mode all --max-instances 3 --llm mock --time-limit 60

    # Full run with an OpenAI-compatible backend
    python run_batch_droc.py --mode all --llm openai --time-limit 300 --workers 2

    # 继续上次中断
    python run_batch_droc.py --mode all --llm openai --time-limit 300 --resume

在 PowerShell / cmd 中运行
"""
import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
from src.utils.io import read_instance
from src.llm.client_factory import create_llm_client, is_llm_available
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.pipelines.droc_pipeline import run_droc_solver
from src.pipelines.dynamic_dispatch_pipeline import DynamicDispatchPipeline, load_scenario

OUTPUT_DIR = Path("results/droc_batch")
STATIC_CSV = OUTPUT_DIR / "static_results.csv"
DYN_CSV    = OUTPUT_DIR / "dynamic_results.csv"
EPOCH_CSV  = OUTPUT_DIR / "dynamic_epoch_results.csv"


# ─────────────────────────────────────────────────────────────────
# Static
# ─────────────────────────────────────────────────────────────────
@dataclass
class StaticResult:
    instance: str = ""
    size: int = 0
    llm: str = ""
    time_limit: int = 0
    droc_iterations: int = 0
    vehicles: int = 0
    distance: float = 0.0
    feasible: bool = False
    late_violations: int = 0
    cap_violations: int = 0
    runtime_sec: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def solve_static_droc(args_tuple):
    inst_path, base_solution, time_limit, max_iterations, llm_name = args_tuple
    t0 = time.time()
    try:
        inst = read_instance(str(inst_path))
        llm = create_llm_client(llm_name)

        solution, gen_result, exp_result = run_droc_solver(
            instance=inst,
            llm_client=llm,
            max_iterations=max_iterations,
            time_limit=float(time_limit),
            droc_timeout=float(time_limit + 60),
            output_dir=None,
            incumbent=base_solution,
        )

        return StaticResult(
            instance=inst.name,
            size=inst.size,
            llm=llm_name,
            time_limit=time_limit,
            droc_iterations=gen_result.iterations if gen_result else 0,
            vehicles=solution.vehicles_used if solution else 0,
            distance=solution.total_distance if solution else 0.0,
            feasible=solution.feasible if solution else False,
            late_violations=solution.late_violations if solution else -1,
            cap_violations=solution.capacity_violations if solution else -1,
            runtime_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        return StaticResult(
            instance=Path(inst_path).stem,
            llm=llm_name,
            time_limit=time_limit,
            runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


# ─────────────────────────────────────────────────────────────────
# Dynamic
# ─────────────────────────────────────────────────────────────────
@dataclass
class EpochRow:
    scenario: str; base: str; epoch: int
    served_rate: float; feasible: bool
    late_violations: int; cap_violations: int
    runtime_sec: float; orders_dispatched: int
    orders_pending: int; distance: float; error: str = ""


@dataclass
class DynamicResult:
    scenario: str = ""
    base_instance: str = ""
    num_epochs: int = 0
    llm: str = ""
    time_limit: int = 0
    overall_served_rate: float = 0.0
    overall_feasible: bool = False
    total_late: int = 0
    total_cap_viol: int = 0
    total_distance: float = 0.0
    total_runtime_sec: float = 0.0
    error: str = ""
    epochs: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def solve_dynamic_droc(args_tuple):
    scen_path, base_path, time_limit, max_iterations, llm_name = args_tuple
    t0 = time.time()
    try:
        scenario = load_scenario(str(scen_path))
        base = read_instance(str(base_path))
        llm = create_llm_client(llm_name)

        droc_timeout = max(time_limit * scenario.num_epochs, time_limit * 3)

        pipeline = DynamicDispatchPipeline(
            solver="droc",
            llm_client=llm,
            time_limit_per_epoch=float(time_limit),
            droc_timeout=float(droc_timeout),
            output_dir=OUTPUT_DIR / "dynamic",
        )
        result = pipeline.run(scenario=scenario, base_instance=base)

        epoch_rows = [
            EpochRow(
                scenario=scenario.name, base=base.name,
                epoch=er.epoch, served_rate=er.served_rate,
                feasible=er.feasible, late_violations=er.late_violations,
                cap_violations=er.capacity_violations,
                runtime_sec=er.runtime_sec,
                orders_dispatched=er.orders_dispatched,
                orders_pending=er.orders_pending,
                distance=er.total_distance,
            )
            for er in result.epoch_results
        ]

        return DynamicResult(
            scenario=scenario.name,
            base_instance=base.name,
            num_epochs=result.num_epochs,
            llm=llm_name,
            time_limit=time_limit,
            overall_served_rate=result.overall_served_rate,
            overall_feasible=result.overall_feasible,
            total_late=result.total_late_violations,
            total_cap_viol=result.total_capacity_violations,
            total_distance=result.total_distance,
            total_runtime_sec=round(time.time() - t0, 2),
            epochs=epoch_rows,
        )
    except Exception as e:
        return DynamicResult(
            scenario=Path(scen_path).stem,
            base_instance=Path(base_path).stem,
            llm=llm_name,
            time_limit=time_limit,
            total_runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
            epochs=[],
        )


# ─────────────────────────────────────────────────────────────────
# CSV utilities
# ─────────────────────────────────────────────────────────────────
STATIC_FIELDS = [
    "timestamp", "instance", "size", "llm", "time_limit",
    "droc_iterations", "vehicles", "distance", "feasible",
    "late_violations", "capacity_violations", "runtime_sec", "error",
]
DYN_FIELDS = [
    "timestamp", "scenario", "base_instance", "num_epochs", "llm",
    "time_limit", "overall_served_rate", "overall_feasible",
    "total_late_violations", "total_capacity_violations",
    "total_distance", "total_runtime_sec", "error",
]
EPOCH_FIELDS = [
    "scenario", "base", "epoch", "served_rate", "feasible",
    "late_violations", "capacity_violations", "runtime_sec",
    "orders_dispatched", "orders_pending", "distance", "error",
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
# OR-Tools baseline (for incumbent warm-start)
# ─────────────────────────────────────────────────────────────────
def load_ortools_baseline(csv_path: Path) -> dict[str, dict]:
    """Load OR-Tools results to get incumbent solutions for warm-start."""
    baselines = {}
    if not csv_path.exists():
        return baselines
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("feasible") == "True" and row.get("instance"):
                baselines[row["instance"]] = {
                    "vehicles": int(row["vehicles"]),
                    "distance": float(row["distance"]),
                    "runtime": float(row["runtime_sec"]),
                }
    return baselines


def get_incumbent(inst_path: str, baselines: dict):
    """Return a minimal incumbent dict for warm-starting DRoC."""
    import dataclasses
    from src.domain.schema import RouteSolution

    name = Path(inst_path).stem
    if name not in baselines:
        return None
    b = baselines[name]
    # Return a minimal RouteSolution so DRoC knows the baseline quality
    return RouteSolution(
        routes=[[]],  # placeholder, will be replaced by DRoC
        vehicles_used=b["vehicles"],
        total_distance=b["distance"],
        total_duration=0.0,  # baseline CSV does not store duration
        feasible=True,
        late_violations=0,
        capacity_violations=0,
    )





# ─────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────
def print_static_summary(results: list[StaticResult]):
    total = len(results)
    ok = [r for r in results if r.feasible and not r.error]
    err = [r for r in results if r.error]
    print(f"\n{'='*65}")
    print(f"  DRoC STATIC  ({total} instances)")
    print(f"{'='*65}")
    print(f"  可行率:   {len(ok)}/{total}  ({len(ok)/max(total,1)*100:.1f}%)")
    print(f"  错误数:   {len(err)}")
    if ok:
        v = [r.vehicles for r in ok]; d = [r.distance for r in ok]
        t = [r.runtime_sec for r in ok]
        print(f"  车辆数:   avg={sum(v)/len(v):.1f}  min={min(v)}  max={max(v)}")
        print(f"  距离:     avg={sum(d)/len(d):.0f}  min={min(d):.0f}  max={max(d):.0f}")
        print(f"  运行时间: avg={sum(t)/len(t):.1f}s  min={min(t):.1f}s  max={max(t):.1f}s")
        iters = [r.droc_iterations for r in ok if r.droc_iterations > 0]
        if iters:
            print(f"  DRoC迭代: avg={sum(iters)/len(iters):.1f}  max={max(iters)}")
        by_size = {}
        for r in ok:
            by_size.setdefault(r.size, []).append(r)
        for sz in sorted(by_size):
            grp = by_size[sz]
            print(f"    [{sz:>4} 节点] {len(grp):>3} 个 | "
                  f"avg_v={sum(x.vehicles for x in grp)/len(grp):.1f}  "
                  f"avg_d={sum(x.distance for x in grp)/len(grp):.0f}")
    if err:
        print(f"\n  错误 ({len(err)}):")
        for r in err[:5]:
            print(f"    {r.instance}: {r.error[:80]}")


def print_dynamic_summary(results: list[DynamicResult]):
    total = len(results)
    ok = [r for r in results if r.overall_feasible and not r.error]
    err = [r for r in results if r.error]
    served = [r.overall_served_rate for r in results if not r.error]
    times  = [r.total_runtime_sec for r in results if not r.error]
    print(f"\n{'='*65}")
    print(f"  DRoC DYNAMIC  ({total} scenarios)")
    print(f"{'='*65}")
    print(f"  可行率:   {len(ok)}/{total}  ({len(ok)/max(total,1)*100:.1f}%)")
    print(f"  错误数:   {len(err)}")
    if served:
        print(f"  服务率:   avg={sum(served)/len(served)*100:.1f}%  "
              f"min={min(served)*100:.1f}%  max={max(served)*100:.1f}%")
    if times:
        print(f"  运行时间: avg={sum(times)/len(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")
    if err:
        print(f"\n  错误 ({len(err)}):")
        for r in err[:5]:
            print(f"    {r.scenario}: {r.error[:80]}")


# ─────────────────────────────────────────────────────────────────
# Main batch
# ─────────────────────────────────────────────────────────────────
def run_static(time_limit, max_instances, resume, workers, llm, max_iterations, rerun_failed=False):
    static_dir = Path("data/VRPTW/ORTEC/static")
    ortools_csv = Path("results/ortools_batch/static_results.csv")

    print(f"\n[DRoC 静态] 扫描 {static_dir} ...")
    files = sorted(static_dir.glob("ORTEC-SYNTH-*.txt"))
    print(f"[DRoC 静态] 共 {len(files)} 个实例 | LLM={llm} | max_iter={max_iterations}")

    baselines = load_ortools_baseline(ortools_csv)
    print(f"[DRoC 静态] 加载 OR-Tools baseline: {len(baselines)} 个可行解")

    # --rerun-failed: 识别 feasible=False 的实例，清除旧记录，重新跑
    if rerun_failed and STATIC_CSV.exists():
        import pandas as pd
        df = pd.read_csv(STATIC_CSV)
        failed_rows = df[df["feasible"] == False]
        if len(failed_rows) > 0:
            print(f"[DRoC 静态] --rerun-failed: 发现 {len(failed_rows)} 个失败实例，清除旧记录 ...")
            # 仅保留 feasible=True 的行
            df_clean = df[df["feasible"] == True]
            df_clean.to_csv(STATIC_CSV, index=False)
            failed_names = set(failed_rows["instance"].tolist())
            pending = [f for f in files if f.stem in failed_names]
            print(f"[DRoC 静态] 将重新跑 {len(pending)} 个失败实例")
        else:
            print(f"[DRoC 静态] --rerun-failed: 无失败实例需要重跑")
            return
    else:
        done_set = load_done(STATIC_CSV, "instance") if resume else set()
        pending  = [f for f in files if f.stem not in done_set]
        if max_instances:
            pending = pending[:max_instances]
        total = len(pending)
        print(f"[DRoC 静态] {'断点续跑' if resume else '从头'} | {total} 待跑 | {len(done_set)} 已完成")

    total = len(pending)
    if not total:
        print("[DRoC 静态] 全部完成，无事可做")
        return

    start = time.time(); done = 0; results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        tasks = [
            (str(f), get_incumbent(str(f), baselines), time_limit, max_iterations, llm)
            for f in pending
        ]
        futures = {ex.submit(solve_static_droc, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result(); results.append(r)
            append_row(STATIC_CSV, STATIC_FIELDS, {
                "timestamp": r.timestamp, "instance": r.instance,
                "size": r.size, "llm": r.llm, "time_limit": r.time_limit,
                "droc_iterations": r.droc_iterations,
                "vehicles": r.vehicles, "distance": round(r.distance, 2),
                "feasible": r.feasible,
                "late_violations": r.late_violations,
                "capacity_violations": r.cap_violations,
                "runtime_sec": r.runtime_sec, "error": r.error,
            })
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:30]})"
            print(f"  [{done:3d}/{total}] {r.instance:<42s}  "
                  f"v={r.vehicles:3d}  d={r.distance:12.1f}  "
                  f"feas={r.feasible}  iter={r.droc_iterations}  "
                  f"t={r.runtime_sec:6.1f}s  {status}  ETA={eta:.0f}m")

    print_static_summary(results)


def run_dynamic(time_limit, max_instances, resume, workers, llm, max_iterations, rerun_failed=False):
    dynamic_dir = Path("data/VRPTW/ORTEC/dynamic")
    static_dir  = Path("data/VRPTW/ORTEC/static")
    droc_timeout_global = time_limit * 10  # hard cap

    print(f"\n[DRoC 动态] 扫描 {dynamic_dir} ...")
    dyn_files = sorted(dynamic_dir.glob("ORTEC-SYNTH-*_dyn5e.json"))
    print(f"[DRoC 动态] 共 {len(dyn_files)} 个场景 | LLM={llm} | max_iter={max_iterations}")

    # --rerun-failed: 识别 overall_feasible=False 的场景，清除旧记录，重新跑
    if rerun_failed and DYN_CSV.exists():
        import pandas as pd
        df = pd.read_csv(DYN_CSV)
        failed_rows = df[df["overall_feasible"] == False]
        if len(failed_rows) > 0:
            print(f"[DRoC 动态] --rerun-failed: 发现 {len(failed_rows)} 个失败场景，清除旧记录 ...")
            # 仅保留 feasible=True 的行
            df_clean = df[df["overall_feasible"] == True]
            df_clean.to_csv(DYN_CSV, index=False)
            # 同时清除对应的 epoch 记录
            if EPOCH_CSV.exists():
                df_ep = pd.read_csv(EPOCH_CSV)
                kept_scenarios = set(df_clean["scenario"].tolist())
                df_ep_clean = df_ep[df_ep["scenario"].isin(kept_scenarios)]
                df_ep_clean.to_csv(EPOCH_CSV, index=False)
            failed_scenarios = set(failed_rows["scenario"].tolist())
            pending = [f for f in dyn_files if f.stem in failed_scenarios]
            print(f"[DRoC 动态] 将重新跑 {len(pending)} 个失败场景")
        else:
            print(f"[DRoC 动态] --rerun-failed: 无失败场景需要重跑")
            return
    else:
        done_set = load_done(DYN_CSV, "scenario") if resume else set()
        pending  = [f for f in dyn_files if f.stem not in done_set]
        if max_instances:
            pending = pending[:max_instances]
        total = len(pending)
        print(f"[DRoC 动态] {'断点续跑' if resume else '从头'} | {total} 待跑 | {len(done_set)} 已完成")

    tasks = []
    for f in pending:
        base_stem = f.stem.replace("_dyn5e", "")
        base_path = static_dir / f"{base_stem}.txt"
        if base_path.exists():
            tasks.append((str(f), str(base_path), time_limit, max_iterations, llm))
        else:
            print(f"  [SKIP] 找不到 base: {base_stem}.txt")

    total = len(tasks)
    if not total:
        print("[DRoC 动态] 全部完成，无事可做")
        return

    start = time.time(); done = 0; results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(solve_dynamic_droc, t): t for t in tasks}
        for fut in as_completed(futures):
            r = fut.result(); results.append(r)
            append_row(DYN_CSV, DYN_FIELDS, {
                "timestamp": r.timestamp, "scenario": r.scenario,
                "base_instance": r.base_instance, "num_epochs": r.num_epochs,
                "llm": r.llm, "time_limit": r.time_limit,
                "overall_served_rate": f"{r.overall_served_rate:.4f}",
                "overall_feasible": r.overall_feasible,
                "total_late_violations": r.total_late,
                "total_capacity_violations": r.total_cap_viol,
                "total_distance": round(r.total_distance, 2),
                "total_runtime_sec": r.total_runtime_sec,
                "error": r.error,
            })
            for ep in r.epochs:
                append_row(EPOCH_CSV, EPOCH_FIELDS, {
                    "scenario": ep.scenario, "base": ep.base,
                    "epoch": ep.epoch, "served_rate": f"{ep.served_rate:.4f}",
                    "feasible": ep.feasible,
                    "late_violations": ep.late_violations,
                    "capacity_violations": ep.cap_violations,
                    "runtime_sec": ep.runtime_sec,
                    "orders_dispatched": ep.orders_dispatched,
                    "orders_pending": ep.orders_pending,
                    "distance": round(ep.distance, 2),
                    "error": ep.error,
                })
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:30]})"
            print(f"  [{done:3d}/{total}] {r.scenario:<50s}  "
                  f"served={r.overall_served_rate*100:5.1f}%  "
                  f"feas={r.overall_feasible}  "
                  f"t={r.total_runtime_sec:6.1f}s  "
                  f"{status}  ETA={eta:.0f}m")

    print_dynamic_summary(results)


# ─────────────────────────────────────────────────────────────────
# Solomon
# ─────────────────────────────────────────────────────────────────
SOLOMON_STATIC_CSV = OUTPUT_DIR / "solomon_results.csv"


def run_solomon(time_limit, max_instances, resume, workers, llm, max_iterations):
    """Run DRoC on Solomon 100 benchmark (56 instances, C1/C2/R1/R2/RC1/RC2 families)."""
    from src.data.solomon_loader import load_solomon_instance

    solomon_dir = Path("data/VRPTW/Solomon")
    solomon_files = sorted(solomon_dir.glob("*.vrp"))
    print(f"\n[DRoC Solomon] 扫描 {solomon_dir} ...")
    print(f"[DRoC Solomon] 共 {len(solomon_files)} 个实例 | LLM={llm} | max_iter={max_iterations}")

    done_set = load_done(SOLOMON_STATIC_CSV, "instance") if resume else set()
    pending = [f for f in solomon_files if f.stem not in done_set]
    if max_instances:
        pending = pending[:max_instances]
    total = len(pending)
    print(f"[DRoC Solomon] {'断点续跑' if resume else '从头'} | {total} 待跑 | {len(done_set)} 已完成")

    if not total:
        print("[DRoC Solomon] 全部完成，无事可做")
        return

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for f in pending:
            try:
                inst = load_solomon_instance(str(f))
                llm_client = create_llm_client(llm)
                task = (
                    str(f),           # inst_path
                    None,             # no OR-Tools baseline incumbent for Solomon
                    time_limit,
                    max_iterations,
                    llm,
                    inst,              # pass loaded instance directly
                )
                futures[ex.submit(_solve_solomon_droc, task)] = f
            except Exception as e:
                print(f"  [SKIP] {f.stem}: 加载失败: {e}")

        done = 0
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = StaticResult(
                    instance=f.stem,
                    llm=llm,
                    time_limit=time_limit,
                    runtime_sec=0.0,
                    error=f"{type(e).__name__}: {e}",
                )
            results.append(r)

            row = {
                "timestamp": r.timestamp,
                "instance": r.instance,
                "size": r.size,
                "llm": r.llm,
                "time_limit": r.time_limit,
                "droc_iterations": r.droc_iterations,
                "vehicles": r.vehicles,
                "distance": round(r.distance, 2),
                "feasible": r.feasible,
                "late_violations": r.late_violations,
                "capacity_violations": r.cap_violations,
                "runtime_sec": r.runtime_sec,
                "error": r.error,
            }
            append_row(SOLOMON_STATIC_CSV, STATIC_FIELDS, row)

            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:30]})"
            print(f"  [{done:3d}/{total}] {r.instance:<10s}  "
                  f"v={r.vehicles:3d}  d={r.distance:10.1f}  "
                  f"feas={r.feasible}  iter={r.droc_iterations}  "
                  f"t={r.runtime_sec:6.1f}s  {status}  ETA={eta:.0f}m")

    print_static_summary(results)


def _solve_solomon_droc(args_tuple):
    """Solve a single Solomon instance with DRoC."""
    inst_path, _, time_limit, max_iterations, llm_name, inst = args_tuple
    t0 = time.time()
    try:
        llm = create_llm_client(llm_name)
        solution, gen_result, exp_result = run_droc_solver(
            instance=inst,
            llm_client=llm,
            max_iterations=max_iterations,
            time_limit=float(time_limit),
            droc_timeout=float(time_limit + 60),
            output_dir=None,
            incumbent=None,  # no incumbent for Solomon (no OR-Tools baseline)
        )
        return StaticResult(
            instance=inst.name or Path(inst_path).stem,
            size=inst.size,
            llm=llm_name,
            time_limit=time_limit,
            droc_iterations=gen_result.iterations if gen_result else 0,
            vehicles=solution.vehicles_used if solution else 0,
            distance=solution.total_distance if solution else 0.0,
            feasible=solution.feasible if solution else False,
            late_violations=solution.late_violations if solution else -1,
            cap_violations=solution.capacity_violations if solution else -1,
            runtime_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        return StaticResult(
            instance=Path(inst_path).stem,
            llm=llm_name,
            time_limit=time_limit,
            runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="DRoC 大规模 VRPTW 对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
注意事项：
  - DRoC 需要 LLM API 调用，会产生费用
  - 推荐先用 --llm mock 验证脚本无误后再切真实 API
  - --llm openai uses OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL
  - OR-Tools baseline 结果会被自动加载作为 incumbent（热启动）

示例:
  # Mock 测试（不花 API 费用）
  python run_batch_droc.py --mode all --llm mock --max-instances 3 --time-limit 60

  # 正式运行
  python run_batch_droc.py --mode all --llm openai --time-limit 300 --workers 2

  # 仅动态，续跑
  python run_batch_droc.py --mode dynamic --llm openai --resume

  # 重跑所有失败的实例（静态 + 动态）
  python run_batch_droc.py --mode all --llm openai --time-limit 300 --rerun-failed

  # 仅重跑静态失败实例
  python run_batch_droc.py --mode static --llm openai --time-limit 300 --rerun-failed

  # 仅重跑动态失败场景
  python run_batch_droc.py --mode dynamic --llm openai --time-limit 300 --rerun-failed

输出文件:
  results/droc_batch/static_results.csv
  results/droc_batch/dynamic_results.csv
  results/droc_batch/dynamic_epoch_results.csv
""",
    )
    p.add_argument(
        "--mode", choices=["static", "dynamic", "solomon", "all"], default="all",
        help="实验类型: static(ORTEC静态), dynamic(ORTEC动态), solomon(Solomon100), all(全部，默认)",
    )
    p.add_argument(
        "--llm", default="openai",
        help="LLM client name: openai, anthropic, or mock (default: openai)",
    )
    p.add_argument(
        "--time-limit", type=int, default=300,
        help="每 epoch / 软时间限制，秒 (默认: 300)",
    )
    p.add_argument(
        "--max-iterations", type=int, default=4,
        help="DRoC 最大自迭代次数 (默认: 4)",
    )
    p.add_argument(
        "--max-instances", type=int, default=None,
        help="每类最多跑多少个 (默认: 全部)",
    )
    p.add_argument(
        "--workers", type=int, default=2,
        help="并发线程数 (默认: 2，DRoC 有 API 限流，建议 <=2)",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument(
        "--rerun-failed", action="store_true", default=False,
        help="识别并重新运行失败的实例（feasible=False），覆盖旧结果",
    )
    args = p.parse_args()

    # 检查 LLM 可用性
    if args.llm != "mock" and not is_llm_available(args.llm):
        print(f"[WARN] LLM '{args.llm}' 不可用，将使用 mock")
        args.llm = "mock"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"#  DRoC 批量实验")
    print(f"#  模式: {args.mode}  |  LLM: {args.llm}  |  max_iter: {args.max_iterations}")
    print(f"#  时间限制: {args.time_limit}s  |  并发: {args.workers}  |  断点续跑: {args.resume}")
    print(f"{'#'*65}")

    t_start = time.time()

    if args.mode in ("static", "all"):
        run_static(
            time_limit=args.time_limit,
            max_instances=args.max_instances,
            resume=args.resume,
            workers=args.workers,
            llm=args.llm,
            max_iterations=args.max_iterations,
            rerun_failed=args.rerun_failed,
        )

    if args.mode in ("dynamic", "all"):
        run_dynamic(
            time_limit=args.time_limit,
            max_instances=args.max_instances,
            resume=args.resume,
            workers=args.workers,
            llm=args.llm,
            max_iterations=args.max_iterations,
            rerun_failed=args.rerun_failed,
        )

    if args.mode in ("solomon", "all"):
        run_solomon(
            time_limit=args.time_limit,
            max_instances=args.max_instances,
            resume=args.resume,
            workers=args.workers,
            llm=args.llm,
            max_iterations=args.max_iterations,
        )

    elapsed = time.time() - t_start
    print(f"\n{'#'*65}")
    print(f"#  完成  (总耗时: {elapsed/60:.1f} 分钟)")
    print(f"#  静态: {STATIC_CSV}")
    print(f"#  动态: {DYN_CSV}")
    print(f"#  Epoch: {EPOCH_CSV}")
    print(f"{'#'*65}")


if __name__ == "__main__":
    main()
