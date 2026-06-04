"""
OR-Tools 大规模对比实验
静态 VRPTW + 动态分批调度
适用：ORTEC-SYNTH-300/500/800 全套数据

使用方法：
    python run_batch.py                      # 默认配置（见下方 CONFIG）
    python run_batch.py --help                # 查看所有选项

    # 指定参数
    python run_batch.py --mode all --time-limit 120 --workers 4 --resume

在 PowerShell / cmd 中运行，不要在 Python IDE 内运行（避免 multiprocessing 权限问题）
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

# ─────────────────────────────────────────────────────────────────
# 配置（根据需要修改这里）
# ─────────────────────────────────────────────────────────────────
DEFAULT_TIME_LIMIT   = 120   # 秒 / instance，或每个 epoch
DEFAULT_WORKERS      = 4
DEFAULT_MODE         = "all"  # "static" | "dynamic" | "all"
DEFAULT_MAX_INSTANCES = None  # None = 全部，数字 = 限制数量
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, ".")
from src.utils.io import read_instance
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.pipelines.dynamic_dispatch_pipeline import DynamicDispatchPipeline, load_scenario


# ─────────────────────────────────────────────────────────────────
# Static 结果结构
# ─────────────────────────────────────────────────────────────────
@dataclass
class StaticResult:
    timestamp: str = ""
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

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "instance": self.instance,
            "size": self.size,
            "solver": self.solver,
            "time_limit_sec": self.time_limit,
            "vehicles": self.vehicles,
            "distance": round(self.distance, 2),
            "feasible": self.feasible,
            "late_violations": self.late_violations,
            "capacity_violations": self.cap_violations,
            "runtime_sec": round(self.runtime_sec, 2),
            "error": self.error,
        }


def solve_static(inst_path: str, time_limit: int) -> StaticResult:
    """求解单个静态实例。"""
    t0 = time.time()
    ts = datetime.now().isoformat()
    try:
        inst = read_instance(str(inst_path))
        solution, _ = solve_ortools(inst, time_limit=float(time_limit), warm_start=None)

        if solution is None:
            return StaticResult(
                timestamp=ts, instance=inst.name, size=inst.size,
                solver="ortools", time_limit=time_limit,
                runtime_sec=time.time() - t0,
                error="No solution returned",
            )

        # 可行性验证
        report = check_feasibility(inst, solution)

        return StaticResult(
            timestamp=ts,
            instance=inst.name,
            size=inst.size,
            solver="ortools",
            time_limit=time_limit,
            vehicles=solution.vehicles_used,
            distance=solution.total_distance,
            feasible=report.feasible,
            late_violations=solution.late_violations,
            cap_violations=solution.capacity_violations,
            runtime_sec=time.time() - t0,
        )
    except Exception as e:
        return StaticResult(
            timestamp=ts,
            instance=Path(inst_path).stem,
            solver="ortools",
            time_limit=time_limit,
            runtime_sec=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )


# ─────────────────────────────────────────────────────────────────
# Dynamic 结果结构
# ─────────────────────────────────────────────────────────────────
@dataclass
class EpochRow:
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

    def to_row(self) -> dict:
        return {
            "scenario": self.scenario,
            "base": self.base,
            "epoch": self.epoch,
            "served_rate": f"{self.served_rate:.4f}",
            "feasible": self.feasible,
            "late_violations": self.late_violations,
            "capacity_violations": self.cap_violations,
            "runtime_sec": round(self.runtime_sec, 2),
            "orders_dispatched": self.orders_dispatched,
            "orders_pending": self.orders_pending,
            "distance": round(self.distance, 2),
            "error": self.error,
        }


@dataclass
class DynamicResult:
    timestamp: str = ""
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
    epochs: list = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "scenario": self.scenario,
            "base_instance": self.base_instance,
            "num_epochs": self.num_epochs,
            "solver": self.solver,
            "time_limit_sec": self.time_limit,
            "overall_served_rate": f"{self.overall_served_rate:.4f}",
            "overall_feasible": self.overall_feasible,
            "total_late_violations": self.total_late,
            "total_capacity_violations": self.total_cap_viol,
            "total_distance": round(self.total_distance, 2),
            "total_runtime_sec": round(self.total_runtime_sec, 2),
            "error": self.error,
        }


def solve_dynamic(scen_path: str, base_path: str, time_limit: int) -> DynamicResult:
    """求解单个动态场景（epoch by epoch）。"""
    t0 = time.time()
    ts = datetime.now().isoformat()
    try:
        scenario = load_scenario(str(scen_path))
        base = read_instance(str(base_path))

        pipeline = DynamicDispatchPipeline(
            solver="ortools",
            llm_client=None,
            time_limit_per_epoch=float(time_limit),
            droc_timeout=float(time_limit * 3),
            output_dir=OUTPUT_DIR / "dynamic",
        )
        result = pipeline.run(scenario=scenario, base_instance=base)

        epoch_rows = [
            EpochRow(
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
        ]

        return DynamicResult(
            timestamp=ts,
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
            total_runtime_sec=time.time() - t0,
            epochs=epoch_rows,
        )
    except Exception as e:
        return DynamicResult(
            timestamp=ts,
            scenario=Path(scen_path).stem,
            base_instance=Path(base_path).stem,
            num_epochs=0,
            solver="ortools",
            time_limit=time_limit,
            total_runtime_sec=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
            epochs=[],
        )


# ─────────────────────────────────────────────────────────────────
# CSV 工具
# ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/ortools_batch")
STATIC_CSV = OUTPUT_DIR / "static_results.csv"
DYN_CSV     = OUTPUT_DIR / "dynamic_results.csv"
EPOCH_CSV   = OUTPUT_DIR / "dynamic_epoch_results.csv"

STATIC_FIELDS = [
    "timestamp", "instance", "size", "solver", "time_limit_sec",
    "vehicles", "distance", "feasible", "late_violations",
    "capacity_violations", "runtime_sec", "error",
]
DYN_FIELDS = [
    "timestamp", "scenario", "base_instance", "num_epochs", "solver",
    "time_limit_sec", "overall_served_rate", "overall_feasible",
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
# 汇总打印
# ─────────────────────────────────────────────────────────────────
def print_static_summary(results: list[StaticResult]):
    total = len(results)
    ok = [r for r in results if r.feasible and not r.error]
    err = [r for r in results if r.error]

    print(f"\n{'='*65}")
    print(f"  STATIC  ({total} instances)")
    print(f"{'='*65}")
    print(f"  可行率:   {len(ok)}/{total}  ({len(ok)/max(total,1)*100:.1f}%)")
    print(f"  错误数:   {len(err)}")
    if ok:
        v = [r.vehicles for r in ok]
        d = [r.distance for r in ok]
        t = [r.runtime_sec for r in ok]
        print(f"  车辆数:   avg={sum(v)/len(v):.1f}  min={min(v)}  max={max(v)}")
        print(f"  距离:     avg={sum(d)/len(d):.0f}  min={min(d):.0f}  max={max(d):.0f}")
        print(f"  运行时间: avg={sum(t)/len(t):.1f}s  min={min(t):.1f}s  max={max(t):.1f}s")

        by_size = {}
        for r in ok:
            by_size.setdefault(r.size, []).append(r)
        for sz in sorted(by_size):
            grp = by_size[sz]
            print(f"    [{sz:>4} 节点] {len(grp):>3} 个 | "
                  f"avg_v={sum(x.vehicles for x in grp)/len(grp):.1f}  "
                  f"avg_d={sum(x.distance for x in grp)/len(grp):.0f}")

    if err:
        print(f"\n  错误详情:")
        for r in err[:5]:
            print(f"    {r.instance}: {r.error[:80]}")
        if len(err) > 5:
            print(f"    ... 还有 {len(err)-5} 个错误")


def print_dynamic_summary(results: list[DynamicResult]):
    total = len(results)
    ok = [r for r in results if r.overall_feasible and not r.error]
    err = [r for r in results if r.error]

    print(f"\n{'='*65}")
    print(f"  DYNAMIC  ({total} scenarios)")
    print(f"{'='*65}")
    print(f"  可行率:   {len(ok)}/{total}  ({len(ok)/max(total,1)*100:.1f}%)")
    print(f"  错误数:   {len(err)}")
    served = [r.overall_served_rate for r in results if not r.error]
    times   = [r.total_runtime_sec for r in results if not r.error]
    if served:
        print(f"  服务率:   avg={sum(served)/len(served)*100:.1f}%  "
              f"min={min(served)*100:.1f}%  max={max(served)*100:.1f}%")
    if times:
        print(f"  运行时间: avg={sum(times)/len(times):.1f}s  "
              f"min={min(times):.1f}s  max={max(times):.1f}s")

    by_epochs = {}
    for r in results:
        by_epochs.setdefault(r.num_epochs, []).append(r)
    for ep in sorted(by_epochs):
        grp = by_epochs[ep]
        ok_ep = [r for r in grp if r.overall_feasible and not r.error]
        print(f"    [{ep} epochs] {len(grp):>3} 个 | 可行={len(ok_ep)} "
              f"| avg_served={sum(r.overall_served_rate for r in grp)/len(grp)*100:.1f}%")

    if err:
        print(f"\n  错误详情:")
        for r in err[:5]:
            print(f"    {r.scenario}: {r.error[:80]}")
        if len(err) > 5:
            print(f"    ... 还有 {len(err)-5} 个错误")


# ─────────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────────
def run_static(time_limit: int, max_instances: int | None, resume: bool, workers: int):
    static_dir = Path("data/VRPTW/ORTEC/static")
    print(f"\n[静态] 扫描 {static_dir} ...")
    files = sorted(static_dir.glob("ORTEC-SYNTH-*.txt"))
    print(f"[静态] 共 {len(files)} 个实例")

    done_set = load_done(STATIC_CSV, "instance") if resume else set()
    pending  = [f for f in files if f.stem not in done_set]
    if max_instances:
        pending = pending[:max_instances]
    total = len(pending)

    print(f"[静态] {'已跳过' if resume else '从头'} | {total} 个待跑  |  {len(done_set)} 个已完成")
    if not total:
        print("[静态] 全部完成，无事可做")
        return

    start = time.time()
    done  = 0
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(solve_static, str(f), time_limit): f for f in pending}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            append_row(STATIC_CSV, STATIC_FIELDS, r.to_row())
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:30]})"
            print(f"  [{done:3d}/{total}] {r.instance:<42s}  "
                  f"v={r.vehicles:3d}  d={r.distance:12.1f}  "
                  f"feas={r.feasible}  t={r.runtime_sec:6.1f}s  "
                  f"{status}  ETA={eta:.0f}m")

    print_static_summary(results)


def run_dynamic(time_limit: int, max_instances: int | None, resume: bool, workers: int):
    dynamic_dir = Path("data/VRPTW/ORTEC/dynamic")
    static_dir  = Path("data/VRPTW/ORTEC/static")
    print(f"\n[动态] 扫描 {dynamic_dir} ...")
    dyn_files = sorted(dynamic_dir.glob("ORTEC-SYNTH-*_dyn5e.json"))
    print(f"[动态] 共 {len(dyn_files)} 个场景")

    done_set = load_done(DYN_CSV, "scenario") if resume else set()
    pending  = [f for f in dyn_files if f.stem not in done_set]
    if max_instances:
        pending = pending[:max_instances]
    total = len(pending)

    # 配对 base instance
    tasks = []
    for f in pending:
        base_stem = f.stem.replace("_dyn5e", "")
        base_path = static_dir / f"{base_stem}.txt"
        if base_path.exists():
            tasks.append((str(f), str(base_path)))
        else:
            print(f"  [SKIP] 找不到 base: {base_stem}.txt")

    total = len(tasks)
    print(f"[动态] {'已跳过' if resume else '从头'} | {total} 个待跑  |  {len(done_set)} 个已完成")
    if not total:
        print("[动态] 全部完成，无事可做")
        return

    start = time.time()
    done  = 0
    results = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(solve_dynamic, p[0], p[1], time_limit): p for p in tasks}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            append_row(DYN_CSV,   DYN_FIELDS,   r.to_row())
            for ep in r.epochs:
                append_row(EPOCH_CSV, EPOCH_FIELDS, ep.to_row())
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
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="OR-Tools 大规模 ORTEC VRPTW 对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_batch.py                              # 默认配置
  python run_batch.py --mode all --time-limit 120   # 全部 120s
  python run_batch.py --mode static --workers 4     # 仅静态
  python run_batch.py --mode dynamic --resume        # 继续上次动态
  python run_batch.py --mode all --max-instances 20  # 每类各 20 个

输出文件:
  results/ortools_batch/static_results.csv         # 静态实验结果
  results/ortools_batch/dynamic_results.csv        # 动态实验结果（汇总）
  results/ortools_batch/dynamic_epoch_results.csv  # 动态实验结果（每个 epoch 详情）
""",
    )
    p.add_argument(
        "--mode", choices=["static", "dynamic", "all"],
        default=DEFAULT_MODE,
        help=f"实验类型 (默认: {DEFAULT_MODE})",
    )
    p.add_argument(
        "--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
        help=f"每 instance / 每 epoch 的时间限制，秒 (默认: {DEFAULT_TIME_LIMIT})",
    )
    p.add_argument(
        "--max-instances", type=int, default=DEFAULT_MAX_INSTANCES,
        help="每个类型最多跑多少个实例 (默认: 全部)",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"并发线程数 (默认: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--resume", action="store_true", default=True,
        help="从上次中断处继续（默认开启）",
    )
    p.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="从头开始，忽略已有结果",
    )
    args = p.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"#  OR-Tools 批量实验")
    print(f"#  模式: {args.mode}  |  时间限制: {args.time_limit}s  |  并发: {args.workers}")
    print(f"#  断点续跑: {args.resume}  |  最大实例: {args.max_instances or '全部'}")
    print(f"{'#'*65}")

    t_start = time.time()

    if args.mode in ("static", "all"):
        run_static(args.time_limit, args.max_instances, args.resume, args.workers)

    if args.mode in ("dynamic", "all"):
        run_dynamic(args.time_limit, args.max_instances, args.resume, args.workers)

    elapsed = time.time() - t_start
    print(f"\n{'#'*65}")
    print(f"#  完成  (总耗时: {elapsed/60:.1f} 分钟)")
    print(f"#  静态: {STATIC_CSV}")
    print(f"#  动态: {DYN_CSV}")
    print(f"#  Epoch: {EPOCH_CSV}")
    print(f"{'#'*65}")


if __name__ == "__main__":
    main()
