"""
RQ1: Solomon 100 Family Shift 实验
=====================================
验证 DRoC 在跨 family 泛化（tight TW → loose TW）中的表现。

Family Shift 实验设计：
  - Seen (训练/参考): Solomon C1/R1/RC1 (tight TW)
  - Unseen (测试): Solomon C2/R2/RC2 (loose TW)

对比方法:
  - OR-Tools cold-start (CP-SAT, 无 warm-start)
  - DRoC (Scene Card + Self-Debug + RAG)

每个算例 × 3 随机种子 → 3 runs/method

使用方法:
    # 快速验证（3个算例，mock LLM）
    python run_rq1_solomon.py --max-instances 3 --llm mock

    # 正式运行（OR-Tools baseline + DRoC，56个算例，3 seeds）
    python run_rq1_solomon.py --llm openai --time-limit 300 --baseline-time-limit 120 --workers 2 --seeds 42,43,44

    # 仅跑 OR-Tools baseline（不调用 LLM，快速）
    python run_rq1_solomon.py --baseline-only --workers 4

    # 仅跑 DRoC（基于已有 baseline 结果续跑）
    python run_rq1_solomon.py --droc-only --llm openai --resume

输出:
    results/rq1_solomon/
    ├── solomon_baseline_results.csv    # OR-Tools baseline
    ├── solomon_droc_results.csv        # DRoC 结果
    └── solomon_comparison_results.csv  # 对比结果（gap、胜率等）
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, ".")
from src.data.solomon_loader import load_solomon_instance, _derive_family
from src.solvers.ortools_solver import solve_ortools
from src.solvers.feasibility_checker import check_feasibility
from src.llm.client_factory import create_llm_client, is_llm_available
from src.pipelines.droc_pipeline import run_droc_solver


# ─── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/rq1_solomon")
BASELINE_CSV = OUTPUT_DIR / "solomon_baseline_results.csv"
DROC_CSV = OUTPUT_DIR / "solomon_droc_results.csv"
COMPARISON_CSV = OUTPUT_DIR / "solomon_comparison_results.csv"

# Solomon 100 families
SOLOMON_FAMILIES = {
    "C1": [f"C10{i}" for i in range(1, 10)],   # 9 instances
    "C2": [f"C20{i}" for i in range(1, 10)],   # 9 instances
    "R1": [f"R10{i}" for i in range(1, 12)],   # 11 instances
    "R2": [f"R20{i}" for i in range(1, 12)],   # 11 instances
    "RC1": [f"RC10{i}" for i in range(1, 9)],  # 8 instances
    "RC2": [f"RC20{i}" for i in range(1, 9)],  # 8 instances
}

# Family labels for plot grouping
TIGHT_FAMILIES = {"C1", "R1", "RC1"}   # tight time windows
LOOSE_FAMILIES = {"C2", "R2", "RC2"}   # loose time windows

SOLOMON_DIR = Path("data/VRPTW/Solomon")


# ─── Dataclasses ─────────────────────────────────────────────────────────────
@dataclass
class BaselineResult:
    instance: str = ""
    family: str = ""
    size: int = 100
    seed: int = 42
    vehicles: int = 0
    distance: float = 0.0
    feasible: bool = False
    runtime_sec: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DrocResult:
    instance: str = ""
    family: str = ""
    size: int = 100
    seed: int = 42
    droc_iterations: int = 0
    vehicles: int = 0
    distance: float = 0.0
    feasible: bool = False
    late_violations: int = 0
    cap_violations: int = 0
    llm_latency_ms: float = 0.0
    runtime_sec: float = 0.0
    error: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ─── CSV helpers ─────────────────────────────────────────────────────────────
BASELINE_FIELDS = [
    "timestamp", "instance", "family", "size", "seed",
    "vehicles", "distance", "feasible", "runtime_sec", "error",
]
DROC_FIELDS = [
    "timestamp", "instance", "family", "size", "seed",
    "droc_iterations", "vehicles", "distance", "feasible",
    "late_violations", "cap_violations", "llm_latency_ms",
    "runtime_sec", "error",
]
COMPARISON_FIELDS = [
    "instance", "family", "size", "seed",
    # Baseline
    "baseline_vehicles", "baseline_distance", "baseline_feasible",
    # DRoC
    "droc_vehicles", "droc_distance", "droc_feasible", "droc_iterations",
    # Gaps
    "gap_vehicles", "gap_distance",  # positive = DRoC better
    "droc_wins_vehicles", "droc_wins_distance",
    "lexicographic_winner",  # "droc" or "baseline" or "tie"
    "error",
]


def load_done(csv_path: Path, key_col: str, seed_col: str = "seed") -> set[tuple]:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {
            (row[key_col], int(row.get(seed_col, 42)))
            for row in csv.DictReader(f)
            if row.get(key_col) and row.get(seed_col)
        }


def append_row(csv_path: Path, fields: list[str], row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─── Solvers ─────────────────────────────────────────────────────────────────
def run_single_baseline(args_tuple):
    inst_name, seed, time_limit = args_tuple
    t0 = time.time()
    try:
        inst_path = SOLOMON_DIR / f"{inst_name}.vrp"
        inst = load_solomon_instance(str(inst_path))
        family = _derive_family(inst_name) or "Unknown"

        sol, meta = solve_ortools(inst, time_limit=time_limit)
        validation = check_feasibility(inst, sol)
        return BaselineResult(
            instance=inst_name,
            family=family,
            size=inst.size,
            seed=seed,
            vehicles=sol.vehicles_used,
            distance=sol.total_distance,
            feasible=validation.feasible,
            runtime_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        family = _derive_family(inst_name) or "Unknown"
        return BaselineResult(
            instance=inst_name,
            family=family,
            seed=seed,
            runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


def run_single_droc(args_tuple):
    inst_name, seed, time_limit, max_iterations, llm_name = args_tuple
    t0 = time.time()
    try:
        inst_path = SOLOMON_DIR / f"{inst_name}.vrp"
        inst = load_solomon_instance(str(inst_path))
        family = _derive_family(inst_name) or "Unknown"
        inst.name = inst_name  # ensure name is set

        llm = create_llm_client(llm_name)
        solution, gen_result, exp_result = run_droc_solver(
            instance=inst,
            llm_client=llm,
            max_iterations=max_iterations,
            time_limit=float(time_limit),
            droc_timeout=float(time_limit + 60),
            output_dir=None,
            incumbent=None,  # no incumbent for Solomon
        )

        return DrocResult(
            instance=inst_name,
            family=family,
            size=inst.size,
            seed=seed,
            droc_iterations=gen_result.iterations if gen_result else 0,
            vehicles=solution.vehicles_used if solution else 0,
            distance=solution.total_distance if solution else 0.0,
            feasible=solution.feasible if solution else False,
            late_violations=solution.late_violations if solution else -1,
            cap_violations=solution.capacity_violations if solution else -1,
            llm_latency_ms=gen_result.llm_latency_ms if gen_result else 0.0,
            runtime_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        family = _derive_family(inst_name) or "Unknown"
        return DrocResult(
            instance=inst_name,
            family=family,
            seed=seed,
            runtime_sec=round(time.time() - t0, 2),
            error=f"{type(e).__name__}: {e}",
        )


# ─── Comparison ───────────────────────────────────────────────────────────────
def lexicographic_compare(r) -> tuple[int, float]:
    """Return (vehicles, distance) tuple — lower is better."""
    return (r.vehicles if r.vehicles else 999, r.distance if r.distance else float("inf"))


def build_comparison():
    """Build comparison CSV from baseline + DRoC results."""
    import pandas as pd

    baseline_df = pd.read_csv(BASELINE_CSV) if BASELINE_CSV.exists() else None
    droc_df = pd.read_csv(DROC_CSV) if DROC_CSV.exists() else None

    if baseline_df is None or droc_df is None:
        print("[WARN] 无法构建对比表：缺少 baseline 或 DRoC 结果")
        return

    merged = baseline_df.merge(
        droc_df,
        on=["instance", "family", "size", "seed"],
        suffixes=("_baseline", "_droc"),
        how="outer",
    )

    rows = []
    for _, r in merged.iterrows():
        row = {
            "instance": r["instance"],
            "family": r["family"],
            "size": r["size"],
            "seed": r["seed"],
            "baseline_vehicles": r.get("vehicles_baseline", -1),
            "baseline_distance": r.get("distance_baseline", -1.0),
            "baseline_feasible": r.get("feasible_baseline", False),
            "droc_vehicles": r.get("vehicles_droc", -1),
            "droc_distance": r.get("distance_droc", -1.0),
            "droc_feasible": r.get("feasible_droc", False),
            "droc_iterations": r.get("droc_iterations", -1),
            "gap_vehicles": 0,
            "gap_distance": 0.0,
            "droc_wins_vehicles": 0,
            "droc_wins_distance": 0,
            "lexicographic_winner": "tie",
            "error": r.get("error_droc", "") or r.get("error_baseline", ""),
        }

        b_veh = r.get("vehicles_baseline", -1)
        d_veh = r.get("vehicles_droc", -1)
        b_dist = r.get("distance_baseline", -1.0)
        d_dist = r.get("distance_droc", -1.0)

        if b_veh >= 0 and d_veh >= 0 and r.get("feasible_baseline") and r.get("feasible_droc"):
            row["gap_vehicles"] = int(b_veh - d_veh)
            row["gap_distance"] = round(b_dist - d_dist, 2)
            row["droc_wins_vehicles"] = 1 if d_veh < b_veh else 0
            row["droc_wins_distance"] = 1 if d_dist < b_dist else 0

            if d_veh < b_veh:
                row["lexicographic_winner"] = "droc"
            elif d_veh > b_veh:
                row["lexicographic_winner"] = "baseline"
            else:
                if d_dist < b_dist:
                    row["lexicographic_winner"] = "droc"
                elif d_dist > b_dist:
                    row["lexicographic_winner"] = "baseline"
                else:
                    row["lexicographic_winner"] = "tie"

        rows.append(row)

    df = pd.DataFrame(rows, columns=COMPARISON_FIELDS)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"[对比] 已写入 {COMPARISON_CSV} ({len(df)} 行)")


def print_summary():
    """Print summary statistics grouped by family (seen/unseen)."""
    import pandas as pd

    if not COMPARISON_CSV.exists():
        print("[WARN] 对比表不存在，跳过汇总")
        return

    df = pd.read_csv(COMPARISON_CSV)
    print(f"\n{'=' * 70}")
    print("RQ1 SOLOMON FAMILY SHIFT SUMMARY")
    print(f"{'=' * 70}")

    # Split by seen/unseen
    seen_mask = df["family"].isin(list(TIGHT_FAMILIES))
    unseen_mask = df["family"].isin(list(LOOSE_FAMILIES))

    for split_name, mask in [("Seen (tight TW, C1/R1/RC1)", seen_mask), ("Unseen (loose TW, C2/R2/RC2)", unseen_mask)]:
        split = df[mask]
        if split.empty:
            continue

        wins_v = split["droc_wins_vehicles"].sum()
        wins_d = split["droc_wins_distance"].sum()
        total = len(split)
        tie = (split["lexicographic_winner"] == "tie").sum()
        droc_win = (split["lexicographic_winner"] == "droc").sum()
        base_win = (split["lexicographic_winner"] == "baseline").sum()

        avg_gap_v = split["gap_vehicles"].mean()
        avg_gap_d = split["gap_distance"].mean()

        print(f"\n  [{split_name}] n={total}")
        print(f"  DRoC 胜车辆数:   {wins_v}/{total} ({wins_v/max(total,1)*100:.1f}%)")
        print(f"  DRoC 胜总距离:   {wins_d}/{total} ({wins_d/max(total,1)*100:.1f}%)")
        print(f"  Lexicographic:  DRoC胜={droc_win}  基线胜={base_win}  平局={tie}")
        print(f"  平均车辆数差距:  {avg_gap_v:+.2f} (正=DRoC用车更少)")
        print(f"  平均距离差距:    {avg_gap_d:+12.2f} (正=DRoC距离更短)")

    # Per family
    print(f"\n  按 Family 分组:")
    for family in sorted(df["family"].unique()):
        fam = df[df["family"] == family]
        wins = fam["droc_wins_vehicles"].sum()
        total = len(fam)
        lex_wins = (fam["lexicographic_winner"] == "droc").sum()
        print(f"    {family}: n={total} | DRoC胜车辆={wins}/{total} | Lex胜={lex_wins}/{total}")

    print(f"\n{'=' * 70}")


# ─── Main experiment functions ────────────────────────────────────────────────
def run_baseline_experiment(seeds: list[int], time_limit: int, max_instances: Optional[int], workers: int):
    """Run OR-Tools baseline on all Solomon instances × seeds."""
    print(f"\n[OR-Tools Baseline] 时间限制={time_limit}s | seeds={seeds}")

    all_instances = []
    for fam, names in SOLOMON_FAMILIES.items():
        for name in names:
            for seed in seeds:
                all_instances.append((name, seed, time_limit))

    if max_instances:
        all_instances = all_instances[:max_instances]

    done_set = load_done(BASELINE_CSV, "instance", "seed")
    pending = [t for t in all_instances if (t[0], t[1]) not in done_set]
    total = len(pending)
    print(f"  共 {len(all_instances)} runs | {total} 待跑 | {len(done_set)} 已完成")

    if not total:
        print("  全部完成，无事可做")
        return

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_baseline, t): t for t in pending}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            append_row(BASELINE_CSV, BASELINE_FIELDS, {
                "timestamp": r.timestamp, "instance": r.instance,
                "family": r.family, "size": r.size, "seed": r.seed,
                "vehicles": r.vehicles, "distance": round(r.distance, 2),
                "feasible": r.feasible,
                "runtime_sec": r.runtime_sec, "error": r.error,
            })
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:25]})"
            print(f"  [{done:3d}/{total}] {r.instance:<6s} seed={r.seed}  "
                  f"v={r.vehicles:3d}  d={r.distance:10.1f}  "
                  f"feas={r.feasible}  {status}  ETA={eta:.0f}m")

    print(f"\n[OR-Tools Baseline] 完成！耗时 {time.time()-start:.1f}s")


def run_droc_experiment(seeds: list[int], time_limit: int, max_iterations: int,
                        max_instances: Optional[int], workers: int, llm: str, resume: bool):
    """Run DRoC on all Solomon instances × seeds."""
    print(f"\n[DRoC Solomon] LLM={llm} | 时间限制={time_limit}s | max_iter={max_iterations} | seeds={seeds}")

    all_instances = []
    for fam, names in SOLOMON_FAMILIES.items():
        for name in names:
            for seed in seeds:
                all_instances.append((name, seed, time_limit, max_iterations, llm))

    if max_instances:
        all_instances = all_instances[:max_instances]

    done_set = load_done(DROC_CSV, "instance", "seed")
    pending = [t for t in all_instances if (t[0], t[1]) not in done_set]
    total = len(pending)
    print(f"  共 {len(all_instances)} runs | {total} 待跑 | {len(done_set)} 已完成")

    if not total:
        print("  全部完成，无事可做")
        return

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_single_droc, t): t for t in pending}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            append_row(DROC_CSV, DROC_FIELDS, {
                "timestamp": r.timestamp, "instance": r.instance,
                "family": r.family, "size": r.size, "seed": r.seed,
                "droc_iterations": r.droc_iterations,
                "vehicles": r.vehicles, "distance": round(r.distance, 2),
                "feasible": r.feasible,
                "late_violations": r.late_violations,
                "cap_violations": r.cap_violations,
                "llm_latency_ms": r.llm_latency_ms,
                "runtime_sec": r.runtime_sec, "error": r.error,
            })
            done += 1
            elapsed = time.time() - start
            eta = (elapsed / done) * (total - done) / 60 if done else 0
            status = "OK" if not r.error else f"ERR({r.error[:25]})"
            print(f"  [{done:3d}/{total}] {r.instance:<6s} seed={r.seed}  "
                  f"v={r.vehicles:3d}  d={r.distance:10.1f}  "
                  f"feas={r.feasible}  iter={r.droc_iterations}  "
                  f"t={r.runtime_sec:6.1f}s  {status}  ETA={eta:.0f}m")

    print(f"\n[DRoC Solomon] 完成！耗时 {time.time()-start:.1f}s")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="RQ1 Solomon Family Shift 实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 一次性跑完（OR-Tools baseline + DRoC，3 seeds，推荐）
  python run_rq1_solomon.py --llm openai --time-limit 300 --baseline-time-limit 120 --workers 2 --seeds 42,43,44

  # 验证脚本（mock LLM，3个算例）
  python run_rq1_solomon.py --baseline-only --workers 4 --time-limit 30 --max-instances 3

输出:
  results/rq1_solomon/solomon_baseline_results.csv
  results/rq1_solomon/solomon_droc_results.csv
  results/rq1_solomon/solomon_comparison_results.csv  (跑完自动生成)
""",
    )
    p.add_argument("--llm", default="openai", help="LLM client name (default: openai)")
    p.add_argument("--time-limit", type=int, default=300, help="DRoC 求解时间限制 (秒，默认: 300)")
    p.add_argument("--baseline-time-limit", type=int, default=120, help="OR-Tools baseline 时间限制 (秒，默认: 120)")
    p.add_argument("--max-iterations", type=int, default=4, help="DRoC 最大迭代次数 (默认: 4)")
    p.add_argument("--seeds", default="42", help="逗号分隔的随机种子 (默认: 42)")
    p.add_argument("--workers", type=int, default=2, help="并发数 (默认: 2)")
    p.add_argument("--max-instances", type=int, default=None,
                   help="每个 family 最多数例数 (默认: None=全部)")
    p.add_argument("--resume", action="store_true", default=True,
                   help="断点续跑 (默认: True)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="从头开始（忽略已完成）")
    p.add_argument("--baseline-only", action="store_true",
                   help="仅跑 OR-Tools baseline")
    p.add_argument("--droc-only", action="store_true",
                   help="仅跑 DRoC（基于已有 baseline）")

    args = p.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    if args.llm != "mock" and not is_llm_available(args.llm):
        print(f"[WARN] LLM '{args.llm}' 不可用，回退到 mock")
        args.llm = "mock"

    print(f"\n{'#' * 65}")
    print(f"#  RQ1 Solomon Family Shift 实验")
    print(f"#  LLM={args.llm} | time_limit={args.time_limit}s | seeds={seeds}")
    print(f"#  max_iterations={args.max_iterations} | workers={args.workers}")
    print(f"#  resume={args.resume} | baseline_only={args.baseline_only} | droc_only={args.droc_only}")
    print(f"{'#' * 65}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Run baseline
    if not args.droc_only:
        run_baseline_experiment(
            seeds=seeds,
            time_limit=args.baseline_time_limit,
            max_instances=args.max_instances,
            workers=args.workers,
        )

    # Run DRoC
    if not args.baseline_only:
        run_droc_experiment(
            seeds=seeds,
            time_limit=args.time_limit,
            max_iterations=args.max_iterations,
            max_instances=args.max_instances,
            workers=args.workers,
            llm=args.llm,
            resume=args.resume,
        )

    # Build comparison and print summary
    if not args.baseline_only:
        print("\n" + "=" * 70)
        print("构建对比结果...")
        build_comparison()
        print_summary()

    total_time = time.time() - t_start
    print(f"\n完成！总耗时: {total_time / 60:.1f} 分钟")
    print(f"输出目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
