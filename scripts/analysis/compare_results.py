"""
OR-Tools vs DRoC 对比报告

使用方法：
    python compare_results.py                    # 生成完整报告
    python compare_results.py --summary           # 仅打印汇总，不生成文件
    python compare_results.py --instances 20     # 仅看前 20 个

输出：
    results/comparison_report/static_comparison.csv
    results/comparison_report/dynamic_comparison.csv
    results/comparison_report/summary.txt         # 可读的汇总报告
"""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

OT_DIR  = Path("results/ortools_batch")
DROC_DIR = Path("results/droc_batch")
OUT_DIR  = Path("results/comparison_report")


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────
@dataclass
class StaticRow:
    instance: str
    size: int
    ot_vehicles: int;   ot_distance: float;  ot_feasible: bool;  ot_runtime: float
    droc_vehicles: int; droc_distance: float; droc_feasible: bool; droc_runtime: float
    droc_iterations: int; droc_error: str
    vehicle_gap: float;  distance_gap: float;  feasible_match: bool

@dataclass
class DynamicRow:
    scenario: str; base: str; num_epochs: int
    ot_served: float;  ot_feasible: bool;  ot_runtime: float;  ot_distance: float
    droc_served: float; droc_feasible: bool; droc_runtime: float; droc_distance: float
    droc_error: str
    served_diff: float; feasible_match: bool


# ─────────────────────────────────────────────────────────────────
# Load CSVs
# ─────────────────────────────────────────────────────────────────
def load_static_ot() -> dict[str, dict]:
    rows = {}
    p = OT_DIR / "static_results.csv"
    if not p.exists():
        return rows
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["instance"]] = {
                "size": int(r["size"]),
                "vehicles": int(r["vehicles"]),
                "distance": float(r["distance"]),
                "feasible": r["feasible"] == "True",
                "runtime": float(r["runtime_sec"]),
            }
    return rows


def load_static_droc() -> dict[str, dict]:
    rows = {}
    p = DROC_DIR / "static_results.csv"
    if not p.exists():
        return rows
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["instance"]] = {
                "size": int(r["size"]),
                "vehicles": int(r["vehicles"]),
                "distance": float(r["distance"]),
                "feasible": r["feasible"] == "True",
                "runtime": float(r["runtime_sec"]),
                "droc_iterations": int(r["droc_iterations"]),
                "error": r.get("error", ""),
            }
    return rows


def load_dynamic_ot() -> dict[str, dict]:
    rows = {}
    p = OT_DIR / "dynamic_results.csv"
    if not p.exists():
        return rows
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["scenario"]] = {
                "base": r["base_instance"],
                "num_epochs": int(r["num_epochs"]),
                "served": float(r["overall_served_rate"]),
                "feasible": r["overall_feasible"] == "True",
                "runtime": float(r["total_runtime_sec"]),
                "distance": float(r["total_distance"]),
            }
    return rows


def load_dynamic_droc() -> dict[str, dict]:
    rows = {}
    p = DROC_DIR / "dynamic_results.csv"
    if not p.exists():
        return rows
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["scenario"]] = {
                "base": r["base_instance"],
                "num_epochs": int(r["num_epochs"]),
                "served": float(r["overall_served_rate"]),
                "feasible": r["overall_feasible"] == "True",
                "runtime": float(r["total_runtime_sec"]),
                "distance": float(r["total_distance"]),
                "error": r.get("error", ""),
            }
    return rows


# ─────────────────────────────────────────────────────────────────
# Compare
# ─────────────────────────────────────────────────────────────────
def compare_static():
    ot   = load_static_ot()
    droc = load_static_droc()

    all_names = sorted(set(ot.keys()) & set(droc.keys()))
    rows = []

    for name in all_names:
        o = ot[name]; d = droc[name]

        v_gap = (d["vehicles"] - o["vehicles"]) / max(o["vehicles"], 1) * 100
        d_gap = (d["distance"] - o["distance"]) / max(o["distance"], 1) * 100

        rows.append({
            "instance": name,
            "size": o["size"],
            "ot_vehicles": o["vehicles"], "droc_vehicles": d["vehicles"],
            "vehicle_gap_pct": round(v_gap, 2),
            "ot_distance": round(o["distance"], 1), "droc_distance": round(d["distance"], 1),
            "distance_gap_pct": round(d_gap, 2),
            "ot_feasible": o["feasible"], "droc_feasible": d["feasible"],
            "ot_runtime": round(o["runtime"], 1), "droc_runtime": round(d["runtime"], 1),
            "droc_iterations": d["droc_iterations"],
            "droc_error": d["error"],
            "winner": (
                "DRoC" if v_gap < -1 or (v_gap < 1 and d_gap < -1)
                else "OR-Tools" if v_gap > 1 or (v_gap > -1 and d_gap > 1)
                else "tie"
            ),
        })

    return rows


def compare_dynamic():
    ot   = load_dynamic_ot()
    droc = load_dynamic_droc()

    all_names = sorted(set(ot.keys()) & set(droc.keys()))
    rows = []

    for name in all_names:
        o = ot[name]; d = droc[name]

        served_diff = (d["served"] - o["served"]) * 100  # percentage points

        rows.append({
            "scenario": name,
            "base": o["base"],
            "num_epochs": o["num_epochs"],
            "ot_served_pct": round(o["served"] * 100, 1),
            "droc_served_pct": round(d["served"] * 100, 1),
            "served_diff_pp": round(served_diff, 1),
            "ot_feasible": o["feasible"], "droc_feasible": d["feasible"],
            "ot_distance": round(o["distance"], 1),
            "droc_distance": round(d["distance"], 1),
            "ot_runtime": round(o["runtime"], 1),
            "droc_runtime": round(d["runtime"], 1),
            "droc_error": d["error"],
            "winner": (
                "DRoC" if served_diff > 5
                else "OR-Tools" if served_diff < -5
                else "tie"
            ),
        })

    return rows


# ─────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────
def generate_summary(static_rows, dynamic_rows) -> str:
    lines = []
    W = 65
    def hr(): lines.append("=" * W)
    def sub(t): lines.append(f"  {t}")

    hr()
    lines.append("  OR-Tools vs DRoC  实验汇总报告")
    lines.append(f"  生成时间: auto")
    hr()

    # Static
    if static_rows:
        ot_feas = [r for r in static_rows if r["ot_feasible"]]
        dr_feas = [r for r in static_rows if r["droc_feasible"]]
        both_feas = [r for r in static_rows if r["ot_feasible"] and r["droc_feasible"]]
        wins = [r for r in static_rows if r["winner"] == "DRoC"]
        wins_ot = [r for r in static_rows if r["winner"] == "OR-Tools"]
        ties = [r for r in static_rows if r["winner"] == "tie"]

        lines.append(f"\n  【静态 VRPTW】  (共 {len(static_rows)} 个实例)")
        lines.append(f"  OR-Tools 可行:  {len(ot_feas)}/{len(static_rows)}  ({len(ot_feas)/len(static_rows)*100:.1f}%)")
        lines.append(f"  DRoC 可行:     {len(dr_feas)}/{len(static_rows)}  ({len(dr_feas)/len(static_rows)*100:.1f}%)")
        if both_feas:
            v_gaps = [r["vehicle_gap_pct"] for r in both_feas]
            d_gaps = [r["distance_gap_pct"] for r in both_feas]
            lines.append(f"\n  仅比较双方均可行时的结果 ({len(both_feas)} 个):")
            lines.append(f"  车辆数差距: avg={sum(v_gaps)/len(v_gaps):+.2f}%  "
                         f"DRoC胜={len(wins)}  OR-Tools胜={len(wins_ot)}  tie={len(ties)}")
            lines.append(f"  距离差距:   avg={sum(d_gaps)/len(d_gaps):+.2f}%")
            lines.append(f"  最佳改善:   DRoC少用 {-min(v_gaps):.1f}% 车辆, {-min(d_gaps):.1f}% 距离")
            lines.append(f"  最差恶化:   DRoC多用 {max(v_gaps):.1f}% 车辆, {max(d_gaps):.1f}% 距离")
        lines.append(f"\n  胜负统计 (基于车辆数):")
        lines.append(f"  DRoC 胜:    {len(wins)}  ({len(wins)/len(static_rows)*100:.1f}%)")
        lines.append(f"  OR-Tools 胜: {len(wins_ot)}  ({len(wins_ot)/len(static_rows)*100:.1f}%)")
        lines.append(f"  平局:       {len(ties)}  ({len(ties)/len(static_rows)*100:.1f}%)")
        lines.append(f"\n  按规模分层:")
        for sz in sorted({r["size"] for r in static_rows}):
            grp = [r for r in static_rows if r["size"] == sz]
            g = both_feas = [r for r in grp if r["ot_feasible"] and r["droc_feasible"]]
            vg = [r["vehicle_gap_pct"] for r in g] if g else [0]
            lines.append(f"    [{sz:>4} 节点] {len(grp):>3} 个 | "
                         f"OT可行={sum(1 for r in grp if r['ot_feasible'])} "
                         f"DRoC可行={sum(1 for r in grp if r['droc_feasible'])} "
                         f"| avg_v_gap={sum(vg)/max(len(vg),1):+.1f}%")

    # Dynamic
    if dynamic_rows:
        ot_feas = [r for r in dynamic_rows if r["ot_feasible"]]
        dr_feas = [r for r in dynamic_rows if r["droc_feasible"]]
        both = [r for r in dynamic_rows if r["ot_feasible"] and r["droc_feasible"]]
        wins_dr = [r for r in dynamic_rows if r["winner"] == "DRoC"]
        wins_ot = [r for r in dynamic_rows if r["winner"] == "OR-Tools"]
        ties = [r for r in dynamic_rows if r["winner"] == "tie"]
        served_diffs = [r["served_diff_pp"] for r in dynamic_rows]

        lines.append(f"\n  【动态调度】  (共 {len(dynamic_rows)} 个场景)")
        lines.append(f"  OR-Tools 可行:  {len(ot_feas)}/{len(dynamic_rows)}  ({len(ot_feas)/len(dynamic_rows)*100:.1f}%)")
        lines.append(f"  DRoC 可行:     {len(dr_feas)}/{len(dynamic_rows)}  ({len(dr_feas)/len(dynamic_rows)*100:.1f}%)")
        lines.append(f"\n  整体服务率:")
        lines.append(f"  OR-Tools: avg={sum(r['ot_served_pct'] for r in dynamic_rows)/len(dynamic_rows):.1f}%  "
                     f"min={min(r['ot_served_pct'] for r in dynamic_rows):.1f}%  "
                     f"max={max(r['ot_served_pct'] for r in dynamic_rows):.1f}%")
        lines.append(f"  DRoC:     avg={sum(r['droc_served_pct'] for r in dynamic_rows)/len(dynamic_rows):.1f}%  "
                     f"min={min(r['droc_served_pct'] for r in dynamic_rows):.1f}%  "
                     f"max={max(r['droc_served_pct'] for r in dynamic_rows):.1f}%")
        lines.append(f"  平均服务率差距: DRoC - OR-Tools = {sum(served_diffs)/len(served_diffs):+.1f} pp")
        lines.append(f"\n  胜负统计 (服务率 >5pp 差距为胜):")
        lines.append(f"  DRoC 胜:    {len(wins_dr)}  ({len(wins_dr)/len(dynamic_rows)*100:.1f}%)")
        lines.append(f"  OR-Tools 胜: {len(wins_ot)}  ({len(wins_ot)/len(dynamic_rows)*100:.1f}%)")
        lines.append(f"  平局:       {len(ties)}  ({len(ties)/len(dynamic_rows)*100:.1f}%)")

        errors_dr = [r for r in dynamic_rows if r["droc_error"]]
        if errors_dr:
            lines.append(f"\n  DRoC 错误 ({len(errors_dr)} 个):")
            for r in errors_dr[:5]:
                lines.append(f"    {r['scenario']}: {r['droc_error'][:80]}")

    hr()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="OR-Tools vs DRoC 对比报告")
    p.add_argument("--summary", action="store_true", help="仅打印汇总，不写文件")
    p.add_argument("--instances", type=int, default=None, help="仅显示前 N 个实例")
    args = p.parse_args()

    print("\n[对比] 加载 OR-Tools 结果 ...")
    print("[对比] 加载 DRoC 结果 ...")

    static_rows  = compare_static()
    dynamic_rows = compare_dynamic()

    if not static_rows and not dynamic_rows:
        print("错误：找不到结果文件。请先运行 run_batch.py 和 run_batch_droc.py")
        return

    print(f"[对比] 静态: {len(static_rows)} 个共同实例 | 动态: {len(dynamic_rows)} 个共同场景")

    summary = generate_summary(static_rows, dynamic_rows)
    print("\n" + summary)

    if args.summary:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save static comparison
    if static_rows:
        fields = [
            "instance", "size",
            "ot_vehicles", "droc_vehicles", "vehicle_gap_pct",
            "ot_distance", "droc_distance", "distance_gap_pct",
            "ot_feasible", "droc_feasible",
            "ot_runtime", "droc_runtime", "droc_iterations",
            "winner", "droc_error",
        ]
        with open(OUT_DIR / "static_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in static_rows[:args.instances] if args.instances else static_rows:
                w.writerow(r)

    # Save dynamic comparison
    if dynamic_rows:
        fields = [
            "scenario", "base", "num_epochs",
            "ot_served_pct", "droc_served_pct", "served_diff_pp",
            "ot_feasible", "droc_feasible",
            "ot_distance", "droc_distance",
            "ot_runtime", "droc_runtime",
            "winner", "droc_error",
        ]
        with open(OUT_DIR / "dynamic_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in dynamic_rows[:args.instances] if args.instances else dynamic_rows:
                w.writerow(r)

    # Save text summary
    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"\n报告已保存至 {OUT_DIR}/")
    print(f"  summary.txt")


if __name__ == "__main__":
    main()
