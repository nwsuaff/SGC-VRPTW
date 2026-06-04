import sys, csv
from pathlib import Path

result_dir = Path("results/test_warmstart")
csv_file = result_dir / "comparison_results.csv"

failed = []
with open(csv_file) as f:
    for row in csv.DictReader(f):
        if row["droc_feasible"].strip().lower() == "false":
            failed.append(row)

print(f"FAILED instances ({len(failed)}/10):")
for row in failed:
    # DRoC解 vs baseline解
    d_v = int(row["droc_vehicles"])
    d_d = float(row["droc_distance"])
    b_v = int(row["baseline_vehicles"])
    b_d = float(row["baseline_distance"])
    improvement = (b_v - d_v) / b_v * 100
    print(f"\n  {row['instance']}:")
    print(f"    DRoC:      {d_v} vehicles, dist={d_d:.1f}")
    print(f"    Baseline:  {b_v} vehicles, dist={b_d:.1f}")
    print(f"    Vehicle improvement: {improvement:+.1f}% (DRoC uses {d_v-b_v:+d} fewer vehicles)")
