import csv
from pathlib import Path

result_dir = Path("results/test_warmstart")
csv_file = result_dir / "comparison_results.csv"

# Read failed instances
with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        feas = row["droc_feasible"].strip().lower()
        if feas == "false":
            print(f"FAIL: {row['instance']} | DRoC: {row['droc_vehicles']}v/{row['droc_distance']}d | Baseline: {row['baseline_vehicles']}v/{row['baseline_distance']}d | Baseline feas={row['baseline_feasible']}")
