"""Aggregate per-instance results into a single CSV after all runs complete.

Run this after all gen_batch_commands.txt jobs have finished:
    python aggregate_results.py

Reads all results/gh800/<instance>/*_droc_*/experiment_result.json files and produces:
    results/gh800_droc_aggregated.csv
    results/gh800_droc_summary.csv
"""
import json
import pandas as pd
from pathlib import Path


def aggregate_results(
    results_dir: str = "results/gh800",
    output_csv: str = "results/gh800_droc_aggregated.csv",
    summary_csv: str = "results/gh800_droc_summary.csv",
):
    results_dir = Path(results_dir)
    rows = []

    for instance_dir in sorted(results_dir.iterdir()):
        if not instance_dir.is_dir():
            continue

        # run_droc_solver saves to output_dir/{instance}_droc_{timestamp}/experiment_result.json
        exp_dir = instance_dir  # instance_dir already = results/gh800/<instance>
        exp_file = None
        for sub in exp_dir.iterdir():
            if sub.is_dir() and sub.name.startswith(instance_dir.name + "_droc_"):
                candidate = sub / "experiment_result.json"
                if candidate.exists():
                    exp_file = candidate
                    break

        if exp_file is None:
            print(f"  [SKIP] {instance_dir.name}: no experiment_result.json found in subdirectories")
            continue

        try:
            with open(exp_file, encoding="utf-8") as f:
                exp = json.load(f)
        except Exception as e:
            print(f"  [ERROR] {instance_dir.name}: {e}")
            continue

        row = {
            "instance": instance_dir.name,
            "solver": exp.get("solver", ""),
            "vehicles_used": exp.get("vehicles_used"),
            "total_distance": exp.get("total_distance"),
            "feasible": exp.get("feasible"),
            "late_violations": exp.get("late_violations"),
            "capacity_violations": exp.get("capacity_violations"),
            "runtime_sec": exp.get("runtime_sec"),
            "experiment_id": exp.get("experiment_id", ""),
            "droc_success": exp.get("metadata", {}).get("code_success"),
            "droc_iterations": exp.get("metadata", {}).get("code_iterations"),
            "llm_latency_ms": exp.get("metadata", {}).get("llm_latency_ms"),
            "error_message": exp.get("metadata", {}).get("error_message"),
        }
        rows.append(row)

    if not rows:
        print("No results found!")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\nAggregated {len(df)} results -> {output_csv}")

    # Summary by instance type
    df["type"] = df["instance"].str.extract(r"^([A-Z]+)")

    summary_rows = []
    for (solver_type, feasible), grp in df.groupby(["type", "feasible"]):
        summary_rows.append({
            "type": solver_type,
            "feasible": feasible,
            "count": len(grp),
            "avg_vehicles": grp["vehicles_used"].mean(),
            "avg_distance": grp["total_distance"].mean(),
            "avg_runtime_sec": grp["runtime_sec"].mean(),
        })

    # Also per-feasibility summary
    for feasible_val, grp in df.groupby("feasible"):
        summary_rows.append({
            "type": "ALL",
            "feasible": feasible_val,
            "count": len(grp),
            "avg_vehicles": grp["vehicles_used"].mean(),
            "avg_distance": grp["total_distance"].mean(),
            "avg_runtime_sec": grp["runtime_sec"].mean(),
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(summary_csv, index=False)
    print(f"Summary {summary_csv}")
    print(f"\nTotal: {len(df)}, Feasible: {df['feasible'].sum()}, "
          f"Infeasible: {(~df['feasible']).sum()}")
    print(f"\nBy type:")
    for t, grp in df.groupby("type"):
        ok = grp["feasible"].sum()
        print(f"  {t}: {ok}/{len(grp)} feasible  "
              f"avg_vehicles={grp['vehicles_used'].mean():.1f}  "
              f"avg_distance={grp['total_distance'].mean():.1f}")

    return df


if __name__ == "__main__":
    aggregate_results()
