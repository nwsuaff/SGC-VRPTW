"""Evaluate Experiment Results - Load and analyze experiment output."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_results(output_dir: Path) -> dict:
    """Load and evaluate experiment results from output directory.
    
    Args:
        output_dir: Directory containing experiment results.
    
    Returns:
        Dictionary with evaluation statistics.
    """
    results_csv = output_dir / "comparison_results.csv"
    summary_csv = output_dir / "summary.csv"
    
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        stats = _compute_statistics(df)
        _print_report(stats)
        return stats
    elif summary_csv.exists():
        df = pd.read_csv(summary_csv)
        print(df.to_string())
        return {}
    else:
        logger.warning(f"No results found in {output_dir}")
        return {}


def _compute_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics from results DataFrame."""
    stats = {
        "total_instances": len(df),
        "droc_success_rate": df["droc_success"].mean() if "droc_success" in df.columns else 0,
        "droc_feasible_rate": df["droc_feasible"].mean() if "droc_feasible" in df.columns else 0,
        "baseline_feasible_rate": df["baseline_feasible"].mean() if "baseline_feasible" in df.columns else 0,
    }
    
    if "droc_vehicles" in df.columns and "baseline_vehicles" in df.columns:
        valid = df[df["droc_feasible"] == True]
        if len(valid) > 0:
            stats["droc_mean_vehicles"] = valid["droc_vehicles"].mean()
            stats["droc_mean_distance"] = valid["droc_distance"].mean()
        
        baseline_valid = df[df["baseline_feasible"] == True]
        if len(baseline_valid) > 0:
            stats["baseline_mean_vehicles"] = baseline_valid["baseline_vehicles"].mean()
            stats["baseline_mean_distance"] = baseline_valid["baseline_distance"].mean()
        
        if "vehicle_improvement" in df.columns:
            imp = df[df["vehicle_improvement"].notna()]
            if len(imp) > 0:
                stats["mean_vehicle_improvement"] = imp["vehicle_improvement"].mean()
                stats["droc_wins_rate"] = imp["droc_wins"].mean() if "droc_wins" in imp.columns else 0
    
    return stats


def _print_report(stats: dict) -> None:
    """Print evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total instances: {stats.get('total_instances', 'N/A')}")
    print()
    print(f"DRoC success rate: {stats.get('droc_success_rate', 0):.1%}")
    print(f"DRoC feasible rate: {stats.get('droc_feasible_rate', 0):.1%}")
    print(f"Baseline feasible rate: {stats.get('baseline_feasible_rate', 0):.1%}")
    print()
    if "droc_mean_vehicles" in stats:
        print(f"DRoC mean vehicles: {stats['droc_mean_vehicles']:.2f}")
        print(f"Baseline mean vehicles: {stats['baseline_mean_vehicles']:.2f}")
        print(f"Mean vehicle improvement: {stats.get('mean_vehicle_improvement', 0):.2f}%")
        print(f"DRoC wins rate: {stats.get('droc_wins_rate', 0):.1%}")
    print("=" * 60)
