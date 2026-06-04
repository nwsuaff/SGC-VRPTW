"""Comparison Visualisation — DRoC vs Baseline Solvers.

Generates 6 plots that compare DRoC against a classical baseline solver
(e.g. OR-Tools, PyVRP, Gurobi):

  1. Vehicles bar chart       — vehicles used per instance
  2. Distance bar chart      — total distance per instance
  3. Feasible rate bars     — overall feasibility rate
  4. Improvement histogram   — distribution of vehicle-improvement percentages
  5. Runtime bar chart      — wall-clock time per instance
  6. Scatter plot           — DRoC distance vs baseline distance (Pareto check)

Expected CSV columns (from ``run_comparison`` in comparison_pipeline.py):
  ``instance``, ``size``, ``seed``,
  ``droc_vehicles``, ``droc_distance``, ``droc_feasible``, ``droc_runtime``,
  ``droc_iterations``,
  ``baseline_vehicles``, ``baseline_distance``, ``baseline_feasible``,
  ``baseline_runtime``,
  ``vehicle_improvement``, ``distance_improvement``,
  ``droc_wins``, ``droc_success``

Usage:
    python -m src.analysis.comparison_plots \
        --results results/comparison/comparison_results.csv \
        --output results/comparison/plots \
        --baseline ortools

    from src.analysis.comparison_plots import plot_comparison_summary
    paths = plot_comparison_summary("results/comparison/comparison_results.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.analysis.plot_utils import (
    setup_style,
    PROPOSED, BASELINE, FAIL_RED,
    save_fig, save_caption, label_panel, empty_panel,
)

logger = logging.getLogger(__name__)

try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np
    import pandas as _pd
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_summary(
    results_csv: str | Path,
    output_dir: str | Path = "results/comparison/plots",
    baseline_method: str = "ortools",
    instance_limit: int = 30,
) -> dict[str, Path]:
    """Generate all 6 comparison plots.

    Args:
        results_csv:    Path to the comparison results CSV.
        output_dir:     Directory for saved PNG files.
        baseline_method: Display name for the baseline solver.
        instance_limit:  Cap the x-axis at N instances for readability.

    Returns:
        Dict mapping each plot name to its absolute PNG path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    setup_style()
    df = _pd.read_csv(results_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # ── 1. Vehicles bar chart ────────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(13, 5))
    _plot_vehicles(df, ax, baseline_method, instance_limit)
    p = save_fig(fig, "vehicles_comparison", out)
    paths.update(p)
    _plt.close(fig)

    # ── 2. Distance bar chart ───────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(13, 5))
    _plot_distance(df, ax, baseline_method, instance_limit)
    p = save_fig(fig, "distance_comparison", out)
    paths.update(p)
    _plt.close(fig)

    # ── 3. Feasible rate ────────────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(4, 3))
    _plot_feasible_rate(df, ax, baseline_method)
    p = save_fig(fig, "feasible_rate", out)
    paths.update(p)
    _plt.close(fig)

    # ── 4. Improvement histogram ────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    _plot_improvement(df, ax)
    p = save_fig(fig, "improvement_distribution", out)
    paths.update(p)
    _plt.close(fig)

    # ── 5. Runtime ──────────────────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(10, 4))
    _plot_runtime(df, ax, baseline_method, instance_limit)
    p = save_fig(fig, "runtime_comparison", out)
    paths.update(p)
    _plt.close(fig)

    # ── 6. Scatter ──────────────────────────────────────────────────────────
    fig, ax = _plt.subplots(figsize=(5, 5))
    _plot_scatter(df, ax)
    p = save_fig(fig, "scatter_comparison", out)
    paths.update(p)
    _plt.close(fig)

    logger.info("Comparison plots saved to %s  (%d files)", out, len(paths))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Individual plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _feasible_mask(df: _pd.DataFrame) -> _pd.DataFrame:
    """Return rows where both DRoC and baseline produced feasible solutions."""
    return df[
        df["droc_feasible"].fillna(False) &
        df["baseline_feasible"].fillna(False)
    ].copy()


def _plot_vehicles(df: _pd.DataFrame, ax, baseline: str, limit: int) -> None:
    """Bar chart: vehicles used per instance."""
    valid = _feasible_mask(df)
    if valid.empty:
        empty_panel(ax, "No valid comparisons")
        return

    grouped = (valid.groupby("instance")
               [["droc_vehicles", "baseline_vehicles"]]
               .mean()
               .reset_index()
               .sort_values("baseline_vehicles")
               .head(limit))

    x = _np.arange(len(grouped))
    w = 0.35

    ax.bar(x - w/2, grouped["baseline_vehicles"], w,
           label=f"Baseline ({baseline})", color=BASELINE, alpha=0.82)
    ax.bar(x + w/2, grouped["droc_vehicles"],    w,
           label="DRoC",                      color=PROPOSED,  alpha=0.82)

    ax.set_xlabel("Instance", fontsize=9)
    ax.set_ylabel("Vehicles used", fontsize=9)
    ax.set_title(f"Vehicle count: DRoC vs {baseline}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["instance"], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_distance(df: _pd.DataFrame, ax, baseline: str, limit: int) -> None:
    """Bar chart: total distance per instance (x1000 for readability)."""
    valid = _feasible_mask(df)
    if valid.empty:
        empty_panel(ax, "No valid comparisons")
        return

    grouped = (valid.groupby("instance")
               [["droc_distance", "baseline_distance"]]
               .mean()
               .reset_index()
               .sort_values("baseline_distance")
               .head(limit))

    x = _np.arange(len(grouped))
    w = 0.35

    ax.bar(x - w/2, grouped["baseline_distance"] / 1000, w,
           label=f"Baseline ({baseline})", color=BASELINE, alpha=0.82)
    ax.bar(x + w/2, grouped["droc_distance"] / 1000,    w,
           label="DRoC",                      color=PROPOSED,  alpha=0.82)

    ax.set_xlabel("Instance", fontsize=9)
    ax.set_ylabel("Total distance (x1000)", fontsize=9)
    ax.set_title(f"Total distance: DRoC vs {baseline}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["instance"], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_feasible_rate(df: _pd.DataFrame, ax, baseline: str) -> None:
    """Side-by-side bar: feasible rate of DRoC vs baseline."""
    n = len(df)
    if n == 0:
        empty_panel(ax, "No data")
        return

    droc_rate  = df["droc_feasible"].fillna(False).sum()  / n
    base_rate  = df["baseline_feasible"].fillna(False).sum() / n

    bars = ax.bar(["Baseline", "DRoC"], [base_rate, droc_rate],
                   color=[BASELINE, PROPOSED], alpha=0.82)

    ax.set_ylabel("Feasible rate", fontsize=9)
    ax.set_title(f"Feasibility: DRoC vs {baseline}", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    for bar, rate in zip(bars, [base_rate, droc_rate]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_improvement(df: _pd.DataFrame, ax) -> None:
    """Histogram of vehicle-improvement percentages."""
    valid = df[df["vehicle_improvement"].notna()].copy()
    if valid.empty:
        empty_panel(ax, "No improvement data")
        return

    vals = valid["vehicle_improvement"].values

    ax.hist(vals, bins=20, color=PROPOSED, alpha=0.72, edgecolor="white")
    ax.axvline(0, color=FAIL_RED, linestyle="--",
               linewidth=2, label="No improvement")
    ax.axvline(_np.mean(vals), color=BASELINE, linestyle="-",
               linewidth=2, label=f"Mean: {_np.mean(vals):.1f}%")

    ax.set_xlabel("Vehicle improvement (%)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Distribution of DRoC vehicle improvement vs baseline", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_runtime(df: _pd.DataFrame, ax, baseline: str, limit: int) -> None:
    """Bar chart: wall-clock runtime per instance (top N only)."""
    cols = ["droc_runtime", "baseline_runtime"]
    valid = df[[c in df.columns for c in cols]].dropna(subset=cols).copy()

    if valid.empty:
        empty_panel(ax, "No runtime data")
        return

    grouped = (valid.groupby("instance")[cols]
                .mean()
                .reset_index()
                .head(limit))

    x = _np.arange(len(grouped))
    w = 0.35

    ax.bar(x - w/2, grouped["baseline_runtime"], w,
           label=f"Baseline ({baseline})", color=BASELINE, alpha=0.82)
    ax.bar(x + w/2, grouped["droc_runtime"],    w,
           label="DRoC (incl. LLM)",      color=PROPOSED,  alpha=0.82)

    ax.set_xlabel("Instance", fontsize=9)
    ax.set_ylabel("Runtime (seconds)", fontsize=9)
    ax.set_title(f"Runtime: DRoC vs {baseline}", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["instance"], rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_scatter(df: _pd.DataFrame, ax) -> None:
    """Scatter: baseline distance vs DRoC distance.

    Points below the diagonal indicate DRoC outperforms the baseline.
    """
    valid = _feasible_mask(df)
    if valid.empty:
        empty_panel(ax, "No valid data")
        return

    x = valid["baseline_distance"] / 1000
    y = valid["droc_distance"] / 1000

    ax.scatter(x, y, color=PROPOSED, alpha=0.65, s=45, edgecolors="white")

    max_val = max(x.max(), y.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2,
            label="Equal performance")

    droc_wins = (y < x).sum()
    total = len(valid)
    ax.text(0.05, 0.95, f"DRoC wins: {droc_wins}/{total}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Baseline distance (x1000)", fontsize=9)
    ax.set_ylabel("DRoC distance (x1000)", fontsize=9)
    ax.set_title("DRoC vs baseline: distance comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate DRoC vs baseline comparison plots")
    p.add_argument("--results",  required=True, help="Comparison results CSV")
    p.add_argument("--output",  default="results/comparison/plots", help="Output directory")
    p.add_argument("--baseline", default="ortools", help="Baseline solver name")
    p.add_argument("--limit",   type=int, default=30, help="Instance cap for bar charts")
    args = p.parse_args()

    plot_comparison_summary(args.results, args.output, args.baseline, args.limit)
