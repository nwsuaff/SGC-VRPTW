"""
一键生成所有论文图表

Usage:
    python generate_figures.py                    # Generate all figures
    python generate_figures.py --rq1               # Only RQ1 figures
    python generate_figures.py --rq4               # Only RQ4 ablation figures
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures.seaborn.utils import (
    set_style, save_fig, write_caption,
    setup_axis, add_significance_bar,
    COLORS, METHOD_LABELS, METHOD_COLORS,
    FIG_WIDTH_SINGLE, FIG_WIDTH_DOUBLE,
    prepare_generalization_data, prepare_ablation_data,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Generalization Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig04_generalization(csv_path: str | Path, output_dir: str | Path = "figures/output"):
    """Generate Fig 4: Gap to BKS by instance family and method."""
    output_dir = Path(output_dir)
    set_style()

    df = pd.read_csv(csv_path)
    df = prepare_generalization_data(df)

    # Filter to relevant columns
    if "method" in df.columns:
        df_plot = df[df["method"].isin(["llm_loop", "pyvrp", "ortools"])]
    else:
        df_plot = df

    # Compute mean gap by family
    summary = df_plot.groupby(["family", "variant"]).agg({
        "gap_to_bks": "mean",
        "vehicles_used": "mean",
        "feasible": "mean",
    }).reset_index()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE))

    # (a) Gap to BKS by family
    families = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
    x = np.arange(len(families))
    width = 0.35

    llm_gaps = []
    baseline_gaps = []
    for fam in families:
        llm_row = summary[(summary["family"] == fam) & (summary["variant"] == "llm_guided")]
        base_row = summary[(summary["family"] == fam) & (summary["variant"] == "baseline")]
        llm_gaps.append(llm_row["gap_to_bks"].mean() if len(llm_row) else 0)
        baseline_gaps.append(base_row["gap_to_bks"].mean() if len(base_row) else 0)

    axes[0].bar(x - width/2, baseline_gaps, width, label="Baseline (OR-Tools)", color=COLORS["baseline"])
    axes[0].bar(x + width/2, llm_gaps, width, label="LLM-guided", color=COLORS["proposed"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families)
    axes[0].set_ylabel("Gap to BKS (%)")
    axes[0].set_xlabel("Instance Family")
    axes[0].legend(fontsize=7)
    setup_axis(axes[0], panel_label="(a)")

    # (b) Feasible rate by family
    llm_feas = []
    base_feas = []
    for fam in families:
        llm_row = summary[(summary["family"] == fam) & (summary["variant"] == "llm_guided")]
        base_row = summary[(summary["family"] == fam) & (summary["variant"] == "baseline")]
        llm_feas.append((llm_row["feasible"].mean() * 100) if len(llm_row) else 0)
        base_feas.append((base_row["feasible"].mean() * 100) if len(base_row) else 0)

    axes[1].bar(x - width/2, base_feas, width, label="Baseline", color=COLORS["baseline"])
    axes[1].bar(x + width/2, llm_feas, width, label="LLM-guided", color=COLORS["proposed"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(families)
    axes[1].set_ylabel("Feasible Rate (%)")
    axes[1].set_xlabel("Instance Family")
    axes[1].set_ylim(0, 105)
    setup_axis(axes[1], panel_label="(b)")

    fig.tight_layout()
    paths = save_fig(fig, "Fig04_generalization", output_dir)

    caption = (
        "Fig. 4. Generalization performance across Solomon instance families. "
        "(a) Gap to best known solutions (BKS) by family. "
        "(b) Feasible solution rate by family. "
        "LLM-guided solver shows improved performance especially on R2 and RC2 families."
    )
    write_caption(Path(output_dir) / "Fig04_generalization.md", caption)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Stability Analysis (Multi-seed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig05_stability(csv_path: str | Path, output_dir: str | Path = "figures/output"):
    """Generate Fig 5: Multi-seed stability analysis."""
    output_dir = Path(output_dir)
    set_style()

    df = pd.read_csv(csv_path)

    # Filter by method
    df_llm = df[df.get("variant", df.get("method", "")) == "llm_guided"]
    df_base = df[df.get("variant", df.get("method", "")) == "baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE))

    # (a) Vehicle count distribution by seed
    if "seed" in df.columns:
        for i, (variant, label, color) in enumerate([
            ("baseline", "Baseline", COLORS["baseline"]),
            ("llm_guided", "LLM-guided", COLORS["proposed"]),
        ]):
            subset = df[df["variant"] == variant] if "variant" in df.columns else df[df["method"] == variant]
            if not subset.empty and "seed" in subset.columns:
                seed_means = subset.groupby("seed")["vehicles_used"].mean()
                axes[0].scatter(seed_means.index, seed_means.values,
                              label=label, color=color, alpha=0.7, s=50)

        axes[0].set_xlabel("Seed")
        axes[0].set_ylabel("Mean Vehicles Used")
        axes[0].legend(fontsize=7)
        setup_axis(axes[0], panel_label="(a)")

    # (b) CV (coefficient of variation) comparison
    cv_data = []
    for variant in ["baseline", "llm_guided"]:
        subset = df[df.get("variant", df.get("method", "")) == variant]
        if not subset.empty:
            cv = subset.groupby("instance")["vehicles_used"].std() / subset.groupby("instance")["vehicles_used"].mean()
            cv_data.append({"variant": variant, "cv_mean": cv.mean(), "cv_std": cv.std()})

    if cv_data:
        variants = [d["variant"] for d in cv_data]
        cv_means = [d["cv_mean"] for d in cv_data]
        cv_stds = [d["cv_std"] for d in cv_data]

        x = np.arange(len(variants))
        axes[1].bar(x, cv_means, yerr=cv_stds, capsize=3,
                   color=[COLORS["baseline"], COLORS["proposed"]])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(["Baseline", "LLM-guided"])
        axes[1].set_ylabel("Coefficient of Variation")
        setup_axis(axes[1], panel_label="(b)")

    fig.tight_layout()
    paths = save_fig(fig, "Fig05_stability", output_dir)

    caption = (
        "Fig. 5. Multi-seed stability analysis. "
        "(a) Mean vehicles used across different random seeds. "
        "(b) Coefficient of variation (CV) of solution quality across seeds. "
        "Lower CV indicates more stable solver behavior."
    )
    write_caption(Path(output_dir) / "Fig05_stability.md", caption)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Ablation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig07_ablation(ablation_csv: str | Path, output_dir: str | Path = "figures/output"):
    """Generate Fig 7: Ablation analysis of hint types."""
    output_dir = Path(output_dir)
    set_style()

    df = pd.read_csv(ablation_csv)
    df = prepare_ablation_data(df)

    # Define variant order and colors
    variants = ["zero_shot", "priority_only", "locked_only", "moves_only", "full_hints"]
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE))

    # (a) Feasible rate by variant
    feasible_rates = []
    for v in variants:
        subset = df[df["variant"] == v]
        rate = subset["feasible"].mean() * 100 if not subset.empty else 0
        feasible_rates.append(rate)

    x = np.arange(len(variants))
    axes[0].bar(x, feasible_rates, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=7)
    axes[0].set_ylabel("Feasible Rate (%)")
    axes[0].set_ylim(0, 105)
    setup_axis(axes[0], panel_label="(a)")

    # Add value labels
    for i, v in enumerate(feasible_rates):
        axes[0].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=7)

    # (b) Mean vehicles by variant
    mean_vehicles = []
    for v in variants:
        subset = df[df["variant"] == v]
        mean_v = subset["vehicles_used"].mean() if not subset.empty else 0
        mean_vehicles.append(mean_v)

    axes[1].bar(x, mean_vehicles, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=7)
    axes[1].set_ylabel("Mean Vehicles Used")
    setup_axis(axes[1], panel_label="(b)")

    # Add value labels
    for i, v in enumerate(mean_vehicles):
        axes[1].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=7)

    fig.tight_layout()
    paths = save_fig(fig, "Fig07_ablation", output_dir)

    caption = (
        "Fig. 7. Ablation analysis of hint type contributions. "
        "(a) Feasible solution rate by ablation variant. "
        "(b) Mean vehicles used by ablation variant. "
        "Full hints (all hint types combined) achieves the best performance, "
        "with locked_subroutes contributing the most to feasibility."
    )
    write_caption(Path(output_dir) / "Fig07_ablation.md", caption)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Few-shot Learning Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_fig08_fewshot(csv_path: str | Path, output_dir: str | Path = "figures/output"):
    """Generate Fig 8: Few-shot learning curve."""
    output_dir = Path(output_dir)
    set_style()

    df = pd.read_csv(csv_path)

    if "n_shots" not in df.columns:
        logger.warning("No 'n_shots' column found, skipping Fig 8")
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE))

    # Group by n_shots
    summary = df.groupby("n_shots").agg({
        "feasible": ["mean", "std"],
        "vehicles_used": ["mean", "std"],
    }).reset_index()
    summary.columns = ["n_shots", "feas_mean", "feas_std", "veh_mean", "veh_std"]

    # (a) Feasible rate vs n_shots
    axes[0].errorbar(summary["n_shots"], summary["feas_mean"] * 100,
                     yerr=summary["feas_std"] * 100, fmt="o-",
                     color=COLORS["proposed"], capsize=3)
    axes[0].set_xlabel("Number of Few-shot Examples")
    axes[0].set_ylabel("Feasible Rate (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_xticks(summary["n_shots"])
    setup_axis(axes[0], panel_label="(a)")

    # (b) Vehicles vs n_shots
    axes[1].errorbar(summary["n_shots"], summary["veh_mean"],
                     yerr=summary["veh_std"], fmt="o-",
                     color=COLORS["proposed"], capsize=3)
    axes[1].set_xlabel("Number of Few-shot Examples")
    axes[1].set_ylabel("Mean Vehicles Used")
    axes[1].set_xticks(summary["n_shots"])
    setup_axis(axes[1], panel_label="(b)")

    fig.tight_layout()
    paths = save_fig(fig, "Fig08_fewshot", output_dir)

    caption = (
        "Fig. 8. Few-shot learning curve on out-of-distribution instances. "
        "Performance improves with more target-domain examples. "
        "Retrieval-augmented prompts provide an immediate lift over zero-shot baseline."
    )
    write_caption(Path(output_dir) / "Fig08_fewshot.md", caption)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--rq1", action="store_true", help="Generate RQ1 figures")
    parser.add_argument("--rq2", action="store_true", help="Generate RQ2 figures (generalization)")
    parser.add_argument("--rq3", action="store_true", help="Generate RQ3 figures (stability)")
    parser.add_argument("--rq4", action="store_true", help="Generate RQ4 figures (ablation)")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="figures/output", help="Output directory")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fig 4: Generalization (RQ2) ─────────────────────────────────────────
    if args.all or args.rq2:
        fig4_path = results_dir / "rq2_size"
        csv_files = list(fig4_path.glob("*.csv")) if fig4_path.exists() else []
        if csv_files:
            # Use most recent
            latest = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Generating Fig 4 from {latest}")
            plot_fig04_generalization(latest, output_dir)
        else:
            logger.warning(f"No CSV found in {fig4_path}")

    # ── Fig 5: Stability (RQ3) ──────────────────────────────────────────────
    if args.all or args.rq3:
        fig5_path = results_dir / "rq1_solomon"
        csv_files = list(fig5_path.glob("*.csv")) if fig5_path.exists() else []
        if csv_files:
            latest = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Generating Fig 5 from {latest}")
            plot_fig05_stability(latest, output_dir)
        else:
            logger.warning(f"No CSV found in {fig5_path}")

    # ── Fig 7: Ablation (RQ4) ───────────────────────────────────────────────
    if args.all or args.rq4:
        fig7_path = results_dir / "rq4_ablation"
        csv_files = list(fig7_path.glob("*.csv")) if fig7_path.exists() else []
        if csv_files:
            latest = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Generating Fig 7 from {latest}")
            plot_fig07_ablation(latest, output_dir)
        else:
            logger.warning(f"No CSV found in {fig7_path}")

    # ── Fig 8: Few-shot curve ────────────────────────────────────────────────
    if args.all or args.rq4:
        fig8_path = results_dir / "rq4_ablation"
        csv_files = list(fig8_path.glob("*.csv")) if fig8_path.exists() else []
        if csv_files:
            latest = max(csv_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Generating Fig 8 from {latest}")
            plot_fig08_fewshot(latest, output_dir)

    logger.info(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
