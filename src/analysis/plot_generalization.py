"""Figure 4 — Cross-scenario Generalization (RQ1).

Generates a 3-panel figure:

  Panel (a)  Grouped bar chart of lexicographic gap on Solomon families
              (Seen = tight-TW families, Unseen = loose-TW families).

  Panel (b)  Line plot of gap-to-reference (%) on Homberger instances
              vs. problem scale (200–1000 customers), comparing
              DRoC / LLM-loop against PyVRP cold-start.

  Panel (c)  Box plot of gap-to-reference on realistic routing / ORTEC
              instances.  Falls back to Homberger grouped by family if
              ORTEC data is absent.

Usage:
    python -m src.analysis.plot_generalization \
        --results results/batch_001.csv \
        --bks    data/bks_solomon.csv \
        --output figures/

    # Programmatic call
    from src.analysis.plot_generalization import plot_generalization
    paths = plot_generalization("results/batch_001.csv", "data/bks.csv")
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.analysis.plot_utils import (
    setup_style,
    PROPOSED, BASELINE, ACCENT, FAIL_RED,
    droc_colors, boxplot_colors,
    save_fig, save_caption, label_panel,
    prepare_results_df, family_shift_label, empty_panel,
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

def plot_generalization(
    batch_csv: str | Path,
    bks_csv: str | Path | None = None,
    output_dir: str | Path = "figures",
    caption_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate Fig. 4: Cross-scenario Generalization.

    Args:
        batch_csv:  CSV with columns ``instance``, ``family``, ``size``,
                    ``method``, ``vehicles_used``, ``total_distance``, ``feasible``.
                    May also contain ``dataset`` / ``source`` and ``method``.
        bks_csv:    Optional BKS CSV with ``instance_name, best_vehicles,
                    best_distance`` columns.
        output_dir: Directory for saved figures.
        caption_path: Optional path for the caption draft (overrides auto-save).

    Returns:
        Dict mapping ``"Fig4_generalization.pdf"`` and ``".png"`` to absolute paths.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required: pip install matplotlib numpy pandas")

    setup_style()
    df = prepare_results_df(_pd.read_csv(batch_csv), bks_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    _panel_family_shift(df, axes[0])
    _panel_size_degradation(df, axes[1])
    _panel_realistic_routing(df, axes[2])

    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        label_panel(ax, chr(97 + i), x=-0.10, y=1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    paths = save_fig(fig, "Fig4_generalization", out)
    _plt.close(fig)

    cap = (
        "Fig. 4. Cross-scenario generalisation under family, size, and realistic-routing "
        "shifts. (a) Lexicographic performance of DRoC on seen (tight time-window) and "
        "unseen (loose time-window) Solomon families. (b) Gap to reference (%) on "
        "Homberger instances as a function of problem scale (200–1000 customers), "
        "comparing DRoC against PyVRP cold-start. (c) Distribution of gap to reference "
        "on realistic routing instances."
    )
    cap_path = caption_path or (out / "Fig4_generalization_caption.md")
    save_caption(cap, "Fig4_generalization", out)

    logger.info("Fig. 4 saved: %s", list(paths.values()))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Panel (a) — Family shift
# ─────────────────────────────────────────────────────────────────────────────

def _panel_family_shift(df: _pd.DataFrame, ax) -> None:
    """Grouped bar chart: seen vs unseen families on Solomon instances."""
    sol = df[df.get("dataset", "unknown").str.contains("solomon", case=False, na=False)].copy()
    if sol.empty:
        empty_panel(ax, "(a) No Solomon data")
        return

    sol["shift_type"] = sol["family"].apply(family_shift_label)

    families = ["C", "R", "RC"]
    seen_data: dict[str, list[float]] = {f: [] for f in families}
    unseen_data: dict[str, list[float]] = {f: [] for f in families}

    for _, row in sol.iterrows():
        fam = row.get("family", "")
        prefix = fam[0] if isinstance(fam, str) and fam else "R"
        gap = row.get("lex_gap", 0)
        if row["shift_type"].startswith("Seen"):
            seen_data.get(prefix, []).append(gap)
        else:
            unseen_data.get(prefix, []).append(gap)

    x = _np.arange(len(families))
    w = 0.35

    seen_vals  = [_np.median(seen_data.get(f,   [0])) for f in families]
    unseen_vals = [_np.median(unseen_data.get(f, [0])) for f in families]

    ax.bar(x - w/2, seen_vals,   w, label="Seen (tight TW)",   color=BASELINE,   alpha=0.85)
    ax.bar(x + w/2, unseen_vals, w, label="Unseen (loose TW)", color=ACCENT,      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylabel("Lexicographic gap", fontsize=9)
    ax.set_title("(a) Family shift", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (b) — Size degradation
# ─────────────────────────────────────────────────────────────────────────────

def _panel_size_degradation(df: _pd.DataFrame, ax) -> None:
    """Line plot: gap-to-reference (%) vs problem size on Homberger."""
    hom = df[df.get("dataset", "unknown").str.contains("homberger", case=False, na=False)].copy()
    if hom.empty:
        empty_panel(ax, "(b) No Homberger data")
        return

    if "size" not in hom.columns:
        empty_panel(ax, "(b) 'size' column missing")
        return

    sizes = sorted(hom["size"].dropna().unique())
    # The two methods we expect in batch CSV output
    methods = ["pyvrp", "llm_loop", "droc"]

    colors = droc_colors(len(methods))
    labels = {
        "pyvrp":    "PyVRP (cold)",
        "llm_loop": "LLM loop",
        "droc":     "DRoC",
    }

    for color, method in zip(colors, methods):
        sub = hom[hom.get("method", "") == method]
        if sub.empty:
            continue

        means, stds = [], []
        for sz in sizes:
            vals = sub[sub["size"] == sz]["gap_to_bks"].dropna()
            if len(vals):
                means.append(vals.mean())
                stds.append(vals.std())
            else:
                means.append(_np.nan)
                stds.append(_np.nan)

        means = _np.array(means, dtype=float)
        stds  = _np.array(stds,  dtype=float)

        ax.plot(sizes, means, color=color,
                marker="o", markersize=4, linewidth=1.5,
                label=labels.get(method, method))
        mask = ~_np.isnan(means) & ~_np.isnan(stds)
        if mask.any():
            ax.fill_between(
                _np.array(sizes)[mask],
                (means - stds)[mask],
                (means + stds)[mask],
                color=color, alpha=0.12,
            )

    ax.set_xlabel("Instance size (customers)", fontsize=9)
    ax.set_ylabel("Gap to reference (%)", fontsize=9)
    ax.set_title("(b) Size degradation", fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# Panel (c) — Realistic routing
# ─────────────────────────────────────────────────────────────────────────────

def _panel_realistic_routing(df: _pd.DataFrame, ax) -> None:
    """Box plot of gap-to-reference on realistic routing / ORTEC instances."""
    ortec = df[df.get("dataset", "unknown").isin(
        ["ortec", "rrnco", "realistic", "rrnco_full"]
    )].copy()

    data_df = ortec if not ortec.empty else df[
        df.get("dataset", "unknown").str.contains("homberger", case=False, na=False)
    ].copy()

    if data_df.empty:
        empty_panel(ax, "(c) No realistic routing data")
        return

    data_df = data_df[data_df["family"].notna()]
    if data_df.empty:
        empty_panel(ax, "(c) No family labels")
        return

    groups = sorted(data_df["family"].dropna().unique())
    box_data = [
        data_df[data_df["family"] == g]["gap_to_bks"].dropna().values
        for g in groups
    ]
    # Remove empty groups
    valid = [(g, d) for g, d in zip(groups, box_data) if len(d) > 0]
    if not valid:
        empty_panel(ax, "(c) No valid groups")
        return
    groups, box_data = zip(*valid)
    groups = list(groups)

    bp = ax.boxplot(
        list(box_data), labels=groups,
        patch_artist=True, widths=0.5,
        medianprops=dict(color="white", linewidth=1.5),
    )
    colors = boxplot_colors(len(groups))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Gap to reference (%)", fontsize=9)
    ax.set_title("(c) Realistic routing", fontsize=10)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate Fig. 4: Generalization")
    p.add_argument("--results", required=True, help="Path to batch results CSV")
    p.add_argument("--bks",      default=None, help="Path to BKS CSV")
    p.add_argument("--output",   default="figures", help="Output directory")
    args = p.parse_args()

    plot_generalization(
        batch_csv=args.results,
        bks_csv=args.bks,
        output_dir=args.output,
    )
