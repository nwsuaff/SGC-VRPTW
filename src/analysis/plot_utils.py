"""Shared plotting utilities for VRPTW LLM Solver figures.

This module provides a unified style system, color palette, and helper
functions used across all figure-generation modules. All figures follow
Automation in Construction (AutCon) guidelines:
  - Minimal text inside figures (axes + legend only)
  - One main conclusion per figure
  - Output as PDF (vector, 300 DPI) + PNG (raster, 150 DPI)
  - Font: DejaVu Sans / Arial 7-9 pt
  - Color-blind safe palette

Usage:
    from src.analysis.plot_utils import (
        setup_style, PROPOSED, BASELINE, ACCENT_COLORS, save_fig
    )
    setup_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    # ... plot ...
    save_fig(fig, "Fig_foo", "figures")
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — color-blind safe, roughly matched to d3.schemeCategory10
# ─────────────────────────────────────────────────────────────────────────────

PROPOSED   = "#4CAF50"  # DRoC / LLM-proposed method
BASELINE   = "#2196F3"  # Classical baseline (OR-Tools, PyVRP, Gurobi)
ZERO_SHOT  = "#607D8B"  # Zero-shot / cold-start
HINT_FULL  = "#FF9800"  # Full-hints variant
HINT_PRI   = "#3F51B5"  # Priority-only hint
HINT_LCK   = "#9C27B0"  # Locked-only hint
HINT_MOV   = "#E91E63"  # Moves-only hint
FAIL_RED   = "#F44336"
ACCENT     = "#FF5722"
GRAY_LIGHT = "#ECEFF1"
GRAY_MED   = "#90A4AE"

# Extended palette for boxplots with many groups
SEABORN_PALETTE = [
    "#4CAF50", "#2196F3", "#FF9800", "#9C27B0",
    "#F44336", "#00BCD4", "#FFEB3B", "#795548",
]


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib style helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_style(
    font_family: str = "DejaVu Sans",
    font_size: int = 9,
    spine_top: bool = False,
    spine_right: bool = False,
) -> None:
    """Configure matplotlib rcParams for AutCon-compatible publication figures.

    Call this once at the start of each figure-generating script.

    Args:
        font_family: Primary font (DejaVu Sans works cross-platform without
                     extra fonts; Arial is preferred for final submission).
        font_size:   Base font size in pt.
        spine_top:   Whether to show top spine.
        spine_right: Whether to show right spine.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.titleweight": "bold",
        "axes.spines.top": spine_top,
        "axes.spines.right": spine_right,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": "white",
        "axes_facecolor": "#FAFAFA",
        "figure.facecolor": "white",
        "axes.unicode_minus": False,
    })


def autcon_single(width_in: float = 3.5, height_in: float = 2.6) -> tuple[float, float]:
    """Return (width, height) for a single-column AutCon figure in inches."""
    return width_in, height_in


def autcon_double(width_in: float = 7.0, height_in: float = 3.5) -> tuple[float, float]:
    """Return (width, height) for a double-column AutCon figure in inches."""
    return width_in, height_in


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers
# ─────────────────────────────────────────────────────────────────────────────

def droc_colors(n: int = 5) -> list[str]:
    """Return a colour list for n groups, cycling through the standard palette."""
    base = [PROPOSED, BASELINE, ZERO_SHOT, HINT_FULL, HINT_PRI, HINT_LCK, HINT_MOV]
    return (base * ((n // len(base)) + 1))[:n]


def boxplot_colors(n: int) -> list[str]:
    """Return n distinct box colours from the seaborn-inspired palette."""
    return (SEABORN_PALETTE * ((n // len(SEABORN_PALETTE)) + 1))[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Figure saving
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(
    fig,
    name: str,
    output_dir: str | Path,
    dpi_pdf: int = 300,
    dpi_png: int = 150,
    bbox_inches: str = "tight",
) -> dict[str, Path]:
    """Save a figure in both PDF and PNG formats.

    Args:
        fig:          Matplotlib Figure object.
        name:         Base filename without extension.
        output_dir:   Output directory.
        dpi_pdf:      DPI for the PDF (vector output, ignored by most backends).
        dpi_png:      DPI for the PNG raster.
        bbox_inches:  Passed to ``savefig(bbox_inches=...)``.

    Returns:
        Dict mapping ``name + ".pdf"`` and ``name + ".png"`` to their absolute paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    pdf_path = out / f"{name}.pdf"
    png_path = out / f"{name}.png"

    fig.savefig(pdf_path, bbox_inches=bbox_inches, dpi=dpi_pdf, format="pdf")
    fig.savefig(png_path, bbox_inches=bbox_inches, dpi=dpi_png, format="png")
    paths[f"{name}.pdf"] = pdf_path
    paths[f"{name}.png"] = png_path

    logger.info("Saved %s  ->  %s  +  %s", name, pdf_path, png_path)
    return paths


def save_caption(
    caption_text: str,
    name: str,
    output_dir: str | Path,
) -> Path:
    """Write a caption draft alongside the figure.

    Args:
        caption_text: Plain-text caption (no "Fig X. " prefix).
        name:         Base filename (e.g. "Fig4_generalization").
        output_dir:   Output directory.

    Returns:
        Path to the written caption file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    caption_path = out / f"{name}_caption.md"
    caption_path.write_text(caption_text, encoding="utf-8")
    logger.info("Caption written to %s", caption_path)
    return caption_path


# ─────────────────────────────────────────────────────────────────────────────
# Panel labelling helper
# ─────────────────────────────────────────────────────────────────────────────

def label_panel(ax, label: str = "a", x: float = -0.10, y: float = 1.05) -> None:
    """Add a panel label (e.g. '(a)') to the top-left corner of an Axes.

    Args:
        ax:    Matplotlib Axes.
        label: Label text.
        x:     x position relative to axes coordinates.
        y:     y position relative to axes coordinates.
    """
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


# ─────────────────────────────────────────────────────────────────────────────
# Data-preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def prepare_results_df(
    df,
    bks_csv: Path | None = None,
) -> "pd.DataFrame":
    """Add BKS and gap columns to a batch-results DataFrame.

    Adds the following columns if they are not already present:
      - ``bks_vehicles``, ``bks_distance``: from ``bks_csv`` or in-group minimum
      - ``lex_gap``: lexicographic gap = (vehicles - BKS_v) * 10000 + (dist - BKS_d)
      - ``gap_to_bks``: percentage gap in total distance to BKS
      - ``dataset``: inferred from the ``source`` column (defaults to "unknown")

    Args:
        df:       Batch results DataFrame. Expected columns include
                  ``instance``, ``vehicles_used``, ``total_distance``, ``feasible``.
        bks_csv:  Optional path to a BKS CSV with columns
                  ``instance_name, best_vehicles, best_distance``.

    Returns:
        Copy of ``df`` with additional columns.
    """
    import pandas as pd

    df = df.copy()

    # Normalise source -> dataset
    if "dataset" not in df.columns and "source" in df.columns:
        df["dataset"] = df["source"]

    # Merge BKS
    if bks_csv:
        bks = pd.read_csv(bks_csv).rename(columns={
            "instance_name": "instance",
            "best_vehicles": "bks_vehicles",
            "best_distance": "bks_distance",
        })
        df = df.merge(bks[["instance", "bks_vehicles", "bks_distance"]],
                      on="instance", how="left")

    # Fall back to in-group minimum
    if "bks_vehicles" not in df.columns:
        df["bks_vehicles"] = df.groupby("instance")["vehicles_used"].transform("min")
        df["bks_distance"] = df.groupby("instance")["total_distance"].transform("min")

    # Lexicographic gap (vehicles primary, distance secondary)
    df["lex_gap"] = (
        (df["vehicles_used"] - df["bks_vehicles"]) * 10000
        + (df["total_distance"] - df["bks_distance"]).fillna(0)
    )

    # Percentage gap
    df["gap_to_bks"] = (
        (df["total_distance"] - df["bks_distance"]).fillna(0)
        / df["bks_distance"].replace(0, 1)
        * 100
    )

    return df


def family_shift_label(family: str) -> str:
    """Classify a Solomon-family label as 'Seen' or 'Unseen'.

    Convention:
      - Tight TW families (C1, R1, RC1) are "seen" during training.
      - Loose TW families (C2, R2, RC2) are "unseen".
    """
    if family in ("C1", "R1", "RC1"):
        return "Seen (tight TW)"
    if family in ("C2", "R2", "RC2"):
        return "Unseen (loose TW)"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Empty-panel placeholder
# ─────────────────────────────────────────────────────────────────────────────

def empty_panel(ax, message: str = "No data available") -> None:
    """Show a grey placeholder in a panel when no data is present."""
    ax.text(0.5, 0.5, message,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#9E9E9E")
    ax.set_xticks([])
    ax.set_yticks([])
