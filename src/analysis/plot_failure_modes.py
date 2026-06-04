"""Figure 9 — Failure Mode Analysis (RQ2 / RQ4).

Generates two figure families:

  Fig 9  —  Two-panel failure mode analysis:
    Panel (a)  Funnel chart: how many LLM outputs survive each filtering layer
                (valid JSON → hints accepted → warm-start applied → improved).
    Panel (b)  Bar chart of failure category frequencies across all instances.

  Fig 7  —  Two-panel ablation contribution study (DRoC component ablation):
    Panel (a)  Mean vehicle count across ablation variants
                (zero-shot, priority-only, locked-only, moves-only, full-hints).
    Panel (b)  Feasibility rate (%) across ablation variants.

Both figures read from event logs and ablation CSVs respectively.

Usage:
    python -m src.analysis.plot_failure_modes failure-modes \
        --events results/events.jsonl \
        --output figures/

    python -m src.analysis.plot_failure_modes ablation \
        --results results/ablation_001.csv \
        --output figures/
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from src.analysis.plot_utils import (
    setup_style,
    PROPOSED, BASELINE, ZERO_SHOT, HINT_FULL,
    HINT_PRI, HINT_LCK, HINT_MOV,
    FAIL_RED, ACCENT,
    droc_colors, save_fig, save_caption, label_panel, empty_panel,
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
# Public API — failure modes (Fig 9)
# ─────────────────────────────────────────────────────────────────────────────

def plot_failure_modes(
    events_jsonl: str | Path,
    output_dir: str | Path = "figures",
    caption_path: str | Path | None = None,
) -> dict[str, Path]:
    """Generate Fig. 9: Failure Mode Analysis.

    Args:
        events_jsonl: Path to the event-log JSONL file produced by
                      ``EventLogger``.
        output_dir:   Output directory.
        caption_path:  Override path for caption draft.

    Returns:
        Dict mapping saved-filename to absolute Path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    setup_style()
    _, events = _load_events(events_jsonl)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    _panel_funnel(events, axes[0])
    _panel_failure_bars(events, axes[1])

    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        label_panel(ax, chr(97 + i), x=-0.10, y=1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    paths = save_fig(fig, "Fig9_failure_modes", out)
    _plt.close(fig)

    cap = (
        "Fig. 9. Failure mode analysis of the solver-grounded LLM framework. "
        "(a) Filtering funnel: fraction of LLM outputs surviving each validation "
        "layer (valid JSON, hints accepted, warm-start applied, improved). "
        "(b) Frequency of individual failure categories across all benchmark instances."
    )
    save_caption(cap, "Fig9_failure_modes", out)

    logger.info("Fig. 9 saved: %s", list(paths.values()))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Public API — ablation (Fig 7)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping from variant name (CSV column) → display label → colour
_VARIANT_META = {
    "zero_shot":    ("Zero-shot",          ZERO_SHOT),
    "priority_only": ("Priority only",      HINT_PRI),
    "locked_only":   ("Locked only",        HINT_LCK),
    "moves_only":    ("Moves only",         HINT_MOV),
    "full_hints":    ("Full hints",         HINT_FULL),
    "retrieval_only": ("Retrieval only",    BASELINE),
    "param_only":    ("Param only",         ACCENT),
    "dual_channel":  ("Dual channel",       PROPOSED),
}


def plot_ablation(
    ablation_csv: str | Path,
    output_dir: str | Path = "figures",
    caption_path: str | Path | None = None,
    variants: Optional[list[str]] = None,
) -> dict[str, Path]:
    """Generate Fig. 7: Ablation Study.

    Args:
        ablation_csv: CSV with columns ``instance``, ``variant``, ``vehicles_used``,
                     ``feasible``. Typically produced by ``run_ablation``.
        output_dir:  Output directory.
        caption_path: Override path for caption draft.
        variants:     Subset of variants to plot (default: first 5 standard ones).

    Returns:
        Dict mapping saved-filename to absolute Path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")

    setup_style()
    df = _pd.read_csv(ablation_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if variants is None:
        variants = ["zero_shot", "priority_only", "locked_only", "moves_only", "full_hints"]
    available = [v for v in variants if v in df["variant"].values]
    if not available:
        logger.warning("No ablation variants found in %s", ablation_csv)
        return {}

    fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    _panel_ablation_vehicles(df, available, axes[0])
    _panel_ablation_feasibility(df, available, axes[1])

    for i, ax in enumerate(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
        label_panel(ax, chr(97 + i), x=-0.10, y=1.05)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    paths = save_fig(fig, "Fig7_ablation", out)
    _plt.close(fig)

    cap = (
        "Fig. 7. Ablation analysis: contribution of each guidance type to "
        "LLM-guided solver performance. (a) Mean vehicle count across ablation "
        "variants. Zero-shot = solver without guidance. Full hints = all guidance "
        "types combined. (b) Feasibility rate (%) by guidance type."
    )
    save_caption(cap, "Fig7_ablation", out)

    # Write stats table
    _write_ablation_stats(df, available, out)

    logger.info("Fig. 7 saved: %s", list(paths.values()))
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers — failure modes
# ─────────────────────────────────────────────────────────────────────────────

def _load_events(path: str | Path):
    """Load event-log JSONL.  Returns (meta_list, events_list)."""
    from src.utils.event_logger import load_events_from_jsonl
    return load_events_from_jsonl(path)


def _panel_funnel(events: list, ax) -> None:
    """Horizontal funnel: fraction surviving each filtering layer."""
    total = sum(1 for e in events if getattr(e.phase, "value", "") == "llm_query")

    stages = {
        "LLM queries":        total,
        "Valid JSON":         sum(1 for e in events
                                   if getattr(e.phase, "value", "") == "parse"
                                   and getattr(e.failure_mode, "value", "") == "none"),
        "Hints accepted":     sum(1 for e in events
                                   if getattr(e.phase, "value", "") == "hint_transform"
                                   and getattr(e, "warmstart_accepted", False)),
        "Warm-start applied": sum(1 for e in events
                                   if getattr(e.phase, "value", "") == "final_solve"),
        "Improved":           sum(1 for e in events
                                   if getattr(e.improvement, "value", "") == "improved"),
    }

    max_val = max(stages.values()) or 1
    colors = [FAIL_RED, ACCENT, HINT_FULL, BASELINE, PROPOSED]

    for i, (label, count) in enumerate(stages.items()):
        width = count / max_val
        x = (1 - width) / 2
        ax.barh(i, width, left=x, height=0.55,
                color=colors[i % len(colors)], alpha=0.80)
        ax.text(0.5, i, f"{label}: {count}",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="white")

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised proportion", fontsize=9)
    ax.set_title("(a) Filtering funnel", fontsize=10)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=8)


def _panel_failure_bars(events: list, ax) -> None:
    """Bar chart of failure category frequencies."""
    failure_events = [
        e for e in events
        if getattr(e.failure_mode, "value", "none") not in ("none", "")
    ]

    if not failure_events:
        empty_panel(ax, "(b) No failures recorded")
        return

    counts = Counter(getattr(e.failure_mode, "value", "unknown") for e in failure_events)
    modes  = list(counts.keys())
    values = list(counts.values())
    colors = [FAIL_RED, ACCENT, "#FFC107", "#9C27B0", "#3F51B5", "#009688"]

    bars = ax.bar(modes, values, color=colors[:len(modes)], alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Failure mode", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("(b) Failure category frequency", fontsize=10)
    ax.tick_params(axis="x", rotation=25, labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers — ablation
# ─────────────────────────────────────────────────────────────────────────────

def _panel_ablation_vehicles(df: _pd.DataFrame, variants: list[str], ax) -> None:
    """Mean vehicle count ± std by ablation variant."""
    x = range(len(variants))

    means = [df[df["variant"] == v]["vehicles_used"].mean() for v in variants]
    stds  = [df[df["variant"] == v]["vehicles_used"].std()  for v in variants]

    colors = [_VARIANT_META.get(v, (v, "#999999"))[1] for v in variants]

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  capsize=3, error_kw={"linewidth": 1})
    ax.set_xticks(list(x))
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax.set_ylabel("Mean vehicles used", fontsize=9)
    ax.set_title("(a) Vehicle count by guidance type", fontsize=10)

    for bar, val, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)


def _panel_ablation_feasibility(df: _pd.DataFrame, variants: list[str], ax) -> None:
    """Feasibility rate (%) by ablation variant."""
    x = range(len(variants))

    rates  = [df[df["variant"] == v]["feasible"].mean() * 100 for v in variants]
    colors = [_VARIANT_META.get(v, (v, "#999999"))[1] for v in variants]

    bars = ax.bar(x, rates, color=colors, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax.set_ylabel("Feasible rate (%)", fontsize=9)
    ax.set_title("(b) Feasibility by guidance type", fontsize=10)
    ax.set_ylim(0, 115)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)


def _write_ablation_stats(df: _pd.DataFrame, variants: list[str], out: Path) -> None:
    """Write a markdown table of ablation statistics."""
    lines = [
        "# Fig. 7 — Ablation Statistics\n",
        "| Variant | Mean vehicles | Std | Feasible rate |",
        "|---------|---------------|-----|---------------|",
    ]
    for v in variants:
        sub = df[df["variant"] == v]
        if sub.empty:
            continue
        label = _VARIANT_META.get(v, (v, ""))[0]
        lines.append(
            f"| {label} | {sub['vehicles_used'].mean():.3f} "
            f"| {sub['vehicles_used'].std():.3f} "
            f"| {sub['feasible'].mean()*100:.0f}% |"
        )
    Path(out / "Fig7_ablation_stats.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pf = sub.add_parser("failure-modes", help="Generate Fig. 9 (failure modes)")
    pf.add_argument("--events", required=True)
    pf.add_argument("--output", default="figures")

    pa = sub.add_parser("ablation", help="Generate Fig. 7 (ablation)")
    pa.add_argument("--results", required=True, help="Ablation CSV path")
    pa.add_argument("--output",  default="figures")

    args = p.parse_args()

    if args.cmd == "failure-modes":
        plot_failure_modes(args.events, args.output)
    elif args.cmd == "ablation":
        plot_ablation(args.results, args.output)
    else:
        p.print_help()
