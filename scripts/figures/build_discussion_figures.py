#!/usr/bin/env python3
"""Build data-grounded discussion figures for the manuscript.

The figures summarize protocol sensitivity, evidence boundaries, and missing
experiment requirements. They do not introduce new experimental results.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "evidence" / "figure_sources"
OUT = ROOT / "paper" / "manuscript" / "fig" / "final"

PAL = {
    "ink": "#1E2732",
    "muted": "#66737D",
    "grid": "#DFE6EA",
    "wash": "#F7F9F9",
    "blue": "#2F4B68",
    "blue_light": "#7B8FA1",
    "teal": "#4F7E72",
    "ochre": "#9B6A53",
    "moss": "#74836A",
    "rust": "#8F5C5C",
    "gray": "#A9B5BD",
}


def apply_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "axes.edgecolor": PAL["ink"],
            "axes.labelcolor": PAL["ink"],
            "xtick.color": PAL["ink"],
            "ytick.color": PAL["ink"],
            "grid.color": PAL["grid"],
            "grid.linewidth": 0.42,
            "axes.linewidth": 0.72,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(
        -0.105,
        1.075,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=PAL["ink"],
    )
    ax.set_title(title, loc="left", pad=4.5, fontsize=7.35, fontweight="bold", color=PAL["ink"])


def despine(ax: plt.Axes, axis: str | None = "x") -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.62)
        spine.set_color(PAL["ink"])
    ax.tick_params(axis="both", length=2.3, width=0.55, direction="out", colors=PAL["ink"])
    if axis:
        ax.grid(True, axis=axis, color=PAL["grid"], linewidth=0.36, alpha=0.86)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    ax.margins(x=0.02, y=0.06)


def wrap_labels(labels: list[str], width: int = 23) -> list[str]:
    return ["\n".join(textwrap.wrap(str(label), width=width)) for label in labels]


def build_protocol_evidence() -> None:
    agg = pd.read_csv(SRC / "aggregation_sensitivity_source.csv")
    profile = pd.read_csv(SRC / "iter06_evidence_atlas_feasibility_profile.csv")
    inventory = pd.read_csv(SRC / "protocol_evidence_inventory.csv")

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.06, 0.94], height_ratios=[1.03, 0.97], wspace=0.30, hspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    agg = agg.copy()
    agg["label"] = agg["block"] + " " + agg["scale"].astype(str)
    agg = agg.sort_values(["block", "scale"], ascending=[True, True]).reset_index(drop=True)
    ypos = np.arange(len(agg))[::-1]
    ax.hlines(ypos, agg["latest_rate"], agg["retry_rate"], color=PAL["grid"], linewidth=3.2, zorder=1)
    ax.scatter(agg["latest_rate"], ypos, s=42, color=PAL["blue_light"], edgecolor="white", linewidth=0.5, label="Latest retained run", zorder=3)
    ax.scatter(agg["retry_rate"], ypos, s=42, color=PAL["ochre"], edgecolor="white", linewidth=0.5, label="Bounded retry success", zorder=3)
    for y, row in zip(ypos, agg.itertuples(index=False)):
        delta = row.retry_rate - row.latest_rate
        ax.text(row.retry_rate + 1.0, y, f"+{delta:.1f} pp", va="center", ha="left", fontsize=6.1, color=PAL["muted"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(agg["label"])
    ax.set_xlim(65, 103.5)
    ax.set_xlabel("SGC feasible scenarios or instances (%)")
    panel_label(ax, "A", "Protocol sensitivity")
    despine(ax, "x")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3)

    ax = fig.add_subplot(gs[0, 1])
    p = profile.copy()
    p["row"] = p["dataset"].str.replace(" ORTEC", "", regex=False) + " " + p["scale"].astype(str)
    p["col"] = np.where(
        p["method"].eq("OR-Tools"),
        "OR-Tools",
        np.where(p["protocol"].str.contains("retry", case=False), "SGC retry", "SGC latest"),
    )
    row_order = ["Static 300", "Static 500", "Static 800", "Dynamic 300", "Dynamic 500", "Dynamic 800"]
    col_order = ["OR-Tools", "SGC latest", "SGC retry"]
    mat = p.pivot(index="row", columns="col", values="rate").reindex(row_order)[col_order]
    annot = p.pivot(index="row", columns="col", values="success").reindex(row_order)[col_order].astype("Int64").astype(str)
    nmat = p.pivot(index="row", columns="col", values="n").reindex(row_order)[col_order].astype("Int64").astype(str)
    labels = mat.copy().astype(str)
    for r in mat.index:
        for c in mat.columns:
            labels.loc[r, c] = f"{mat.loc[r, c]:.1f}\n{annot.loc[r, c]}/{nmat.loc[r, c]}"
    sns.heatmap(
        mat,
        ax=ax,
        cmap=sns.light_palette(PAL["teal"], as_cmap=True),
        vmin=30,
        vmax=100,
        annot=labels,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Feasibility (%)", "shrink": 0.76},
        annot_kws={"fontsize": 5.8, "color": PAL["ink"]},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=28)
    panel_label(ax, "B", "Feasibility profile")

    ax = fig.add_subplot(gs[1, 0])
    inv = inventory.copy()
    inv["duplicate_ratio"] = inv["duplicate_units"] / inv["unique_units"].replace(0, np.nan) * 100
    inv = inv.sort_values("duplicate_ratio", ascending=True)
    ax.barh(inv["block"], inv["duplicate_ratio"], color=PAL["blue"], alpha=0.83, height=0.58)
    for y, row in enumerate(inv.itertuples(index=False)):
        ax.text(row.duplicate_ratio + 2.0, y, f"{row.duplicate_units}/{row.unique_units}", va="center", ha="left", fontsize=6.2, color=PAL["muted"])
    ax.set_xlabel("Duplicate attempt records per unique unit (%)")
    ax.set_xlim(0, max(inv["duplicate_ratio"].max() * 1.23, 10))
    panel_label(ax, "C", "Attempt-level duplication")
    despine(ax, "x")

    ax = fig.add_subplot(gs[1, 1])
    inv["eligibility"] = inv["eligible_quality_units"] / inv["unique_units"].replace(0, np.nan) * 100
    inv = inv.sort_values("eligibility", ascending=True)
    colors = [PAL["rust"] if val < 30 else PAL["ochre"] if val < 80 else PAL["teal"] for val in inv["eligibility"]]
    ax.barh(inv["block"], inv["eligibility"], color=colors, alpha=0.9, height=0.58)
    for y, row in enumerate(inv.itertuples(index=False)):
        ax.text(row.eligibility + 1.8, y, f"{int(row.eligible_quality_units)}/{int(row.unique_units)}", va="center", ha="left", fontsize=6.2, color=PAL["muted"])
    ax.set_xlabel("Units eligible for paired quality claims (%)")
    ax.set_xlim(0, 105)
    panel_label(ax, "D", "Quality-claim eligibility")
    despine(ax, "x")

    save(fig, "Figure_9_Discussion_Protocol_Evidence")


def build_evidence_readiness() -> None:
    ledger = pd.read_csv(SRC / "iter06_evidence_atlas_claim_ledger.csv")
    needed = pd.read_csv(SRC / "needed_data_schema.csv")
    debug = pd.read_csv(SRC / "debug_iteration_boxplot_source.csv")

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 0.92], height_ratios=[1.08, 0.92], wspace=0.34, hspace=0.42)

    ax = fig.add_subplot(gs[0, :])
    led = ledger.copy()
    led["claim_label"] = [
        "Static ORTEC\nfeasibility",
        "Dynamic dispatch\nservice",
        "Homberger\nquality",
        "Runtime\ntrade-off",
        "Self-debug\nmechanism",
        "Component\nablation",
        "Anytime and\nreplicates",
    ]
    strength_map = {"supported": 2.0, "conditional": 1.15, "descriptive": 0.72, "missing": 0.0}
    led["score"] = led["strength"].map(strength_map)
    x = np.arange(len(led))
    colors = [PAL["teal"] if s >= 1.9 else PAL["ochre"] if s >= 1 else PAL["blue_light"] if s > 0 else PAL["rust"] for s in led["score"]]
    ax.bar(x, led["score"], color=colors, width=0.66)
    for i, row in enumerate(led.itertuples(index=False)):
        marker = "rerun" if bool(row.rerun_needed) else "ready"
        ypos = row.score + 0.07 if row.score > 0 else 0.15
        ax.text(i, ypos, marker, ha="center", va="bottom", fontsize=5.8, color=PAL["muted"])
        if row.score > 0:
            ax.text(i, 0.06, f"n={int(row.rows)}", ha="center", va="bottom", fontsize=5.8, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(led["claim_label"])
    ax.set_ylim(0, 2.45)
    ax.set_yticks([0, 0.75, 1.15, 2.0])
    ax.set_yticklabels(["missing", "descriptive", "conditional", "supported"])
    ax.set_ylabel("Evidence status")
    panel_label(ax, "A", "Claim-level evidence readiness")
    despine(ax, "y")

    ax = fig.add_subplot(gs[1, 0])
    dbg = debug[(debug["metric"].eq("runtime_sec"))].copy()
    dbg["scale"] = dbg["scale"].astype(str)
    dbg["iteration_group"] = dbg["iteration_group"].astype(str)
    order = sorted(dbg["iteration_group"].astype(str).unique(), key=lambda v: (len(v), v))
    sns.boxplot(
        data=dbg,
        x="iteration_group",
        y="value",
        order=order,
        ax=ax,
        fliersize=1.4,
        linewidth=0.58,
        color=PAL["blue_light"],
    )
    for i, group in enumerate(order):
        n = int((dbg["iteration_group"] == group).sum())
        ax.text(i, dbg["value"].max() + 10, f"n={n}", ha="center", va="bottom", fontsize=5.9, color=PAL["muted"])
    ax.set_xlabel("Accepted or terminal generation iteration")
    ax.set_ylabel("Runtime (s)")
    ax.set_ylim(40, dbg["value"].max() + 52)
    panel_label(ax, "B", "Debug-iteration runtime diagnostics")
    despine(ax, "y")

    ax = fig.add_subplot(gs[1, 1])
    needed = needed.copy()
    priority_order = ["P0", "P1", "P2"]
    needed["priority"] = pd.Categorical(needed["priority"], categories=priority_order, ordered=True)
    counts = needed.groupby("priority", observed=False).size().reindex(priority_order, fill_value=0)
    ax.bar(counts.index.astype(str), counts.values, color=[PAL["rust"], PAL["ochre"], PAL["moss"]], width=0.58)
    for i, val in enumerate(counts.values):
        ax.text(i, val + 0.08, str(int(val)), ha="center", va="bottom", fontsize=6.4, color=PAL["ink"])
    ax.set_ylabel("Required data files")
    ax.set_xlabel("Priority")
    ax.set_ylim(0, max(counts.max() + 1.1, 2))
    panel_label(ax, "C", "Missing experiment inputs")
    despine(ax, "y")

    right = ax.twinx()
    right.set_yticks([])
    right.set_ylabel("")
    notes = [
        "P0: matched ablation; unique served-order recomputation",
        "P1: attempt-level debug logs; anytime traces; seed replicates",
        "P2: clean Homberger rerun",
    ]
    ax.text(
        1.08,
        0.92,
        "\n".join(notes),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.05,
        color=PAL["muted"],
        linespacing=1.28,
    )

    save(fig, "Figure_10_Discussion_Evidence_Readiness")


def _rounded(ax: plt.Axes, x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 0.7, r: float = 0.02, z: int = 1) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={r}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=z,
        )
    )


def build_protocol_evidence_ccfa() -> None:
    agg = pd.read_csv(SRC / "aggregation_sensitivity_source.csv")
    profile = pd.read_csv(SRC / "iter06_evidence_atlas_feasibility_profile.csv")
    inventory = pd.read_csv(SRC / "protocol_evidence_inventory.csv")

    fig = plt.figure(figsize=(7.2, 5.08))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 0.92, 0.88], height_ratios=[1.02, 0.92], wspace=0.52, hspace=0.46)

    ax = fig.add_subplot(gs[:, 0])
    a = agg.copy()
    a["label"] = a["block"] + " " + a["scale"].astype(str)
    a["group"] = a["block"].map({"Static": "Static", "Dynamic": "Dynamic"})
    a = a.sort_values(["block", "scale"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(a))[::-1]
    colors = a["block"].map({"Static": PAL["blue"], "Dynamic": PAL["teal"]})
    ax.hlines(y, a["latest_rate"], a["retry_rate"], color="#DDE5E8", linewidth=3.1, zorder=1)
    ax.scatter(a["latest_rate"], y, s=46, color="white", edgecolor=colors, linewidth=1.05, zorder=3, label="Latest run")
    ax.scatter(a["retry_rate"], y, s=46, color=colors, edgecolor="white", linewidth=0.55, zorder=4, label="Retry success")
    for yy, row, color in zip(y, a.itertuples(index=False), colors, strict=True):
        ax.text((row.retry_rate + row.latest_rate) / 2, yy + 0.18, f"+{row.retry_rate - row.latest_rate:.1f} pp", va="center", ha="center", fontsize=5.65, color=color)
    ax.axvline(90, color=PAL["grid"], lw=0.65, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(a["label"])
    ax.set_xlim(64, 104)
    ax.set_xlabel("Feasible scenarios or instances (%)")
    panel_label(ax, "A", "Protocol lift under bounded retry")
    despine(ax, "x")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=PAL["blue"], markersize=4.8, label="Latest retained"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PAL["blue"], markeredgecolor="white", markersize=4.8, label="Retry success"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.00, -0.185),
        ncol=2,
        frameon=False,
        fontsize=5.9,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    ax = fig.add_subplot(gs[0, 1:])
    p = profile.copy()
    p["row"] = p["dataset"].str.replace(" ORTEC", "", regex=False) + " " + p["scale"].astype(str)
    p["col"] = np.where(
        p["method"].eq("OR-Tools"),
        "OR-Tools",
        np.where(p["protocol"].str.contains("retry", case=False), "SGC retry", "SGC latest"),
    )
    row_order = ["Static 300", "Static 500", "Static 800", "Dynamic 300", "Dynamic 500", "Dynamic 800"]
    col_order = ["OR-Tools", "SGC latest", "SGC retry"]
    mat = p.pivot(index="row", columns="col", values="rate").reindex(row_order)[col_order]
    annot = p.pivot(index="row", columns="col", values="success").reindex(row_order)[col_order].astype("Int64").astype(str)
    nmat = p.pivot(index="row", columns="col", values="n").reindex(row_order)[col_order].astype("Int64").astype(str)
    labels = mat.copy().astype(str)
    for r in mat.index:
        for c in mat.columns:
            labels.loc[r, c] = f"{mat.loc[r, c]:.0f}\n{annot.loc[r, c]}/{nmat.loc[r, c]}"
    cmap = sns.light_palette(PAL["teal"], as_cmap=True)
    sns.heatmap(
        mat,
        ax=ax,
        cmap=cmap,
        vmin=30,
        vmax=100,
        annot=labels,
        fmt="",
        linewidths=0.55,
        linecolor="white",
        cbar=False,
        annot_kws={"fontsize": 5.8, "color": PAL["ink"]},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    panel_label(ax, "B", "Feasibility support matrix")

    ax = fig.add_subplot(gs[1, 1:])
    inv = inventory.copy()
    inv["duplicate"] = inv["duplicate_units"] / inv["unique_units"].replace(0, np.nan) * 100
    inv["quality"] = inv["eligible_quality_units"] / inv["unique_units"].replace(0, np.nan) * 100
    order = ["ORTEC static", "ORTEC dynamic", "Dynamic epochs", "Homberger"]
    inv["block"] = pd.Categorical(inv["block"], categories=order, ordered=True)
    inv = inv.sort_values("block")
    y = np.arange(len(inv))[::-1]
    ax.hlines(y, inv["quality"], inv["duplicate"], color="#DDE5E8", linewidth=2.4, zorder=1)
    ax.scatter(inv["duplicate"], y, s=52, color=PAL["rust"], edgecolor="white", linewidth=0.55, label="duplicate attempts", zorder=4)
    ax.scatter(inv["quality"], y, s=52, color=PAL["teal"], edgecolor="white", linewidth=0.55, label="paired-quality eligible", zorder=4)
    for yy, row in zip(y, inv.itertuples(index=False), strict=True):
        ax.text(row.duplicate + 2.1, yy + 0.13, f"{int(row.duplicate_units)}/{int(row.unique_units)}", fontsize=5.55, color=PAL["rust"], va="center")
        if abs(row.duplicate - row.quality) < 16 and row.quality > 5:
            qx, qha = row.quality - 2.4, "right"
        else:
            qx, qha = row.quality + 2.1, "left"
        ax.text(qx, yy - 0.15, f"{int(row.eligible_quality_units)}/{int(row.unique_units)}", fontsize=5.55, color=PAL["teal"], va="center", ha=qha)
    ax.set_yticks(y)
    ax.set_yticklabels(inv["block"].astype(str))
    ax.set_xlim(-3, 125)
    ax.set_ylim(-0.25, len(inv) - 0.38)
    ax.set_xlabel("Share of unique units (%)")
    panel_label(ax, "C", "Duplication versus paired support")
    despine(ax, "x")
    c_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PAL["rust"], markeredgecolor="white", markersize=4.6, label="Duplicate attempts"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PAL["teal"], markeredgecolor="white", markersize=4.6, label="Paired-quality eligible"),
    ]
    ax.legend(handles=c_handles, loc="upper left", bbox_to_anchor=(0.0, 1.00), frameon=False, ncol=2, fontsize=5.55, handletextpad=0.25, columnspacing=0.55, borderaxespad=0.0)

    save(fig, "Figure_9_Discussion_Protocol_Evidence")


def build_evidence_readiness_ccfa() -> None:
    ledger = pd.read_csv(SRC / "iter06_evidence_atlas_claim_ledger.csv")
    needed = pd.read_csv(SRC / "needed_data_schema.csv")
    debug = pd.read_csv(SRC / "debug_iteration_boxplot_source.csv")

    fig = plt.figure(figsize=(7.2, 5.05))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.22, 0.86, 0.92], height_ratios=[0.98, 1.02], wspace=0.42, hspace=0.48)

    ax = fig.add_subplot(gs[0, :])
    led = ledger.copy()
    led["claim_label"] = [
        "Static\nfeas.",
        "Dynamic\nservice",
        "Homberger\nquality",
        "Runtime\ntrade-off",
        "Self-debug\nlogs",
        "Component\nablation",
        "Anytime /\nreplicates",
    ]
    status_order = ["missing", "descriptive", "conditional", "supported"]
    ymap = {s: i for i, s in enumerate(status_order)}
    display_strength = led["strength"].replace({"descriptive": "descriptive", "conditional": "conditional", "supported": "supported", "missing": "missing"})
    led["y"] = display_strength.map(ymap)
    colors = {"supported": PAL["teal"], "conditional": PAL["ochre"], "descriptive": PAL["blue_light"], "missing": PAL["rust"]}
    x = np.arange(len(led))
    ax.scatter(x, led["y"], s=360, marker="s", color=[colors[s] for s in display_strength], edgecolor="white", linewidth=0.9, zorder=3)
    for xi, row, strength in zip(x, led.itertuples(index=False), display_strength, strict=True):
        txt = f"n={int(row.rows)}" if int(row.rows) > 0 else "new"
        ax.text(xi, ymap[strength], txt, ha="center", va="center", fontsize=5.65, color="white", fontweight="bold", zorder=4)
        ax.text(xi, ymap[strength] + 0.34, "rerun" if bool(row.rerun_needed) else "ready", ha="center", va="bottom", fontsize=5.35, color=PAL["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels(led["claim_label"])
    ax.set_yticks(range(len(status_order)))
    ax.set_yticklabels(status_order)
    ax.set_ylim(-0.55, 3.75)
    ax.set_xlim(-0.55, len(led) - 0.45)
    ax.grid(True, axis="y", color=PAL["grid"], linewidth=0.42)
    ax.set_axisbelow(True)
    panel_label(ax, "A", "Claim-level evidence matrix")
    despine(ax, "y")

    ax = fig.add_subplot(gs[1, 0])
    dbg = debug[debug["metric"].eq("runtime_sec")].copy()
    order = ["0", "1", "2+"]
    sns.boxplot(
        data=dbg,
        x="iteration_group",
        y="value",
        order=order,
        ax=ax,
        width=0.52,
        fliersize=0,
        linewidth=0.68,
        color="#E9EEF0",
        boxprops={"edgecolor": PAL["blue"], "linewidth": 0.68},
        medianprops={"color": PAL["rust"], "linewidth": 0.78},
        whiskerprops={"color": PAL["blue"], "linewidth": 0.62},
        capprops={"color": PAL["blue"], "linewidth": 0.62},
    )
    sns.stripplot(data=dbg, x="iteration_group", y="value", order=order, ax=ax, color=PAL["blue"], size=2.0, alpha=0.35, jitter=0.22)
    for i, group in enumerate(order):
        n = int((dbg["iteration_group"].astype(str) == group).sum())
        ax.text(i, dbg["value"].max() + 16, f"n={n}", ha="center", va="bottom", fontsize=5.8, color=PAL["muted"])
    ax.set_xlabel("Accepted or terminal generation iteration")
    ax.set_ylabel("Runtime (s)")
    ax.set_ylim(40, dbg["value"].max() + 55)
    panel_label(ax, "B", "Self-debug cost is diagnostic")
    despine(ax, "y")

    ax = fig.add_subplot(gs[1, 1:])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_label(ax, "C", "Extension inputs for broader claims")
    pmap = {"P0": PAL["rust"], "P1": PAL["ochre"], "P2": PAL["moss"]}
    display_rows = {
        "matched_ablation_runs.csv": ("Matched ablation", "module effect", "run"),
        "dynamic_served_orders_recomputed.csv": ("Served-order recomputation", "served rate", "scen.-epoch"),
        "attempt_level_debug_logs.csv": ("Attempt debug logs", "error taxonomy", "attempt"),
        "anytime_quality_traces.csv": ("Anytime traces", "budget curve", "timepoint"),
        "protocol_replicates_by_seed.csv": ("Seed replicates", "CI bands", "unit-seed"),
        "homberger_rerun_clean.csv": ("Clean Homberger rerun", "transfer check", "inst.-seed"),
    }
    ordered = needed.assign(
        basename=needed["target_file"].map(lambda value: Path(str(value)).name),
        rank=needed["priority"].map({"P0": 0, "P1": 1, "P2": 2}),
    ).sort_values(["rank", "basename"])

    ax.text(0.050, 0.855, "Priority", ha="left", va="center", fontsize=5.55, color=PAL["muted"], fontweight="bold")
    ax.text(0.205, 0.855, "Input file", ha="left", va="center", fontsize=5.55, color=PAL["muted"], fontweight="bold")
    ax.text(0.610, 0.855, "Claim enabled", ha="left", va="center", fontsize=5.55, color=PAL["muted"], fontweight="bold")
    ax.text(0.910, 0.855, "Unit", ha="right", va="center", fontsize=5.55, color=PAL["muted"], fontweight="bold")
    ax.plot([0.045, 0.925], [0.825, 0.825], color=PAL["grid"], lw=0.62, clip_on=False)

    row_h = 0.108
    y_start = 0.772
    for idx, row in enumerate(ordered.itertuples(index=False)):
        name, claim, unit = display_rows.get(row.basename, (row.basename.replace(".csv", ""), "broader evidence", "logged unit"))
        yy = y_start - idx * row_h
        bg = "#FFFFFF" if idx % 2 == 0 else "#F8FAFA"
        ax.add_patch(Rectangle((0.045, yy - 0.045), 0.880, 0.082, facecolor=bg, edgecolor=PAL["grid"], linewidth=0.35, zorder=1))
        _rounded(ax, 0.058, yy - 0.025, 0.085, 0.047, pmap[str(row.priority)], pmap[str(row.priority)], lw=0, r=0.018, z=2)
        ax.text(0.100, yy - 0.001, str(row.priority), ha="center", va="center", fontsize=5.75, color="white", fontweight="bold", zorder=3)
        ax.text(0.205, yy + 0.012, name, ha="left", va="center", fontsize=5.85, color=PAL["ink"], fontweight="bold", zorder=3)
        ax.text(0.205, yy - 0.019, row.basename, ha="left", va="center", fontsize=4.75, color=PAL["muted"], zorder=3)
        ax.text(0.610, yy - 0.001, claim, ha="left", va="center", fontsize=5.35, color=PAL["ink"], zorder=3)
        ax.text(0.910, yy - 0.001, unit, ha="right", va="center", fontsize=5.05, color=PAL["muted"], zorder=3)

    save(fig, "Figure_10_Discussion_Evidence_Readiness")


def main() -> None:
    apply_style()
    build_protocol_evidence_ccfa()
    build_evidence_readiness_ccfa()


if __name__ == "__main__":
    main()
