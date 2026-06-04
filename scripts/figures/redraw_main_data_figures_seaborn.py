#!/usr/bin/env python3
"""Redraw the manuscript's main data figures with seaborn/matplotlib.

The script uses the checked CSV sources under ``evidence/figure_sources``
and exports editable SVG, LaTeX-ready PDF, and PNG QA previews. It also syncs
the five main figures into ``paper/manuscript/fig/final`` because the organized
manuscript package references those filenames.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "evidence" / "figure_sources"
OUT = ROOT / "evidence" / "qa" / "generated_figures" / "seaborn_redraw"
MAIN = ROOT / "paper" / "manuscript" / "fig" / "final"
QA = ROOT / "evidence" / "qa"
AUDIT = ROOT / "evidence" / "figure_audits" / "figure_audit_seaborn_redraw.md"
NEEDED_DATA = ROOT / "evidence" / "figure_audits" / "needed_data_for_final_figures.md"
NEEDED_SCHEMA = ROOT / "evidence" / "figure_sources" / "needed_data_schema.csv"

FIGURE_MAP = {
    "Figure_1_Static_ORTEC": "Figure_1_Static_ORTEC",
    "Figure_2_Dynamic_Epoch_Trajectory": "Figure_2_Dynamic_Epoch_Trajectory",
    "Figure_3_Instance_Outcome_Map": "Figure_3_Instance_Outcome_Map",
    "Figure_4_Runtime_Tradeoff": "Figure_4_Runtime_Tradeoff",
    "Figure_5_Homberger_Family_Profile": "Figure_5_Homberger_Family_Profile",
}

SCALE_ORDER = ["300", "500", "800"]
HOM_SCALES = [200, 400, 600, 800]
FAMILY_ORDER = ["C1", "C2", "R1", "R2", "RC1", "RC2"]

COL = {
    "ortools": "#2F4B68",
    "droc_latest": "#4F7E72",
    "droc_retry": "#9B6A53",
    "teal": "#4F7E72",
    "blue": "#2F4B68",
    "orange": "#9B6A53",
    "red": "#8F5C5C",
    "gray": "#A9B5BD",
    "light": "#F1F4F5",
    "grid": "#DFE6EA",
    "ink": "#1E2732",
}
SCALE_COLORS = {"300": "#2F4B68", "500": "#4F7E72", "800": "#9B6A53"}
METHOD_COLORS = {
    "OR-Tools": COL["ortools"],
    "SGC, single retained run": COL["droc_latest"],
    "SGC, bounded-retry success": COL["droc_retry"],
}
STATE_ORDER = ["both", "droc_only", "ort_only", "neither"]
STATE_LABELS = {
    "both": "Both feasible",
    "droc_only": "SGC only",
    "ort_only": "OR-Tools only",
    "neither": "Neither",
}
STATE_COLORS = {
    "both": "#2F4B68",
    "droc_only": "#4F7E72",
    "ort_only": "#9B6A53",
    "neither": "#CED6DB",
}


def configure_style() -> None:
    sns.set_theme(
        context="paper",
        style="ticks",
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.grid": False,
            "grid.color": COL["grid"],
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def read_source(name: str) -> pd.DataFrame:
    path = SRC / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in df.columns:
        if col.endswith("feasible") or col in {"ort_feasible", "droc_feasible", "dual_feasible_clean"}:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    if "scale" in df.columns:
        df["scale"] = df["scale"].astype(str)
    return df


def panel_label(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        va="top",
        ha="left",
        color=COL["ink"],
    )
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color=COL["ink"])


def clean_axis(ax: plt.Axes, grid: str | None = "y") -> None:
    sns.despine(ax=ax, trim=False)
    ax.tick_params(axis="both", colors="#3B424A", length=2.5)
    ax.xaxis.label.set_color(COL["ink"])
    ax.yaxis.label.set_color(COL["ink"])
    if grid:
        ax.grid(axis=grid, color=COL["grid"], linewidth=0.45, alpha=0.8)
        ax.set_axisbelow(True)


def save_and_sync(fig: plt.Figure, stem: str) -> dict[str, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    MAIN.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for ext in ["svg", "pdf", "png"]:
        path = OUT / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=600, facecolor="white")
        else:
            fig.savefig(path, facecolor="white")
        paths[ext] = path
        shutil.copyfile(path, MAIN / f"{FIGURE_MAP[stem]}.{ext}")
    plt.close(fig)
    return paths


def add_reference_lines(ax: plt.Axes, x_zero: bool = False, y_zero: bool = False) -> None:
    if x_zero:
        ax.axvline(0, color="#6A737D", lw=0.7, ls=(0, (3, 3)), zorder=0)
    if y_zero:
        ax.axhline(0, color="#6A737D", lw=0.7, ls=(0, (3, 3)), zorder=0)


def fig1_static_ortec() -> None:
    static = read_source("static_ortec_merged.csv")
    static["scale"] = pd.Categorical(static["scale"], SCALE_ORDER, ordered=True)

    rows = []
    for scale in SCALE_ORDER:
        retry = static[(static["scale"].astype(str) == scale) & (static["rule"] == "retry")]
        latest = static[(static["scale"].astype(str) == scale) & (static["rule"] == "latest")]
        for method, feasible, base in [
            ("OR-Tools", retry["ort_feasible"], retry),
            ("SGC, single retained run", latest["droc_feasible"], latest),
            ("SGC, bounded-retry success", retry["droc_feasible"], retry),
        ]:
            rows.append(
                {
                    "scale": scale,
                    "method": method,
                    "rate": feasible.mean() * 100,
                    "count": int(feasible.sum()),
                    "n": len(base),
                }
            )
    feas = pd.DataFrame(rows)
    dual = static[(static["rule"] == "retry") & static["ort_feasible"] & static["droc_feasible"]].copy()
    dual["scale"] = dual["scale"].astype(str)

    fig = plt.figure(figsize=(7.25, 3.05))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.15], wspace=0.42)
    ax0, ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panel_label(ax0, "a", "Feasibility by scale")
    y_pos = {scale: idx for idx, scale in enumerate(SCALE_ORDER[::-1])}
    y_offsets = {"OR-Tools": -0.085, "SGC, single retained run": 0.0, "SGC, bounded-retry success": 0.085}
    n_by_scale = {
        scale: int(feas[feas["scale"] == scale]["n"].max())
        for scale in SCALE_ORDER
    }
    for scale in SCALE_ORDER:
        sub = feas[feas["scale"] == scale].sort_values("rate")
        y = y_pos[scale]
        ax0.plot(sub["rate"], [y] * len(sub), color="#B8C1CA", lw=1.1, zorder=1)
        for _, row in sub.iterrows():
            yy = y + y_offsets[row["method"]]
            ax0.scatter(
                row["rate"],
                yy,
                s=34,
                color=METHOD_COLORS[row["method"]],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax0.text(
                row["rate"] - 2.0 if row["rate"] > 94 else row["rate"] + 1.4,
                yy + 0.02,
                f"{row['count']}",
                fontsize=6.2,
                color=METHOD_COLORS[row["method"]],
                va="center",
                ha="right" if row["rate"] > 94 else "left",
            )
    ax0.set_yticks([y_pos[s] for s in SCALE_ORDER])
    ax0.set_yticklabels([f"{s}\n(n={n_by_scale[s]})" for s in SCALE_ORDER])
    ax0.set_xlim(0, 108)
    ax0.set_xlabel("Feasible units (%)")
    ax0.set_ylabel("Customer scale")
    clean_axis(ax0, "x")
    ax0.legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white", label=m) for m, c in METHOD_COLORS.items()],
        loc="lower left",
        bbox_to_anchor=(-0.01, -0.36),
        ncol=1,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    panel_label(ax1, "b", "Dual-feasible vehicle counts")
    max_vehicle = float(max(dual["ort_vehicles"].max(), dual["droc_vehicles"].max()))
    min_vehicle = float(min(dual["ort_vehicles"].min(), dual["droc_vehicles"].min()))
    ax1.plot([min_vehicle, max_vehicle], [min_vehicle, max_vehicle], color="#98A2AD", lw=0.8, ls=(0, (3, 3)), zorder=0)
    sns.scatterplot(
        data=dual,
        x="ort_vehicles",
        y="droc_vehicles",
        hue="scale",
        palette=SCALE_COLORS,
        s=25,
        alpha=0.76,
        linewidth=0.25,
        edgecolor="white",
        ax=ax1,
        legend=False,
    )
    ax1.set_xlabel("OR-Tools vehicles")
    ax1.set_ylabel("SGC vehicles")
    ax1.set_aspect("equal", adjustable="box")
    ax1.text(0.04, 0.95, "Below diagonal: fewer vehicles", transform=ax1.transAxes, fontsize=6.4, va="top", color="#525B66")
    clean_axis(ax1, "both")

    panel_label(ax2, "c", "Distance improvement")
    sns.boxenplot(
        data=dual,
        x="distance_improvement",
        y="scale",
        order=SCALE_ORDER,
        hue="scale",
        palette=SCALE_COLORS,
        width=0.52,
        linewidth=0.8,
        saturation=0.82,
        showfliers=False,
        legend=False,
        ax=ax2,
    )
    rng = np.random.default_rng(4)
    jittered = dual.copy()
    jittered["_jitter"] = jittered["scale"].map({s: i for i, s in enumerate(SCALE_ORDER)}).astype(float) + rng.normal(0, 0.055, len(jittered))
    for scale in SCALE_ORDER:
        sub = jittered[jittered["scale"] == scale]
        ax2.scatter(
            sub["distance_improvement"],
            sub["_jitter"],
            s=9,
            color=SCALE_COLORS[scale],
            alpha=0.38,
            linewidth=0,
            zorder=2,
        )
    add_reference_lines(ax2, x_zero=True)
    ax2.set_xlabel("Distance reduction relative to OR-Tools (%)")
    ax2.set_ylabel("")
    clean_axis(ax2, "x")
    ax2.set_xlim(min(-12, dual["distance_improvement"].min() - 2), max(36, dual["distance_improvement"].max() + 2))

    fig.align_ylabels([ax0, ax1, ax2])
    save_and_sync(fig, "Figure_1_Static_ORTEC")


def fig2_dynamic_epoch() -> None:
    unit = read_source("dynamic_epoch_unit_latest_source.csv")
    summ = read_source("dynamic_epoch_latest_source.csv")
    unit["scale"] = unit["scale"].astype(str)
    summ["scale"] = summ["scale"].astype(str)
    unit["served_pct"] = unit["served_rate"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.95), sharey=True)
    rng = np.random.default_rng(12)

    for ax, scale, label in zip(axes, SCALE_ORDER, ["a", "b", "c"]):
        panel_label(ax, label, f"Scale {scale} dispatch process")
        for method in ["OR-Tools", "SGC, single retained run"]:
            s = summ[(summ["scale"] == scale) & (summ["method"] == method)].sort_values("epoch")
            if s.empty:
                continue
            color = METHOD_COLORS[method]
            ax.fill_between(
                s["epoch"].to_numpy(float),
                s["served_q1_pct"].to_numpy(float),
                s["served_q3_pct"].to_numpy(float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.plot(
                s["epoch"],
                s["served_median_pct"],
                marker="s" if method == "OR-Tools" else "o",
                ms=3.1,
                lw=1.3,
                ls=(0, (3, 2)) if method == "OR-Tools" else "-",
                color=color,
                label=method,
                zorder=3,
            )
            raw = unit[(unit["scale"] == scale) & (unit["method"] == method)].copy()
            raw["_x"] = raw["epoch"] + rng.normal(0, 0.035, len(raw))
            ax.scatter(raw["_x"], raw["served_pct"], s=3.8, color=color, alpha=0.08, linewidth=0, zorder=1)
        final = (
            summ[(summ["scale"] == scale) & (summ["epoch"] == summ["epoch"].max())]
            .set_index("method")["served_median_pct"]
        )
        if {"OR-Tools", "SGC, single retained run"}.issubset(final.index):
            delta = final["SGC, single retained run"] - final["OR-Tools"]
            ax.text(
                0.98,
                0.08,
                f"Final Δ={delta:+.2f} pp",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=COL["teal"],
            )
        ax.set_xlim(-0.18, 4.18)
        ax.set_ylim(0, 108)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel("Dispatch epoch")
        clean_axis(ax, "y")
        ax.text(0.02, 0.92, "Median + IQR; pale points are unit logs", transform=ax.transAxes, fontsize=6.2, color="#5B6470")
    axes[0].set_ylabel("Logged served rate (%)")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, handlelength=1.8)
    fig.subplots_adjust(bottom=0.24, wspace=0.22)
    save_and_sync(fig, "Figure_2_Dynamic_Epoch_Trajectory")


def _state_matrix(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str], int]:
    sub = df[df["block"] == block].copy()
    sub["scale"] = sub["scale"].astype(str)
    max_n = int(sub.groupby("scale").size().max())
    matrix = np.full((len(SCALE_ORDER), max_n), np.nan)
    code = {state: idx for idx, state in enumerate(STATE_ORDER)}
    for r, scale in enumerate(SCALE_ORDER):
        s = sub[sub["scale"] == scale].sort_values("ort_runtime")
        matrix[r, : len(s)] = [code.get(v, np.nan) for v in s["state"]]
    return matrix, SCALE_ORDER, max_n


def fig3_instance_outcome_map() -> None:
    outcome = read_source("instance_outcome_map_source.csv")
    outcome["scale"] = outcome["scale"].astype(str)

    cmap = ListedColormap([STATE_COLORS[s] for s in STATE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(STATE_ORDER) + 0.5, 1), cmap.N)
    fig = plt.figure(figsize=(7.25, 3.35))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.4, 1.35], hspace=0.54, wspace=0.22)
    map_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    comp_axes = [fig.add_subplot(gs[i, 1]) for i in range(2)]

    for ax, block, label, title in zip(
        map_axes,
        ["static", "dynamic"],
        ["a", "c"],
        ["Static unit-level feasibility states", "Dynamic scenario-level feasibility states"],
    ):
        matrix, ylabels, max_n = _state_matrix(outcome, block)
        panel_label(ax, label, title)
        ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set_xticks([0, max_n // 2, max_n - 1])
        ax.set_xticklabels(["fast", "median", "slow"])
        ax.set_xlabel("Units ordered by OR-Tools runtime" if block == "dynamic" else "")
        ax.set_ylabel("Scale")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax, block, label in zip(comp_axes, ["static", "dynamic"], ["b", "d"]):
        panel_label(ax, label, "Composition")
        comp = (
            outcome[outcome["block"] == block]
            .groupby(["scale", "state"])
            .size()
            .rename("n")
            .reset_index()
        )
        totals = comp.groupby("scale")["n"].transform("sum")
        comp["pct"] = comp["n"] / totals * 100
        left = np.zeros(len(SCALE_ORDER))
        y = np.arange(len(SCALE_ORDER))
        for state in STATE_ORDER:
            vals = [
                float(comp[(comp["scale"] == scale) & (comp["state"] == state)]["pct"].sum())
                for scale in SCALE_ORDER
            ]
            ax.barh(y, vals, left=left, color=STATE_COLORS[state], height=0.58, edgecolor="white", linewidth=0.4)
            for yi, val, base in zip(y, vals, left):
                if val >= 9:
                    ax.text(
                        base + val / 2,
                        yi,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color="white" if state in {"both", "droc_only", "ort_only"} else "#3A424B",
                    )
            left += np.asarray(vals)
        ax.set_yticks(y)
        ax.set_yticklabels(SCALE_ORDER)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share (%)" if block == "dynamic" else "")
        clean_axis(ax, "x")

    handles = [Patch(facecolor=STATE_COLORS[s], edgecolor="none", label=STATE_LABELS[s]) for s in STATE_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=4, handlelength=1.1, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.18)
    save_and_sync(fig, "Figure_3_Instance_Outcome_Map")


def fig4_runtime_tradeoff() -> None:
    static = read_source("static_ortec_merged.csv")
    dynamic = read_source("dynamic_ortec_merged.csv")
    static = static[(static["rule"] == "retry") & static["ort_feasible"] & static["droc_feasible"]].copy()
    dynamic = dynamic[dynamic["rule"] == "retry"].copy()
    static["scale"] = static["scale"].astype(str)
    dynamic["scale"] = dynamic["scale"].astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05))
    specs = [
        (axes[0], static, "vehicle_improvement", "a", "Static: compute cost versus vehicle reduction", "Vehicle reduction (%)"),
        (axes[1], dynamic, "served_gain_pp", "b", "Dynamic: compute cost versus served-rate gain", "Served-rate gain (percentage points)"),
    ]
    for ax, df, metric, label, title, ylabel in specs:
        panel_label(ax, label, title)
        ax.axvspan(1, max(65, df["runtime_ratio"].max() * 1.08), color="#F7F9FA", zorder=0)
        add_reference_lines(ax, y_zero=True)
        ax.axvline(1, color="#6A737D", lw=0.7, ls=(0, (3, 3)), zorder=1)
        if len(df) > 12:
            try:
                sns.kdeplot(
                    data=df,
                    x="runtime_ratio",
                    y=metric,
                    levels=4,
                    color="#8D99A6",
                    linewidths=0.5,
                    alpha=0.35,
                    log_scale=(True, False),
                    ax=ax,
                )
            except Exception:
                pass
        sns.scatterplot(
            data=df,
            x="runtime_ratio",
            y=metric,
            hue="scale",
            hue_order=SCALE_ORDER,
            palette=SCALE_COLORS,
            s=21,
            alpha=0.70,
            linewidth=0.25,
            edgecolor="white",
            ax=ax,
            legend=False,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Runtime ratio (SGC / OR-Tools, log scale)")
        ax.set_ylabel(ylabel)
        clean_axis(ax, "both")
        finite = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["runtime_ratio", metric])
        med_ratio = finite["runtime_ratio"].median()
        med_metric = finite[metric].median()
        ax.text(
            0.04,
            0.95,
            f"n={len(finite)}, median ratio={med_ratio:.1f}x\nmedian gain={med_metric:.1f}",
            transform=ax.transAxes,
            fontsize=6.2,
            va="top",
            color="#4F5863",
        )
    axes[0].legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=SCALE_COLORS[s], markeredgecolor="white", label=s) for s in SCALE_ORDER],
        title="Scale",
        loc="lower center",
        bbox_to_anchor=(1.08, -0.31),
        ncol=3,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.subplots_adjust(bottom=0.25, wspace=0.28)
    save_and_sync(fig, "Figure_4_Runtime_Tradeoff")


def _heatmap_matrix(df: pd.DataFrame, value: str) -> pd.DataFrame:
    mat = (
        df.assign(scale=df["scale"].astype(int))
        .pivot(index="family", columns="scale", values=value)
        .reindex(index=FAMILY_ORDER, columns=HOM_SCALES)
    )
    return mat


def fig5_homberger_family_profile() -> None:
    hom = read_source("homberger_family_profile_source.csv")
    hom["scale"] = hom["scale"].astype(int)

    fig = plt.figure(figsize=(7.25, 2.95))
    gs = fig.add_gridspec(1, 3, wspace=0.37)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    matrices = [
        (_heatmap_matrix(hom, "dual_feasible"), "a", "Dual-feasible rows", "Count out of 10", "crest", 0, 10, None),
        (_heatmap_matrix(hom, "vehicle_median_reduction"), "b", "Median vehicle reduction", "Reduction (%)", "BrBG", -40, 40, 0),
        (_heatmap_matrix(hom, "distance_median_reduction"), "c", "Median distance reduction", "Reduction (%)", "BrBG", -25, 25, 0),
    ]
    for ax, (mat, label, title, cbar_label, cmap, vmin, vmax, center) in zip(axes, matrices):
        panel_label(ax, label, title)
        mask = mat.isna()
        annot = mat.copy()
        if title == "Dual-feasible rows":
            annot = annot.map(lambda x: "" if pd.isna(x) else f"{int(x)}")
        else:
            annot = annot.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        sns.heatmap(
            mat,
            mask=mask,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            annot=annot,
            fmt="",
            annot_kws={"fontsize": 6.2},
            linewidths=0.55,
            linecolor="white",
            cbar=True,
            cbar_kws={"label": cbar_label, "shrink": 0.72, "pad": 0.025},
            ax=ax,
        )
        ax.set_xlabel("Customers")
        ax.set_ylabel("Family" if ax is axes[0] else "")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if mask.any().any():
            for y, family in enumerate(mat.index):
                for x, scale in enumerate(mat.columns):
                    if pd.isna(mat.loc[family, scale]):
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="#F1F3F5", edgecolor="white", lw=0.55, hatch="///", alpha=0.9))
    fig.text(0.5, 0.015, "Hatched cells indicate no eligible dual-feasible paired comparison in the checked result file.", ha="center", fontsize=6.7, color="#59636E")
    fig.subplots_adjust(bottom=0.18)
    save_and_sync(fig, "Figure_5_Homberger_Family_Profile")


def write_audit_report() -> None:
    static = read_source("static_ortec_merged.csv")
    dynamic = read_source("dynamic_ortec_merged.csv")
    epoch = read_source("dynamic_epoch_unit_latest_source.csv")
    outcome = read_source("instance_outcome_map_source.csv")
    hom = read_source("homberger_family_profile_source.csv")

    lines = [
        "# Seaborn Figure Redraw Audit",
        "",
        "## Runtime",
        "",
        f"- Python backend: seaborn/matplotlib via `{Path(__import__('sys').executable)}`.",
        f"- Source CSV folder: `{SRC.relative_to(ROOT)}`.",
        f"- Editable/vector output folder: `{OUT.relative_to(ROOT)}`.",
        f"- Manuscript figure folder synced: `{MAIN.relative_to(ROOT)}`.",
        "",
        "## Data Sources Checked",
        "",
        "| Figure | Source rows | Core evidence | Redraw decision |",
        "|---|---:|---|---|",
        f"| Figure 1 | {len(static)} | Static ORTEC feasibility, vehicles, distance | Dumbbell feasibility + paired scatter + boxen/strip distribution |",
        f"| Figure 2 | {len(epoch)} | Dynamic epoch trajectories | Median/IQR lines with faint unit-level points |",
        f"| Figure 3 | {len(outcome)} | Instance/scenario feasibility states | Runtime-ordered outcome heatmap plus composition bars |",
        f"| Figure 4 | {len(static) + len(dynamic)} | Runtime-benefit trade-off | Log-scale scatter with reference regions and density contours |",
        f"| Figure 5 | {len(hom)} | Homberger family-scale conditionality | Seaborn heatmaps with hatching for missing paired comparisons |",
        "",
        "## Audit Findings",
        "",
        "- The previous main figures were data-driven, but several panels had large unused whitespace and weak hierarchy at manuscript scale.",
        "- The dynamic trajectory figure was visually under-informative because the median curves are close to deterministic; the redraw adds unit-level traces and IQR context.",
        "- The outcome-map figure now pairs tile-level evidence with composition summaries, so the reader can see both individual units and aggregate state shares.",
        "- The Homberger figure remains a conditional generalization figure; missing cells are explicitly marked rather than visually smoothed over.",
        "- No new experimental numbers were fabricated; all plotted values come from the existing checked CSV sources.",
    ]
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_needed_data_list() -> None:
    """Write a concrete list of missing data needed for final-grade figures."""
    rows = [
        {
            "target_file": "evidence/figure_sources/matched_ablation_runs.csv",
            "priority": "P0",
            "figure_use": "Strict ablation figure; replace current diagnostic-only ablation discussion",
            "unit": "one row per matched run",
            "required_columns": "dataset,unit_id,scale,seed,variant,variant_label,aggregation_rule,time_limit_sec,llm_model,solver_name,solver_version,feasible,served_rate,vehicles,distance,runtime_sec,llm_time_sec,parse_time_sec,solver_time_sec,debug_iterations,failure_category,baseline_feasible,baseline_vehicles,baseline_distance,baseline_runtime_sec",
            "minimum_coverage": "same units, seeds, and time budgets for solver_only, llm_code_only, scene_card_no_debug, scene_card_self_debug, no_retrieval, no_incumbent, full_droc",
            "reason": "The current repository has protocol/debug boxplots but no matched causal ablation CSV.",
        },
        {
            "target_file": "evidence/figure_sources/dynamic_served_orders_recomputed.csv",
            "priority": "P0",
            "figure_use": "Dynamic served-rate figure and final served-rate gain statistics",
            "unit": "one row per scenario-epoch-attempt-method",
            "required_columns": "scenario,scale,epoch,method,attempt_id,seed,total_orders,unique_served_orders,duplicate_served_orders,served_rate_recomputed,served_rate_logged,feasible,runtime_sec",
            "minimum_coverage": "all ORTEC dynamic scenarios and epochs used in Figure 2",
            "reason": "Current logged SGC served rates can exceed 1.0; final figures should recompute from unique served orders.",
        },
        {
            "target_file": "evidence/figure_sources/attempt_level_debug_logs.csv",
            "priority": "P1",
            "figure_use": "Failure diagnosis and self-debugging mechanism figure",
            "unit": "one row per generation/debug attempt",
            "required_columns": "dataset,unit_id,scale,seed,variant,attempt_id,iteration,prompt_tokens,completion_tokens,parse_status,verifier_status,failure_category,accepted,runtime_sec,vehicles,distance,served_rate",
            "minimum_coverage": "all failed and recovered SGC runs in ORTEC static and dynamic settings",
            "reason": "A reviewer-facing diagnosis needs attempt-level error categories, not only final feasible/infeasible labels.",
        },
        {
            "target_file": "evidence/figure_sources/anytime_quality_traces.csv",
            "priority": "P1",
            "figure_use": "Anytime budget-quality curve",
            "unit": "one row per method-unit-seed-time checkpoint",
            "required_columns": "dataset,unit_id,scale,seed,method,time_checkpoint_sec,incumbent_feasible,incumbent_vehicles,incumbent_distance,served_rate,best_known_vehicles,best_known_distance",
            "minimum_coverage": "at least ORTEC static and dynamic, 3-5 checkpoints per run",
            "reason": "The current runtime figure shows final overhead only; anytime traces would support deployment budget claims.",
        },
        {
            "target_file": "evidence/figure_sources/protocol_replicates_by_seed.csv",
            "priority": "P1",
            "figure_use": "Uncertainty intervals for feasibility and quality improvements",
            "unit": "one row per unit-method-seed",
            "required_columns": "dataset,unit_id,scale,seed,method,aggregation_rule,feasible,vehicles,distance,served_rate,runtime_sec",
            "minimum_coverage": "3-5 seeds on the same representative ORTEC subset",
            "reason": "Most current figures are deterministic or log-aggregation based; seed replicates would justify uncertainty bands.",
        },
        {
            "target_file": "evidence/figure_sources/homberger_rerun_clean.csv",
            "priority": "P2",
            "figure_use": "Upgrade Homberger from conditional heatmap to stronger generalization evidence",
            "unit": "one row per instance-method-seed",
            "required_columns": "instance,family,scale,seed,method,feasible,vehicles,distance,runtime_sec,bks_vehicles,bks_distance,failure_category",
            "minimum_coverage": "all 240 checked Homberger instances, with zero baseline distance/vehicle rows handled by logged reason",
            "reason": "Current Homberger evidence has many non-dual-feasible cells, so the claim must remain conditional.",
        },
    ]
    df = pd.DataFrame(rows)
    NEEDED_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(NEEDED_SCHEMA, index=False)

    lines = [
        "# Needed Data for Final-Grade Manuscript Figures",
        "",
        "This list separates what can be plotted from the current checked CSV files from what still needs new or cleaned experiments. The current main figures do not fabricate missing data; the items below are needed only if the manuscript wants stronger ablation, dynamic, and uncertainty claims.",
        "",
        "| Priority | Target CSV | Figure use | Minimum coverage | Why needed |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | `{row['target_file']}` | {row['figure_use']} | {row['minimum_coverage']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Column Schemas",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### `{row['target_file']}`",
                "",
                f"- Unit: {row['unit']}",
                f"- Required columns: `{row['required_columns']}`",
                "",
            ]
        )
    NEEDED_DATA.parent.mkdir(parents=True, exist_ok=True)
    NEEDED_DATA.write_text("\n".join(lines), encoding="utf-8")


def make_contact_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageOps

    paths = [OUT / f"{stem}.png" for stem in FIGURE_MAP]
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        thumb = ImageOps.contain(img, (590, 330), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (630, 385), "white")
        canvas.paste(thumb, ((630 - thumb.width) // 2, 38))
        draw = ImageDraw.Draw(canvas)
        draw.text((14, 12), path.stem, fill=(30, 34, 40))
        thumbs.append(canvas)
    sheet = Image.new("RGB", (1260, 1155), (245, 247, 249))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % 2) * 630, (idx // 2) * 385))
    QA.mkdir(parents=True, exist_ok=True)
    out = QA / "seaborn_redraw_main_figures_contact_sheet.png"
    sheet.save(out)
    return out


def main() -> None:
    configure_style()
    fig1_static_ortec()
    fig2_dynamic_epoch()
    fig3_instance_outcome_map()
    fig4_runtime_tradeoff()
    fig5_homberger_family_profile()
    write_audit_report()
    write_needed_data_list()
    sheet = make_contact_sheet()
    print(f"Redrew figures in {OUT.relative_to(ROOT)}")
    print(f"Synced manuscript figures to {MAIN.relative_to(ROOT)}")
    print(f"QA contact sheet: {sheet.relative_to(ROOT)}")
    print(f"Audit report: {AUDIT.relative_to(ROOT)}")
    print(f"Needed data list: {NEEDED_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
