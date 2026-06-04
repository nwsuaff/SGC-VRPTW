"""All Figures — Run all paper figures at once.

This module is the single entry point for regenerating every figure
in the paper. It calls each figure-generating module in sequence and
reports which outputs were produced.

Expected input files:
  results/comparison/comparison_results.csv   → comparison plots
  results/batch_001.csv                     → Fig 4 (generalization)
  results/ablation_001.csv                  → Fig 7 (ablation)
  results/events.jsonl                       → Fig 9 (failure modes)

All outputs are written to ``figures/`` (PDF + PNG) with caption drafts
and statistics sidecars.

Usage:
    # Run everything
    python -m src.analysis.all_figures

    # Run specific figures only
    python -m src.analysis.all_figures --what comparison
    python -m src.analysis.all_figures --what generalization
    python -m src.analysis.all_figures --what ablation
    python -m src.analysis.all_figures --what failure_modes
    python -m src.analysis.all_figures --what all
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.analysis.plot_utils import setup_style, save_fig

logger = logging.getLogger(__name__)

try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared defaults (override with CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT  = Path("figures")
DEFAULT_RESULTS    = Path("results/comparison/comparison_results.csv")
DEFAULT_BATCH      = Path("results/batch_001.csv")
DEFAULT_ABLATION   = Path("results/ablation_001.csv")
DEFAULT_EVENTS     = Path("results/events.jsonl")
DEFAULT_BKS       = Path("data/bks_solomon.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ─────────────────────────────────────────────────────────────────────────────

def _run_comparison(output: Path) -> dict[str, Path]:
    """Generate DRoC vs baseline comparison plots."""
    from src.analysis.comparison_plots import plot_comparison_summary
    results_csv = DEFAULT_RESULTS
    if not results_csv.exists():
        logger.warning("Comparison CSV not found: %s  — skipping comparison plots", results_csv)
        return {}
    return plot_comparison_summary(str(results_csv), output, baseline_method="ortools")


def _run_generalization(output: Path) -> dict[str, Path]:
    """Generate Fig. 4: Cross-scenario Generalization."""
    from src.analysis.plot_generalization import plot_generalization
    batch_csv = DEFAULT_BATCH
    bks_csv   = DEFAULT_BKS if DEFAULT_BKS.exists() else None
    if not batch_csv.exists():
        logger.warning("Batch CSV not found: %s  — skipping Fig. 4", batch_csv)
        return {}
    return plot_generalization(str(batch_csv), str(bks_csv) if bks_csv else None, output)


def _run_ablation(output: Path) -> dict[str, Path]:
    """Generate Fig. 7: Ablation Study."""
    from src.analysis.plot_failure_modes import plot_ablation
    ablation_csv = DEFAULT_ABLATION
    if not ablation_csv.exists():
        logger.warning("Ablation CSV not found: %s  — skipping Fig. 7", ablation_csv)
        return {}
    return plot_ablation(str(ablation_csv), output)


def _run_failure_modes(output: Path) -> dict[str, Path]:
    """Generate Fig. 9: Failure Mode Analysis."""
    from src.analysis.plot_failure_modes import plot_failure_modes
    events_jsonl = DEFAULT_EVENTS
    if not events_jsonl.exists():
        logger.warning("Events JSONL not found: %s  — skipping Fig. 9", events_jsonl)
        return {}
    return plot_failure_modes(str(events_jsonl), output)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    output: Path | str = DEFAULT_OUTPUT,
    what: str = "all",
) -> dict[str, Path]:
    """Generate all (or a subset of) paper figures.

    Args:
        output: Root output directory.
        what:   Which figure group to generate:
                "all"           — everything
                "comparison"    — DRoC vs baseline
                "generalization" — Fig 4
                "ablation"       — Fig 7
                "failure_modes"  — Fig 9

    Returns:
        Dict mapping each generated file (relative to output) to its absolute Path.
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required: pip install matplotlib numpy pandas")

    setup_style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    all_paths: dict[str, Path] = {}

    runners = {
        "comparison":     _run_comparison,
        "generalization": _run_generalization,
        "ablation":       _run_ablation,
        "failure_modes":  _run_failure_modes,
    }

    if what == "all":
        groups = list(runners.keys())
    else:
        groups = [what] if what in runners else []

    for group in groups:
        logger.info("─── Generating: %s ───", group)
        try:
            paths = runners[group](output)
            for k, v in paths.items():
                all_paths[k] = v
            logger.info("  %d file(s) generated", len(paths))
        except Exception as exc:
            logger.error("  FAILED %s: %s", group, exc, exc_info=True)

    logger.info(
        "\n=== Done: %d file(s) in %s ===",
        len(all_paths), output.resolve()
    )

    for name, path in sorted(all_paths.items()):
        logger.info("  %s", path)

    return all_paths


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Generate all (or selected) paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m src.analysis.all_figures
  python -m src.analysis.all_figures --what comparison
  python -m src.analysis.all_figures --what generalization
  python -m src.analysis.all_figures --output my_figures/
        """,
    )
    p.add_argument(
        "--what", default="all",
        choices=["all", "comparison", "generalization", "ablation", "failure_modes"],
        help="Which figure group to generate (default: all)",
    )
    p.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output directory (default: figures/)",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_all(output=args.output, what=args.what)
