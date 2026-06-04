"""Statistical tests and summary utilities for experiment analysis.

Provides paired Wilcoxon tests, bootstrap confidence intervals, effect size
computation, and structured comparison summaries for paper results.

This module is also accessible as src.analysis.__init__.
"""

from __future__ import annotations

from src.analysis.statistical_tests import (
    wilcoxon_paired_test,
    bootstrap_ci,
    compute_effect_size,
    summarize_comparison,
)

__all__ = [
    "wilcoxon_paired_test",
    "bootstrap_ci",
    "compute_effect_size",
    "summarize_comparison",
]
