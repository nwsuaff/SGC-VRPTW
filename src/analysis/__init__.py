"""Statistical tests and summary utilities for experiment analysis.

Provides paired Wilcoxon tests, bootstrap confidence intervals, effect size
computation, and structured comparison summaries for paper results.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def wilcoxon_paired_test(
    results_a: list[float],
    results_b: list[float],
    metric: str = "vehicles_used",
) -> dict[str, Any]:
    """Perform a paired Wilcoxon signed-rank test.

    Tests whether the median difference between two paired samples is zero.

    Args:
        results_a: Values from method A (e.g., llm_loop).
        results_b: Values from method B (e.g., pyvrp_only).
        metric: Name of the metric being compared (for reporting).

    Returns:
        Dict with keys: statistic, p_value, n, conclusion.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {
            "statistic": None,
            "p_value": None,
            "n": len(results_a),
            "conclusion": "scipy not installed",
            "metric": metric,
        }

    if len(results_a) != len(results_b):
        raise ValueError(f"Length mismatch: {len(results_a)} vs {len(results_b)}")

    diffs = [a - b for a, b in zip(results_a, results_b)]
    # Remove zeros (ties)
    diffs_nonzero = [d for d in diffs if d != 0]

    if len(diffs_nonzero) < 5:
        return {
            "statistic": None,
            "p_value": None,
            "n": len(results_a),
            "n_nonzero": len(diffs_nonzero),
            "conclusion": "insufficient non-tie pairs",
            "metric": metric,
        }

    stat, p_value = wilcoxon(diffs_nonzero)

    conclusion = "significant" if p_value < 0.05 else "not significant"
    better = (
        "A"
        if np.median(diffs_nonzero) < 0
        else "B"
        if np.median(diffs_nonzero) > 0
        else "tie"
    )

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": len(results_a),
        "n_nonzero": len(diffs_nonzero),
        "median_diff": float(np.median(diffs_nonzero)),
        "mean_diff": float(np.mean(diffs_nonzero)),
        "better_method": better,
        "conclusion": f"{conclusion} (p={'<0.001' if p_value < 0.001 else f'{p_value:.4f}'})",
        "metric": metric,
    }


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    stat: str = "mean",
) -> dict[str, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Args:
        values: Sample values.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g., 0.95 for 95% CI).
        stat: Statistic to compute ('mean', 'median').

    Returns:
        Dict with observed, ci_lower, ci_upper, se.
    """
    values_arr = np.array(values, dtype=float)
    n = len(values_arr)

    if stat == "mean":
        observed = float(np.mean(values_arr))
        resampled = [
            float(np.mean(np.random.choice(values_arr, size=n, replace=True)))
            for _ in range(n_bootstrap)
        ]
    elif stat == "median":
        observed = float(np.median(values_arr))
        resampled = [
            float(np.median(np.random.choice(values_arr, size=n, replace=True)))
            for _ in range(n_bootstrap)
        ]
    else:
        raise ValueError(f"Unknown stat: {stat}")

    alpha = 1 - ci
    ci_lower = float(np.percentile(resampled, 100 * alpha / 2))
    ci_upper = float(np.percentile(resampled, 100 * (1 - alpha / 2)))
    se = float(np.std(resampled))

    return {
        "observed": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": se,
        "n": n,
        "ci_level": ci,
        "stat": stat,
    }


def compute_effect_size(
    results_a: list[float],
    results_b: list[float],
) -> dict[str, float]:
    """Compute effect size (Cohen's d) for paired samples.

    Args:
        results_a: Values from method A.
        results_b: Values from method B.

    Returns:
        Dict with Cohen's d, interpretation, and Cliff's delta.
    """
    diffs = np.array([a - b for a, b in zip(results_a, results_b)], dtype=float)

    # Cohen's d for paired samples = mean(diff) / std(diff)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff > 1e-9:
        cohens_d = mean_diff / std_diff
    else:
        cohens_d = 0.0

    # Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    # Cliff's delta (dominance measure for ordinal data)
    n = len(diffs)
    DOM = sum(
        1 if diffs[i] > diffs[j] else -1 if diffs[i] < diffs[j] else 0
        for i in range(n)
        for j in range(i + 1, n)
    )
    n_pairs = n * (n - 1) // 2
    cliffs_delta = DOM / n_pairs if n_pairs > 0 else 0.0

    return {
        "cohens_d": cohens_d,
        "interpretation": interpretation,
        "cliffs_delta": cliffs_delta,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
    }


def summarize_comparison(
    results_a: list[float],
    results_b: list[float],
    metric: str,
    method_a: str = "Method A",
    method_b: str = "Method B",
) -> dict[str, Any]:
    """Produce a comprehensive comparison summary for two methods.

    Args:
        results_a: Metric values for method A.
        results_b: Metric values for method B.
        metric: Name of the metric.
        method_a: Display name for method A.
        method_b: Display name for method B.

    Returns:
        Comprehensive dict with all statistics.
    """
    a_arr = np.array(results_a, dtype=float)
    b_arr = np.array(results_b, dtype=float)

    wilcoxon_result = wilcoxon_paired_test(results_a, results_b, metric)
    effect = compute_effect_size(results_a, results_b)

    ci_a = bootstrap_ci(results_a, stat="mean")
    ci_b = bootstrap_ci(results_b, stat="mean")

    wins_a = sum(1 for a, b in zip(results_a, results_b) if a < b)
    wins_b = sum(1 for a, b in zip(results_a, results_b) if a > b)
    ties = sum(1 for a, b in zip(results_a, results_b) if a == b)
    total = len(results_a)

    return {
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "n_instances": total,
        f"{method_a}_mean": float(np.mean(a_arr)),
        f"{method_a}_std": float(np.std(a_arr)),
        f"{method_a}_median": float(np.median(a_arr)),
        f"{method_a}_ci": [ci_a["ci_lower"], ci_a["ci_upper"]],
        f"{method_b}_mean": float(np.mean(b_arr)),
        f"{method_b}_std": float(np.std(b_arr)),
        f"{method_b}_median": float(np.median(b_arr)),
        f"{method_b}_ci": [ci_b["ci_lower"], ci_b["ci_upper"]],
        "wilcoxon": wilcoxon_result,
        "effect_size": effect,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "win_rate_a": wins_a / total if total > 0 else 0,
        "win_rate_b": wins_b / total if total > 0 else 0,
    }
