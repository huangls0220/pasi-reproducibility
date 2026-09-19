"""Statistical analysis (Section 14).

- Mean, std, 95% CI
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d effect size
- Holm-Bonferroni correction
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats


def mean_ci(data: np.ndarray, confidence: float = 0.95) -> dict:
    """Mean and confidence interval."""
    n = len(data)
    if n < 2:
        return {"mean": float(np.mean(data)) if n > 0 else float("nan"),
                "std": float(np.std(data, ddof=1)) if n > 1 else float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan"), "n": n}
    m = float(np.mean(data))
    s = float(np.std(data, ddof=1))
    se = s / math.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    return {"mean": m, "std": s, "ci_lower": m - t_val * se, "ci_upper": m + t_val * se, "n": n}


def paired_t_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 2:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n, "test": "paired_t"}
    t_stat, p_val = stats.ttest_rel(a, b)
    return {"statistic": float(t_stat), "p_value": float(p_val), "n": n, "test": "paired_t"}


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n, "test": "wilcoxon"}
    diff = a - b
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "n": n, "test": "wilcoxon"}
    try:
        w_stat, p_val = stats.wilcoxon(a, b, zero_method="zsplit")
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan"), "n": n, "test": "wilcoxon"}
    return {"statistic": float(w_stat), "p_value": float(p_val), "n": n, "test": "wilcoxon"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    n = len(diff)
    if n < 2:
        return float("nan")
    d = float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-12))
    return d


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni corrected p-values."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])
    adjusted = [0.0] * n
    for rank, (p, idx) in enumerate(indexed):
        adjusted[idx] = min(1.0, p * (n - rank))
    # Ensure monotonicity
    for i in range(n - 1):
        adj_idx_i = indexed[i][1]
        adj_idx_j = indexed[i + 1][1]
        if adjusted[adj_idx_j] < adjusted[adj_idx_i]:
            adjusted[adj_idx_j] = adjusted[adj_idx_i]
    return adjusted


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation for paired samples (effect size for Wilcoxon)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    diff = (a - b)[mask]
    diff = diff[np.abs(diff) > 1e-12]
    n = len(diff)
    if n == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diff))
    r_pos = ranks[diff > 0].sum()
    r_neg = ranks[diff < 0].sum()
    total = n * (n + 1) / 2
    return float((r_pos - r_neg) / total)


def shapiro_normal(diff: np.ndarray, alpha: float = 0.05) -> bool:
    """Shapiro-Wilk normality of paired differences (True = looks normal)."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    if len(diff) < 3 or np.allclose(diff, diff[0]):
        return True  # degenerate: t-test handles constants gracefully enough
    try:
        _, p = stats.shapiro(diff)
    except ValueError:
        return True
    return bool(p > alpha)


def compare_methods(
    results: dict[str, dict[str, list[float]]],
    reference: str = "PRIME",
    metrics: Optional[list[str]] = None,
) -> list[dict]:
    """Compare each method against reference across all metrics (Section 14).

    Test choice per comparison: paired t-test when the paired differences
    pass Shapiro-Wilk normality, Wilcoxon signed-rank otherwise.  Effect
    size: Cohen's d (t) or rank-biserial correlation (Wilcoxon).  All raw
    p-values are Holm-Bonferroni corrected jointly.

    results: {method: {metric: [values across seeds, pairing by position]}}
    """
    if metrics is None:
        all_metrics = set()
        for m_results in results.values():
            all_metrics.update(m_results.keys())
        metrics = sorted(all_metrics)

    ref_data = results.get(reference)
    if ref_data is None:
        raise ValueError(f"Reference method '{reference}' not in results")

    rows = []
    for method in results:
        if method == reference:
            continue
        for metric in metrics:
            if metric not in ref_data or metric not in results[method]:
                continue
            a = np.array(ref_data[metric], dtype=float)
            b = np.array(results[method][metric], dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            a, b = a[mask], b[mask]
            if len(a) < 2:
                continue
            diff = a - b
            ci = mean_ci(diff)
            normal = shapiro_normal(diff)
            if normal:
                t_res = paired_t_test(a, b)
                p_val, stat = t_res["p_value"], t_res["statistic"]
                effect = cohens_d(a, b)
                test_name = "paired_t"
            else:
                w_res = wilcoxon_test(a, b)
                p_val, stat = w_res["p_value"], w_res["statistic"]
                effect = rank_biserial(a, b)
                test_name = "wilcoxon"

            rows.append({
                "metric": metric,
                "method_a": reference,
                "method_b": method,
                "n": len(a),
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_diff": ci["mean"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "statistic": stat,
                "raw_p": p_val,
                "effect_size": effect,
                "normality_ok": normal,
                "test_name": test_name,
            })

    if rows:
        adjusted = holm_bonferroni([r["raw_p"] for r in rows])
        for i, r in enumerate(rows):
            r["adjusted_p"] = adjusted[i]

    return rows
