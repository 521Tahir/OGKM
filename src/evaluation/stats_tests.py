"""
Statistical significance tests for evaluation results (Week 10).

Provides paired t-test and Wilcoxon signed-rank test for comparing
pipeline mode scores on main evaluation metrics.

Paper: Table 11 — Statistical Significance of Mode Comparisons
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from statistics import mean, stdev
from typing import Optional

from ..utils.logging_utils import get_logger

logger = get_logger("evaluation.stats_tests")


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SignificanceResult:
    """Result of a statistical significance test."""
    metric:     str
    mode_a:     str
    mode_b:     str
    test:       str     # "paired_ttest" or "wilcoxon"
    statistic:  float
    p_value:    float
    significant: bool   # p < alpha
    alpha:      float
    mean_a:     float
    mean_b:     float
    delta:      float   # mean_b - mean_a (positive = B better)

    def to_dict(self) -> dict:
        return {
            "metric":      self.metric,
            "mode_a":      self.mode_a,
            "mode_b":      self.mode_b,
            "test":        self.test,
            "statistic":   round(float(self.statistic), 4),
            "p_value":     round(float(self.p_value), 4),
            "significant": bool(self.significant),
            "alpha":       float(self.alpha),
            "mean_a":      round(float(self.mean_a), 4),
            "mean_b":      round(float(self.mean_b), 4),
            "delta":       round(float(self.delta), 4),
        }


# ── Statistical tests ──────────────────────────────────────────────────────────

def paired_ttest(
    scores_a: list[float],
    scores_b: list[float],
    mode_a: str = "A",
    mode_b: str = "B",
    metric: str = "score",
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Paired t-test: H0 = no difference in means between mode_a and mode_b.

    Uses scipy if available, falls back to manual implementation.
    """
    n = min(len(scores_a), len(scores_b))
    if n < 2:
        return SignificanceResult(
            metric=metric, mode_a=mode_a, mode_b=mode_b,
            test="paired_ttest", statistic=0.0, p_value=1.0,
            significant=False, alpha=alpha,
            mean_a=mean(scores_a) if scores_a else 0,
            mean_b=mean(scores_b) if scores_b else 0,
            delta=0.0,
        )

    diffs = [scores_b[i] - scores_a[i] for i in range(n)]
    d_mean = mean(diffs)
    d_std  = stdev(diffs) if len(diffs) > 1 else 0.0

    if d_std == 0:
        t_stat = 0.0
        p_val  = 1.0
    else:
        t_stat = d_mean / (d_std / math.sqrt(n))
        p_val  = _t_pvalue(abs(t_stat), df=n - 1)

    return SignificanceResult(
        metric=metric,
        mode_a=mode_a,
        mode_b=mode_b,
        test="paired_ttest",
        statistic=t_stat,
        p_value=p_val,
        significant=p_val < alpha,
        alpha=alpha,
        mean_a=mean(scores_a[:n]),
        mean_b=mean(scores_b[:n]),
        delta=d_mean,
    )


def wilcoxon_signed_rank(
    scores_a: list[float],
    scores_b: list[float],
    mode_a: str = "A",
    mode_b: str = "B",
    metric: str = "score",
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Wilcoxon signed-rank test: non-parametric alternative to paired t-test.

    Uses scipy if available, falls back to normal approximation for large n.
    """
    n = min(len(scores_a), len(scores_b))
    if n < 2:
        return SignificanceResult(
            metric=metric, mode_a=mode_a, mode_b=mode_b,
            test="wilcoxon", statistic=0.0, p_value=1.0,
            significant=False, alpha=alpha,
            mean_a=mean(scores_a) if scores_a else 0,
            mean_b=mean(scores_b) if scores_b else 0,
            delta=0.0,
        )

    diffs = [scores_b[i] - scores_a[i] for i in range(n)]

    # Try scipy first
    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        nonzero = [(i, d) for i, d in enumerate(diffs) if d != 0]
        if not nonzero:
            w_stat, p_val = 0.0, 1.0
        else:
            d_vals = [d for _, d in nonzero]
            w_stat, p_val = scipy_wilcoxon(d_vals, alternative="two-sided")
        return SignificanceResult(
            metric=metric, mode_a=mode_a, mode_b=mode_b,
            test="wilcoxon", statistic=float(w_stat), p_value=float(p_val),
            significant=bool(float(p_val) < alpha), alpha=alpha,
            mean_a=mean(scores_a[:n]), mean_b=mean(scores_b[:n]),
            delta=mean(diffs),
        )
    except ImportError:
        pass

    # Manual normal approximation (valid for n > 10)
    w_stat, p_val = _wilcoxon_approx(diffs)
    return SignificanceResult(
        metric=metric, mode_a=mode_a, mode_b=mode_b,
        test="wilcoxon", statistic=w_stat, p_value=p_val,
        significant=p_val < alpha, alpha=alpha,
        mean_a=mean(scores_a[:n]), mean_b=mean(scores_b[:n]),
        delta=mean(diffs),
    )


# ── Multi-mode comparison ──────────────────────────────────────────────────────

class SignificanceTester:
    """
    Run paired comparisons between all mode pairs on multiple metrics.

    Parameters
    ----------
    alpha : float
        Significance threshold (default 0.05).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare_all_pairs(
        self,
        mode_scores: dict[str, list[float]],
        metric: str = "score",
        test: str = "both",
    ) -> list[SignificanceResult]:
        """
        Run t-test and/or Wilcoxon for all pairs of modes.

        Parameters
        ----------
        mode_scores : {mode: [score_per_query]}
        test : "ttest" | "wilcoxon" | "both"
        """
        results = []
        modes = list(mode_scores.keys())
        for mode_a, mode_b in combinations(modes, 2):
            a = mode_scores[mode_a]
            b = mode_scores[mode_b]
            if test in ("ttest", "both"):
                results.append(paired_ttest(a, b, mode_a, mode_b, metric, self.alpha))
            if test in ("wilcoxon", "both"):
                results.append(wilcoxon_signed_rank(a, b, mode_a, mode_b, metric, self.alpha))
        return results

    def significance_table(
        self,
        results: list[SignificanceResult],
    ) -> str:
        """Generate a markdown table of significance results."""
        lines = [
            "| Metric | Mode A | Mode B | Test | Δ mean | t/W stat | p-value | Sig? |",
            "|--------|--------|--------|------|-------:|:--------:|:-------:|:----:|",
        ]
        for r in results:
            sig_str = "**✓**" if r.significant else "✗"
            lines.append(
                f"| {r.metric} | {r.mode_a} | {r.mode_b} "
                f"| {r.test[:7]} "
                f"| {r.delta:+.3f} "
                f"| {r.statistic:.3f} "
                f"| {r.p_value:.4f} "
                f"| {sig_str} |"
            )
        return "\n".join(lines)

    def summary(self, results: list[SignificanceResult]) -> str:
        """One-paragraph summary of significance findings."""
        n_total = len(results)
        n_sig   = sum(1 for r in results if r.significant)
        sig_pairs = [
            f"{r.mode_a}→{r.mode_b} ({r.metric}, p={r.p_value:.3f})"
            for r in results if r.significant
        ]
        if not sig_pairs:
            return (
                f"No statistically significant differences found "
                f"(α={self.alpha}, {n_total} tests)."
            )
        return (
            f"{n_sig}/{n_total} tests significant at α={self.alpha}. "
            f"Significant pairs: {'; '.join(sig_pairs[:5])}."
        )


# ── Internal math helpers ──────────────────────────────────────────────────────

def _t_pvalue(t: float, df: int) -> float:
    """Two-tailed p-value from t-distribution. Approximation via regularized incomplete beta."""
    try:
        from scipy.stats import t as scipy_t
        return float(2 * scipy_t.sf(abs(t), df=df))
    except ImportError:
        pass
    # Abramowitz & Stegun approximation
    x = df / (df + t * t)
    p_half = _beta_inc(df / 2, 0.5, x) / 2
    return min(1.0, 2 * p_half)


def _beta_inc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function via continued fraction (simplified)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    # Use log-gamma approximation
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    # Gauss-Legendre quadrature over [0, x] (5 points)
    nodes   = [0.046910, 0.230765, 0.500000, 0.769235, 0.953090]
    weights = [0.118463, 0.239314, 0.284444, 0.239314, 0.118463]
    result  = 0.0
    for t_node, w in zip(nodes, weights):
        t_scaled = t_node * x
        if 0 < t_scaled < 1:
            result += w * math.exp(
                (a - 1) * math.log(t_scaled) + (b - 1) * math.log(1 - t_scaled) - lbeta
            )
    return max(0.0, min(1.0, result * x))


def _wilcoxon_approx(diffs: list[float]) -> tuple[float, float]:
    """Wilcoxon W statistic with normal approximation for n > 10."""
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0

    abs_diffs = sorted(enumerate(abs(d) for d in nonzero), key=lambda x: x[1])
    ranks = {}
    i = 0
    while i < n:
        j = i
        while j < n - 1 and abs_diffs[j + 1][1] == abs_diffs[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[abs_diffs[k][0]] = avg_rank
        i = j + 1

    w_plus  = sum(ranks[i] for i, d in enumerate(nonzero) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(nonzero) if d < 0)
    w_stat  = min(w_plus, w_minus)

    # Normal approximation
    mu    = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return w_stat, 1.0
    z = (w_stat - mu) / sigma
    p_val = 2 * _normal_sf(abs(z))
    return w_stat, p_val


def _normal_sf(z: float) -> float:
    """Survival function of standard normal (upper tail)."""
    return 0.5 * math.erfc(z / math.sqrt(2))
