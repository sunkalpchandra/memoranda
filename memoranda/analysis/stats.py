"""Small statistics helpers used across analyses (all non-parametric / resampling)."""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def auc_effect(a: np.ndarray, b: np.ndarray) -> float:
    """Probability that a random draw from ``a`` exceeds one from ``b`` (Mann–Whitney AUC)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u = sps.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(u / (len(a) * len(b)))


def mannwhitney(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    r = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(r.statistic), float(r.pvalue)


def cliffs_delta(a, b) -> float:
    return 2 * auc_effect(a, b) - 1


def fisher_enrichment(in_group: np.ndarray, is_hit: np.ndarray) -> tuple[float, float, int, int, int, int]:
    """Odds ratio & p for hits being enriched in group. Returns (OR, p, a, b, c, d)."""
    a = int((in_group & is_hit).sum())
    b = int((in_group & ~is_hit).sum())
    c = int((~in_group & is_hit).sum())
    d = int((~in_group & ~is_hit).sum())
    orat, p = sps.fisher_exact([[a, b], [c, d]])
    return float(orat), float(p), a, b, c, d


def fdr_bh(p: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg. Returns (reject, q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_out = np.empty(n)
    q_out[order] = np.clip(q, 0, 1)
    return q_out <= alpha, q_out


def bootstrap_ci(x: np.ndarray, fn=np.mean, n_boot: int = 2000, ci: float = 95, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    boots = np.array([fn(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(fn(x)), float(lo), float(hi)


def perm_test_diff(a: np.ndarray, b: np.ndarray, n_perm: int = 5000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    """Two-sided permutation test of stat(a) - stat(b)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    obs = stat(a) - stat(b)
    pooled = np.concatenate([a, b])
    n = len(a)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(stat(pooled[:n]) - stat(pooled[n:])) >= abs(obs):
            cnt += 1
    return float(obs), (cnt + 1) / (n_perm + 1)


def spearman(a, b) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan, np.nan
    r = sps.spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue)
