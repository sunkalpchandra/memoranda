import numpy as np

from memoranda.analysis import stats as S


def test_auc_effect_extremes():
    assert S.auc_effect([1, 2, 3], [4, 5, 6]) == 0.0
    assert S.auc_effect([4, 5, 6], [1, 2, 3]) == 1.0
    assert abs(S.auc_effect([1, 2, 3], [1, 2, 3]) - 0.5) < 1e-9


def test_fisher_enrichment_counts():
    g = np.array([1, 1, 1, 0, 0, 0], bool)
    h = np.array([1, 1, 0, 0, 0, 0], bool)
    orat, p, a, b, c, d = S.fisher_enrichment(g, h)
    assert (a, b, c, d) == (2, 1, 0, 3)
    assert orat == np.inf


def test_fdr_monotone():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    rej, q = S.fdr_bh(p)
    assert q[0] <= q[1] <= q[2] <= q[3]
    assert rej[0] and not rej[3]


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(5, 1, 200)
    m, lo, hi = S.bootstrap_ci(x)
    assert lo < m < hi
    assert lo < 5.3 and hi > 4.7


def test_perm_test_detects_shift():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 60), rng.normal(1.5, 1, 60)
    d, p = S.perm_test_diff(a, b, n_perm=500)
    assert d < 0 and p < 0.05
