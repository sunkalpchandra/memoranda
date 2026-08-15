import numpy as np
from scipy.spatial.distance import pdist, squareform

from memoranda.analysis import rsa


def _rdm(n, d, seed):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return squareform(pdist(X, "correlation"))


def test_compare_identical_is_one():
    D = _rdm(20, 5, 0)
    assert abs(rsa.compare_rdms(D, D) - 1) < 1e-9


def test_partial_removes_shared_component():
    C = _rdm(25, 6, 2)
    A = C + 0.3 * _rdm(25, 6, 3)
    B = C + 0.3 * _rdm(25, 6, 4)
    full = rsa.compare_rdms(A, B)
    part = rsa.partial_compare(A, B, [C])
    assert full > 0.5
    assert part < full


def test_rdm_regression_recovers_predictor():
    P1, P2 = _rdm(30, 4, 5), _rdm(30, 4, 6)
    T = P1 + 0.05 * _rdm(30, 4, 7)
    out = rsa.rdm_regression(T, {"p1": P1, "p2": P2})
    assert out["beta_p1"] > 0.8
    assert abs(out["beta_p2"]) < 0.2
    assert out["r2"] > 0.8


def test_perm_p_significant_for_related():
    D = _rdm(30, 5, 8)
    rho, p = rsa.perm_p(D, D + 0.01 * _rdm(30, 5, 9), n_perm=200)
    assert rho > 0.9 and p < 0.01
