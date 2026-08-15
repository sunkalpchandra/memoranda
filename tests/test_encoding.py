import numpy as np

from memoranda.analysis import encoding as E


def test_cv_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 50))
    w = rng.normal(size=50)
    y = X @ w + rng.normal(scale=0.5, size=60)
    r = E.cv_score(X, y, n_pcs=None)
    assert r > 0.6


def test_cv_null_near_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 50))
    y = rng.normal(size=60)
    r = E.cv_score(X, y, n_pcs=20)
    assert abs(r) < 0.5


def test_tuning_reliability_high_for_reliable_unit():
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(30), 6)
    tuning = rng.normal(size=30) * 5
    R = (tuning[labels] + rng.normal(size=len(labels)))[:, None]
    rel = E.tuning_reliability(R, labels)
    assert rel[0] > 0.8
