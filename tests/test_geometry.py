import numpy as np

from memoranda.analysis import geometry as G


def test_cosine_dist_diag_zero():
    X = np.random.default_rng(0).normal(size=(10, 5))
    D = G.cosine_dist_matrix(X)
    assert np.allclose(np.diag(D), 0)
    assert np.allclose(D, D.T)


def test_outlier_has_larger_nn_and_centroid_distance():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(20, 8)) + 5, rng.normal(size=(1, 8)) - 5])
    D = G.cosine_dist_matrix(X)
    nn = G.nn_distance(D)
    cd = G.centroid_distance(X)
    assert nn[-1] > nn[:-1].max()
    assert cd[-1] > cd[:-1].max()


def test_category_typicality_leave_one_out():
    X = np.array([[1, 0], [1, 0.1], [0, 1], [0.1, 1.0]])
    lab = np.array(["a", "a", "b", "b"])
    t = G.category_typicality(X, lab)
    assert np.all(t > 0.9)
    single = G.category_typicality(X, np.array(["a", "b", "c", "d"]))
    assert np.all(np.isnan(single))
