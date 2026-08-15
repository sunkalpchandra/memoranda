import numpy as np

from memoranda import neural


def test_count_in_windows():
    spikes = np.array([0.1, 0.5, 0.9, 1.3, 2.0])
    onsets = np.array([0.0, 1.0])
    c = neural.count_in_windows(spikes, onsets, (0.2, 1.0))
    assert c.tolist() == [2.0, 1.0]  # [0.2,1.0): 0.5,0.9 ; [1.2,2.0): 1.3


def test_permuted_anova_detects_selective_unit():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(20), 6)
    R = rng.poisson(2.0, size=(len(labels), 2)).astype(float)
    R[labels == 7, 0] += 8  # unit 0 selective for image 7, unit 1 not
    F, p = neural.permuted_anova(R, labels, n_perm=300)
    assert p[0] < 0.01
    assert p[1] > 0.05
    pref, t, pp = neural.permuted_max_vs_rest(R, labels, n_perm=300)
    assert pref[0] == 7 and pp[0] < 0.01


def test_depth_of_selectivity_and_sparseness():
    one_hot = np.zeros(10)
    one_hot[3] = 5.0
    assert abs(neural.depth_of_selectivity(one_hot) - 1.0) < 1e-9
    assert neural.sparseness(one_hot) < 0.15
    flat = np.ones(10)
    assert abs(neural.depth_of_selectivity(flat)) < 1e-9
    assert abs(neural.sparseness(flat) - 1.0) < 1e-9
