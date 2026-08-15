import numpy as np
import pandas as pd

from memoranda.analysis import behavior as B


def test_per_subject_slopes_and_group_test():
    rng = np.random.default_rng(0)
    rows = []
    for s in range(8):
        x = rng.normal(size=40)
        loads = rng.integers(1, 4, size=40)
        y = 0.3 * x + 0.1 * loads + rng.normal(scale=0.5, size=40)
        rows += [{"subject": s, "x": xi, "y": yi, "loads": li} for xi, yi, li in zip(x, y, loads)]
    t = pd.DataFrame(rows)
    sl = B.per_subject_slopes(t, "x", "y")
    assert len(sl) == 8
    g = B.group_test(sl)
    assert g["mean_slope"] > 0 and g["p"] < 0.01


def test_add_similarity_uses_encoded_set(monkeypatch):
    # fake embedding: three orthogonal pictures + one identical to the first
    X = np.eye(3)
    X = np.vstack([X, X[0]])
    pos = {"a": 0, "b": 1, "c": 2, "a2": 3}
    monkeypatch.setattr(B, "_embed", lambda *args, **kw: (X, pos))
    t = pd.DataFrame({"enc1": ["a", "a"], "enc2": ["b", None], "enc3": [None, None], "probe": ["c", "a2"]})
    out = B.add_similarity(t, "m", "l", "v", "toy")
    assert np.isclose(out.loc[0, "toy_sim_probe_max"], 0.0)
    assert np.isclose(out.loc[1, "toy_sim_probe_max"], 1.0)
    assert np.isclose(out.loc[0, "toy_sim_within"], 0.0)
    assert np.isnan(out.loc[1, "toy_sim_within"])
