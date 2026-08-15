import numpy as np

from memoranda import features as Fs


def test_save_and_load_roundtrip(tmp_path):
    uids = ["img_0001", "img_0002", "img_0003"]
    feats = {("layer1", "gap"): np.arange(6, dtype=np.float32).reshape(3, 2), ("layer1", "rp"): np.ones((3, 4), np.float32)}
    p = Fs.save_features("toy", uids, feats, root=tmp_path)
    assert p.exists()
    layers = Fs.list_layers("toy", root=tmp_path)
    assert set(layers) == {("layer1", "gap"), ("layer1", "rp")}
    X, stored = Fs.load_features("toy", "layer1", "gap", root=tmp_path)
    assert X.shape == (3, 2) and list(stored) == uids
    assert np.allclose(X, feats[("layer1", "gap")])


def test_taxonomy_loads_and_is_well_formed():
    from memoranda.models.zeroshot import load_taxonomy

    tax = load_taxonomy()
    assert set(tax) >= {"category", "attributes", "fine"}
    assert all(len(v) >= 2 for v in tax["category"].values())
    assert all(len(v) == 2 for v in tax["attributes"].values())
    assert len(tax["fine"]) > 30


def test_load_space_orders_and_normalises(tmp_path, monkeypatch):
    from memoranda import features as Fs

    uids = ["img_0001", "img_0002", "img_0003"]
    X = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]], np.float32)
    Fs.save_features("toy2", uids, {("l", "gap"): X}, root=tmp_path)
    monkeypatch.setattr(Fs, "FEATURES", tmp_path)
    Y, order = Fs.load_space("toy2", "l", "gap", uids=["img_0003", "img_0001"], normalize=True)
    assert list(order) == ["img_0003", "img_0001"]
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0)
    assert np.allclose(Y[1], [1.0, 0.0])
