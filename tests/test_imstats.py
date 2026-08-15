import numpy as np

from memoranda import imstats


def test_flat_image_stats():
    arr = np.full((64, 96, 3), 128, dtype=np.uint8)
    s = imstats.compute_all(arr)
    assert abs(s["luminance"] - 128 / 255) < 1e-3
    assert s["rms_contrast"] < 1e-6
    assert s["colorfulness"] < 1e-6
    assert s["edge_density"] == 0.0
    assert abs(s["aspect"] - 1.5) < 1e-9


def test_noise_has_high_entropy_and_flat_spectrum():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    s = imstats.compute_all(arr)
    assert s["gray_entropy"] > 5.0
    assert s["spectral_slope"] > -1.0  # white noise ≈ 0, natural images ≈ -2


def test_all_keys_present():
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, 16:] = 255
    s = imstats.compute_all(arr)
    assert set(s) == set(imstats.STAT_NAMES)
    assert s["edge_density"] > 0
