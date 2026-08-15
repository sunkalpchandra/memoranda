import numpy as np

from memoranda import images as I


def test_fix_orientation_shape():
    arr = np.zeros((400, 300, 3), dtype=np.uint8)
    out = I.fix_orientation(arr)
    assert out.shape == (300, 400, 3)
    assert out.flags["C_CONTIGUOUS"]


def test_fix_orientation_is_clockwise():
    # mark the top-left corner of the *stored* array; after a clockwise
    # rotation it must land in the top-right corner of the upright picture
    arr = np.zeros((4, 3, 3), dtype=np.uint8)
    arr[0, 0] = 255
    out = I.fix_orientation(arr)
    assert out[0, -1].sum() == 255 * 3
    assert out[0, 0].sum() == 0


def test_dhash_stable_and_length():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    h1 = I.dhash(arr)
    h2 = I.dhash(arr)
    assert h1 == h2
    assert len(h1) == 64
    assert set(h1) <= {"0", "1"}


def test_hamming():
    assert I.hamming("0000", "0000") == 0
    assert I.hamming("0000", "0101") == 2


def test_is_blank():
    assert I.is_blank(np.full((10, 10, 3), 255, dtype=np.uint8))
    rng = np.random.default_rng(1)
    assert not I.is_blank(rng.integers(0, 255, size=(10, 10, 3), dtype=np.uint8))
