"""Image handling: orientation fix, hashing, saving, loading.

The NWB files store every template as a (400, 300, 3) uint8 array that is the
original 300×400 landscape picture rotated 90° counter-clockwise (a MATLAB
row/column artefact). ``fix_orientation`` undoes that.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import IMAGES


def fix_orientation(arr: np.ndarray) -> np.ndarray:
    """Rotate the stored array 90° clockwise so the picture is upright.

    Verified visually on several templates (people upright, text readable).
    """
    return np.ascontiguousarray(np.rot90(arr, k=-1))


def sha1_of_array(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dhash(arr: np.ndarray, size: int = 8) -> str:
    """Difference hash (perceptual). Robust to resampling / minor recompression."""
    im = Image.fromarray(arr).convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b, strict=True))


def image_path(subject: int, session: int, name: str, root: Path | None = None) -> Path:
    root = root or IMAGES
    return root / f"sub-{subject:02d}" / f"ses-{session}" / f"{name}.png"


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, format="PNG", optimize=True)


def load_rgb(path: Path | str) -> Image.Image:
    return Image.open(path).convert("RGB")


def is_blank(arr: np.ndarray, tol: float = 1.0) -> bool:
    """True for the all-white/black null placeholder (``image_999``)."""
    return float(arr.std()) < tol
