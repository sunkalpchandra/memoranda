"""On-disk feature store: one .npz per (model), keyed by image_uid order.

Layout: ``data/features/<model>.npz`` with arrays named ``<layer>__<view>`` and an
``image_uid`` string array giving the row order. Loading helpers return
(features, uids) so downstream code never has to worry about alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dedup import unique_images
from .paths import FEATURES, MANIFESTS, ROOT


def feature_path(model: str, root: Path | None = None) -> Path:
    return (root or FEATURES) / f"{model}.npz"


def save_features(model: str, uids: list[str], feats: dict[tuple[str, str], np.ndarray], root: Path | None = None) -> Path:
    p = feature_path(model, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"{layer}__{view}": v.astype(np.float16) for (layer, view), v in feats.items()}
    np.savez_compressed(p, image_uid=np.array(uids), **arrays)
    return p


def list_layers(model: str, root: Path | None = None) -> list[tuple[str, str]]:
    with np.load(feature_path(model, root)) as z:
        return [tuple(k.split("__")) for k in z.files if k != "image_uid"]


def load_features(model: str, layer: str, view: str = "gap", root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    with np.load(feature_path(model, root)) as z:
        return z[f"{layer}__{view}"].astype(np.float32), z["image_uid"]


def load_aligned(model: str, layer: str, uids: list[str] | np.ndarray, view: str = "gap") -> np.ndarray:
    """Features for an arbitrary list of uids (rows in that order)."""
    X, stored = load_features(model, layer, view)
    pos = {u: i for i, u in enumerate(stored)}
    idx = np.array([pos[u] for u in uids])
    return X[idx]


def unique_image_table() -> pd.DataFrame:
    """The canonical list of unique images with resolvable absolute paths."""
    df = pd.read_csv(MANIFESTS / "images.csv")
    u = unique_images(df).copy()
    u["abs_path"] = [str(ROOT / "data" / p) for p in u["path"]]
    return u
