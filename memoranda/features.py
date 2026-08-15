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


def load_space(
    model: str,
    layer: str,
    view: str = "gap",
    uids: list[str] | np.ndarray | None = None,
    center: bool = False,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Unified loader used by the analysis scripts.

    Handles the special-cased CLIP image embedding (``clip_vitb32``/``embed``) stored by
    script 12 as well as regular layer arrays. Optionally centres (subtract the mean over
    all stored images) and L2-normalises rows so that ``X @ X.T`` is a cosine similarity.
    Returns ``(X, uids)`` in the requested order (or stored order when ``uids`` is None).
    """
    if model == "clip_vitb32" and layer == "embed":
        with np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True) as z:
            X, stored = z["embed"].astype(np.float32), z["image_uid"].astype(str)
    else:
        X, stored = load_features(model, layer, view)
    X = X.astype(np.float64)
    if center:
        X = X - X.mean(0)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    if uids is None:
        return X, stored
    pos = {u: i for i, u in enumerate(stored)}
    idx = np.array([pos[u] for u in uids])
    return X[idx], np.asarray(uids)
