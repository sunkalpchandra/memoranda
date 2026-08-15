"""Representational-geometry descriptors of individual images.

Given an (n, D) feature matrix these functions describe *where* each image sits
relative to the others: how far from the centre, how isolated, how typical of
its category. All distances are cosine unless stated.
"""

from __future__ import annotations

import numpy as np


def l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def cosine_dist_matrix(X: np.ndarray) -> np.ndarray:
    Z = l2norm(X)
    return np.clip(1.0 - Z @ Z.T, 0, 2)


def centroid_distance(X: np.ndarray) -> np.ndarray:
    """Cosine distance of each row to the mean vector."""
    Z = l2norm(X)
    c = Z.mean(0)
    c = c / (np.linalg.norm(c) + 1e-12)
    return 1.0 - Z @ c


def nn_distance(D: np.ndarray, k: int = 1) -> np.ndarray:
    """Mean distance to the k nearest other items."""
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].mean(1)


def mean_distance(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    return (D.sum(1) - np.diag(D)) / max(n - 1, 1)


def local_density(D: np.ndarray, k: int = 5) -> np.ndarray:
    """1 − mean distance to k nearest neighbours (higher = denser neighbourhood)."""
    return 1.0 - nn_distance(D, k)


def category_typicality(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Cosine similarity of each item to the centroid of its own category (leave-one-out)."""
    Z = l2norm(X)
    out = np.full(len(Z), np.nan)
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        S = Z[idx].sum(0)
        for i in idx:
            cen = S - Z[i]
            cen = cen / (np.linalg.norm(cen) + 1e-12)
            out[i] = Z[i] @ cen
    return out


def describe(X: np.ndarray, labels: np.ndarray | None = None, prefix: str = "") -> dict[str, np.ndarray]:
    D = cosine_dist_matrix(X)
    out = {
        f"{prefix}centroid_dist": centroid_distance(X),
        f"{prefix}nn1_dist": nn_distance(D, 1),
        f"{prefix}nn5_dist": nn_distance(D, 5),
        f"{prefix}mean_dist": mean_distance(D),
    }
    if labels is not None:
        out[f"{prefix}cat_typicality"] = category_typicality(X, labels)
    return out
