"""Representational similarity analysis between neural populations and DNN layers.

Neural RDMs are built per screening session from the (units × images) matrix
of mean firing rates in the 200–1000 ms window. Model RDMs come from the
feature store restricted to the same images. Comparison metric is Spearman ρ
over the upper triangle; significance by permuting image labels; reliability
by split-half over the six repetitions of each image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial.distance import pdist, squareform

from .. import neural
from ..features import load_features
from ..paths import FEATURES, MANIFESTS


# ------------------------------------------------------------------- neural RDMs
def session_response_matrix(subject: int, session: int = 1, region: str | None = None, split: int | None = None, seed: int = 0):
    """Return (rates: images × units, image_uids, unit_ids).

    ``split``: None → all reps; 0/1 → random half of the reps per image (seeded).
    ``region``: 'MTL' | 'MFC' | None (all).
    """
    spikes = neural.load_spikes(subject, session)
    pres = neural.load_presentations(subject, session)
    pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
    if session == 2:
        pres = pres[pres.role.isin(["enc1", "enc2", "enc3"])].reset_index(drop=True)
    units_tab = neural.load_units_table(subject, session)
    if region is not None:
        keep_units = set(units_tab.loc[units_tab.region == region, "unit"])
        spikes = {u: st for u, st in spikes.items() if u in keep_units}
    if not spikes:
        return None, None, None
    R, units = neural.rate_matrix(spikes, pres["onset"].to_numpy(), neural.RESP_WIN)
    labels = pres["image_uid"].to_numpy()
    if split is not None:
        rng = np.random.default_rng(seed)
        mask = np.zeros(len(labels), bool)
        for u in np.unique(labels):
            idx = np.where(labels == u)[0]
            rng.shuffle(idx)
            half = idx[: len(idx) // 2] if split == 0 else idx[len(idx) // 2 :]
            mask[half] = True
        R, labels = R[mask], labels[mask]
    uids = np.unique(labels)
    M = np.stack([R[labels == u].mean(0) for u in uids])
    return M, uids, np.array(units)


def neural_rdm(M: np.ndarray, metric: str = "correlation", zscore_units: bool = True) -> np.ndarray:
    X = M.copy()
    if zscore_units:
        sd = X.std(0, ddof=1)
        sd[sd == 0] = 1.0
        X = (X - X.mean(0)) / sd
    if metric == "correlation" and X.shape[1] < 2:
        metric = "euclidean"
    D = squareform(pdist(X, metric=metric))
    return np.nan_to_num(D, nan=np.nanmax(D) if np.isfinite(D).any() else 1.0)


# -------------------------------------------------------------------- model RDMs
def _clip_embed(uids):
    z = np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True)
    pos = {u: i for i, u in enumerate(z["image_uid"])}
    return z["embed"][[pos[u] for u in uids]]


def model_rdm(model: str, layer: str, view: str, uids: np.ndarray, metric: str = "correlation") -> np.ndarray:
    if model == "clip_vitb32" and layer == "embed":
        X = _clip_embed(uids)
    else:
        X, stored = load_features(model, layer, view)
        pos = {u: i for i, u in enumerate(stored)}
        X = X[[pos[u] for u in uids]]
    X = X.astype(np.float64)
    X = X[:, X.std(0) > 0] if X.shape[1] > 1 else X
    return squareform(pdist(X, metric=metric))


def category_rdm(uids: np.ndarray, labels_df: pd.DataFrame | None = None) -> np.ndarray:
    labels_df = labels_df if labels_df is not None else pd.read_csv(MANIFESTS / "image_labels.csv")
    cat = labels_df.set_index("image_uid").loc[uids, "category"].to_numpy()
    return (cat[:, None] != cat[None, :]).astype(float)


def lowlevel_rdm(uids: np.ndarray, stats_df: pd.DataFrame | None = None) -> np.ndarray:
    stats_df = stats_df if stats_df is not None else pd.read_csv(MANIFESTS / "image_stats.csv")
    X = stats_df.set_index("image_uid").loc[uids].drop(columns=["aspect"]).to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return squareform(pdist(X, metric="euclidean"))


# -------------------------------------------------------------------- comparison
def upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def compare_rdms(a: np.ndarray, b: np.ndarray) -> float:
    x, y = upper(a), upper(b)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan
    return float(sps.spearmanr(x[m], y[m]).statistic)


def perm_p(a: np.ndarray, b: np.ndarray, n_perm: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Permute item labels of ``b`` (rows+cols jointly). One-sided p for ρ_obs > null."""
    rng = np.random.default_rng(seed)
    obs = compare_rdms(a, b)
    n = a.shape[0]
    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        if compare_rdms(a, b[np.ix_(perm, perm)]) >= obs:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def split_half_reliability(subject: int, session: int = 1, region: str | None = None, n_splits: int = 10) -> float:
    """Spearman–Brown corrected split-half reliability of the neural RDM."""
    rs = []
    for s in range(n_splits):
        A, uids, _ = session_response_matrix(subject, session, region, split=0, seed=s)
        B, _, _ = session_response_matrix(subject, session, region, split=1, seed=s)
        if A is None or A.shape[1] == 0:
            return np.nan
        r = compare_rdms(neural_rdm(A), neural_rdm(B))
        rs.append(r)
    r = float(np.nanmean(rs))
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


@dataclass
class RSAResult:
    subject: int
    region: str
    n_units: int
    n_images: int
    model: str
    layer: str
    view: str
    rho: float
    p: float
    reliability: float

    @property
    def rho_ceiling_normalised(self) -> float:
        return self.rho / np.sqrt(self.reliability) if self.reliability and self.reliability > 0 else np.nan


# ------------------------------------------------------------- partial / regression
def _rank(x: np.ndarray) -> np.ndarray:
    return sps.rankdata(x)


def partial_compare(a: np.ndarray, b: np.ndarray, controls: list[np.ndarray]) -> float:
    """Partial Spearman ρ between RDMs a and b controlling for control RDMs.

    Ranks of the upper triangles are residualised on the ranked controls
    (with intercept), then Pearson-correlated.
    """
    x, y = _rank(upper(a)), _rank(upper(b))
    C = np.column_stack([np.ones(len(x))] + [_rank(upper(c)) for c in controls])
    bx, *_ = np.linalg.lstsq(C, x, rcond=None)
    by, *_ = np.linalg.lstsq(C, y, rcond=None)
    rx, ry = x - C @ bx, y - C @ by
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def rdm_regression(target: np.ndarray, predictors: dict[str, np.ndarray]) -> dict[str, float]:
    """Rank-based OLS of the neural RDM on several model RDMs; returns standardised betas + R²."""
    y = _rank(upper(target))
    y = (y - y.mean()) / y.std()
    names = list(predictors)
    X = np.column_stack([_rank(upper(predictors[n])) for n in names]).astype(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - resid.var() / y.var()
    out = {f"beta_{n}": float(b) for n, b in zip(names, beta[1:])}
    out["r2"] = float(r2)
    return out
