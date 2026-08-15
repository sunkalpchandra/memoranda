"""Cross-validated encoding models: DNN features → single-unit tuning curves.

Each unit's target is its vector of mean firing rates over the images shown in
its session (54–63 in screening). Predictors are standardised layer features
reduced by PCA inside each training fold; the ridge penalty is chosen by
efficient leave-one-out inside the training fold (``RidgeCV``). Performance
is Pearson r between held-out predictions and observed rates, and a per-unit
noise ceiling comes from split-half reliability of the tuning curve.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ALPHAS = np.logspace(-2, 4, 13)


def cv_predict(X: np.ndarray, y: np.ndarray, n_folds: int = 6, n_pcs: int | None = 20, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of y from X."""
    n = len(y)
    pred = np.zeros(n)
    kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if n_pcs is not None and Xtr.shape[1] > n_pcs:
            pca = PCA(n_components=min(n_pcs, len(tr) - 1), random_state=seed).fit(Xtr)
            Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
        model = RidgeCV(alphas=ALPHAS).fit(Xtr, y[tr])
        pred[te] = model.predict(Xte)
    return pred


def cv_score(X: np.ndarray, y: np.ndarray, **kw) -> float:
    if y.std() == 0:
        return np.nan
    p = cv_predict(X, y, **kw)
    if p.std() == 0:
        return 0.0
    return float(sps.pearsonr(p, y).statistic)


def cv_scores_multi(X: np.ndarray, Y: np.ndarray, **kw) -> np.ndarray:
    """Score every column of Y (images × units) against the same X."""
    return np.array([cv_score(X, Y[:, j], **kw) for j in range(Y.shape[1])])


def tuning_reliability(R: np.ndarray, labels: np.ndarray, n_splits: int = 10, seed: int = 0) -> np.ndarray:
    """Spearman–Brown split-half reliability of each unit's tuning curve.

    R: events × units firing rates, labels: image id per event.
    """
    rng = np.random.default_rng(seed)
    uids = np.unique(labels)
    out = []
    for _ in range(n_splits):
        A = np.zeros((len(uids), R.shape[1]))
        B = np.zeros_like(A)
        for i, u in enumerate(uids):
            idx = np.where(labels == u)[0]
            rng.shuffle(idx)
            h = len(idx) // 2
            A[i] = R[idx[:h]].mean(0)
            B[i] = R[idx[h:]].mean(0)
        r = np.array([sps.pearsonr(A[:, j], B[:, j]).statistic if A[:, j].std() > 0 and B[:, j].std() > 0 else np.nan for j in range(R.shape[1])])
        out.append(r)
    r = np.nanmean(np.array(out), 0)
    with np.errstate(invalid="ignore"):
        sb = 2 * r / (1 + r)
    return sb
