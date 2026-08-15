#!/usr/bin/env python
"""Stage 68 — time-resolved encoding models for concept cells.

For every screening concept cell (MTL, and MFC as comparison) and 100-ms sliding window
(centres −300…+1100 ms, step 50 ms) we compute the per-picture mean rate and fit the
cross-validated ridge encoding model of ``analysis.encoding.cv_score`` (20 PCs, 6 folds)
from four predictors: CLIP ViT-B/32 ln_post cls, ResNet-50 avgpool, AlexNet conv1 and the
category one-hot. Because CV r is negatively biased under the null, the mean r over the
pre-stimulus windows (centres ≤ −100 ms) is used as an empirical baseline for each
cell × predictor and subtracted ("debiased" curve).

Outputs: results/tables/A4_time_resolved_encoding.csv (window × predictor × region),
results/tables/A4_time_resolved_encoding_per_cell.csv, results/figures/A4_time_resolved_encoding.png
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from memoranda import neural
from memoranda.analysis import encoding as E
from memoranda.features import load_space
from memoranda.log import get_logger
from memoranda.paths import FIGURES, MANIFESTS, TABLES

log = get_logger("trenc")
WIN = 0.1
CENTERS = np.round(np.arange(-0.3, 1.1 + 1e-9, 0.05), 3)
BASE_MAX = -0.1  # centres ≤ this are pre-stimulus baseline windows
N_PCS = 20
N_FOLDS = 6
PREDICTORS = {
    "clip ln_post": ("clip_vitb32", "ln_post", "cls"),
    "resnet50 avgpool": ("resnet50", "avgpool", "gap"),
    "alexnet conv1": ("alexnet", "conv1", "gap"),
    "category one-hot": ("baseline", "category", "-"),
}
COLORS = {"clip ln_post": "#C44E52", "resnet50 avgpool": "#08306b", "alexnet conv1": "#9ecae1", "category one-hot": "#55A868"}


def feats_for(model, layer, view, uids, labels_df):
    if model == "baseline":
        return pd.get_dummies(labels_df.loc[uids, "category"]).to_numpy(float)
    return load_space(model, layer, view, uids=uids)[0]


class FoldCache:
    """Pre-computed fold splits + standardisation + PCA for one X, mirroring E.cv_predict.

    The transforms depend only on X and the fold assignment, so they can be shared across
    all windows and cells of a session; only RidgeCV is refit per target. Results are
    identical to ``E.cv_score`` (checked in ``main``).
    """

    def __init__(self, X: np.ndarray, n_folds: int = N_FOLDS, n_pcs: int | None = N_PCS, seed: int = 0):
        n = len(X)
        kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
        self.folds = []
        for tr, te in kf.split(X):
            Xtr, Xte = X[tr], X[te]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            if n_pcs is not None and Xtr.shape[1] > n_pcs:
                pca = PCA(n_components=min(n_pcs, len(tr) - 1), random_state=seed).fit(Xtr)
                Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
            self.folds.append((tr, te, Xtr, Xte))
        self.n = n

    def score(self, y: np.ndarray) -> float:
        if y.std() == 0:
            return np.nan
        pred = np.zeros(self.n)
        for tr, te, Xtr, Xte in self.folds:
            pred[te] = RidgeCV(alphas=E.ALPHAS).fit(Xtr, y[tr]).predict(Xte)
        if pred.std() == 0:
            return 0.0
        return float(sps.pearsonr(pred, y).statistic)


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "screening") & sel.concept_cell & sel.region.isin(["MTL", "MFC"])]
    labels_df = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    rows = []
    checked = False
    t_all = time.time()
    for s, g in cc.groupby("subject"):
        t0 = time.time()
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        onsets = pres.onset.to_numpy()
        lab = pres.image_uid.to_numpy()
        uids = np.unique(lab)
        G = np.stack([(lab == u).astype(float) for u in uids])  # (n_img, n_events)
        G = G / G.sum(1, keepdims=True)
        caches = {k: FoldCache(feats_for(*v, uids, labels_df)) for k, v in PREDICTORS.items()}
        for r in g.itertuples():
            st = spikes[r.unit]
            for c in CENTERS:
                cnt = neural.count_in_windows(st, onsets, (c - WIN / 2, c + WIN / 2)) / WIN
                y = G @ cnt  # per-picture mean rate
                rec = {"subject": s, "unit": r.unit, "area": r.area, "region": r.region, "t": float(c), "mean_rate": float(y.mean())}
                for k, fc in caches.items():
                    rec[f"r_{k}"] = fc.score(y)
                    if not checked and np.isfinite(rec[f"r_{k}"]):
                        ref = E.cv_score(feats_for(*PREDICTORS[k], uids, labels_df), y, n_pcs=N_PCS)
                        assert abs(ref - rec[f"r_{k}"]) < 1e-9, (ref, rec[f"r_{k}"])
                        checked = True
                rows.append(rec)
        log.info(f"sub-{s:02d}: {len(g)} cells ({time.time() - t0:.0f}s)")
    log.info(f"total {time.time() - t_all:.0f}s")

    per = pd.DataFrame(rows)
    keys = list(PREDICTORS)
    # empirical baseline: mean r over pre-stimulus windows for each cell × predictor
    base = per[per.t <= BASE_MAX + 1e-9].groupby(["subject", "unit"])[[f"r_{k}" for k in keys]].mean()
    base.columns = [f"base_{k}" for k in keys]
    per = per.merge(base, left_on=["subject", "unit"], right_index=True, how="left")
    for k in keys:
        per[f"rdeb_{k}"] = per[f"r_{k}"] - per[f"base_{k}"]
    per.to_csv(TABLES / "A4_time_resolved_encoding_per_cell.csv", index=False)

    out = []
    for (reg, t), gg in per.groupby(["region", "t"]):
        for k in keys:
            r = gg[f"r_{k}"].dropna()
            d = gg[f"rdeb_{k}"].dropna()
            out.append(
                {
                    "region": reg,
                    "t": t,
                    "predictor": k,
                    "n_cells": int(len(r)),
                    "r_mean": float(r.mean()),
                    "r_sem": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else np.nan,
                    "r_debiased_mean": float(d.mean()),
                    "r_debiased_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                    "p_debiased_vs0": float(sps.ttest_1samp(d, 0).pvalue) if len(d) > 2 else np.nan,
                }
            )
    summ = pd.DataFrame(out).sort_values(["region", "predictor", "t"]).reset_index(drop=True)
    summ.to_csv(TABLES / "A4_time_resolved_encoding.csv", index=False)

    # ---- figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    peaks = []
    for ax, reg in zip(axes, ["MTL", "MFC"], strict=True):
        sr = summ[summ.region == reg]
        n = sr.n_cells.max()
        for k in keys:
            d = sr[sr.predictor == k].sort_values("t")
            tms = d.t.to_numpy() * 1000
            m, se = d.r_debiased_mean.to_numpy(), d.r_debiased_sem.to_numpy()
            post = d[d.t > BASE_MAX]
            pk = post.loc[post.r_debiased_mean.idxmax()]
            ax.plot(tms, m, color=COLORS[k], lw=1.8, label=f"{k} (peak {pk.t * 1000:.0f} ms, r = {pk.r_debiased_mean:.2f})")
            ax.fill_between(tms, m - se, m + se, color=COLORS[k], alpha=0.2)
            ax.plot(pk.t * 1000, pk.r_debiased_mean, "v", color=COLORS[k], ms=7, mec="k", mew=0.5)
            peaks.append({"region": reg, "predictor": k, "peak_t_ms": pk.t * 1000, "peak_r_debiased": pk.r_debiased_mean, "peak_sem": pk.r_debiased_sem, "peak_r_raw": pk.r_mean, "baseline_r": float(sr[(sr.predictor == k) & (sr.t <= BASE_MAX + 1e-9)].r_mean.mean()), "n_cells": n})
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(-350, BASE_MAX * 1000 + 50, color="grey", alpha=0.08, lw=0)
        ax.set_xlabel("window centre, time from picture onset (ms)")
        ax.set_title(f"{reg} concept cells (n = {n})")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CV encoding r − pre-stimulus baseline (mean ± sem)")
    for ax in axes:
        ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Time-resolved encoding models (100-ms windows, 20 PCs, 6-fold ridge); shaded = baseline windows; ▼ = peak")
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_time_resolved_encoding.png", dpi=150)
    pk = pd.DataFrame(peaks)
    print(pk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
