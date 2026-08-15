#!/usr/bin/env python
"""Stage 51 — label-shuffle null for encoding models (concept cells, curated layers).

For each MTL/MFC concept cell and each curated predictor we recompute the CV r on
``--n-shuffle`` image-label permutations, giving a per-cell null distribution that
shares the negative bias of small-sample CV. Reports r_obs − mean(null), the per-cell
empirical p, and the fraction of cells significant at p < 0.05.
Output: results/tables/A4_encoding_null.csv (per cell × predictor), A4_encoding_null_summary.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from memoranda import neural
from memoranda.analysis import encoding as E
from memoranda.features import load_features
from memoranda.log import get_logger
from memoranda.paths import FEATURES, MANIFESTS, TABLES

log = get_logger("encnull")
PREDICTORS = [("alexnet", "fc6", "gap"), ("vgg16", "fc7", "gap"), ("resnet50", "avgpool", "gap"), ("clip_vitb32", "ln_post", "cls"), ("dinov2_small", "block8", "cls"), ("baseline", "category", "-")]


def feats_for(model, layer, view, uids, labels_df):
    if model == "baseline":
        return pd.get_dummies(labels_df.loc[uids, "category"]).to_numpy(float)
    if model == "clip_vitb32" and layer == "embed":
        z = np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True)
        X, stored = z["embed"], z["image_uid"]
    else:
        X, stored = load_features(model, layer, view)
    pos = {u: i for i, u in enumerate(stored)}
    return X[[pos[u] for u in uids]].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shuffle", type=int, default=100)
    args = ap.parse_args()
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "screening") & sel.concept_cell]
    labels_df = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    rng = np.random.default_rng(0)
    rows = []
    for s, g in cc.groupby("subject"):
        t0 = time.time()
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        R, units = neural.rate_matrix(spikes, pres.onset.to_numpy(), neural.RESP_WIN)
        lab = pres.image_uid.to_numpy()
        uids = np.unique(lab)
        M = np.stack([R[lab == u].mean(0) for u in uids])
        upos = {u: j for j, u in enumerate(units)}
        Xs = {k: feats_for(*k, uids, labels_df) for k in PREDICTORS}
        for r in g.itertuples():
            y = M[:, upos[r.unit]]
            for k, X in Xs.items():
                obs = E.cv_score(X, y, n_pcs=20)
                null = np.array([E.cv_score(X, y[rng.permutation(len(y))], n_pcs=20) for _ in range(args.n_shuffle)])
                rows.append({"subject": s, "unit": r.unit, "area": r.area, "region": r.region, "model": k[0], "layer": k[1], "r_obs": obs, "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)), "r_debiased": obs - float(np.nanmean(null)), "p_cell": float((np.sum(null >= obs) + 1) / (len(null) + 1))})
        log.info(f"sub-{s:02d} {len(g)} concept cells ({time.time()-t0:.0f}s)")
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A4_encoding_null.csv", index=False)
    summ = per.groupby(["region", "model", "layer"]).agg(n=("r_obs", "size"), r_obs=("r_obs", "mean"), null_mean=("null_mean", "mean"), r_debiased=("r_debiased", "mean"), r_debiased_sem=("r_debiased", lambda x: x.std() / np.sqrt(len(x))), frac_sig=("p_cell", lambda p: (p < 0.05).mean())).reset_index()
    summ.to_csv(TABLES / "A4_encoding_null_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
