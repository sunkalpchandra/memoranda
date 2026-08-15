#!/usr/bin/env python
"""Stage 41 — encoding models for every screening unit × model layer.

For each unit (target = per-image mean rate) and each (model, layer, view) we
compute 6-fold CV Pearson r using ridge on in-fold PCA features. Baselines:
category one-hot, low-level statistics. Noise ceiling per unit: split-half
tuning reliability (Spearman–Brown).

Outputs:
  results/tables/A4_encoding_per_unit.csv    (unit × layer) rows
  results/tables/A4_encoding_summary.csv     mean r by region / concept-cell status / layer
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from memoranda import neural
from memoranda.analysis import encoding as E
from memoranda.dandi import list_assets
from memoranda.features import list_layers, load_features
from memoranda.log import get_logger
from memoranda.models.registry import DEFAULT_MODELS
from memoranda.paths import FEATURES, MANIFESTS, TABLES

log = get_logger("encoding")
VIEWS = ("gap", "cls")


def feats_for(model, layer, view, uids):
    if model == "clip_vitb32" and layer == "embed":
        z = np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True)
        X, stored = z["embed"], z["image_uid"]
    else:
        X, stored = load_features(model, layer, view)
    pos = {u: i for i, u in enumerate(stored)}
    return X[[pos[u] for u in uids]].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--n-pcs", type=int, default=20)
    ap.add_argument("--min-rate", type=float, default=0.2, help="skip units with mean rate below (Hz)")
    args = ap.parse_args()

    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    sel = sel[sel.task == "screening"].set_index(["subject", "unit"])
    labels_df = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv").set_index("image_uid")
    layer_list = [(m, l, v) for m in args.models for (l, v) in list_layers(m) if v in VIEWS] + [("clip_vitb32", "embed", "gap")]

    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        t0 = time.time()
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        R, units = neural.rate_matrix(spikes, pres.onset.to_numpy(), neural.RESP_WIN)
        lab = pres.image_uid.to_numpy()
        uids = np.unique(lab)
        M = np.stack([R[lab == u].mean(0) for u in uids])  # images × units
        rel = E.tuning_reliability(R, lab)
        keep = M.mean(0) >= args.min_rate
        # baselines
        cat = pd.get_dummies(labels_df.loc[uids, "category"]).to_numpy(float)
        low = stats_df.loc[uids].drop(columns=["aspect"]).to_numpy(float)
        Xs = {("baseline", "category", "-"): cat, ("baseline", "lowlevel", "-"): low}
        for key in layer_list:
            Xs[key] = feats_for(*key, uids)
        for (m, l, v), X in Xs.items():
            r = np.full(len(units), np.nan)
            r[keep] = E.cv_scores_multi(X, M[:, keep], n_pcs=args.n_pcs)
            for j, u in enumerate(units):
                info = sel.loc[(s, u)]
                rows.append(
                    {
                        "subject": s,
                        "unit": u,
                        "area": info.area,
                        "region": info.region,
                        "concept_cell": bool(info.concept_cell),
                        "mean_rate": float(M[:, j].mean()),
                        "reliability": float(rel[j]),
                        "model": m,
                        "layer": l,
                        "view": v,
                        "r": float(r[j]),
                    }
                )
        log.info(f"sub-{s:02d} {len(units)} units × {len(Xs)} predictors ({time.time()-t0:.0f}s)")

    per = pd.DataFrame(rows)
    per["r_norm"] = per.r / np.sqrt(per.reliability.clip(lower=0.05))
    per.to_csv(TABLES / "A4_encoding_per_unit.csv", index=False)
    g = per.dropna(subset=["r"]).groupby(["region", "concept_cell", "model", "layer", "view"])
    summ = g.agg(n_units=("r", "size"), r_mean=("r", "mean"), r_median=("r", "median"), r_sem=("r", lambda x: x.std() / np.sqrt(len(x))), r_norm_mean=("r_norm", "mean"), frac_pos=("r", lambda x: (x > 0).mean())).reset_index()
    summ.to_csv(TABLES / "A4_encoding_summary.csv", index=False)
    print(summ[summ.concept_cell & (summ.region == "MTL")].sort_values("r_mean", ascending=False).head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
