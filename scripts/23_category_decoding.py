#!/usr/bin/env python
"""Stage 23 — how categorical is each population? Decode picture category from single-trial
population responses (screening, 200–1000 ms), per session and region.

Leave-one-picture-out: train a multinomial logistic regression on the trials of all other
pictures, test on the held-out picture's 6 trials (so identity can't leak). Categories with
< 4 pictures in the session are dropped; chance = 1/n_categories; we report balanced accuracy
minus chance. Regions: MTL, MFC, amygdala, hippocampus, MTL_concept.
Output: results/tables/A5_category_decoding.csv, results/figures/A5_category_decoding.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from memoranda import neural
from memoranda.dandi import list_assets
from memoranda.log import get_logger
from memoranda.paths import FIGURES, MANIFESTS, TABLES

log = get_logger("catdec")
REGIONS = ["MTL", "MFC", "amygdala", "hippocampus", "MTL_concept"]


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    lab = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        R, units = neural.rate_matrix(spikes, pres.onset.to_numpy(), neural.RESP_WIN)
        pic = pres.image_uid.to_numpy()
        cat = lab.loc[pic, "category"].to_numpy()
        # keep categories with >= 4 pictures
        pics_per_cat = pd.Series(pic).groupby(cat).nunique()
        keep_cats = pics_per_cat[pics_per_cat >= 4].index
        m = np.isin(cat, keep_cats)
        if len(keep_cats) < 3:
            continue
        ut = neural.load_units_table(s, 1).set_index("unit")
        for region in REGIONS:
            if region == "MTL_concept":
                keep_u = sel[(sel.subject == s) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
            elif region in ("MTL", "MFC"):
                keep_u = ut[ut.region == region].index.to_numpy()
            else:
                keep_u = ut[ut.area == region].index.to_numpy()
            umask = np.isin(units, keep_u)
            if umask.sum() < 3:
                continue
            X, y, p = R[m][:, umask], cat[m], pic[m]
            preds = np.empty(len(y), dtype=object)
            for pu in np.unique(p):
                te = p == pu
                tr = ~te
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
                preds[te] = clf.predict(sc.transform(X[te]))
            acc = balanced_accuracy_score(y, preds)
            chance = 1.0 / len(keep_cats)
            rows.append({"subject": s, "region": region, "n_units": int(umask.sum()), "n_categories": len(keep_cats), "n_trials": int(m.sum()), "balanced_acc": acc, "chance": chance, "acc_minus_chance": acc - chance})
        log.info(f"sub-{s:02d} done")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A5_category_decoding_per_session.csv", index=False)
    summ = df.groupby("region").agg(n=("subject", "size"), units_median=("n_units", "median"), acc_minus_chance=("acc_minus_chance", "mean"), sem=("acc_minus_chance", lambda x: x.std() / np.sqrt(len(x)))).reset_index()
    summ["p_wilcoxon"] = [sps.wilcoxon(df[df.region == r].acc_minus_chance).pvalue if (df.region == r).sum() >= 5 else np.nan for r in summ.region]
    summ.to_csv(TABLES / "A5_category_decoding.csv", index=False)
    print(summ.round(4).to_string(index=False))
    a = df[df.region == "amygdala"].set_index("subject").acc_minus_chance
    h = df[df.region == "hippocampus"].set_index("subject").acc_minus_chance
    common = a.index.intersection(h.index)
    if len(common) >= 5:
        print("amygdala vs hippocampus paired Wilcoxon p =", sps.wilcoxon(a[common], h[common]).pvalue, "n =", len(common))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = ["MTL", "MTL_concept", "amygdala", "hippocampus", "MFC"]
    d = summ.set_index("region").reindex(order)
    ax.bar(range(len(d)), d["acc_minus_chance"].to_numpy() * 100, yerr=d["sem"].to_numpy() * 100, color=["#C44E52", "#8C2D3A", "#DD8452", "#55A868", "#4C72B0"], capsize=3)
    for i, r in enumerate(order):
        g = df[df.region == r]
        ax.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(g)), g.acc_minus_chance * 100, s=10, color="k", alpha=0.5)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([f"{r}\n(n={int(n)})" for r, n in zip(order, d.n)], fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("balanced accuracy − chance (%)")
    ax.set_title("Single-trial category decoding, leave-one-picture-out")
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_category_decoding.png", dpi=150)


if __name__ == "__main__":
    main()
