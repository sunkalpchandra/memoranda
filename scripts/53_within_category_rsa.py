#!/usr/bin/env python
"""Stage 53 — within-category RSA: is the MTL–DNN correspondence only categorical?

For each screening session and region we restrict the neural and model RDMs to the
pictures of a single CLIP category (face_person, animal; needs ≥ 8 pictures in the
session) and recompute Spearman ρ with a curated set of layers. Also a "between only"
control: RDM entries for pairs from different categories only.
Output: results/tables/A3_within_category_rsa.csv, results/figures/A3_within_category.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.paths import FIGURES, MANIFESTS, TABLES

LAYERS = [("alexnet", "conv1", "gap"), ("alexnet", "conv4", "gap"), ("resnet50", "layer2", "gap"), ("resnet50", "avgpool", "gap"), ("vit_b_16", "ln", "gap"), ("dinov2_small", "norm", "gap"), ("clip_vitb32", "ln_post", "cls")]
CATS = ["face_person", "animal", "landmark_place", "vehicle"]
MIN_N = 8


def region_matrix(subject, region, sel):
    if region in ("MTL", "MFC"):
        M, uids, _ = rsa.session_response_matrix(subject, 1, region)
        return M, uids
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None
    keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
    mask = np.isin(units, keep)
    return (M[:, mask], uids) if mask.sum() >= 2 else (None, None)


def masked_rho(Dn, Dm, mask_mat):
    iu = np.triu_indices(Dn.shape[0], 1)
    m = mask_mat[iu]
    x, y = Dn[iu][m], Dm[iu][m]
    if m.sum() < 10 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(sps.spearmanr(x, y).statistic)


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        _, uids, _ = rsa.session_response_matrix(s, 1, None)
        cat = labels.loc[uids, "category"].to_numpy()
        Dm = {k: rsa.model_rdm(*k, uids) for k in LAYERS}
        same = cat[:, None] == cat[None, :]
        for region in ("MTL", "MFC", "MTL_concept"):
            M, u2 = region_matrix(s, region, sel)
            if M is None:
                continue
            Dn = rsa.neural_rdm(M)
            for k, D in Dm.items():
                rows.append({"subject": s, "region": region, "scope": "all", "model": k[0], "layer": k[1], "n_images": len(uids), "rho": rsa.compare_rdms(Dn, D)})
                rows.append({"subject": s, "region": region, "scope": "between-category pairs", "model": k[0], "layer": k[1], "n_images": len(uids), "rho": masked_rho(Dn, D, ~same)})
                rows.append({"subject": s, "region": region, "scope": "within-category pairs (all cats)", "model": k[0], "layer": k[1], "n_images": len(uids), "rho": masked_rho(Dn, D, same)})
                for c in CATS:
                    idx = np.where(cat == c)[0]
                    if len(idx) < MIN_N:
                        continue
                    sub = np.ix_(idx, idx)
                    rows.append({"subject": s, "region": region, "scope": f"within {c}", "model": k[0], "layer": k[1], "n_images": len(idx), "rho": rsa.compare_rdms(Dn[sub], D[sub])})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_within_category_rsa_per_session.csv", index=False)

    def _t(x):
        x = x.dropna()
        return pd.Series({"n": len(x), "mean": x.mean(), "sem": x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan, "p": sps.ttest_1samp(x, 0).pvalue if len(x) > 2 else np.nan})

    summ = per.groupby(["region", "scope", "model", "layer"]).rho.apply(_t).unstack().reset_index()
    summ.to_csv(TABLES / "A3_within_category_rsa.csv", index=False)
    show = summ[(summ.region == "MTL") & (summ.layer.isin(["avgpool", "ln_post", "conv1"]))]
    print(show.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 4.6))
    scopes = ["all", "between-category pairs", "within-category pairs (all cats)", "within face_person", "within animal", "within landmark_place", "within vehicle"]
    lay_lbl = [f"{m}\n{l}" for (m, l, v) in LAYERS]
    w = 0.11
    d = summ[summ.region == "MTL"]
    for i, sc in enumerate(scopes):
        g = d[d.scope == sc].set_index(["model", "layer"]).reindex([(m, l) for (m, l, v) in LAYERS])
        ax.bar(np.arange(len(LAYERS)) + (i - 3) * w, g["mean"], w, yerr=g["sem"], label=f"{sc} (n≈{int(g['n'].max()) if g['n'].notna().any() else 0})", capsize=1.5)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(lay_lbl, fontsize=7.5)
    ax.set_ylabel("Spearman ρ (MTL RDM, model RDM), mean ± sem")
    ax.set_title("Is the MTL–DNN correspondence categorical or finer? RSA restricted to subsets of picture pairs")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_within_category.png", dpi=150)


if __name__ == "__main__":
    main()
