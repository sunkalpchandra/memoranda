#!/usr/bin/env python
"""Stage 45 — pooled cross-subject neural RDM.

Every screening session yields an RDM over its own 54–63 images. Since the pool
is shared, we rank-normalise each session's RDM and average entries over all
sessions in which a pair of images was co-shown. The result is a partially
observed 342×342 consensus RDM (with a coverage count per pair) for MTL, MFC,
amygdala, hippocampus and MTL concept cells. We compare it with model RDMs on
observed pairs (weighted Spearman via repetition of pairs by coverage is
overkill; we use plain Spearman on pairs with coverage ≥ 2) and permutation p.
Also: 2-D MDS of the pooled MTL RDM, coloured by category, for the report.
Outputs: results/tables/A3_pooled_rsa.csv, results/figures/A3_pooled_mds.png,
         data/cache/pooled_rdm_<region>.npz
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.manifold import MDS

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import list_layers
from memoranda.models.registry import DEFAULT_MODELS
from memoranda.paths import CACHE, FIGURES, MANIFESTS, TABLES

REGIONS = ["MTL", "MFC", "amygdala", "hippocampus", "MTL_concept"]


def region_matrix(subject, region, sel):
    if region in ("MTL", "MFC"):
        M, uids, _ = rsa.session_response_matrix(subject, 1, region)
        return M, uids
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None
    ut = neural.load_units_table(subject, 1).set_index("unit")
    keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy() if region == "MTL_concept" else ut[ut.area == region].index.to_numpy()
    mask = np.isin(units, keep)
    return (M[:, mask], uids) if mask.sum() >= 2 else (None, None)


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    all_uids = np.array(sorted(labels.index))
    pos = {u: i for i, u in enumerate(all_uids)}
    n = len(all_uids)
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})

    pooled = {}
    for region in REGIONS:
        S = np.zeros((n, n))
        C = np.zeros((n, n))
        for s in subjects:
            M, uids = region_matrix(s, region, sel)
            if M is None:
                continue
            D = rsa.neural_rdm(M)
            iu = np.triu_indices(len(uids), 1)
            r = np.zeros_like(D)
            r[iu] = sps.rankdata(D[iu]) / len(iu[0])  # rank-normalise to (0,1]
            r = r + r.T
            idx = np.array([pos[u] for u in uids])
            S[np.ix_(idx, idx)] += r
            C[np.ix_(idx, idx)] += 1
        np.fill_diagonal(C, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where(C > 0, S / C, np.nan)
        pooled[region] = (R, C)
        np.savez_compressed(CACHE / f"pooled_rdm_{region}.npz", rdm=R, coverage=C, image_uid=all_uids.astype(str))
        print(region, "pairs observed:", int((C > 0).sum() / 2), " coverage≥2:", int((C >= 2).sum() / 2))

    layer_list = [(m, l, v) for m in DEFAULT_MODELS for (l, v) in list_layers(m) if v in ("gap", "cls")] + [("clip_vitb32", "embed", "gap"), ("baseline", "category", "-"), ("baseline", "lowlevel", "-")]
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv")
    rows = []
    rng = np.random.default_rng(0)
    for (m, l, v) in layer_list:
        if m == "baseline":
            Dm = rsa.category_rdm(all_uids, labels.reset_index()) if l == "category" else rsa.lowlevel_rdm(all_uids, stats_df)
        else:
            Dm = rsa.model_rdm(m, l, v, all_uids)
        for region, (R, C) in pooled.items():
            iu = np.triu_indices(n, 1)
            mask = C[iu] >= 2
            x, y = R[iu][mask], Dm[iu][mask]
            rho = sps.spearmanr(x, y).statistic
            # permutation: shuffle image identity of the model RDM
            null = []
            for _ in range(200):
                perm = rng.permutation(n)
                Dp = Dm[np.ix_(perm, perm)]
                null.append(sps.spearmanr(x, Dp[iu][mask]).statistic)
            p = (np.sum(np.array(null) >= rho) + 1) / 201
            rows.append({"region": region, "model": m, "layer": l, "view": v, "n_pairs": int(mask.sum()), "rho": rho, "p_perm": p, "null_sd": float(np.std(null))})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "A3_pooled_rsa.csv", index=False)
    print(res[res.region == "MTL"].sort_values("rho", ascending=False).head(15).round(4).to_string(index=False))
    print(res[res.region == "MFC"].sort_values("rho", ascending=False).head(5).round(4).to_string(index=False))

    # MDS of pooled MTL RDM (impute missing pairs with the mean distance)
    R, C = pooled["MTL"]
    D = np.where(np.isnan(R), np.nanmean(R), R)
    np.fill_diagonal(D, 0)
    seen = C.sum(1) > 0
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(D[np.ix_(seen, seen)])
    cats = labels.loc[all_uids[seen], "category"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in np.unique(cats):
        mm = cats == c
        ax.scatter(emb[mm, 0], emb[mm, 1], s=18, label=f"{c} ({mm.sum()})", alpha=0.8)
    ax.legend(fontsize=7)
    ax.set_title("2-D MDS of the pooled MTL neural RDM (screening; 342 images)\ncoloured by CLIP zero-shot category")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_pooled_mds.png", dpi=150)


if __name__ == "__main__":
    main()
