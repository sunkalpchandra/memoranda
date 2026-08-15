#!/usr/bin/env python
"""Stage 55 — is a face-identity space a better model of MTL structure among face pictures?

Restricting each screening session to pictures with a detected face (MTCNN, ≥ 8 per session):
  * RSA of MTL / MTL-concept RDMs vs VGGFace2 identity RDM, CLIP ln_post, ResNet-50 avgpool,
    AlexNet conv1, and the pooled cross-subject face RDM
  * similarity tuning of face-preferring concept cells (preferred picture contains a face)
    with the VGGFace2 space vs the object models
Also: are concept cells more likely to prefer pictures with more/larger faces than shown?
Output: results/tables/A3_face_space_rsa.csv, A4_face_similarity_tuning.csv, A5_face_preference.csv,
        results/figures/A3_face_space.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.analysis import rsa
from memoranda.analysis import stats as S
from memoranda.dandi import list_assets
from memoranda.features import load_features
from memoranda.paths import FIGURES, MANIFESTS, TABLES

SPACES = {"vggface2 identity": ("facenet_vggface2", "embed", "gap"), "clip ln_post": ("clip_vitb32", "ln_post", "cls"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "dinov2 norm": ("dinov2_small", "norm", "cls"), "alexnet conv1": ("alexnet", "conv1", "gap")}
MIN_N = 8


def load_space(model, layer, view):
    X, stored = load_features(model, layer, view)
    X = X.astype(np.float64)
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, {u: i for i, u in enumerate(stored)}


def main() -> None:
    faces = pd.read_csv(MANIFESTS / "faces.csv").set_index("image_uid")
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})

    # --- RSA among face pictures
    rows = []
    for s in subjects:
        M, uids, units = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        has_face = faces.loc[uids, "face_found"].to_numpy().astype(bool)
        idx = np.where(has_face)[0]
        if len(idx) < MIN_N:
            continue
        keep = sel[(sel.subject == s) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
        cmask = np.isin(units, keep)
        for region, Mr in (("MTL", M), ("MTL_concept", M[:, cmask] if cmask.sum() >= 2 else None)):
            if Mr is None:
                continue
            Dn = rsa.neural_rdm(Mr[idx])
            for name, (m, l, v) in SPACES.items():
                Dm = rsa.model_rdm(m, l, v, uids[idx])
                rows.append({"subject": s, "region": region, "n_face_images": len(idx), "space": name, "rho": rsa.compare_rdms(Dn, Dm)})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_face_space_rsa_per_session.csv", index=False)
    summ = per.groupby(["region", "space"]).rho.agg(["count", "mean", "sem"]).reset_index()
    summ["p"] = [sps.ttest_1samp(per[(per.region == r) & (per.space == sp)].rho.dropna(), 0).pvalue for r, sp in zip(summ.region, summ.space)]
    summ.to_csv(TABLES / "A3_face_space_rsa.csv", index=False)
    print(summ.round(4).to_string(index=False))

    # --- similarity tuning of face-preferring concept cells
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]
    cc = cc[cc.pref_image.map(lambda u: faces.loc[u, "face_found"] == 1 if u in faces.index else False)]
    spaces = {k: load_space(*v) for k, v in SPACES.items()}
    trows = []
    for r in cc.itertuples():
        t = tun[(tun.subject == r.subject) & (tun.unit == r.unit) & (tun.task == "screening")]
        uids = t.image_uid.to_numpy()
        rates = t.rate.to_numpy()
        pref_i = np.where(uids == r.pref_image)[0][0]
        for scope in ("all other pictures", "other face pictures"):
            if scope == "all other pictures":
                mask = np.arange(len(uids)) != pref_i
            else:
                mask = (np.arange(len(uids)) != pref_i) & faces.loc[uids, "face_found"].to_numpy().astype(bool)
            if mask.sum() < 6:
                continue
            for name, (X, pos) in spaces.items():
                sims = X[[pos[u] for u in uids]] @ X[pos[r.pref_image]]
                trows.append({"subject": r.subject, "unit": r.unit, "area": r.area, "scope": scope, "space": name, "n": int(mask.sum()), "rho": sps.spearmanr(rates[mask], sims[mask]).statistic})
    tp = pd.DataFrame(trows)
    tp.to_csv(TABLES / "A4_face_similarity_tuning_per_cell.csv", index=False)
    ts = tp.groupby(["scope", "space"]).rho.agg(["count", "mean", "sem"]).reset_index()
    ts["wilcoxon_p"] = [sps.wilcoxon(tp[(tp.scope == a) & (tp.space == b)].rho.dropna()).pvalue for a, b in zip(ts.scope, ts.space)]
    ts.to_csv(TABLES / "A4_face_similarity_tuning.csv", index=False)
    print(ts.round(4).to_string(index=False))

    # --- do concept cells prefer face pictures / bigger faces?
    img = pd.read_csv(MANIFESTS / "images.csv")
    shown = img[(img.task == "screening") & (img.image_uid != "img_null")][["subject", "image_uid"]].drop_duplicates()
    ccall = sel[(sel.task == "screening") & sel.concept_cell]
    prow = []
    for region in ("MTL", "amygdala", "hippocampus", "MFC"):
        c = ccall[ccall.region == region] if region in ("MTL", "MFC") else ccall[ccall.area == region]
        sh = shown[shown.subject.isin(c.subject.unique())]
        for col in ("face_found", "n_faces", "largest_face_area"):
            a = faces.loc[c.pref_image, col].to_numpy(float)
            b = faces.loc[sh.image_uid, col].to_numpy(float)
            u, p = S.mannwhitney(a, b)
            prow.append({"region": region, "measure": col, "n_cells": len(c), "mean_pref": np.nanmean(a), "mean_shown": np.nanmean(b), "auc": S.auc_effect(a, b), "p": p})
    pf = pd.DataFrame(prow)
    pf.to_csv(TABLES / "A5_face_preference.csv", index=False)
    print(pf.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    d = summ[summ.region == "MTL"].set_index("space").reindex(list(SPACES))
    axes[0].bar(range(len(d)), d["mean"], yerr=d["sem"], color=["#C44E52" if i == 0 else "#8da0cb" for i in range(len(d))], capsize=3)
    axes[0].set_xticks(range(len(d)))
    axes[0].set_xticklabels(d.index, rotation=25, ha="right", fontsize=8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("Spearman ρ (MTL RDM among face pictures)")
    axes[0].set_title(f"RSA restricted to face pictures (n={int(d['count'].max())} sessions)")
    d2 = ts[ts.scope == "other face pictures"].set_index("space").reindex(list(SPACES))
    axes[1].bar(range(len(d2)), d2["mean"], yerr=d2["sem"], color=["#C44E52" if i == 0 else "#8da0cb" for i in range(len(d2))], capsize=3)
    axes[1].set_xticks(range(len(d2)))
    axes[1].set_xticklabels(d2.index, rotation=25, ha="right", fontsize=8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_ylabel("similarity-tuning ρ")
    axes[1].set_title(f"Face-preferring MTL concept cells (n={int(d2['count'].max())}): generalisation over other face pictures")
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_face_space.png", dpi=150)


if __name__ == "__main__":
    main()
