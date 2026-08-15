#!/usr/bin/env python
"""Stage 37 — can DNN features alone predict which pictures a patient's concept cells prefer?

Rows: (subject, screening picture). Label: ≥ 1 MTL concept cell of that subject prefers it.
Leave-one-subject-out logistic regression (L2, standardised, PCA-50) on CLIP / ResNet-50 /
DINOv2 features, category one-hot, low-level statistics and face descriptors; report
LOSO AUC and, for the same design, the "reuse" predictor (fraction of other subjects whose
concept cells preferred the picture). Output: results/tables/A5_predict_preference.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from memoranda import imstats
from memoranda.analysis.tables import subject_image_table
from memoranda.features import load_features
from memoranda.paths import FEATURES, TABLES


def feats(model, layer, view, uids):
    if model == "clip_vitb32" and layer == "embed":
        z = np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True)
        X, stored = z["embed"], z["image_uid"]
    else:
        X, stored = load_features(model, layer, view)
    pos = {u: i for i, u in enumerate(stored)}
    return X[[pos[u] for u in uids]].astype(np.float64)


def loso_auc(X, y, groups, n_pcs=50):
    aucs = []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        if y[te].sum() == 0 or y[te].sum() == te.sum():
            continue
        steps = [StandardScaler()]
        if X.shape[1] > n_pcs:
            steps.append(PCA(n_components=n_pcs, random_state=0))
        steps.append(LogisticRegression(C=0.5, max_iter=2000))
        clf = make_pipeline(*steps).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs) / np.sqrt(len(aucs))), len(aucs)


def main() -> None:
    st = subject_image_table()
    scr = st[st.in_screening & st.n_concept_pref_mtl.notna()].reset_index(drop=True)
    y = (scr.n_concept_pref_mtl > 0).to_numpy().astype(int)
    groups = scr.subject.to_numpy()
    uids = scr.image_uid.to_numpy()
    print("positives:", y.mean().round(3), "n rows", len(y))
    preds = {
        "CLIP embed": feats("clip_vitb32", "embed", "gap", uids),
        "CLIP ln_post": feats("clip_vitb32", "ln_post", "cls", uids),
        "ResNet-50 avgpool": feats("resnet50", "avgpool", "gap", uids),
        "DINOv2 norm": feats("dinov2_small", "norm", "cls", uids),
        "AlexNet conv1": feats("alexnet", "conv1", "gap", uids),
        "category one-hot": pd.get_dummies(scr.category).to_numpy(float),
        "low-level stats": scr[imstats.STAT_NAMES[:-1]].to_numpy(float),
        "faces (found, n, area)": scr[["face_found", "n_faces", "largest_face_area"]].fillna(0).to_numpy(float),
        "reuse (LOSO other-subject rate)": scr[["sternberg_rate_loso"]].fillna(0).to_numpy(float),
    }
    # concept-cell reuse from other subjects: fraction of other subjects with a concept cell preferring the picture
    hit = scr.assign(hit=(scr.n_concept_pref_mtl > 0).astype(int))
    tot = hit.groupby("image_uid").hit.transform("sum")
    cnt = hit.groupby("image_uid").hit.transform("size")
    scr["cc_reuse_loso"] = ((tot - hit.hit) / (cnt - 1).replace(0, np.nan)).fillna(0)
    preds["concept-cell reuse (other subjects)"] = scr[["cc_reuse_loso"]].to_numpy(float)
    rows = []
    for name, X in preds.items():
        auc, sem, n = loso_auc(X, y, groups)
        rows.append({"predictor": name, "loso_auc": auc, "sem": sem, "n_subjects": n})
        print(f"{name:38s} AUC {auc:.3f} ± {sem:.3f} (n={n})")
    pd.DataFrame(rows).to_csv(TABLES / "A5_predict_preference.csv", index=False)


if __name__ == "__main__":
    main()
