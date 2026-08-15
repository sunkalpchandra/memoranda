#!/usr/bin/env python
"""Stage 74 — cross-validated reweighted RSA: does combining layers / models beat the best single layer?

For each screening session the MTL RDM (rank-transformed upper triangle) is regressed with
non-negative least squares on a bank of model RDMs (rank-transformed) — (a) all hooked layers of
one model, (b) the last layer of every model, (c) all layers of all models. Weights are fitted on
the other sessions (stacked) and applied to the held-out session (leave-one-session-out); the score
is Spearman ρ between the weighted prediction and the held-out neural RDM. Compared with the best
single layer chosen on the training sessions (also cross-validated) and with the fixed CLIP ln_post.
Output: results/tables/A3_reweighted_rsa.csv, results/figures/A3_reweighted_rsa.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.optimize import nnls
from scipy.stats import rankdata

from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import list_layers
from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import FIGURES, TABLES

VIEW = {"vit": "cls", "dino": "cls", "clip": "cls", "cnn": "gap"}


def main() -> None:
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    banks = {"all layers, all models": [], "last layer of each model": []}
    per_model = {m: [] for m in DEFAULT_MODELS}
    for m in DEFAULT_MODELS:
        v = VIEW[get_spec(m).family]
        lays = [l for (l, vv) in list_layers(m) if vv == v]
        order = [l for l in get_spec(m).layers if l in lays]
        for l in order:
            banks["all layers, all models"].append((m, l, v))
            per_model[m].append((m, l, v))
        banks["last layer of each model"].append((m, order[-1], v))
    for m in DEFAULT_MODELS:
        banks[f"all layers of {m}"] = per_model[m]

    # cache ranked upper triangles per session
    sess = {}
    for s in subjects:
        M, uids, _ = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        y = rankdata(rsa.upper(rsa.neural_rdm(M)))
        y = (y - y.mean()) / y.std()
        feats = {}
        for key in banks["all layers, all models"]:
            x = rankdata(rsa.upper(rsa.model_rdm(*key, uids)))
            feats[key] = (x - x.mean()) / (x.std() + 1e-12)
        sess[s] = (y, feats)
    subs = sorted(sess)
    rows = []
    for bname, keys in banks.items():
        scores_w, scores_best, scores_clip = [], [], []
        for s in subs:
            train = [t for t in subs if t != s]
            Xtr = np.vstack([np.column_stack([sess[t][1][k] for k in keys]) for t in train])
            ytr = np.concatenate([sess[t][0] for t in train])
            w, _ = nnls(Xtr, ytr)
            Xte = np.column_stack([sess[s][1][k] for k in keys])
            pred = Xte @ w
            scores_w.append(sps.spearmanr(pred, sess[s][0]).statistic if pred.std() > 0 else 0.0)
            # best single layer chosen on training sessions
            means = [np.mean([sps.spearmanr(sess[t][1][k], sess[t][0]).statistic for t in train]) for k in keys]
            kb = keys[int(np.argmax(means))]
            scores_best.append(sps.spearmanr(sess[s][1][kb], sess[s][0]).statistic)
            kc = ("clip_vitb32", "ln_post", "cls")
            scores_clip.append(sps.spearmanr(sess[s][1][kc], sess[s][0]).statistic)
        sw, sb, sc = map(np.array, (scores_w, scores_best, scores_clip))
        rows.append({"bank": bname, "n_predictors": len(keys), "n_sessions": len(subs), "rho_reweighted": sw.mean(), "sem_reweighted": sw.std() / np.sqrt(len(sw)), "rho_best_single_cv": sb.mean(), "rho_clip_ln_post": sc.mean(), "p_reweighted_vs_best": sps.wilcoxon(sw - sb).pvalue if np.any(sw != sb) else np.nan, "p_reweighted_vs_clip": sps.wilcoxon(sw - sc).pvalue if np.any(sw != sc) else np.nan})
    df = pd.DataFrame(rows).sort_values("rho_reweighted", ascending=False)
    df.to_csv(TABLES / "A3_reweighted_rsa.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = df.sort_values("rho_reweighted")
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.rho_reweighted, 0.4, xerr=d.sem_reweighted, color="#C44E52", label="NNLS-reweighted (LOSO CV)", capsize=2)
    ax.barh(y + 0.2, d.rho_best_single_cv, 0.4, color="#8da0cb", label="best single layer (chosen on training sessions)")
    ax.axvline(d.rho_clip_ln_post.iloc[0], color="k", ls="--", lw=0.8, label="fixed CLIP ln_post")
    ax.set_yticks(y)
    ax.set_yticklabels(d.bank, fontsize=8)
    ax.set_xlabel("held-out Spearman ρ with MTL RDM (mean over sessions)")
    ax.set_title("Reweighted RSA: combining layers/models vs the best single layer")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_reweighted_rsa.png", dpi=150)


if __name__ == "__main__":
    main()
