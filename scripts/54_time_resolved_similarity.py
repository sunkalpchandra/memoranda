#!/usr/bin/env python
"""Stage 54 — when does a concept cell's generalisation along DNN similarity emerge?

Per MTL concept cell (screening) and 100-ms sliding window (step 25 ms, −200…1200 ms):
Spearman ρ between the cell's windowed rate on non-preferred pictures and their late-layer
similarity (ResNet-50 avgpool, CLIP ln_post) or early-layer similarity (AlexNet conv1) to
its preferred picture. Also the plain preferred-vs-nonpreferred rate difference for latency
comparison. Output: results/tables/A4_time_resolved_similarity.csv, results/figures/A4_time_resolved_similarity.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.features import load_features
from memoranda.paths import FEATURES, FIGURES, MANIFESTS, TABLES

WIN = 0.1
CENTERS = np.arange(-0.2 + WIN / 2, 1.2 - WIN / 2 + 1e-9, 0.025)
SPACES = {"alexnet conv1": ("alexnet", "conv1", "gap"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "clip ln_post": ("clip_vitb32", "ln_post", "cls")}


def load_space(model, layer, view):
    if model == "clip_vitb32" and layer == "embed":
        z = np.load(FEATURES / "clip_vitb32_embed.npz", allow_pickle=True)
        X, stored = z["embed"], z["image_uid"]
    else:
        X, stored = load_features(model, layer, view)
    X = X.astype(np.float64)
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, {u: i for i, u in enumerate(stored)}


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]
    spaces = {k: load_space(*v) for k, v in SPACES.items()}
    rows = []
    for s, g in cc.groupby("subject"):
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        onsets = pres.onset.to_numpy()
        lab = pres.image_uid.to_numpy()
        uids = np.unique(lab)
        for r in g.itertuples():
            st = spikes[r.unit]
            pref_i = np.where(uids == r.pref_image)[0][0]
            mask = np.arange(len(uids)) != pref_i
            sims = {k: X[[pos[u] for u in uids]] @ X[pos[r.pref_image]] for k, (X, pos) in spaces.items()}
            for c in CENTERS:
                cnt = neural.count_in_windows(st, onsets, (c - WIN / 2, c + WIN / 2)) / WIN
                M = np.array([cnt[lab == u].mean() for u in uids])
                rec = {"subject": s, "unit": r.unit, "area": r.area, "t": round(float(c), 4), "pref_minus_nonpref": float(M[pref_i] - M[mask].mean())}
                for k, sm in sims.items():
                    rec[f"rho_{k}"] = sps.spearmanr(M[mask], sm[mask]).statistic if M[mask].std() > 0 else np.nan
                rows.append(rec)
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A4_time_resolved_similarity_per_cell.csv", index=False)
    cols = [f"rho_{k}" for k in SPACES] + ["pref_minus_nonpref"]
    summ = per.groupby("t")[cols].agg(["mean", "sem"]).reset_index()
    summ.columns = ["t"] + [f"{a}_{b}" for a, b in summ.columns[1:]]
    summ.to_csv(TABLES / "A4_time_resolved_similarity.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"alexnet conv1": "#9ecae1", "resnet50 avgpool": "#08306b", "clip ln_post": "#C44E52"}
    for k in SPACES:
        ax.plot(summ.t * 1000, summ[f"rho_{k}_mean"], color=colors[k], label=f"ρ(rate, sim to preferred) — {k}", lw=1.8)
        ax.fill_between(summ.t * 1000, summ[f"rho_{k}_mean"] - summ[f"rho_{k}_sem"], summ[f"rho_{k}_mean"] + summ[f"rho_{k}_sem"], color=colors[k], alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(summ.t * 1000, summ["pref_minus_nonpref_mean"], color="grey", ls="--", lw=1.2, label="preferred − non-preferred rate (Hz, right axis)")
    ax2.set_ylabel("Δ rate (Hz)", color="grey")
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time from picture onset (ms)")
    ax.set_ylabel("Spearman ρ (mean ± sem over MTL concept cells)")
    ax.set_title(f"Time course of similarity tuning, {per.groupby(['subject','unit']).ngroups} MTL concept cells (100-ms windows)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_time_resolved_similarity.png", dpi=150)
    pk = summ.loc[summ["rho_clip ln_post_mean"].idxmax()]
    pk2 = summ.loc[summ["pref_minus_nonpref_mean"].idxmax()]
    print(f"peak similarity tuning (CLIP) at {pk.t*1000:.0f} ms, ρ={pk['rho_clip ln_post_mean']:.3f}; peak pref-nonpref at {pk2.t*1000:.0f} ms")


if __name__ == "__main__":
    main()
