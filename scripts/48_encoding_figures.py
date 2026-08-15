#!/usr/bin/env python
"""Stage 48 — figures/tables for the encoding-model sweep (A4).

* layer curves of mean CV r for MTL concept cells, MTL non-concept, MFC concept
* best predictor per unit → distribution of "best model / best layer depth" for concept cells
* per-unit r vs noise ceiling scatter for the best late layer
* model comparison table (mean r at best layer, fraction of ceiling)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import FIGURES, TABLES

VIEW_PREF = {"vit": "cls", "dino": "cls", "clip": "cls", "cnn": "gap"}
COLORS = {"MTL concept": "#C44E52", "MTL non-concept": "#DD8452", "MFC concept": "#4C72B0", "MFC non-concept": "#8da0cb"}


def main() -> None:
    per = pd.read_csv(TABLES / "A4_encoding_per_unit.csv")
    per = per.dropna(subset=["r"])
    per["group"] = np.where(per.region == "MTL", "MTL", "MFC") + np.where(per.concept_cell, " concept", " non-concept")
    per["r_clip"] = per.r.clip(lower=-1, upper=1)

    # layer curves
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, m in zip(axes.flat, DEFAULT_MODELS):
        view = VIEW_PREF[get_spec(m).family]
        lay = [l for l in get_spec(m).layers if ((per.model == m) & (per.layer == l) & (per.view == view)).any()]
        for gname in COLORS:
            d = per[(per.group == gname) & (per.model == m) & (per.view == view)]
            g = d.groupby("layer").r.agg(["mean", "sem"]).reindex(lay)
            ax.errorbar(range(len(lay)), g["mean"], g["sem"], fmt="o-", ms=4, color=COLORS[gname], label=f"{gname} (n={d.unit.nunique()})", capsize=2)
        for base, ls in (("category", "--"), ("lowlevel", ":")):
            b = per[(per.group == "MTL concept") & (per.model == "baseline") & (per.layer == base)].r.mean()
            ax.axhline(b, color="grey", ls=ls, lw=1, label=f"MTL concept vs {base}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(lay)))
        ax.set_xticklabels(lay, rotation=60, fontsize=7)
        ax.set_title(f"{m} ({view})")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("cross-validated Pearson r\n(mean ± sem over units)")
    axes[1, 0].set_ylabel("CV Pearson r")
    axes[0, 0].legend(fontsize=6.5)
    fig.suptitle("Encoding models: DNN layer features → single-unit tuning curves (screening, 6-fold CV ridge)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_encoding_layer_curves.png", dpi=150)
    plt.close(fig)

    # best layer per model per group + comparison table
    rows = []
    for gname in COLORS:
        d = per[per.group == gname]
        for m in DEFAULT_MODELS + ["baseline"]:
            dm = d[d.model == m]
            if not len(dm):
                continue
            g = dm.groupby(["layer", "view"]).r.agg(["mean", "sem", "size"]).reset_index()
            best = g.loc[g["mean"].idxmax()]
            dd = dm[(dm.layer == best.layer) & (dm.view == best["view"])]
            rel = dd.reliability.clip(lower=0.05)
            rows.append(
                {
                    "group": gname,
                    "model": m,
                    "best_layer": best.layer,
                    "view": best["view"],
                    "n_units": int(best["size"]),
                    "r_mean": best["mean"],
                    "r_sem": best["sem"],
                    "r_median": dd.r.median(),
                    "frac_r_pos": (dd.r > 0).mean(),
                    "r_over_ceiling": float((dd.r / np.sqrt(rel)).mean()),
                    "wilcoxon_p": sps.wilcoxon(dd.r).pvalue if len(dd) > 5 else np.nan,
                }
            )
    bl = pd.DataFrame(rows)
    bl.to_csv(TABLES / "A4_encoding_best_layer.csv", index=False)
    print(bl.round(3).to_string(index=False))

    # per-unit best predictor among DNN layers (MTL concept cells): which model & relative depth
    dnn = per[(per.model != "baseline") & (per.group == "MTL concept")]
    idx = dnn.groupby(["subject", "unit"]).r.idxmax()
    best = dnn.loc[idx]
    depth = []
    for r in best.itertuples():
        lay = list(get_spec(r.model).layers)
        depth.append(lay.index(r.layer) / max(len(lay) - 1, 1) if r.layer in lay else np.nan)
    best = best.assign(rel_depth=depth)
    best[["subject", "unit", "area", "model", "layer", "view", "r", "reliability", "rel_depth"]].to_csv(TABLES / "A4_best_predictor_per_concept_cell.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    best.model.value_counts().reindex(DEFAULT_MODELS).plot.bar(ax=axes[0], color="#C44E52")
    axes[0].set_title("Best-predicting model per MTL concept cell")
    axes[0].set_ylabel("cells")
    axes[1].hist(best.rel_depth.dropna(), bins=np.linspace(0, 1, 11), color="#C44E52", edgecolor="w")
    axes[1].set_title("Relative depth of best layer (0=first, 1=last)")
    axes[1].set_xlabel("relative depth")
    d = per[(per.group == "MTL concept") & (per.model == "resnet50") & (per.layer == "avgpool")]
    axes[2].scatter(d.reliability, d.r, s=14, alpha=0.7, color="#C44E52")
    xx = np.linspace(0, 1, 50)
    axes[2].plot(xx, np.sqrt(xx), "k--", lw=0.8, label="ceiling √reliability")
    axes[2].set_xlabel("tuning-curve split-half reliability")
    axes[2].set_ylabel("CV r (ResNet-50 avgpool)")
    axes[2].set_title("MTL concept cells: prediction vs ceiling")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_encoding_best.png", dpi=150)


if __name__ == "__main__":
    main()
