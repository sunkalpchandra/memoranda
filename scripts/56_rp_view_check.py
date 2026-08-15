#!/usr/bin/env python
"""Stage 56 — robustness: does keeping spatial layout (random-projection view) change the
layer-depth picture? Compare `gap` vs `rp` views for every layer of AlexNet, VGG-16,
ResNet-50 and ConvNeXt against the MTL RDM (per session), plus Euclidean vs correlation
distance for model RDMs. Output: results/tables/A3_rp_view_check.csv, results/figures/A3_rp_view_check.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import list_layers
from memoranda.models.registry import get_spec
from memoranda.paths import FIGURES, TABLES

MODELS = ["alexnet", "vgg16", "resnet50", "convnext_tiny"]


def main() -> None:
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        M, uids, _ = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        Dn = rsa.neural_rdm(M)
        for m in MODELS:
            for layer, view in list_layers(m):
                if view not in ("gap", "rp"):
                    continue
                for metric in ("correlation", "euclidean"):
                    Dm = rsa.model_rdm(m, layer, view, uids, metric=metric)
                    rows.append({"subject": s, "model": m, "layer": layer, "view": view, "metric": metric, "rho": rsa.compare_rdms(Dn, Dm)})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_rp_view_check_per_session.csv", index=False)
    summ = per.groupby(["model", "layer", "view", "metric"]).rho.agg(["count", "mean", "sem"]).reset_index()
    summ["p"] = [sps.ttest_1samp(per[(per.model == a) & (per.layer == b) & (per.view == c) & (per.metric == d)].rho.dropna(), 0).pvalue for a, b, c, d in zip(summ.model, summ.layer, summ["view"], summ.metric)]
    summ.to_csv(TABLES / "A3_rp_view_check.csv", index=False)
    print(summ[summ.metric == "correlation"].round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for ax, m in zip(axes, MODELS):
        lay = list(get_spec(m).layers)
        for view, col in (("gap", "#C44E52"), ("rp", "#4C72B0")):
            for metric, ls in (("correlation", "-"), ("euclidean", "--")):
                d = summ[(summ.model == m) & (summ["view"] == view) & (summ.metric == metric)].set_index("layer").reindex(lay)
                ax.errorbar(range(len(lay)), d["mean"], d["sem"], fmt="o" + ls, ms=3, color=col, label=f"{view} / {metric}", capsize=2, alpha=0.9 if metric == "correlation" else 0.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(lay)))
        ax.set_xticklabels(lay, rotation=60, fontsize=7)
        ax.set_title(m)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Spearman ρ with MTL RDM (mean ± sem, sessions)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Robustness: pooled (gap) vs spatially-preserving random-projection (rp) features; correlation vs Euclidean RDMs")
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_rp_view_check.png", dpi=150)


if __name__ == "__main__":
    main()
