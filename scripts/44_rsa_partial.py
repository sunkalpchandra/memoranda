#!/usr/bin/env python
"""Stage 44 — do DNN layers explain MTL geometry *beyond* category and low-level structure?

Per screening session and region (MTL, MTL_concept, MFC, amygdala, hippocampus):
  * partial Spearman ρ(neural, model | category RDM, low-level RDM) for a curated set of layers
  * partial ρ(neural, category | model)   — does category survive controlling the model?
  * rank RDM regression: neural ~ category + lowlevel + model  (standardised betas)
Outputs: results/tables/A3_partial_rsa.csv, results/figures/A3_partial_rsa.png
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

LAYERS = [
    ("alexnet", "conv1", "gap"),
    ("alexnet", "conv4", "gap"),
    ("alexnet", "fc7", "gap"),
    ("vgg16", "conv3_3", "gap"),
    ("vgg16", "fc7", "gap"),
    ("resnet50", "layer2", "gap"),
    ("resnet50", "avgpool", "gap"),
    ("convnext_tiny", "logits", "gap"),
    ("vit_b_16", "ln", "gap"),
    ("dinov2_small", "norm", "gap"),
    ("clip_vitb32", "block5", "cls"),
    ("clip_vitb32", "ln_post", "cls"),
    ("clip_vitb32", "embed", "gap"),
]
REGIONS = ["MTL", "MTL_concept", "MFC", "amygdala", "hippocampus"]


def region_matrix(subject, region, sel):
    if region == "MFC":
        return rsa.session_response_matrix(subject, 1, "MFC")
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None
    if region == "MTL":
        return M, uids
    ut = neural.load_units_table(subject, 1).set_index("unit")
    if region == "MTL_concept":
        keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
    else:
        keep = ut[ut.area == region].index.to_numpy()
    mask = np.isin(units, keep)
    return (M[:, mask], uids) if mask.sum() >= 2 else (None, None)


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        _, uids, _ = rsa.session_response_matrix(s, 1, None)
        Dcat = rsa.category_rdm(uids, labels)
        Dlow = rsa.lowlevel_rdm(uids, stats_df)
        Dm = {k: rsa.model_rdm(*k, uids) for k in LAYERS}
        for region in REGIONS:
            M, u2 = region_matrix(s, region, sel)
            if M is None:
                continue
            Dn = rsa.neural_rdm(M)
            for k, D in Dm.items():
                reg = rsa.rdm_regression(Dn, {"category": Dcat, "lowlevel": Dlow, "model": D})
                rows.append(
                    {
                        "subject": s,
                        "region": region,
                        "n_units": M.shape[1],
                        "model": k[0],
                        "layer": k[1],
                        "view": k[2],
                        "rho": rsa.compare_rdms(Dn, D),
                        "rho_partial_cat_low": rsa.partial_compare(Dn, D, [Dcat, Dlow]),
                        "rho_partial_cat": rsa.partial_compare(Dn, D, [Dcat]),
                        "rho_cat": rsa.compare_rdms(Dn, Dcat),
                        "rho_cat_partial_model": rsa.partial_compare(Dn, Dcat, [D]),
                        **reg,
                    }
                )
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_partial_rsa_per_session.csv", index=False)

    def _t(x):
        x = x.dropna()
        if len(x) < 3:
            return pd.Series({"mean": np.nan, "sem": np.nan, "p": np.nan, "n": len(x)})
        return pd.Series({"mean": x.mean(), "sem": x.std() / np.sqrt(len(x)), "p": sps.ttest_1samp(x, 0).pvalue, "n": len(x)})

    out = []
    for (region, m, l, v), g in per.groupby(["region", "model", "layer", "view"]):
        rec = {"region": region, "model": m, "layer": l, "view": v}
        for col in ("rho", "rho_partial_cat_low", "rho_partial_cat", "rho_cat", "rho_cat_partial_model", "beta_model", "beta_category", "beta_lowlevel", "r2"):
            t = _t(g[col])
            rec[f"{col}_mean"] = t["mean"]
            rec[f"{col}_sem"] = t["sem"]
            rec[f"{col}_p"] = t["p"]
        rec["n_sessions"] = int(_t(g["rho"])["n"])
        out.append(rec)
    summ = pd.DataFrame(out)
    summ.to_csv(TABLES / "A3_partial_rsa.csv", index=False)
    cols = ["region", "model", "layer", "rho_mean", "rho_partial_cat_low_mean", "rho_partial_cat_low_p", "rho_cat_partial_model_mean", "rho_cat_partial_model_p", "beta_model_mean", "beta_category_mean", "r2_mean"]
    print(summ[summ.region.isin(["MTL", "MTL_concept"])][cols].round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    labs = [f"{m}\n{l}" for (m, l, v) in LAYERS]
    for ax, region in zip(axes, ["MTL", "MTL_concept"]):
        d = summ[summ.region == region].set_index(["model", "layer"]).reindex([(m, l) for (m, l, v) in LAYERS])
        x = np.arange(len(d))
        ax.bar(x - 0.2, d.rho_mean, 0.4, yerr=d.rho_sem, color="#C44E52", label="ρ(neural, model)", capsize=2)
        ax.bar(x + 0.2, d.rho_partial_cat_low_mean, 0.4, yerr=d.rho_partial_cat_low_sem, color="#DD8452", label="partial ρ | category, low-level", capsize=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=6.5, rotation=45, ha="right")
        ax.set_title(f"{region}: model RDM vs neural RDM (n={int(d.n_sessions.max())} sessions)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Spearman ρ (mean ± sem)")
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_partial_rsa.png", dpi=150)


if __name__ == "__main__":
    main()
