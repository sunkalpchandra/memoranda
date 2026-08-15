#!/usr/bin/env python
"""Stage 43 — figures for the RSA sweep: layer-depth curves per model, region comparison,
best-layer summary. Reads results/tables/A3_rsa_summary.csv and A3_rsa_per_session.csv."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import FIGURES, TABLES

REGION_COLORS = {"MTL": "#C44E52", "MFC": "#4C72B0", "MTL_concept": "#8C2D3A", "amygdala": "#DD8452", "hippocampus": "#55A868"}


def preferred_view(model: str) -> str:
    return "cls" if get_spec(model).family in ("vit", "dino", "clip") else "gap"


def layer_curves(summ: pd.DataFrame, per: pd.DataFrame) -> None:
    models = DEFAULT_MODELS
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, m in zip(axes.flat, models):
        layers = list(get_spec(m).layers)
        view = preferred_view(m)
        for region in ("MTL", "MFC", "MTL_concept"):
            d = summ[(summ.region == region) & (summ.model == m) & (summ.view == view)].set_index("layer")
            d = d.reindex([l for l in layers if l in d.index])
            x = np.arange(len(d))
            ax.errorbar(x, d.rho_mean, d.rho_sem, fmt="o-", ms=4, color=REGION_COLORS[region], label=region, capsize=2)
        for base, ls in (("category", "--"), ("lowlevel", ":")):
            b = summ[(summ.region == "MTL") & (summ.model == "baseline") & (summ.layer == base)]
            if len(b):
                ax.axhline(b.rho_mean.iloc[0], color="grey", ls=ls, lw=1, label=f"MTL vs {base} RDM")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=60, fontsize=7)
        ax.set_title(f"{m} ({view})", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Spearman ρ (neural RDM, model RDM)\nmean ± s.e.m. over sessions")
    axes[1, 0].set_ylabel("Spearman ρ")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Representational similarity between human single-neuron populations and DNN layers (screening task)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_rsa_layer_curves.png", dpi=150)
    plt.close(fig)


def best_layer_bars(summ: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in ("MTL", "MFC", "MTL_concept", "amygdala", "hippocampus"):
        for m in DEFAULT_MODELS:
            d = summ[(summ.region == region) & (summ.model == m)]
            if not len(d):
                continue
            best = d.loc[d.rho_mean.idxmax()]
            rows.append({"region": region, "model": m, "best_layer": best.layer, "view": best.view, "rho_mean": best.rho_mean, "rho_sem": best.rho_sem, "p": best.p, "rho_norm_mean": best.rho_norm_mean, "n_sessions": best.n_sessions})
        for base in ("category", "lowlevel"):
            d = summ[(summ.region == region) & (summ.model == "baseline") & (summ.layer == base)]
            if len(d):
                b = d.iloc[0]
                rows.append({"region": region, "model": f"baseline:{base}", "best_layer": "-", "view": "-", "rho_mean": b.rho_mean, "rho_sem": b.rho_sem, "p": b.p, "rho_norm_mean": b.rho_norm_mean, "n_sessions": b.n_sessions})
    bl = pd.DataFrame(rows)
    bl.to_csv(TABLES / "A3_rsa_best_layer.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    order = DEFAULT_MODELS + ["baseline:category", "baseline:lowlevel"]
    w = 0.27
    for i, region in enumerate(("MTL", "MFC", "MTL_concept")):
        d = bl[bl.region == region].set_index("model").reindex(order)
        ax.bar(np.arange(len(order)) + (i - 1) * w, d.rho_mean, w, yerr=d.rho_sem, color=REGION_COLORS[region], label=region, capsize=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([o.replace("baseline:", "") for o in order], rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman ρ at best layer (mean ± sem)")
    ax.set_title("Best-layer RSA per model: MTL vs MFC (screening sessions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_rsa_best_layer.png", dpi=150)
    plt.close(fig)
    return bl


def region_contrast(per: pd.DataFrame) -> pd.DataFrame:
    """Paired MTL vs MFC comparison at each layer (sessions with both regions)."""
    rows = []
    for (m, l, v), g in per.groupby(["model", "layer", "view"]):
        a = g[g.region == "MTL"].set_index("subject").rho
        b = g[g.region == "MFC"].set_index("subject").rho
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        t, p = sps.ttest_rel(a[common], b[common])
        rows.append({"model": m, "layer": l, "view": v, "n": len(common), "mtl_minus_mfc": float((a[common] - b[common]).mean()), "t": t, "p": p})
    df = pd.DataFrame(rows).sort_values("p")
    df.to_csv(TABLES / "A3_rsa_mtl_vs_mfc.csv", index=False)
    return df


def main() -> None:
    summ = pd.read_csv(TABLES / "A3_rsa_summary.csv")
    per = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    layer_curves(summ, per)
    bl = best_layer_bars(summ, per)
    print(bl.round(4).to_string(index=False))
    rc = region_contrast(per)
    print(rc.head(10).round(4).to_string(index=False))
    print("MFC top:")
    print(summ[summ.region == "MFC"].sort_values("rho_mean", ascending=False).head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
