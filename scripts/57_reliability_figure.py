#!/usr/bin/env python
"""Stage 57 — how reliable are the neural RDMs, and how does that bound RSA?

Split-half reliability of each session's RDM vs number of units, per region, and the
implied ceiling √reliability alongside the observed best-layer ρ.
Output: results/figures/A3_reliability.png, results/tables/A3_reliability_summary.csv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memoranda.paths import FIGURES, TABLES

COL = {"MTL": "#C44E52", "MFC": "#4C72B0", "MTL_concept": "#8C2D3A", "amygdala": "#DD8452", "hippocampus": "#55A868", "all": "grey"}


def main() -> None:
    rel = pd.read_csv(TABLES / "A3_rsa_reliability.csv")
    per = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    best = per[(per.model == "clip_vitb32") & (per.layer == "ln_post") & (per.view == "cls")][["subject", "region", "rho"]]
    d = rel.merge(best, on=["subject", "region"], how="left")
    summ = d.groupby("region").agg(n_sessions=("subject", "size"), units_median=("n_units", "median"), reliability_mean=("reliability", "mean"), reliability_median=("reliability", "median"), rho_clip_mean=("rho", "mean")).reset_index()
    summ["ceiling_sqrt_rel"] = np.sqrt(summ.reliability_mean.clip(lower=0))
    summ["rho_over_ceiling"] = summ.rho_clip_mean / summ.ceiling_sqrt_rel
    summ.to_csv(TABLES / "A3_reliability_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for region in ("MTL", "MFC", "MTL_concept", "amygdala", "hippocampus"):
        g = d[d.region == region]
        axes[0].scatter(g.n_units, g.reliability, s=22, color=COL[region], label=region, alpha=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("units in population")
    axes[0].set_ylabel("split-half reliability of RDM (Spearman–Brown)")
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("RDM reliability vs population size")
    axes[0].legend(fontsize=7)
    for region in ("MTL", "MFC", "MTL_concept", "amygdala", "hippocampus"):
        g = d[d.region == region]
        axes[1].scatter(g.reliability, g.rho, s=22, color=COL[region], alpha=0.8, label=region)
    xx = np.linspace(0, max(d.reliability.max(), 0.3), 50)
    axes[1].plot(xx, np.sqrt(xx), "k--", lw=0.8, label="ceiling √reliability")
    axes[1].set_xlabel("RDM reliability")
    axes[1].set_ylabel("ρ with CLIP ln_post RDM")
    axes[1].set_title("Observed RSA vs noise ceiling, per session")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_reliability.png", dpi=150)


if __name__ == "__main__":
    main()
