#!/usr/bin/env python
"""Stage 50 — do amygdala and hippocampus (and MFC areas) prefer different layer depths?

Uses the per-cell similarity-tuning ρ (script 47). Layers are mapped to relative depth
(0 = first hooked layer, 1 = last) and pooled across models into 5 depth bins; per area we
plot mean ρ vs depth and test early (bins 1–2) vs late (bins 4–5) with a paired Wilcoxon
over cells. Output: results/tables/A4_area_depth_profile.csv, results/figures/A4_area_depth.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.models.registry import get_spec
from memoranda.paths import FIGURES, TABLES

AREAS = ["amygdala", "hippocampus", "dACC", "preSMA"]
COL = {"amygdala": "#DD8452", "hippocampus": "#55A868", "dACC": "#4C72B0", "preSMA": "#8172B2"}


def main() -> None:
    per = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    per = per[per.group.str.contains("concept") & ~per.group.str.contains("non")]
    depth = []
    for r in per.itertuples():
        lay = list(get_spec(r.model).layers)
        depth.append(lay.index(r.layer) / max(len(lay) - 1, 1) if r.layer in lay else np.nan)
    per["rel_depth"] = depth
    per["bin"] = pd.cut(per.rel_depth, [-0.01, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5])
    cell = per.groupby(["area", "subject", "unit", "bin"], observed=True).rho.mean().reset_index()
    summ = cell.groupby(["area", "bin"], observed=True).rho.agg(["mean", "sem", "count"]).reset_index()
    rows = []
    for a in AREAS:
        c = cell[cell.area == a].pivot_table(index=["subject", "unit"], columns="bin", values="rho", observed=True)
        if len(c) < 5:
            continue
        early = c[[1, 2]].mean(1)
        late = c[[4, 5]].mean(1)
        p = sps.wilcoxon(late - early).pvalue
        rows.append({"area": a, "n_cells": len(c), "early_mean": early.mean(), "late_mean": late.mean(), "late_minus_early": (late - early).mean(), "wilcoxon_p": p})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "A4_area_depth_profile.csv", index=False)
    summ.to_csv(TABLES / "A4_area_depth_bins.csv", index=False)
    print(res.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for a in AREAS:
        d = summ[summ.area == a]
        if not len(d):
            continue
        ax.errorbar(d["bin"].astype(int), d["mean"], d["sem"], fmt="o-", color=COL[a], label=f"{a} (n={int(d['count'].max())})", capsize=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("relative layer depth (bin 1 = earliest, 5 = last), pooled over 8 models")
    ax.set_ylabel("similarity-tuning ρ (mean ± sem over concept cells)")
    ax.set_title("Layer-depth preference by area")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_area_depth.png", dpi=150)


if __name__ == "__main__":
    main()
