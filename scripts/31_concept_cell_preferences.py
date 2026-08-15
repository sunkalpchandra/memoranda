#!/usr/bin/env python
"""Stage 31 — what do concept cells prefer?

For every concept cell (screening + Sternberg) we look up the CLIP category /
attributes of its preferred image and compare against the base rate of images
that were actually shown to that cell (so per-subject stimulus composition is
controlled). Enrichment by Fisher exact on (preferred, shown-but-not-preferred),
BH-FDR across categories; broken down by area.

Outputs: results/tables/A5_pref_category_enrichment.csv, A5_pref_attributes.csv,
         results/figures/A5_concept_cell_preferences.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memoranda.analysis import stats as S
from memoranda.paths import FIGURES, MANIFESTS, TABLES

ATTRS = ["famous", "face_visible", "child", "smiling", "emotional", "single_object", "natural", "indoor", "text_present", "colorful"]


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    lab = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    img = pd.read_csv(MANIFESTS / "images.csv")
    img = img[img.image_uid != "img_null"]
    cc = sel[sel.concept_cell].copy()
    cc["category"] = lab.loc[cc.pref_image, "category"].to_numpy()
    for a in ATTRS:
        cc[f"attr_{a}"] = lab.loc[cc.pref_image, f"attr_{a}"].to_numpy()
    cc.to_csv(TABLES / "A5_concept_cells.csv", index=False)

    # base rate: every (cell, shown image) pair
    rows = []
    for task in ("screening", "sternberg"):
        for area_group, areas in {"MTL": ["amygdala", "hippocampus"], "amygdala": ["amygdala"], "hippocampus": ["hippocampus"], "MFC": ["dACC", "preSMA", "vmPFC"]}.items():
            c = cc[(cc.task == task) & cc.area.isin(areas)]
            if len(c) < 5:
                continue
            shown = img[(img.task == task)][["subject", "image_uid"]].drop_duplicates()
            shown = shown.merge(lab[["category"]], left_on="image_uid", right_index=True)
            # expected distribution: mean over cells of that cell's session composition
            comp = shown.groupby("subject").category.value_counts(normalize=True).unstack(fill_value=0)
            exp = comp.reindex(c.subject).mean(0)
            obs = c.category.value_counts(normalize=True)
            for cat in sorted(lab.category.unique()):
                n_pref = int((c.category == cat).sum())
                # Fisher on pooled counts (pref vs shown-not-pref)
                shown_c = shown[shown.subject.isin(c.subject.unique())]
                a = n_pref
                b = len(c) - n_pref
                cc_ = int((shown_c.category == cat).sum())
                d = len(shown_c) - cc_
                from scipy.stats import fisher_exact

                orat, p = fisher_exact([[a, b], [cc_, d]])
                rows.append(
                    {
                        "task": task,
                        "area": area_group,
                        "category": cat,
                        "n_concept_cells": len(c),
                        "n_pref": n_pref,
                        "frac_pref": n_pref / len(c),
                        "frac_shown_expected": float(exp.get(cat, 0.0)),
                        "odds_ratio": orat,
                        "p": p,
                    }
                )
    enr = pd.DataFrame(rows)
    enr["q"] = np.nan
    for (task, area), g in enr.groupby(["task", "area"]):
        enr.loc[g.index, "q"] = S.fdr_bh(g.p.to_numpy())[1]
    enr.to_csv(TABLES / "A5_pref_category_enrichment.csv", index=False)
    print(enr[(enr.area == "MTL")].round(3).to_string(index=False))

    # attributes: preferred image vs shown images (Mann–Whitney AUC), MTL screening
    arows = []
    for task in ("screening", "sternberg"):
        for area_group, areas in {"MTL": ["amygdala", "hippocampus"], "MFC": ["dACC", "preSMA", "vmPFC"], "amygdala": ["amygdala"], "hippocampus": ["hippocampus"]}.items():
            c = cc[(cc.task == task) & cc.area.isin(areas)]
            shown = img[(img.task == task) & img.subject.isin(c.subject.unique())][["subject", "image_uid"]].drop_duplicates()
            for a in ATTRS:
                x = c[f"attr_{a}"].to_numpy()
                y = lab.loc[shown.image_uid, f"attr_{a}"].to_numpy()
                u, p = S.mannwhitney(x, y)
                arows.append({"task": task, "area": area_group, "attribute": a, "n_cells": len(c), "mean_pref": float(np.mean(x)) if len(x) else np.nan, "mean_shown": float(np.mean(y)), "auc": S.auc_effect(x, y), "p": p})
    at = pd.DataFrame(arows)
    at["q"] = np.nan
    for (task, area), g in at.groupby(["task", "area"]):
        at.loc[g.index, "q"] = S.fdr_bh(g.p.to_numpy())[1]
    at.to_csv(TABLES / "A5_pref_attributes.csv", index=False)
    print(at[at.area == "MTL"].round(3).to_string(index=False))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    e = enr[(enr.task == "screening") & (enr.area == "MTL")].sort_values("frac_pref")
    y = np.arange(len(e))
    axes[0].barh(y - 0.2, e.frac_shown_expected, 0.4, color="#bbbbbb", label="shown (expected)")
    axes[0].barh(y + 0.2, e.frac_pref, 0.4, color=["#C44E52" if q < 0.05 else "#4C72B0" for q in e.q], label="preferred by MTL concept cells")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(e.category)
    axes[0].set_xlabel("fraction")
    axes[0].set_title(f"Preferred-image category, MTL concept cells (screening, n={int(e.n_concept_cells.iloc[0])})\nred: FDR q<0.05")
    axes[0].legend(fontsize=8)
    a = at[(at.task == "screening") & (at.area == "MTL")].sort_values("auc")
    axes[1].barh(a.attribute, a.auc - 0.5, color=["#C44E52" if q < 0.05 else "#4C72B0" for q in a.q])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("AUC − 0.5 (preferred image > shown images)")
    axes[1].set_title("CLIP attributes of preferred images vs. all shown images")
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_concept_cell_preferences.png", dpi=150)


if __name__ == "__main__":
    main()
