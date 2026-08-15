#!/usr/bin/env python
"""Stage 33 — animacy and area: do animate images recruit more concept cells, and where?

Per subject, images are split into animate (face_person, animal, scene_with_people)
vs inanimate; for each area we compute the mean number of concept cells preferring
an image of each class (rate per image) and test the paired difference across
subjects (Wilcoxon). Also a per-category × area table and Mormann-style test of
animal-preferring cells in amygdala vs hippocampus, right vs left.
Outputs: results/tables/A5_animacy_by_area.csv, A5_category_by_area.csv,
         results/figures/A5_animacy.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.paths import FIGURES, MANIFESTS, TABLES

ANIMATE = {"face_person", "animal", "scene_with_people"}
AREAS = ["amygdala", "hippocampus", "dACC", "preSMA", "vmPFC"]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="use the selection-corrected concept_cell_strict criterion")
    args = ap.parse_args()
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    if args.strict:
        sel["concept_cell"] = sel["concept_cell_strict"]
    SUF = "_strict" if args.strict else ""
    sel = sel[sel.task == "screening"]
    lab = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    img = pd.read_csv(MANIFESTS / "images.csv")
    shown = img[(img.task == "screening") & (img.image_uid != "img_null")][["subject", "image_uid"]].drop_duplicates()
    shown["category"] = lab.loc[shown.image_uid, "category"].to_numpy()
    shown["animate"] = shown.category.isin(ANIMATE)
    cc = sel[sel.concept_cell].copy()
    cc["category"] = lab.loc[cc.pref_image, "category"].to_numpy()
    cc["animate"] = cc.category.isin(ANIMATE)

    rows = []
    for area in AREAS + ["MTL", "MFC"]:
        c = cc[cc.area == area] if area in AREAS else cc[cc.region == area]
        subs = sorted(c.subject.unique())
        an, inan = [], []
        for s in subs:
            sh = shown[shown.subject == s]
            cs = c[c.subject == s]
            n_an_img, n_in_img = sh.animate.sum(), (~sh.animate).sum()
            an.append(cs.animate.sum() / max(n_an_img, 1))
            inan.append((~cs.animate).sum() / max(n_in_img, 1))
        an, inan = np.array(an), np.array(inan)
        try:
            wp = sps.wilcoxon(an - inan).pvalue if len(subs) >= 5 else np.nan
        except ValueError:
            wp = np.nan
        rows.append(
            {
                "area": area,
                "n_subjects": len(subs),
                "n_concept_cells": len(c),
                "cells_per_animate_image": an.mean(),
                "cells_per_inanimate_image": inan.mean(),
                "ratio": an.mean() / max(inan.mean(), 1e-9),
                "frac_cells_animate": c.animate.mean() if len(c) else np.nan,
                "frac_images_animate_shown": shown[shown.subject.isin(subs)].animate.mean(),
                "wilcoxon_p": wp,
            }
        )
    an_df = pd.DataFrame(rows)
    an_df.to_csv(TABLES / f"A5_animacy_by_area{SUF}.csv", index=False)
    print(an_df.round(3).to_string(index=False))

    # category × area: cells per shown image of that category
    rows = []
    for area in AREAS:
        c = cc[cc.area == area]
        subs = c.subject.unique()
        sh = shown[shown.subject.isin(subs)]
        for cat in sorted(lab.category.unique()):
            n_img = (sh.category == cat).sum()
            rows.append({"area": area, "category": cat, "n_cells": int((c.category == cat).sum()), "n_images_shown": int(n_img), "cells_per_image": (c.category == cat).sum() / max(n_img, 1)})
    ca = pd.DataFrame(rows)
    ca.to_csv(TABLES / f"A5_category_by_area{SUF}.csv", index=False)
    piv = ca.pivot(index="category", columns="area", values="cells_per_image")[AREAS]
    print(piv.round(3))

    # Mormann-style: fraction of animal-preferring concept cells, amygdala vs hippocampus, R vs L
    m = cc[cc.region == "MTL"].copy()
    m["animal"] = m.category == "animal"
    tab = m.groupby(["area", "hemisphere"]).animal.agg(["mean", "sum", "size"])
    print(tab)
    ct = pd.crosstab(m.area, m.animal)
    print("amygdala vs hippocampus animal-pref Fisher p =", sps.fisher_exact(ct.values)[1])
    amy = m[m.area == "amygdala"]
    ct_h = pd.crosstab(amy.hemisphere, amy.animal).reindex(index=["R", "L"], columns=[True, False], fill_value=0)
    p_rl = sps.fisher_exact(ct_h.values)[1]
    print("right vs left amygdala animal-pref Fisher p =", p_rl, ct_h.values.tolist())
    tab["fisher_p_R_vs_L_amygdala"] = p_rl
    tab.to_csv(TABLES / f"A5_animal_cells_area_hemisphere{SUF}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    d = an_df[an_df.area.isin(AREAS)]
    x = np.arange(len(d))
    axes[0].bar(x - 0.2, d.cells_per_animate_image, 0.4, color="#C44E52", label="animate images")
    axes[0].bar(x + 0.2, d.cells_per_inanimate_image, 0.4, color="#4C72B0", label="inanimate images")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{a}\np={p:.2g}" for a, p in zip(d.area, d.wilcoxon_p)], fontsize=8)
    axes[0].set_ylabel("concept cells per shown image")
    axes[0].set_title("Animate vs inanimate images (screening)")
    axes[0].legend()
    im = axes[1].imshow(piv.values, cmap="magma", aspect="auto")
    axes[1].set_xticks(range(len(AREAS)))
    axes[1].set_xticklabels(AREAS, rotation=30, fontsize=8)
    axes[1].set_yticks(range(len(piv)))
    axes[1].set_yticklabels(piv.index, fontsize=8)
    axes[1].set_title("Concept cells per shown image, by category × area")
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / f"A5_animacy{SUF}.png", dpi=150)


if __name__ == "__main__":
    main()
