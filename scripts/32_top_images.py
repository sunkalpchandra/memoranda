#!/usr/bin/env python
"""Stage 32 — the most and least "neuron-driving" pictures across patients.

Per unique image (shown to ≥3 subjects in screening): concept cells preferring
it summed over subjects, divided by the number of subjects who saw it. Montages
of the top and bottom 24, plus a per-image table used by later analyses.
Output: results/tables/A5_image_neural_score.csv, results/figures/A5_top_bottom_images.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from memoranda.analysis.tables import subject_image_table
from memoranda.paths import FIGURES, ROOT, TABLES


def main() -> None:
    st = subject_image_table()
    scr = st[st.in_screening]
    g = scr.groupby("image_uid").agg(
        n_subjects=("subject", "nunique"),
        concept_pref_total=("n_concept_pref", "sum"),
        concept_pref_mtl_total=("n_concept_pref_mtl", "sum"),
        z_max_mean=("z_max", "mean"),
        z_max_mtl_mean=("z_max_mtl", "mean"),
        n_sternberg=("in_sternberg", "sum"),
        category=("category", "first"),
        fine=("fine", "first"),
    )
    g["concept_per_subject"] = g.concept_pref_total / g.n_subjects
    g["concept_mtl_per_subject"] = g.concept_pref_mtl_total / g.n_subjects
    g["frac_subjects_with_concept"] = scr.assign(hit=scr.n_concept_pref > 0).groupby("image_uid").hit.mean()
    g = g.reset_index().sort_values("concept_per_subject", ascending=False)
    g.to_csv(TABLES / "A5_image_neural_score.csv", index=False)
    paths = st.drop_duplicates("image_uid").set_index("image_uid")
    img = pd.read_csv(ROOT / "data/manifests/images.csv").drop_duplicates("image_uid").set_index("image_uid")

    elig = g[g.n_subjects >= 3]
    top, bot = elig.head(24), elig.tail(24)
    fig, axes = plt.subplots(6, 8, figsize=(16, 12.5))
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(top.iterrows()):
        ax = axes[k // 8, k % 8]
        ax.imshow(Image.open(ROOT / "data" / img.loc[r.image_uid, "path"]).convert("RGB"))
        ax.set_title(f"{r.image_uid[-4:]} · {r.fine[:16]}\n{r.concept_pref_total:.0f} cc / {r.n_subjects} subj = {r.concept_per_subject:.2f}", fontsize=6.5)
    for k, (_, r) in enumerate(bot.iterrows()):
        ax = axes[3 + k // 8, k % 8]
        ax.imshow(Image.open(ROOT / "data" / img.loc[r.image_uid, "path"]).convert("RGB"))
        ax.set_title(f"{r.image_uid[-4:]} · {r.fine[:16]}\n{r.concept_pref_total:.0f} cc / {r.n_subjects} subj", fontsize=6.5)
    axes[0, 0].text(0, 1.45, "TOP 24: most concept cells per patient shown", transform=axes[0, 0].transAxes, fontsize=12, fontweight="bold")
    axes[3, 0].text(0, 1.45, "BOTTOM 24: fewest", transform=axes[3, 0].transAxes, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_top_bottom_images.png", dpi=130)
    print(top[["image_uid", "fine", "n_subjects", "concept_pref_total", "concept_per_subject", "n_sternberg"]].to_string(index=False))
    print(elig.groupby("category").concept_per_subject.agg(["mean", "count"]).sort_values("mean", ascending=False))


if __name__ == "__main__":
    main()
