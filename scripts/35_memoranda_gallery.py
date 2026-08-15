#!/usr/bin/env python
"""Stage 35 — per-subject gallery of the five memoranda with their neural support.

For each subject: the 5 Sternberg pictures, annotated with the number of screening
concept cells (MTL / all) preferring them, the number of Sternberg concept cells
preferring them, and their CLIP category. Output: results/figures/A2_memoranda_gallery.png,
results/tables/A2_memoranda_support.csv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from memoranda.analysis.tables import neural_image_summary, subject_image_table
from memoranda.paths import FIGURES, MANIFESTS, ROOT, TABLES


def main() -> None:
    st = subject_image_table()
    sb = st[st.in_sternberg].copy()
    ns = neural_image_summary("sternberg").rename(columns={"n_concept_pref": "n_cc_sternberg", "n_concept_pref_mtl": "n_cc_sternberg_mtl"})
    sb = sb.merge(ns[["subject", "image_uid", "n_cc_sternberg", "n_cc_sternberg_mtl"]], on=["subject", "image_uid"], how="left").fillna({"n_cc_sternberg": 0, "n_cc_sternberg_mtl": 0})
    img = pd.read_csv(MANIFESTS / "images.csv")
    paths = img[img.task == "sternberg"].set_index(["subject", "image_uid"]).path
    sb = sb.sort_values(["subject", "sternberg_slot"])
    sb[["subject", "sternberg_slot", "image_uid", "category", "fine", "n_concept_pref", "n_concept_pref_mtl", "n_cc_sternberg", "n_cc_sternberg_mtl", "z_max_mtl", "n_subjects_screening", "n_sternberg_other"]].to_csv(TABLES / "A2_memoranda_support.csv", index=False)

    subjects = sorted(sb.subject.unique())
    fig, axes = plt.subplots(len(subjects), 5, figsize=(9, len(subjects) * 1.55))
    for ax in axes.flat:
        ax.axis("off")
    for i, s in enumerate(subjects):
        for j, (_, r) in enumerate(sb[sb.subject == s].iterrows()):
            ax = axes[i, j]
            ax.imshow(Image.open(ROOT / "data" / paths.loc[(s, r.image_uid)]).convert("RGB"))
            ttl = f"{r.category[:12]}\nscr cc {int(r.n_concept_pref) if pd.notna(r.n_concept_pref) else '–'} ({int(r.n_concept_pref_mtl) if pd.notna(r.n_concept_pref_mtl) else '–'} MTL) · WM cc {int(r.n_cc_sternberg)}"
            ax.set_title(ttl, fontsize=5.5, pad=1.5)
        axes[i, 0].text(-0.08, 0.5, f"sub-{s:02d}", transform=axes[i, 0].transAxes, fontsize=8, ha="right", va="center", fontweight="bold")
    fig.suptitle("Sternberg memoranda per subject — screening concept cells preferring each (all / MTL) and Sternberg concept cells", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / "A2_memoranda_gallery.png", dpi=140)
    print(sb.groupby("subject")[["n_concept_pref", "n_cc_sternberg"]].sum().T)


if __name__ == "__main__":
    main()
