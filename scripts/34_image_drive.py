#!/usr/bin/env python
"""Stage 34 — what predicts how strongly an image drives the MTL population overall?

Per (subject, image): mean baseline-normalised response over MTL units (z_mean_mtl) and the
maximum (z_max_mtl). We average over the subjects who saw each image, then relate this
image-level "drive" to image descriptors: category, CLIP attributes, low-level statistics,
DNN geometry (dataset-wide typicality / isolation), and CLIP-space nearest-neighbour
distance. Spearman correlations with BH-FDR, plus a category-adjusted (residualised) version.
Output: results/tables/A5_image_drive_correlates.csv, results/figures/A5_image_drive.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as sps

from memoranda import imstats
from memoranda.analysis import stats as S
from memoranda.paths import FIGURES, TABLES

ATTRS = ["famous", "face_visible", "single_object", "indoor", "natural", "text_present", "colorful", "emotional", "child", "smiling"]


def main() -> None:
    st = pd.read_csv(TABLES / "subject_image_master.csv")
    scr = st[st.in_screening & st.z_mean_mtl.notna()]
    # within-subject z-score the drive so subjects with many units don't dominate
    scr = scr.assign(drive=scr.groupby("subject").z_mean_mtl.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)))
    scr = scr.assign(drive_max=scr.groupby("subject").z_max_mtl.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9)))
    per_img = scr.groupby("image_uid").agg(drive=("drive", "mean"), drive_max=("drive_max", "mean"), n_subj=("subject", "nunique")).reset_index()
    feat_cols = [f"attr_{a}" for a in ATTRS] + imstats.STAT_NAMES[:-1] + [c for c in st.columns if c.startswith(("clip_", "resnet50_", "dinov2_", "alexnet_fc7_")) and not c.startswith(("clip_ws", "resnet50_ws"))]
    img = st.drop_duplicates("image_uid").set_index("image_uid")
    per_img = per_img.merge(img[feat_cols + ["category"]], left_on="image_uid", right_index=True)
    per_img = per_img[per_img.n_subj >= 2]
    per_img.to_csv(TABLES / "A5_image_drive.csv", index=False)

    # category-adjusted drive
    cat_means = per_img.groupby("category").drive.transform("mean")
    per_img["drive_resid"] = per_img.drive - cat_means
    rows = []
    for c in feat_cols:
        r, p = S.spearman(per_img[c], per_img.drive)
        r2, p2 = S.spearman(per_img[c], per_img.drive_resid)
        rows.append({"feature": c, "rho": r, "p": p, "rho_cat_adjusted": r2, "p_cat_adjusted": p2})
    res = pd.DataFrame(rows)
    res["q"] = S.fdr_bh(res.p.to_numpy())[1]
    res["q_cat_adjusted"] = S.fdr_bh(res.p_cat_adjusted.to_numpy())[1]
    res = res.sort_values("p")
    res.to_csv(TABLES / "A5_image_drive_correlates.csv", index=False)
    print(res.head(20).round(4).to_string(index=False))
    kw = sps.kruskal(*[g.drive.to_numpy() for _, g in per_img.groupby("category")])
    print("Kruskal–Wallis drive ~ category:", kw)
    print(per_img.groupby("category").drive.agg(["mean", "count"]).sort_values("mean", ascending=False).round(3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = per_img.groupby("category").drive.mean().sort_values().index
    per_img.boxplot(column="drive", by="category", ax=axes[0], positions=range(len(order)), grid=False)
    axes[0].set_xticklabels(order, rotation=40, ha="right", fontsize=8)
    axes[0].set_title(f"MTL population drive by category (KW p={kw.pvalue:.3g})")
    axes[0].set_ylabel("mean z (within-subject standardised)")
    top = res.head(15).sort_values("rho")
    axes[1].barh(top.feature, top.rho, color=["#C44E52" if q < 0.05 else "#8da0cb" for q in top.q])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_title("Image descriptors vs MTL drive (Spearman; red q<0.05)")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_image_drive.png", dpi=150)


if __name__ == "__main__":
    main()
