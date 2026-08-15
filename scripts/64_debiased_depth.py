#!/usr/bin/env python
"""Stage 64 — debiased encoding performance vs layer depth (from the shuffle nulls, scripts 51).

Combines A4_encoding_null_summary.csv (core set) and A4_encoding_null_summary_depth.csv
(early→late layers of AlexNet, ResNet-50, CLIP + low-level) into one figure per region.
Output: results/figures/A4_debiased_depth.png, results/tables/A4_debiased_all.csv
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from memoranda.paths import FIGURES, TABLES

ORDER = {
    "alexnet": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "logits"],
    "vgg16": ["conv3_3", "conv5_3", "fc6", "fc7"],
    "resnet18": ["layer2", "layer4", "logits"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
    "convnext_tiny": ["stage2", "stage4", "logits"],
    "vit_b_16": ["block2", "block8", "ln"],
    "dinov2_small": ["block2", "block5", "block8", "norm"],
    "clip_vitb32": ["block2", "block5", "block8", "block11", "ln_post", "embed"],
    "clip_rn50": ["layer2", "layer4", "attnpool"],
}
COL = {"alexnet": "#4C72B0", "vgg16": "#8da0cb", "resnet18": "#a1d99b", "resnet50": "#55A868", "convnext_tiny": "#7f7f7f", "vit_b_16": "#8172B2", "dinov2_small": "#DD8452", "clip_vitb32": "#C44E52", "clip_rn50": "#e7969c"}


def main() -> None:
    parts = [pd.read_csv(TABLES / "A4_encoding_null_summary.csv")]
    for extra in ("A4_encoding_null_summary_depth.csv", "A4_encoding_null_summary_wide.csv"):
        if (TABLES / extra).exists():
            parts.append(pd.read_csv(TABLES / extra))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["region", "model", "layer"])
    df.to_csv(TABLES / "A4_debiased_all.csv", index=False)
    print(df[df.region == "MTL"].sort_values(["model", "layer"]).round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, region in zip(axes, ("MTL", "MFC")):
        d = df[df.region == region]
        for m, lay in ORDER.items():
            g = d[d.model == m].set_index("layer").reindex(lay).dropna(subset=["r_debiased"])
            if not len(g):
                continue
            ax.errorbar(range(len(g)), g.r_debiased, g.r_debiased_sem, fmt="o-", color=COL[m], label=m, capsize=2)
            for i, (lname, row) in enumerate(g.iterrows()):
                ax.text(i, row.r_debiased + 0.012, lname, ha="center", fontsize=6, color=COL[m])
        for base, ls in (("category", "--"), ("lowlevel", ":")):
            b = d[(d.model == "baseline") & (d.layer == base)]
            if len(b):
                ax.axhline(b.r_debiased.iloc[0], color="grey", ls=ls, lw=1, label=f"{base} baseline")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{region} concept cells: shuffle-debiased encoding r")
        ax.set_xlabel("layer (early → late)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CV r − mean(shuffle null)  (mean ± sem over cells)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_debiased_depth.png", dpi=150)

    # primary metric after the review: fraction of cells individually significant vs their own shuffle null
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, region in zip(axes, ("MTL", "MFC")):
        d = df[df.region == region]
        for m, lay in ORDER.items():
            g = d[d.model == m].set_index("layer").reindex(lay).dropna(subset=["frac_sig"])
            if len(g) < 2:
                continue
            ax.plot(range(len(g)), g.frac_sig * 100, "o-", color=COL[m], label=m, ms=4)
        for base, ls in (("category", "--"), ("lowlevel", ":")):
            b = d[(d.model == "baseline") & (d.layer == base)]
            if len(b):
                ax.axhline(b.frac_sig.iloc[0] * 100, color="grey", ls=ls, lw=1, label=f"{base} baseline")
        ax.axhline(5, color="k", lw=0.6, ls="-.")
        ax.set_title(f"{region} concept cells: % cells with p < 0.05 vs own shuffle null")
        ax.set_xlabel("layer (early → late, per model)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("% of cells individually significant")
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_fracsig_depth.png", dpi=150)


if __name__ == "__main__":
    main()
