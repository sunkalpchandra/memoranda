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
    "alexnet": ["conv1", "conv3", "conv5", "fc6"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4", "avgpool"],
    "clip_vitb32": ["block2", "block5", "block8", "block11", "ln_post"],
}
COL = {"alexnet": "#4C72B0", "resnet50": "#55A868", "clip_vitb32": "#C44E52"}


def main() -> None:
    parts = [pd.read_csv(TABLES / "A4_encoding_null_summary.csv")]
    depth = TABLES / "A4_encoding_null_summary_depth.csv"
    if depth.exists():
        parts.append(pd.read_csv(depth))
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


if __name__ == "__main__":
    main()
