#!/usr/bin/env python
"""Stage 13 — visual QA sheet of zero-shot labels: images grouped by category."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from memoranda.features import unique_image_table
from memoranda.paths import FIGURES, MANIFESTS


def main() -> None:
    lab = pd.read_csv(MANIFESTS / "image_labels.csv")
    tab = unique_image_table().merge(lab, on="image_uid")
    cats = tab.category.value_counts().index.tolist()
    ncol = 14
    nrows = sum(-(-((tab.category == c).sum()) // ncol) for c in cats)
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.0, nrows * 0.95))
    for ax in axes.flat:
        ax.axis("off")
    r = 0
    for c in cats:
        sub = tab[tab.category == c].sort_values("category_p", ascending=False)
        for k, (_, row) in enumerate(sub.iterrows()):
            ax = axes[r + k // ncol, k % ncol]
            im = Image.open(row.abs_path).convert("RGB")
            im.thumbnail((96, 96))
            ax.imshow(im)
            ax.set_title(f"{row.image_uid[-4:]} {row.category_p:.2f}\n{row.fine[:18]}", fontsize=4.5, pad=1)
        axes[r, 0].text(-0.1, 1.35, f"{c} (n={len(sub)})", transform=axes[r, 0].transAxes, fontsize=8, fontweight="bold")
        r += -(-len(sub) // ncol)
    fig.tight_layout(pad=0.15)
    fig.savefig(FIGURES / "labels_by_category.png", dpi=140)


if __name__ == "__main__":
    main()
