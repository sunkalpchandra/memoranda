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
    ncol = 12
    for c in cats:
        sub = tab[tab.category == c].sort_values("category_p", ascending=False)
        nrow = -(-len(sub) // ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.6, nrow * 1.55), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for k, (_, row) in enumerate(sub.iterrows()):
            ax = axes[k // ncol, k % ncol]
            im = Image.open(row.abs_path).convert("RGB")
            im.thumbnail((160, 160))
            ax.imshow(im)
            ax.set_title(f"{row.image_uid[-4:]}  p={row.category_p:.2f}\n{row.fine[:22]}", fontsize=6, pad=1.5)
        fig.suptitle(f"{c}  (n={len(sub)})", fontsize=11, fontweight="bold")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.9 if nrow > 2 else 0.8, bottom=0.01, hspace=0.45, wspace=0.05)
        fig.savefig(FIGURES / f"labels_{c}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
