#!/usr/bin/env python
"""Stage 04 — contact sheets of the unique stimulus pool and the Sternberg sets.

Writes results/figures/montage_all_unique.png (every unique image, labelled by
uid) and results/figures/montage_sternberg_by_subject.png (5 memoranda per
subject, one row per subject).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from conceptlens.dedup import unique_images
from conceptlens.paths import FIGURES, MANIFESTS, ROOT


def montage_unique(df: pd.DataFrame, ncol: int = 18, thumb: int = 96) -> None:
    u = unique_images(df)
    n = len(u)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.1, nrow * 0.95))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, u.iterrows()):
        im = Image.open(ROOT / "data" / r["path"]).convert("RGB")
        im.thumbnail((thumb, thumb))
        ax.imshow(im)
        ax.set_title(r["image_uid"].replace("img_", ""), fontsize=5, pad=1)
    fig.suptitle(f"{n} unique stimulus images (DANDI 000469)", fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURES / "montage_all_unique.png", dpi=150)
    plt.close(fig)


def montage_sternberg(df: pd.DataFrame, thumb: int = 128) -> None:
    sb = df[(df.task == "sternberg") & (~df.is_null)].sort_values(["subject", "stim_index"])
    subjects = sorted(sb.subject.unique())
    fig, axes = plt.subplots(len(subjects), 5, figsize=(5 * 1.4, len(subjects) * 1.1))
    for ax in axes.flat:
        ax.axis("off")
    for i, s in enumerate(subjects):
        rows = sb[sb.subject == s]
        for j, (_, r) in enumerate(rows.iterrows()):
            im = Image.open(ROOT / "data" / r["path"]).convert("RGB")
            im.thumbnail((thumb, thumb))
            axes[i, j].imshow(im)
            axes[i, j].set_title(f"sub-{s} {r['stim_name']} ({r['image_uid'][-4:]})", fontsize=5, pad=1)
    fig.suptitle("Sternberg memoranda (5 per subject)", fontsize=9)
    fig.tight_layout(pad=0.1)
    fig.savefig(FIGURES / "montage_sternberg_by_subject.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(MANIFESTS / "images.csv")
    montage_unique(df)
    montage_sternberg(df)
    print("wrote montages")
