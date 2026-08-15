#!/usr/bin/env python
"""Stage 12 — CLIP zero-shot category / attribute / fine labels for every unique image.

Writes ``data/manifests/image_labels.csv`` and saves the normalised CLIP image
embedding used for labelling to ``data/features/clip_vitb32_embed.npz``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from conceptlens.features import unique_image_table
from conceptlens.models.zeroshot import ClipZeroShot, label_images, load_taxonomy
from conceptlens.paths import FEATURES, MANIFESTS


def main() -> None:
    table = unique_image_table()
    images = [Image.open(p).convert("RGB") for p in table["abs_path"]]
    zs = ClipZeroShot()
    emb = zs.image_embed(images)
    np.savez_compressed(FEATURES / "clip_vitb32_embed.npz", image_uid=table["image_uid"].to_numpy(), embed=emb)
    labels = label_images(zs, emb, load_taxonomy())
    df = pd.DataFrame({"image_uid": table["image_uid"], **labels})
    df.to_csv(MANIFESTS / "image_labels.csv", index=False)
    print(df.category.value_counts())
    print(df.fine.value_counts().head(25))


if __name__ == "__main__":
    main()
