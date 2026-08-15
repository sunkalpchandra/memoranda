#!/usr/bin/env python
"""Stage 11 — low-level image statistics for every unique image.

Writes ``data/manifests/image_stats.csv`` (one row per image_uid).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from memoranda import imstats
from memoranda.features import unique_image_table
from memoranda.paths import MANIFESTS


def main() -> None:
    table = unique_image_table()
    rows = []
    for _, r in tqdm(table.iterrows(), total=len(table)):
        arr = np.asarray(Image.open(r["abs_path"]).convert("RGB"))
        s = imstats.compute_all(arr)
        s["image_uid"] = r["image_uid"]
        rows.append(s)
    df = pd.DataFrame(rows)[["image_uid", *imstats.STAT_NAMES]]
    df.to_csv(MANIFESTS / "image_stats.csv", index=False)
    print(df.describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
