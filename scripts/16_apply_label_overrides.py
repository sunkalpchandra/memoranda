#!/usr/bin/env python
"""Stage 16 — apply hand-checked category overrides to image_labels.csv.

Keeps CLIP's original decision in ``category_clip`` and writes the corrected label to
``category`` (which every downstream analysis reads). Idempotent.
"""

from __future__ import annotations

import pandas as pd

from memoranda.paths import CONFIGS, MANIFESTS


def main() -> None:
    lab = pd.read_csv(MANIFESTS / "image_labels.csv")
    if "category_clip" not in lab:
        lab.insert(lab.columns.get_loc("category") + 1, "category_clip", lab["category"])
    ov = pd.read_csv(CONFIGS / "label_overrides.csv")
    m = lab.set_index("image_uid")
    changed = 0
    for r in ov.itertuples():
        if r.image_uid in m.index and m.loc[r.image_uid, "category"] != r.category:
            m.loc[r.image_uid, "category"] = r.category
            changed += 1
    m["category_overridden"] = m["category"] != m["category_clip"]
    m.reset_index().to_csv(MANIFESTS / "image_labels.csv", index=False)
    print(f"applied {changed} overrides; {int(m.category_overridden.sum())} images differ from CLIP")
    print(m.category.value_counts())


if __name__ == "__main__":
    main()
