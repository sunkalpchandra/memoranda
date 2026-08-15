"""Assign a dataset-wide ``image_uid`` to every stored template.

Two templates are the *same image* if their pixel sha1 matches (exact) or
their dhash Hamming distance is ≤ ``NEAR_THRESHOLD`` (near-duplicate, e.g.
the same JPEG re-saved). Union–find groups them; the uid is ``img_%04d`` in
order of first appearance (subject, session, stim_index).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .images import hamming

NEAR_THRESHOLD = 6  # bits out of 64


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign_uids(df: pd.DataFrame, near_threshold: int = NEAR_THRESHOLD) -> pd.DataFrame:
    """Return a copy of the image manifest with ``image_uid`` and ``dup_kind`` columns.

    ``dup_kind`` is 'exact' when the row shares a sha1 with an earlier row,
    'near' when linked only through dhash, else 'unique'.
    """
    df = df.sort_values(["subject", "session", "stim_index"]).reset_index(drop=True)
    n = len(df)
    uf = _UF(n)
    kind = np.array(["unique"] * n, dtype=object)

    # exact
    first_by_sha: dict[str, int] = {}
    for i, sha in enumerate(df["sha1"]):
        if sha in first_by_sha:
            uf.union(first_by_sha[sha], i)
            kind[i] = "exact"
        else:
            first_by_sha[sha] = i

    # near (only among non-null images)
    idx = np.where(~df["is_null"].values)[0]
    hashes = df["dhash"].values
    for a_pos, i in enumerate(idx):
        for j in idx[a_pos + 1 :]:
            if uf.find(i) == uf.find(j):
                continue
            if hamming(hashes[i], hashes[j]) <= near_threshold:
                uf.union(i, j)
                if kind[j] == "unique":
                    kind[j] = "near"

    roots = np.array([uf.find(i) for i in range(n)])
    order = {r: k for k, r in enumerate(pd.unique(roots))}
    out = df.copy()
    out["image_uid"] = [f"img_{order[r]:04d}" for r in roots]
    out["dup_kind"] = kind
    # the null placeholder gets a reserved uid
    out.loc[out["is_null"], "image_uid"] = "img_null"
    return out


def unique_images(df: pd.DataFrame) -> pd.DataFrame:
    """One row per uid (first appearance), excluding the null image."""
    d = df[df["image_uid"] != "img_null"]
    return d.drop_duplicates("image_uid").reset_index(drop=True)
