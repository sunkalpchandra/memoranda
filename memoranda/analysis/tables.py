"""Master tables joining images, labels, statistics and neural selectivity.

* :func:`image_table` — one row per unique image (dataset-wide attributes)
* :func:`subject_image_table` — one row per (subject, image) with per-subject
  neural summaries and whether the image became a Sternberg memorandum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..paths import MANIFESTS


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(MANIFESTS / f"{name}.csv")


def image_table() -> pd.DataFrame:
    img = _read("images")
    lab = _read("image_labels")
    stats = _read("image_stats")
    d = img[img.image_uid != "img_null"]
    per = (
        d.groupby("image_uid")
        .agg(
            n_subjects_screening=("subject", lambda s: s[d.loc[s.index, "task"] == "screening"].nunique()),
            n_subjects_sternberg=("subject", lambda s: s[d.loc[s.index, "task"] == "sternberg"].nunique()),
            first_path=("path", "first"),
        )
        .reset_index()
    )
    per["sternberg_rate"] = per.n_subjects_sternberg / per.n_subjects_screening.replace(0, np.nan)
    out = per.merge(lab, on="image_uid", how="left").merge(stats, on="image_uid", how="left")
    faces_csv = MANIFESTS / "faces.csv"
    if faces_csv.exists():
        out = out.merge(pd.read_csv(faces_csv), on="image_uid", how="left")
    return out


def neural_image_summary(task: str = "screening") -> pd.DataFrame:
    """Per (subject, image): #units preferring it, #concept cells preferring it, max z, mean z."""
    sel = _read("unit_selectivity")
    tun = _read("unit_tuning")
    sel = sel[sel.task == task]
    tun = tun[tun.task == task]
    cc = sel[sel.concept_cell]
    pref_cc = cc.groupby(["subject", "pref_image"]).size().rename("n_concept_pref").reset_index().rename(columns={"pref_image": "image_uid"})
    pref_cc_mtl = (
        cc[cc.region == "MTL"].groupby(["subject", "pref_image"]).size().rename("n_concept_pref_mtl").reset_index().rename(columns={"pref_image": "image_uid"})
    )
    pref_any = sel.groupby(["subject", "pref_image"]).size().rename("n_units_pref").reset_index().rename(columns={"pref_image": "image_uid"})
    z = tun.groupby(["subject", "image_uid"]).agg(z_max=("z_base", "max"), z_mean=("z_base", "mean"), rate_mean=("rate", "mean")).reset_index()
    zm = tun[tun.region == "MTL"].groupby(["subject", "image_uid"]).agg(z_max_mtl=("z_base", "max"), z_mean_mtl=("z_base", "mean")).reset_index()
    out = z.merge(zm, how="left").merge(pref_any, how="left").merge(pref_cc, how="left").merge(pref_cc_mtl, how="left")
    for c in ("n_units_pref", "n_concept_pref", "n_concept_pref_mtl"):
        out[c] = out[c].fillna(0).astype(int)
    return out


def subject_image_table() -> pd.DataFrame:
    img = _read("images")
    d = img[img.image_uid != "img_null"]
    scr = d[d.task == "screening"][["subject", "image_uid", "stim_name", "stim_index"]].drop_duplicates(["subject", "image_uid"])
    stb = d[d.task == "sternberg"][["subject", "image_uid", "stim_index"]].rename(columns={"stim_index": "sternberg_slot"})
    t = scr.merge(stb, on=["subject", "image_uid"], how="outer")
    t["in_screening"] = t.stim_index.notna()
    t["in_sternberg"] = t.sternberg_slot.notna()
    t = t.merge(neural_image_summary("screening"), on=["subject", "image_uid"], how="left")
    t = t.merge(image_table().drop(columns=["first_path"]), on="image_uid", how="left")
    # leave-one-subject-out reuse: how often did *other* subjects turn this image into a memorandum?
    n_scr_other = t.n_subjects_screening - t.in_screening.astype(int)
    n_stb_other = t.n_subjects_sternberg - t.in_sternberg.astype(int)
    t["sternberg_rate_loso"] = n_stb_other / n_scr_other.replace(0, np.nan)
    t["n_sternberg_other"] = n_stb_other
    return t
