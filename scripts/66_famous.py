#!/usr/bin/env python
"""Stage 66 — hand-labelled famous people: do concept cells (and the Sternberg selection) care?

Inputs: configs/famous_labels.csv (hand labels for all 342 pictures: famous, identity,
fictional_character, n_people), data/manifests/{images,image_labels,faces,unit_selectivity}.csv.

(a) per-image table joining the hand labels with category / CLIP-famous / MTCNN face columns
(b) MTL screening concept cells: is the preferred picture a famous person more often than the
    person pictures shown to that cell (Fisher exact on pooled counts, Mann–Whitney AUC, and a
    per-cell permutation null that redraws a random person picture from the cell's own screening
    set)?  By area (amygdala / hippocampus) and MFC as a reference.  Two further contrasts use all
    pictures (famous vs everything else; person vs non-person) so the famous effect can be read
    against the plain person/face effect.
(c) Sternberg memoranda vs the rest of each subject's screening set: famous enrichment
    (Fisher exact pooled, mean within-subject AUC), overall and among person pictures.

Outputs: results/tables/A5_famous_preference.csv, results/tables/A5_famous_memoranda.csv,
         results/figures/A5_famous.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from memoranda.analysis import stats as S
from memoranda.analysis.tables import subject_image_table
from memoranda.paths import CONFIGS, FIGURES, MANIFESTS, TABLES

AREA_GROUPS = {
    "MTL": ("region", ["MTL"]),
    "amygdala": ("area", ["amygdala"]),
    "hippocampus": ("area", ["hippocampus"]),
    "MFC": ("region", ["MFC"]),
}
# contrast -> (scope of pictures entering the test, hit column)
CONTRASTS = {
    "famous vs non-famous person": ("person", "famous"),  # restricted to n_people > 0
    "famous vs all other pictures": ("all", "famous"),
    "person vs non-person picture": ("all", "person"),
}
N_PERM = 10000


def image_table() -> pd.DataFrame:
    fam = pd.read_csv(CONFIGS / "famous_labels.csv")
    fam["identity"] = fam.identity.fillna("")
    fam["notes"] = fam.notes.fillna("")
    lab = pd.read_csv(MANIFESTS / "image_labels.csv")[["image_uid", "category", "attr_famous", "attr_face_visible"]]
    faces = pd.read_csv(MANIFESTS / "faces.csv")[["image_uid", "face_found", "n_faces"]]
    it = fam.merge(lab, on="image_uid", how="left").merge(faces, on="image_uid", how="left")
    it["person"] = (it.n_people > 0).astype(int)
    it["famous"] = it.famous.astype(int)
    it["fictional_character"] = it.fictional_character.astype(int)
    return it


def preference(it: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    img = pd.read_csv(MANIFESTS / "images.csv")
    lut = it.set_index("image_uid")
    shown_all = img[(img.task == "screening") & (img.image_uid != "img_null")][["subject", "image_uid"]].drop_duplicates()
    shown_all = shown_all.merge(it[["image_uid", "famous", "person"]], on="image_uid", how="left")
    cc = sel[(sel.task == "screening") & sel.concept_cell].copy()
    cc["famous"] = lut.loc[cc.pref_image, "famous"].to_numpy()
    cc["person"] = lut.loc[cc.pref_image, "person"].to_numpy()

    rows = []
    for contrast, (scope, hit) in CONTRASTS.items():
        for area_name, (col, vals) in AREA_GROUPS.items():
            c = cc[cc[col].isin(vals)]
            shown = shown_all
            if scope == "person":
                c = c[c.person == 1]
                shown = shown_all[shown_all.person == 1]
            if len(c) < 5:
                continue
            shown_c = shown[shown.subject.isin(c.subject.unique())]
            a = int(c[hit].sum())
            b = len(c) - a
            cf = int(shown_c[hit].sum())
            d = len(shown_c) - cf
            orat, p_fisher = fisher_exact([[a, b], [cf, d]])
            x = c[hit].to_numpy(float)
            y = shown_c[hit].to_numpy(float)
            _, p_mw = S.mannwhitney(x, y)
            # per-cell expected fraction from the cell's own session composition + permutation null
            comp = shown.groupby("subject")[hit].mean()
            exp_cell = comp.reindex(c.subject).to_numpy(float)
            pools = {s: g[hit].to_numpy(int) for s, g in shown.groupby("subject")}
            null = np.zeros(N_PERM, int)
            for s, n_s in c.subject.value_counts().items():
                pool = pools[s]
                null += rng.choice(pool, size=(N_PERM, n_s), replace=True).sum(1)
            obs = a
            p_perm = float((np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean())) + 1) / (N_PERM + 1))
            rows.append(
                {
                    "contrast": contrast,
                    "area": area_name,
                    "n_cells": len(c),
                    "n_subjects": c.subject.nunique(),
                    "hit": hit,
                    "n_pref_hit": a,
                    "frac_pref_hit": a / len(c),
                    "frac_shown_hit_pooled": cf / max(len(shown_c), 1),
                    "frac_expected_per_cell": float(np.nanmean(exp_cell)),
                    "excess_per_cell": float(np.nanmean(x - exp_cell)),
                    "odds_ratio": float(orat),
                    "p_fisher": float(p_fisher),
                    "auc": S.auc_effect(x, y),
                    "p_mannwhitney": p_mw,
                    "null_mean_hit": float(null.mean()),
                    "null_sd_hit": float(null.std()),
                    "p_perm": p_perm,
                }
            )
    df = pd.DataFrame(rows)
    df["q_fisher"] = np.nan
    for contrast, g in df.groupby("contrast"):
        df.loc[g.index, "q_fisher"] = S.fdr_bh(g.p_fisher.to_numpy())[1]
    return df


def memoranda(it: pd.DataFrame) -> pd.DataFrame:
    st = subject_image_table()
    st = st.merge(it[["image_uid", "famous", "person", "fictional_character"]], on="image_uid", how="left")
    scr = st[st.in_screening].copy()
    rows = []
    for name, sub, col in (
        ("famous vs rest (all pictures)", scr, "famous"),
        ("famous vs non-famous person (n_people>0)", scr[scr.person == 1], "famous"),
        ("person (n_people>0) vs rest", scr, "person"),
        ("fictional/costumed character vs rest", scr, "fictional_character"),
    ):
        hit = sub[col].to_numpy(bool)
        grp = sub.in_sternberg.to_numpy(bool)
        orat, p, a, b, c, d = S.fisher_enrichment(grp, hit)
        wauc = []
        for _, g in sub.groupby("subject"):
            if g.in_sternberg.sum() >= 2 and (~g.in_sternberg).sum() >= 5:
                wauc.append(S.auc_effect(g.loc[g.in_sternberg, col], g.loc[~g.in_sternberg, col]))
        _, p_mw = S.mannwhitney(sub.loc[sub.in_sternberg, col], sub.loc[~sub.in_sternberg, col])
        rows.append(
            {
                "contrast": name,
                "n_subjects": sub.subject.nunique(),
                "n_memoranda": int(a + b),
                "n_rest": int(c + d),
                "n_memoranda_hit": int(a),
                "frac_memoranda": a / max(a + b, 1),
                "frac_rest": c / max(c + d, 1),
                "odds_ratio": orat,
                "p_fisher": p,
                "auc_pooled": S.auc_effect(sub.loc[sub.in_sternberg, col], sub.loc[~sub.in_sternberg, col]),
                "auc_within_subject_mean": float(np.nanmean(wauc)) if wauc else np.nan,
                "auc_within_subject_sem": float(np.nanstd(wauc) / np.sqrt(len(wauc))) if wauc else np.nan,
                "n_subjects_within": len(wauc),
                "p_mannwhitney": p_mw,
            }
        )
    df = pd.DataFrame(rows)
    df["q_fisher"] = S.fdr_bh(df.p_fisher.to_numpy())[1]
    return df, scr


def figure(pref: pd.DataFrame, mem: pd.DataFrame, scr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1]})
    for ax, contrast in zip(axes[:3], CONTRASTS):
        d = pref[pref.contrast == contrast].set_index("area").reindex(list(AREA_GROUPS)).dropna(subset=["n_cells"])
        y = np.arange(len(d))
        ax.barh(y - 0.2, d.frac_expected_per_cell, 0.4, color="#bbbbbb", label="shown to the cell (expected)")
        ax.barh(y + 0.2, d.frac_pref_hit, 0.4, color=["#C44E52" if p < 0.05 else "#4C72B0" for p in d.p_perm], label="preferred picture")
        for yi, (_, r) in zip(y, d.iterrows()):
            ax.text(max(r.frac_pref_hit, r.frac_expected_per_cell) + 0.01, yi, f"n={int(r.n_cells)}  p={r.p_perm:.3f}" if r.p_perm >= 0.001 else f"n={int(r.n_cells)}  p<0.001", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(d.index)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel(f"fraction {d.hit.iloc[0]}")
        ax.set_title(f"Concept cells (screening): {contrast}\nred: permutation p<0.05", fontsize=10)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    ax = axes[3]
    d = mem.set_index("contrast")
    labels = ["famous vs rest (all pictures)", "famous vs non-famous person (n_people>0)"]
    y = np.arange(len(labels))
    ax.barh(y - 0.2, d.loc[labels, "frac_rest"], 0.4, color="#bbbbbb", label="rest of screening set")
    ax.barh(y + 0.2, d.loc[labels, "frac_memoranda"], 0.4, color=["#C44E52" if q < 0.05 else "#4C72B0" for q in d.loc[labels, "q_fisher"]], label="Sternberg memoranda")
    for yi, lab in zip(y, labels):
        r = d.loc[lab]
        ax.text(max(r.frac_memoranda, r.frac_rest) + 0.01, yi, f"OR={r.odds_ratio:.2f} p={r.p_fisher:.2f}\nwithin-subj AUC={r.auc_within_subject_mean:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(["all pictures", "person pictures"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction famous")
    ax.set_title("Memoranda vs rest of screening set", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_famous.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(0)
    it = image_table()
    n_fam = int(it.famous.sum())
    n_person = int(it.person.sum())
    print(f"{len(it)} pictures: {n_person} with a person, {n_fam} famous, {it.fictional_character.sum()} fictional/costumed, {it.identity.ne('').sum()} identities named")
    print("famous by category:")
    print(pd.crosstab(it.category, it.famous))
    print(f"CLIP attr_famous AUC (hand famous vs not, person pictures): {S.auc_effect(it.loc[(it.person == 1) & (it.famous == 1), 'attr_famous'], it.loc[(it.person == 1) & (it.famous == 0), 'attr_famous']):.3f}")

    pref = preference(it, rng)
    pref.to_csv(TABLES / "A5_famous_preference.csv", index=False)
    print(pref.round(3).to_string(index=False))

    mem, scr = memoranda(it)
    mem.to_csv(TABLES / "A5_famous_memoranda.csv", index=False)
    print(mem.round(3).to_string(index=False))

    figure(pref, mem, scr)


if __name__ == "__main__":
    main()
