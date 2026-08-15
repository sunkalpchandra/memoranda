#!/usr/bin/env python
"""Stage 30 — what was shown, and what makes an image become a Sternberg memorandum?

Analyses
  A1  composition of the unique pool and of per-subject screening sets (categories,
      attributes, low-level statistics)
  A2  Sternberg-5 vs. the rest of each subject's screening set:
        * category enrichment (Fisher exact, BH-FDR)
        * attribute / low-level / DNN-geometry contrasts (Mann–Whitney AUC, BH-FDR)
        * did the selection track our replicated neural selectivity? (rank of memoranda by
          number of concept cells preferring them)
Outputs: results/tables/A1_*.csv, results/tables/A2_*.csv, results/figures/A1_*.png, A2_*.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memoranda import imstats
from memoranda.analysis import geometry as G
from memoranda.analysis import stats as S
from memoranda.analysis.tables import image_table, subject_image_table
from memoranda.features import load_features
from memoranda.paths import FEATURES, FIGURES, TABLES

ATTRS = ["famous", "face_visible", "single_object", "indoor", "natural", "text_present", "colorful", "emotional", "child", "smiling"]
GEOM_SPACES = {
    "clip": ("clip_vitb32", "embed", "gap"),
    "resnet50": ("resnet50", "avgpool", "gap"),
    "dinov2": ("dinov2_small", "norm", "cls"),
    "alexnet_fc7": ("alexnet", "fc7", "gap"),
}


def _load_space(name: str, uids: np.ndarray) -> np.ndarray:
    model, layer, view = GEOM_SPACES[name]
    if model == "clip_vitb32" and layer == "embed":
        z = np.load(FEATURES / "clip_vitb32_embed.npz")
        X, stored = z["embed"], z["image_uid"]
    else:
        X, stored = load_features(model, layer, view)
    pos = {u: i for i, u in enumerate(stored)}
    return X[[pos[u] for u in uids]]


def a1_composition(it: pd.DataFrame, st: pd.DataFrame) -> None:
    comp = it.category.value_counts().rename("n_unique").to_frame()
    # presentation-weighted (each subject's screening set counts once per image)
    scr = st[st.in_screening]
    comp["n_subject_image_pairs"] = scr.category.value_counts()
    comp["frac_unique"] = comp.n_unique / comp.n_unique.sum()
    comp["frac_pairs"] = comp.n_subject_image_pairs / comp.n_subject_image_pairs.sum()
    comp.to_csv(TABLES / "A1_category_composition.csv")

    per_subj = scr.groupby(["subject", "category"]).size().unstack(fill_value=0)
    per_subj.to_csv(TABLES / "A1_category_by_subject.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 2]})
    comp.n_unique.sort_values().plot.barh(ax=axes[0], color="#4C72B0")
    axes[0].set_xlabel("unique images")
    axes[0].set_title("Stimulus pool composition (CLIP zero-shot)")
    (per_subj.div(per_subj.sum(1), axis=0)).plot.bar(stacked=True, ax=axes[1], colormap="tab10", width=0.85)
    axes[1].set_ylabel("fraction of screening set")
    axes[1].set_title("Per-subject screening set composition")
    axes[1].legend(fontsize=6, ncol=3, loc="upper right", bbox_to_anchor=(1.0, -0.15))
    fig.tight_layout()
    fig.savefig(FIGURES / "A1_composition.png", dpi=150)
    plt.close(fig)


def a2_enrichment(st: pd.DataFrame) -> pd.DataFrame:
    scr = st[st.in_screening].copy()
    rows = []
    for c in sorted(scr.category.dropna().unique()):
        orat, p, a, b, cc, d = S.fisher_enrichment((scr.category == c).to_numpy(), scr.in_sternberg.to_numpy())
        rows.append({"category": c, "n_sternberg": a, "n_rest": b, "odds_ratio": orat, "p": p, "frac_in_sternberg": a / max(a + b, 1)})
    df = pd.DataFrame(rows)
    df["q"] = S.fdr_bh(df.p.to_numpy())[1]
    df.to_csv(TABLES / "A2_category_enrichment.csv", index=False)
    return df


def a2_contrasts(scr: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    rows = []
    a_all = scr[scr.in_sternberg]
    b_all = scr[~scr.in_sternberg]
    for c in cols:
        u, p = S.mannwhitney(a_all[c], b_all[c])
        auc = S.auc_effect(a_all[c], b_all[c])
        # within-subject version: mean over subjects of AUC(memoranda vs rest)
        wauc = []
        for _, g in scr.groupby("subject"):
            if g.in_sternberg.sum() >= 2 and (~g.in_sternberg).sum() >= 5:
                wauc.append(S.auc_effect(g.loc[g.in_sternberg, c], g.loc[~g.in_sternberg, c]))
        rows.append(
            {
                "feature": c,
                "mean_sternberg": float(np.nanmean(a_all[c])),
                "mean_rest": float(np.nanmean(b_all[c])),
                "auc": auc,
                "auc_within_subject_mean": float(np.nanmean(wauc)) if wauc else np.nan,
                "auc_within_subject_sem": float(np.nanstd(wauc) / np.sqrt(len(wauc))) if wauc else np.nan,
                "p_mannwhitney": p,
            }
        )
    df = pd.DataFrame(rows)
    df["q"] = S.fdr_bh(df.p_mannwhitney.to_numpy())[1]
    df["block"] = label
    return df


def a2_neural_rank(st: pd.DataFrame) -> pd.DataFrame:
    """Where do the memoranda fall in the subject's screening ranking by neural selectivity?"""
    scr = st[st.in_screening].copy()
    rows = []
    for s, g in scr.groupby("subject"):
        for metric in ("n_concept_pref", "n_concept_pref_mtl", "z_max", "z_max_mtl", "n_units_pref"):
            r = g[metric].rank(ascending=False, method="average")
            n = len(g)
            mem = g.in_sternberg
            rows.append(
                {
                    "subject": s,
                    "metric": metric,
                    "n_images": n,
                    "n_memoranda": int(mem.sum()),
                    "mean_rank_memoranda": float(r[mem].mean()),
                    "mean_pct_rank": float((r[mem] / n).mean()),
                    "n_memoranda_in_top5": int((r[mem] <= 5).sum()),
                    "n_memoranda_in_top10": int((r[mem] <= 10).sum()),
                    "auc": S.auc_effect(g.loc[mem, metric], g.loc[~mem, metric]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A2_neural_rank_of_memoranda.csv", index=False)
    return df


def main() -> None:
    it = image_table()
    st = subject_image_table()

    # DNN geometry, computed dataset-wide (all 342) and within each subject's screening set
    uids_all = it.image_uid.to_numpy()
    for space in GEOM_SPACES:
        X = _load_space(space, uids_all)
        desc = G.describe(X, it.category.to_numpy(), prefix=f"{space}_")
        for k, v in desc.items():
            it[k] = v
    st = st.merge(it[[c for c in it.columns if any(c.startswith(f"{sp}_") for sp in GEOM_SPACES)] + ["image_uid"]], on="image_uid", how="left")
    # within-subject geometry (relative to that subject's own screening set)
    for space in ("clip", "resnet50"):
        col_nn, col_cd = f"{space}_ws_nn1_dist", f"{space}_ws_centroid_dist"
        st[col_nn] = np.nan
        st[col_cd] = np.nan
        for s, g in st[st.in_screening].groupby("subject"):
            X = _load_space(space, g.image_uid.to_numpy())
            D = G.cosine_dist_matrix(X)
            st.loc[g.index, col_nn] = G.nn_distance(D, 1)
            st.loc[g.index, col_cd] = G.centroid_distance(X)
    st.to_csv(TABLES / "subject_image_master.csv", index=False)
    it.to_csv(TABLES / "image_master.csv", index=False)

    a1_composition(it, st)
    enr = a2_enrichment(st)
    print(enr.round(3).to_string(index=False))

    scr = st[st.in_screening]
    blocks = [
        a2_contrasts(scr, [f"attr_{a}" for a in ATTRS] + ["category_p"], "clip_attributes"),
        a2_contrasts(scr, imstats.STAT_NAMES[:-1], "low_level"),
        a2_contrasts(scr, [c for c in st.columns if any(c.startswith(f"{sp}_") for sp in GEOM_SPACES) and c != "clip_ws_nn1_dist"], "dnn_geometry"),
        a2_contrasts(scr, ["n_subjects_screening", "sternberg_rate"], "reuse"),
    ]
    con = pd.concat(blocks, ignore_index=True)
    con.to_csv(TABLES / "A2_feature_contrasts.csv", index=False)
    print(con.sort_values("q")[["block", "feature", "auc", "auc_within_subject_mean", "q"]].head(25).round(3).to_string(index=False))

    nr = a2_neural_rank(st)
    print(nr.groupby("metric")[["mean_pct_rank", "n_memoranda_in_top5", "auc"]].mean().round(3))

    # figure: enrichment + top contrasts
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    e = enr.sort_values("odds_ratio")
    axes[0].barh(e.category, np.log2(e.odds_ratio.replace(0, 0.05)), color=["#C44E52" if q < 0.05 else "#8da0cb" for q in e.q])
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("log2 odds ratio (memoranda vs rest)")
    axes[0].set_title("Category enrichment among Sternberg memoranda\n(red: FDR q<0.05)")
    top = con.sort_values("q").head(20).sort_values("auc")
    axes[1].barh(top.feature, top.auc - 0.5, color=["#C44E52" if q < 0.05 else "#8da0cb" for q in top.q])
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("AUC − 0.5 (memoranda > rest)")
    axes[1].set_title("Top feature contrasts (memoranda vs rest)")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "A2_enrichment_contrasts.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
