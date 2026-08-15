#!/usr/bin/env python
"""Stage 73 — patient-level / cluster-robust versions of the cell-pooled tests (review item H2).

Cells are nested in patients; two patients contribute 42 % of MTL concept cells. For each
headline single-cell claim we report (i) the cell-pooled statistic already in the tables,
(ii) the per-patient mean and a Wilcoxon / t over patients, and (iii) an OLS with cluster-robust
(by patient) standard errors. Also the memoranda face contrast as a within-subject AUC t-test
and family-wide FDR across all A2 contrasts, and a patient-pair-level cross-patient test.
Output: results/tables/A8_cluster_robust.csv, A2_feature_contrasts_familywide.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from memoranda.analysis import stats as S
from memoranda.paths import MANIFESTS, TABLES


def cluster_p(y: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """Intercept-only OLS with cluster-robust SE: (mean, se, p)."""
    y = np.asarray(y, float)
    m = sm.OLS(y, np.ones((len(y), 1))).fit(cov_type="cluster", cov_kwds={"groups": np.asarray(groups)})
    return float(m.params[0]), float(m.bse[0]), float(m.pvalues[0])


def subject_level(df: pd.DataFrame, col: str, subj: str = "subject") -> tuple[int, float, float]:
    g = df.groupby(subj)[col].mean()
    if len(g) < 5:
        return len(g), float(g.mean()), np.nan
    return len(g), float(g.mean()), float(sps.wilcoxon(g).pvalue)


def main() -> None:
    rows = []
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    faces = pd.read_csv(MANIFESTS / "faces.csv").set_index("image_uid")
    img = pd.read_csv(MANIFESTS / "images.csv")

    # 1. similarity tuning
    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    for grp in ("MTL concept", "MTL concept (strict)"):
        for m, l in (("clip_vitb32", "ln_post"), ("resnet50", "avgpool"), ("alexnet", "conv1")):
            d = st[(st.group == grp) & (st.model == m) & (st.layer == l)].dropna(subset=["rho"])
            n_s, mean_s, p_s = subject_level(d, "rho")
            mu, se, p_c = cluster_p(d.rho.to_numpy(), d.subject.to_numpy())
            rows.append({"claim": f"similarity tuning {grp} {m} {l}", "n_cells": len(d), "cell_mean": d.rho.mean(), "cell_p": sps.wilcoxon(d.rho).pvalue, "n_patients": n_s, "patient_mean": mean_s, "patient_p": p_s, "cluster_se": se, "cluster_p": p_c})

    # 2. face preference of MTL concept cells (pref picture has face vs shown composition)
    shown = img[(img.task == "screening") & (img.image_uid != "img_null")][["subject", "image_uid"]].drop_duplicates()
    shown["face"] = faces.loc[shown.image_uid, "face_found"].to_numpy()
    exp = shown.groupby("subject").face.mean()
    for area_lab, mask in (("MTL", sel.region == "MTL"), ("amygdala", sel.area == "amygdala"), ("hippocampus", sel.area == "hippocampus")):
        for crit in ("concept_cell", "concept_cell_strict"):
            c = sel[(sel.task == "screening") & mask & sel[crit]].copy()
            c["face"] = faces.loc[c.pref_image, "face_found"].to_numpy()
            c["excess"] = c.face - c.subject.map(exp)
            n_s, mean_s, p_s = subject_level(c, "excess")
            mu, se, p_c = cluster_p(c.excess.to_numpy(), c.subject.to_numpy())
            rows.append({"claim": f"face preference {area_lab} {crit}", "n_cells": len(c), "cell_mean": c.excess.mean(), "cell_p": sps.wilcoxon(c.excess).pvalue if len(c) > 5 else np.nan, "n_patients": n_s, "patient_mean": mean_s, "patient_p": p_s, "cluster_se": se, "cluster_p": p_c})

    # 3. laterality of similarity tuning (amygdala), patient-level
    d = st[(st.group == "MTL concept") & (st.model == "clip_vitb32") & (st.layer == "ln_post")].merge(sel[sel.task == "screening"][["subject", "unit", "hemisphere"]], on=["subject", "unit"])
    a = d[d.area == "amygdala"]
    ps = a.groupby(["subject", "hemisphere"]).rho.mean().unstack()
    R, L = ps["R"].dropna(), ps["L"].dropna()
    rows.append({"claim": "amygdala similarity tuning R vs L (patient means)", "n_cells": len(a), "cell_mean": a[a.hemisphere == "R"].rho.mean() - a[a.hemisphere == "L"].rho.mean(), "cell_p": sps.mannwhitneyu(a[a.hemisphere == "R"].rho, a[a.hemisphere == "L"].rho).pvalue, "n_patients": len(R) + len(L), "patient_mean": R.mean() - L.mean(), "patient_p": sps.mannwhitneyu(R, L).pvalue, "cluster_se": np.nan, "cluster_p": np.nan, "note": f"R patients {len(R)} (mean {R.mean():.3f}), L patients {len(L)} (mean {L.mean():.3f})"})

    # 4. face-space: object model vs VGGFace2 for face-preferring cells, patient level
    fs = pd.read_csv(TABLES / "A4_face_similarity_tuning_per_cell.csv")
    fs = fs[fs.scope == "other face pictures"]
    piv = fs.pivot_table(index=["subject", "unit"], columns="space", values="rho").reset_index()
    for obj in ("resnet50 avgpool", "clip ln_post"):
        piv["diff"] = piv[obj] - piv["vggface2 identity"]
        n_s, mean_s, p_s = subject_level(piv, "diff")
        mu, se, p_c = cluster_p(piv["diff"].to_numpy(), piv.subject.to_numpy())
        rows.append({"claim": f"face-preferring cells: {obj} − VGGFace2 (other faces)", "n_cells": len(piv), "cell_mean": piv["diff"].mean(), "cell_p": sps.wilcoxon(piv["diff"]).pvalue, "n_patients": n_s, "patient_mean": mean_s, "patient_p": p_s, "cluster_se": se, "cluster_p": p_c})

    # 5. cross-patient tuning: patient-pair level
    xp = pd.read_csv(TABLES / "A5_cross_patient_tuning_pairs.csv")
    xp["pa"] = xp.cell_a.str.split("-").str[0]
    xp["pb"] = xp.cell_b.str.split("-").str[0]
    xp["ppair"] = xp[["pa", "pb"]].min(axis=1) + "_" + xp[["pa", "pb"]].max(axis=1)
    same = xp[~xp.same_pref & xp.same_category].groupby("ppair").rho.mean()
    diff = xp[~xp.same_pref & ~xp.same_category].groupby("ppair").rho.mean()
    common = same.index.intersection(diff.index)
    rows.append({"claim": "cross-patient: same-category − different-category tuning agreement", "n_cells": len(xp), "cell_mean": xp[~xp.same_pref & xp.same_category].rho.mean() - xp[~xp.same_pref & ~xp.same_category].rho.mean(), "cell_p": np.nan, "n_patients": len(common), "patient_mean": float((same[common] - diff[common]).mean()), "patient_p": float(sps.wilcoxon(same[common] - diff[common]).pvalue), "cluster_se": np.nan, "cluster_p": np.nan, "note": "unit = patient pair"})
    sp = xp[xp.same_pref].rho
    lo, hi = sp.mean() - 1.96 * sp.std() / np.sqrt(len(sp)), sp.mean() + 1.96 * sp.std() / np.sqrt(len(sp))
    rows.append({"claim": "cross-patient: same preferred picture agreement", "n_cells": len(sp), "cell_mean": sp.mean(), "cell_p": sps.wilcoxon(sp).pvalue, "n_patients": np.nan, "patient_mean": np.nan, "patient_p": np.nan, "cluster_se": np.nan, "cluster_p": np.nan, "note": f"95% CI {lo:.3f}…{hi:.3f} (n=43 pairs; underpowered)"})

    # 6. memoranda face contrast: within-subject AUC t-test and family-wide FDR
    a2 = pd.read_csv(TABLES / "A2_feature_contrasts.csv")
    a2["q_familywide"] = S.fdr_bh(a2.p_mannwhitney.to_numpy())[1]
    a2["t_within_subject"] = (a2.auc_within_subject_mean - 0.5) / a2.auc_within_subject_sem
    a2["p_within_subject"] = 2 * sps.t.sf(np.abs(a2.t_within_subject), df=19)
    a2.to_csv(TABLES / "A2_feature_contrasts_familywide.csv", index=False)
    for f in ("face_found", "largest_face_area", "n_faces"):
        r = a2[a2.feature == f].iloc[0]
        rows.append({"claim": f"memoranda vs rest: {f}", "n_cells": np.nan, "cell_mean": r.auc, "cell_p": r.p_mannwhitney, "n_patients": 20, "patient_mean": r.auc_within_subject_mean, "patient_p": r.p_within_subject, "cluster_se": r.auc_within_subject_sem, "cluster_p": np.nan, "note": f"block q={r.q:.3f}, family-wide q={r.q_familywide:.3f} over {len(a2)} contrasts"})

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A8_cluster_robust.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
