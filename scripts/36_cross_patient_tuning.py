#!/usr/bin/env python
"""Stage 36 — cross-patient consistency of concept-cell tuning.

Take every pair of screening MTL concept cells from *different* patients. Over the pictures
both patients saw (≥ 10), correlate the two tuning curves (Spearman, preferred pictures of
either cell excluded so the match is not trivial). Compare pairs that prefer the SAME picture
with pairs that prefer different pictures, and with same-category vs different-category
preferences. If tuning were a property of the picture, same-preference pairs should agree.
Output: results/tables/A5_cross_patient_tuning.csv, results/figures/A5_cross_patient_tuning.png
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.paths import FIGURES, MANIFESTS, TABLES


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    lab = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid")
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]
    tun = tun[tun.task == "screening"]
    curves = {(r.subject, r.unit): tun[(tun.subject == r.subject) & (tun.unit == r.unit)].set_index("image_uid").rate for r in cc.itertuples()}
    cells = list(curves)
    pref = {(r.subject, r.unit): r.pref_image for r in cc.itertuples()}
    rows = []
    for a, b in itertools.combinations(cells, 2):
        if a[0] == b[0]:
            continue
        ca, cb = curves[a], curves[b]
        common = ca.index.intersection(cb.index).difference([pref[a], pref[b]])
        if len(common) < 10:
            continue
        rho = sps.spearmanr(ca.loc[common], cb.loc[common]).statistic
        same_pref = pref[a] == pref[b]
        same_cat = lab.loc[pref[a], "category"] == lab.loc[pref[b], "category"]
        rows.append({"cell_a": f"{a[0]}-{a[1]}", "cell_b": f"{b[0]}-{b[1]}", "n_common": len(common), "same_pref": same_pref, "same_category": same_cat, "pref_a": pref[a], "pref_b": pref[b], "rho": rho, "both_pref_shared": (pref[a] in cb.index) and (pref[b] in ca.index)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A5_cross_patient_tuning_pairs.csv", index=False)
    g = df.groupby(["same_pref", "same_category"]).rho.agg(["count", "mean", "sem"]).reset_index()
    print(g.round(4).to_string(index=False))
    a = df[df.same_pref].rho
    b = df[~df.same_pref & df.same_category].rho
    c = df[~df.same_pref & ~df.same_category].rho
    res = {
        "n_same_pref": len(a), "rho_same_pref": a.mean(), "n_diff_pref_same_cat": len(b), "rho_diff_pref_same_cat": b.mean(), "n_diff_cat": len(c), "rho_diff_cat": c.mean(),
        "p_same_vs_samecat": sps.mannwhitneyu(a, b).pvalue if len(a) > 3 else np.nan,
        "p_samecat_vs_diffcat": sps.mannwhitneyu(b, c).pvalue,
        "p_same_pref_vs_zero": sps.wilcoxon(a).pvalue if len(a) > 3 else np.nan,
    }
    pd.DataFrame([res]).to_csv(TABLES / "A5_cross_patient_tuning.csv", index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    data = [a, b, c]
    labels = [f"same preferred\npicture (n={len(a)})", f"different picture,\nsame category (n={len(b)})", f"different\ncategory (n={len(c)})"]
    ax.violinplot(data, showmeans=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("Spearman ρ between tuning curves\n(co-shown pictures, preferred ones excluded)")
    ax.set_title("Cross-patient agreement of MTL concept-cell tuning")
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_cross_patient_tuning.png", dpi=150)


if __name__ == "__main__":
    main()
