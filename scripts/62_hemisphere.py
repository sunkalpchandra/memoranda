#!/usr/bin/env python
"""Stage 62 — laterality: do right and left MTL populations differ in DNN correspondence?

Per screening session: MTL RDM from right-hemisphere units vs left-hemisphere units (each ≥ 3
units) against late layers and the category RDM; paired comparison over sessions with both.
Also similarity tuning of concept cells by hemisphere × area.
Output: results/tables/A3_hemisphere_rsa.csv, A4_hemisphere_similarity.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.paths import MANIFESTS, TABLES

REFS = {"clip ln_post": ("clip_vitb32", "ln_post", "cls"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "alexnet conv1": ("alexnet", "conv1", "gap")}


def main() -> None:
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        M, uids, units = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        ut = neural.load_units_table(s, 1).set_index("unit")
        refs = {k: rsa.model_rdm(*v, uids) for k, v in REFS.items()}
        refs["category"] = rsa.category_rdm(uids, labels)
        for hemi in ("L", "R"):
            mask = np.isin(units, ut[ut.hemisphere == hemi].index.to_numpy())
            if mask.sum() < 3:
                continue
            Dn = rsa.neural_rdm(M[:, mask])
            for k, D in refs.items():
                rows.append({"subject": s, "hemisphere": hemi, "n_units": int(mask.sum()), "reference": k, "rho": rsa.compare_rdms(Dn, D)})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_hemisphere_rsa_per_session.csv", index=False)
    out = []
    for k, g in per.groupby("reference"):
        L = g[g.hemisphere == "L"].set_index("subject").rho
        R = g[g.hemisphere == "R"].set_index("subject").rho
        common = L.index.intersection(R.index)
        out.append({"reference": k, "n_L": len(L), "n_R": len(R), "rho_L": L.mean(), "rho_R": R.mean(), "n_paired": len(common), "paired_p": sps.wilcoxon(R[common] - L[common]).pvalue if len(common) >= 5 else np.nan, "p_L_vs0": sps.ttest_1samp(L, 0).pvalue, "p_R_vs0": sps.ttest_1samp(R, 0).pvalue})
    res = pd.DataFrame(out)
    res.to_csv(TABLES / "A3_hemisphere_rsa.csv", index=False)
    print(res.round(4).to_string(index=False))

    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    st = st[(st.group == "MTL concept") & (st.model == "clip_vitb32") & (st.layer == "ln_post")]
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")[["subject", "session", "unit", "hemisphere"]]
    st = st.merge(sel[sel.session == 1].drop(columns="session"), on=["subject", "unit"])
    g = st.groupby(["area", "hemisphere"]).rho.agg(["count", "mean", "sem"]).reset_index()
    for area in ("amygdala", "hippocampus"):
        a = st[(st.area == area)]
        p = sps.mannwhitneyu(a[a.hemisphere == "R"].rho, a[a.hemisphere == "L"].rho).pvalue if (a.hemisphere == "R").sum() > 3 and (a.hemisphere == "L").sum() > 3 else np.nan
        g.loc[g.area == area, "p_R_vs_L"] = p
    g.to_csv(TABLES / "A4_hemisphere_similarity.csv", index=False)
    print(g.round(4).to_string(index=False))
    # within-patient check for the amygdala: subjects with concept cells in both hemispheres
    a = st[st.area == "amygdala"].groupby(["subject", "hemisphere"]).rho.agg(["mean", "size"]).unstack()
    both = a.dropna()
    if len(both) >= 3:
        d = both[("mean", "R")] - both[("mean", "L")]
        print(f"within-patient amygdala R−L (n={len(both)} patients): mean diff {d.mean():.3f}, Wilcoxon p={sps.wilcoxon(d).pvalue:.3f}, positive in {(d > 0).sum()}/{len(d)}")
        both.to_csv(TABLES / "A4_hemisphere_within_patient.csv")


if __name__ == "__main__":
    main()
