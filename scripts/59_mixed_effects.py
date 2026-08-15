#!/usr/bin/env python
"""Stage 59 — mixed-effects versions of the headline tests (cells nested in patients).

1. Similarity tuning (MTL concept cells, all 8 models pooled): ρ ~ relative layer depth,
   random intercept per subject (and per cell) — is the depth slope significant with the
   nesting respected?
2. RSA at the best late layer (CLIP ln_post): ρ ~ region (MTL vs MFC), random intercept per
   subject (sessions with both regions).
3. Similarity tuning: MTL concept vs MFC concept vs MTL non-concept at late layers with a
   subject random intercept.
Output: results/tables/A8_mixed_effects.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from memoranda.models.registry import get_spec
from memoranda.paths import TABLES


def main() -> None:
    rows = []
    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    depth = []
    for r in st.itertuples():
        lay = list(get_spec(r.model).layers)
        depth.append(lay.index(r.layer) / max(len(lay) - 1, 1) if r.layer in lay else np.nan)
    st["rel_depth"] = depth
    st["cell"] = st.subject.astype(str) + "-" + st.unit.astype(str)
    d = st[(st.group == "MTL concept")].dropna(subset=["rho", "rel_depth"])
    m1 = smf.mixedlm("rho ~ rel_depth", d, groups=d["subject"], re_formula="1", vc_formula={"cell": "0 + C(cell)"}).fit(reml=True)
    rows.append({"test": "similarity tuning ~ rel_depth (MTL concept; RI subject + cell)", "term": "rel_depth", "estimate": m1.params["rel_depth"], "se": m1.bse["rel_depth"], "z": m1.tvalues["rel_depth"], "p": m1.pvalues["rel_depth"], "n_obs": int(m1.nobs), "n_groups": d.subject.nunique()})
    rows.append({"test": "similarity tuning ~ rel_depth (MTL concept; RI subject + cell)", "term": "Intercept", "estimate": m1.params["Intercept"], "se": m1.bse["Intercept"], "z": m1.tvalues["Intercept"], "p": m1.pvalues["Intercept"], "n_obs": int(m1.nobs), "n_groups": d.subject.nunique()})

    late = st[(st.rel_depth >= 0.75)].dropna(subset=["rho"])
    late = late[late.group.isin(["MTL concept", "MFC concept", "MTL non-concept"])]
    late["group"] = pd.Categorical(late.group, ["MTL non-concept", "MTL concept", "MFC concept"])
    m3 = smf.mixedlm("rho ~ C(group)", late, groups=late["subject"], re_formula="1", vc_formula={"cell": "0 + C(cell)"}).fit(reml=True)
    for term in m3.params.index:
        rows.append({"test": "late-layer similarity tuning ~ group (RI subject + cell)", "term": term, "estimate": m3.params[term], "se": m3.bse[term], "z": m3.tvalues[term], "p": m3.pvalues[term], "n_obs": int(m3.nobs), "n_groups": late.subject.nunique()})

    rsa = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    r = rsa[(rsa.model == "clip_vitb32") & (rsa.layer == "ln_post") & (rsa.view == "cls") & rsa.region.isin(["MTL", "MFC"])].dropna(subset=["rho"])
    m2 = smf.mixedlm("rho ~ C(region, Treatment('MFC'))", r, groups=r["subject"]).fit(reml=True)
    for term in m2.params.index:
        rows.append({"test": "RSA ρ (CLIP ln_post) ~ region (RI subject)", "term": term, "estimate": m2.params[term], "se": m2.bse[term], "z": m2.tvalues[term], "p": m2.pvalues[term], "n_obs": int(m2.nobs), "n_groups": r.subject.nunique()})

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A8_mixed_effects.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
