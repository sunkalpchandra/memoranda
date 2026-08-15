#!/usr/bin/env python
"""Stage 25 — Sternberg concept-cell selection on encoding-1 presentations only (paper's rule).

Re-runs the concept-cell test using only the first encoding presentation of each trial and
compares the fraction of concept cells per region with the all-encoding version and the
paper (MTL 20.9 %, MFC 6.7 % for the Sternberg task).
Output: results/tables/A0_sternberg_enc1_selection.csv
"""

from __future__ import annotations

import pandas as pd

from memoranda import neural
from memoranda.dandi import list_assets
from memoranda.paths import MANIFESTS, TABLES


def main() -> None:
    units = pd.read_csv(MANIFESTS / "units.csv")[["subject", "session", "unit", "area", "region"]]
    out = []
    for a in list_assets():
        if a.session != 2:
            continue
        u, _ = neural.analyse_session(a.subject, 2, n_perm=1000, roles=("enc1",))
        out.append(u)
    e1 = pd.concat(out).merge(units, on=["subject", "session", "unit"])
    e1.to_csv(TABLES / "A0_sternberg_enc1_units.csv", index=False)
    all_enc = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    all_enc = all_enc[all_enc.task == "sternberg"]
    rows = []
    for region in ("MTL", "MFC", "amygdala", "hippocampus"):
        m1 = e1[e1.region == region] if region in ("MTL", "MFC") else e1[e1.area == region]
        m2 = all_enc[all_enc.region == region] if region in ("MTL", "MFC") else all_enc[all_enc.area == region]
        rows.append({"region": region, "n_units": len(m1), "frac_concept_enc1": m1.concept_cell.mean(), "n_concept_enc1": int(m1.concept_cell.sum()), "frac_concept_enc1_3": m2.concept_cell.mean(), "n_concept_enc1_3": int(m2.concept_cell.sum())})
    df = pd.DataFrame(rows)
    df["paper"] = df.region.map({"MTL": 0.209, "MFC": 0.067})
    df.to_csv(TABLES / "A0_sternberg_enc1_selection.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
