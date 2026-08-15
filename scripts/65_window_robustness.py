#!/usr/bin/env python
"""Stage 65 — robustness of the headline RSA / similarity-tuning results to the response window.

Windows: 100–600, 200–1000 (default), 300–800, 200–500, 500–1000 ms. For each: per-session
MTL RSA vs CLIP ln_post / ResNet-50 avgpool / AlexNet conv1 / category, and per-cell
similarity tuning (concept cells fixed from the default window).
Output: results/tables/A3_window_robustness.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import load_space
from memoranda.paths import MANIFESTS, TABLES

WINDOWS = {"100-600": (0.1, 0.6), "200-1000": (0.2, 1.0), "300-800": (0.3, 0.8), "200-500": (0.2, 0.5), "500-1000": (0.5, 1.0)}
REFS = {"clip ln_post": ("clip_vitb32", "ln_post", "cls"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "alexnet conv1": ("alexnet", "conv1", "gap")}


def main() -> None:
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    spaces = {}
    for k, v in REFS.items():
        X, stored = load_space(*v, center=True, normalize=True)
        spaces[k] = (X, {u: i for i, u in enumerate(stored)})
    rows = []
    for s in subjects:
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        onsets, lab = pres.onset.to_numpy(), pres.image_uid.to_numpy()
        uids = np.unique(lab)
        ut = neural.load_units_table(s, 1)
        mtl = set(ut.loc[ut.region == "MTL", "unit"])
        sp = {u: st for u, st in spikes.items() if u in mtl}
        if len(sp) < 2:
            continue
        model_rdms = {k: rsa.model_rdm(*v, uids) for k, v in REFS.items()}
        model_rdms["category"] = rsa.category_rdm(uids, labels)
        cells = cc[cc.subject == s]
        for wname, win in WINDOWS.items():
            R, units = neural.rate_matrix(sp, onsets, win)
            M = np.stack([R[lab == u].mean(0) for u in uids])
            Dn = rsa.neural_rdm(M)
            for k, D in model_rdms.items():
                rows.append({"analysis": "RSA MTL", "window": wname, "subject": s, "unit": np.nan, "reference": k, "value": rsa.compare_rdms(Dn, D)})
            upos = {u: j for j, u in enumerate(units)}
            for r in cells.itertuples():
                rates = M[:, upos[r.unit]]
                mask = uids != r.pref_image
                for k, (X, pos) in spaces.items():
                    sims = X[[pos[u] for u in uids]] @ X[pos[r.pref_image]]
                    rows.append({"analysis": "similarity tuning", "window": wname, "subject": s, "unit": r.unit, "reference": k, "value": sps.spearmanr(rates[mask], sims[mask]).statistic if rates[mask].std() > 0 else np.nan})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_window_robustness_per_unit.csv", index=False)
    summ = per.groupby(["analysis", "reference", "window"]).value.agg(["count", "mean", "sem"]).reset_index()
    summ["window"] = pd.Categorical(summ.window, list(WINDOWS))
    summ = summ.sort_values(["analysis", "reference", "window"])
    summ.to_csv(TABLES / "A3_window_robustness.csv", index=False)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
