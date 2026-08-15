#!/usr/bin/env python
"""Stage 58 — group-level permutation test for the RSA result.

Instead of a t-test on Fisher-z across sessions, shuffle picture identity of the model RDM
independently within each session, recompute the mean ρ over sessions, and repeat 2000×.
Reports the permutation p for MTL, MTL concept cells, MFC, amygdala, hippocampus against a
curated set of layers and the category / low-level baselines, plus the MTL − MFC difference.
Output: results/tables/A3_group_permutation.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.paths import MANIFESTS, TABLES

LAYERS = {"alexnet conv1": ("alexnet", "conv1", "gap"), "alexnet conv4": ("alexnet", "conv4", "gap"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "vit ln": ("vit_b_16", "ln", "gap"), "dinov2 norm": ("dinov2_small", "norm", "gap"), "clip ln_post": ("clip_vitb32", "ln_post", "cls")}
N_PERM = 2000


def region_matrix(subject, region, sel):
    if region in ("MTL", "MFC"):
        M, uids, _ = rsa.session_response_matrix(subject, 1, region)
        return M, uids
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None
    ut = neural.load_units_table(subject, 1).set_index("unit")
    keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy() if region == "MTL_concept" else ut[ut.area == region].index.to_numpy()
    mask = np.isin(units, keep)
    return (M[:, mask], uids) if mask.sum() >= 2 else (None, None)


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rng = np.random.default_rng(0)
    # cache neural upper triangles (ranked) and model RDMs per session
    sess = {}
    for s in subjects:
        _, uids, _ = rsa.session_response_matrix(s, 1, None)
        models = {k: rsa.model_rdm(*v, uids) for k, v in LAYERS.items()}
        models["category"] = rsa.category_rdm(uids, labels)
        models["low-level"] = rsa.lowlevel_rdm(uids, stats_df)
        neural_rdms = {}
        for region in ("MTL", "MFC", "MTL_concept", "amygdala", "hippocampus"):
            M, _ = region_matrix(s, region, sel)
            if M is not None:
                neural_rdms[region] = rsa.neural_rdm(M)
        sess[s] = (models, neural_rdms)

    rows = []
    for name in list(LAYERS) + ["category", "low-level"]:
        obs = {}
        null = {}
        for region in ("MTL", "MFC", "MTL_concept", "amygdala", "hippocampus"):
            vals = [rsa.compare_rdms(nr[region], m[name]) for s, (m, nr) in sess.items() if region in nr]
            obs[region] = float(np.nanmean(vals))
            null[region] = np.zeros(N_PERM)
        for k in range(N_PERM):
            for region in obs:
                vals = []
                for s, (m, nr) in sess.items():
                    if region not in nr:
                        continue
                    D = m[name]
                    perm = rng.permutation(D.shape[0])
                    vals.append(rsa.compare_rdms(nr[region], D[np.ix_(perm, perm)]))
                null[region][k] = np.nanmean(vals)
        for region in obs:
            p = (np.sum(null[region] >= obs[region]) + 1) / (N_PERM + 1)
            rows.append({"reference": name, "region": region, "mean_rho": obs[region], "null_mean": float(null[region].mean()), "null_sd": float(null[region].std()), "z": (obs[region] - null[region].mean()) / null[region].std(), "p_perm": p})
        # MTL - MFC
        diff_obs = obs["MTL"] - obs["MFC"]
        diff_null = null["MTL"] - null["MFC"]
        rows.append({"reference": name, "region": "MTL-MFC", "mean_rho": diff_obs, "null_mean": float(diff_null.mean()), "null_sd": float(diff_null.std()), "z": (diff_obs - diff_null.mean()) / diff_null.std(), "p_perm": (np.sum(diff_null >= diff_obs) + 1) / (N_PERM + 1)})
        print(name, {r: round(rows[-1 - i]["p_perm"], 4) for i, r in enumerate(["MTL-MFC"])}, "MTL p=", [r["p_perm"] for r in rows if r["reference"] == name and r["region"] == "MTL"][0])
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A3_group_permutation.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
