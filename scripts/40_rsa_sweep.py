#!/usr/bin/env python
"""Stage 40 — RSA sweep: every screening session × region × model layer.

Regions: MTL, MFC, all, MTL_concept (MTL concept cells only), amygdala, hippocampus.
Baselines: category RDM (CLIP zero-shot), low-level statistics RDM.
Outputs:
  results/tables/A3_rsa_per_session.csv     one row per (subject, region, model, layer, view)
  results/tables/A3_rsa_reliability.csv     split-half reliability per (subject, region)
  results/tables/A3_rsa_summary.csv         mean ρ across sessions, t-test vs 0, ceiling-normalised
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import list_layers
from memoranda.log import get_logger
from memoranda.models.registry import DEFAULT_MODELS
from memoranda.paths import MANIFESTS, TABLES

log = get_logger("rsa")
REGIONS = ["MTL", "MFC", "all", "MTL_concept", "amygdala", "hippocampus"]
VIEWS = ("gap", "cls")


def region_matrix(subject: int, region: str, sel: pd.DataFrame):
    if region == "all":
        return rsa.session_response_matrix(subject, 1, None)
    if region in ("MTL", "MFC"):
        return rsa.session_response_matrix(subject, 1, region)
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None, None
    ut = neural.load_units_table(subject, 1).set_index("unit")
    if region == "MTL_concept":
        keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
    else:
        keep = ut[ut.area == region].index.to_numpy()
    mask = np.isin(units, keep)
    if mask.sum() < 2:
        return None, None, None
    return M[:, mask], uids, units[mask]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = ap.parse_args()

    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv")
    layer_list = [(m, l, v) for m in args.models for (l, v) in list_layers(m) if v in VIEWS]
    layer_list.append(("clip_vitb32", "embed", "gap"))

    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows, rel_rows = [], []
    for s in subjects:
        t0 = time.time()
        # model RDMs for this subject's image set (same for all regions)
        M_all, uids, _ = rsa.session_response_matrix(s, 1, None)
        model_rdms = {(m, l, v): rsa.model_rdm(m, l, v, uids) for (m, l, v) in layer_list}
        model_rdms[("baseline", "category", "-")] = rsa.category_rdm(uids, labels)
        model_rdms[("baseline", "lowlevel", "-")] = rsa.lowlevel_rdm(uids, stats_df)
        for region in REGIONS:
            M, u2, units = region_matrix(s, region, sel)
            if M is None or M.shape[1] < 2:
                continue
            assert np.array_equal(u2, uids)
            Dn = rsa.neural_rdm(M)
            # reliability
            if region in ("MTL", "MFC", "all", "amygdala", "hippocampus"):
                reg_arg = None if region == "all" else region
                if region in ("amygdala", "hippocampus"):
                    rel = _area_reliability(s, region, args.n_splits)
                else:
                    rel = rsa.split_half_reliability(s, 1, reg_arg, n_splits=args.n_splits)
            else:
                rel = _concept_reliability(s, units, args.n_splits)
            rel_rows.append({"subject": s, "region": region, "n_units": M.shape[1], "n_images": len(uids), "reliability": rel})
            for (m, l, v), Dm in model_rdms.items():
                rows.append(
                    {
                        "subject": s,
                        "region": region,
                        "n_units": M.shape[1],
                        "n_images": len(uids),
                        "model": m,
                        "layer": l,
                        "view": v,
                        "rho": rsa.compare_rdms(Dn, Dm),
                        "reliability": rel,
                    }
                )
        log.info(f"sub-{s:02d} done ({time.time()-t0:.1f}s)")

    per = pd.DataFrame(rows)
    per["rho_norm"] = per.rho / np.sqrt(per.reliability.clip(lower=0.01))  # per-session ratio is unstable; prefer the summary's ratio of means
    per.to_csv(TABLES / "A3_rsa_per_session.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(TABLES / "A3_rsa_reliability.csv", index=False)

    def _summ(g):
        z = np.arctanh(np.clip(g.rho.dropna(), -0.999, 0.999))
        t, p = sps.ttest_1samp(z, 0) if len(z) > 2 else (np.nan, np.nan)
        return pd.Series(
            {
                "n_sessions": len(z),
                "rho_mean": float(np.tanh(z.mean())) if len(z) else np.nan,
                "rho_sem": float(g.rho.std() / np.sqrt(len(z))) if len(z) > 1 else np.nan,
                "t": t,
                "p": p,
                "rho_norm_mean": float(np.tanh(z.mean()) / np.sqrt(max(g.reliability.mean(), 1e-3))) if len(z) else np.nan,  # ratio of means (stable)
                "frac_sessions_positive": float((g.rho > 0).mean()),
            }
        )

    summ = per.groupby(["region", "model", "layer", "view"]).apply(_summ).reset_index()
    summ.to_csv(TABLES / "A3_rsa_summary.csv", index=False)
    show = summ[summ.region.isin(["MTL", "MFC"])].sort_values("rho_mean", ascending=False)
    print(show.head(30).round(4).to_string(index=False))


def _area_reliability(subject, area, n_splits):
    ut = neural.load_units_table(subject, 1)
    keep = set(ut.loc[ut.area == area, "unit"])
    return _subset_reliability(subject, keep, n_splits)


def _concept_reliability(subject, units, n_splits):
    return _subset_reliability(subject, set(units.tolist()), n_splits)


def _subset_reliability(subject, keep, n_splits):
    rs = []
    for k in range(n_splits):
        A, uids, ua = rsa.session_response_matrix(subject, 1, None, split=0, seed=k)
        B, _, ub = rsa.session_response_matrix(subject, 1, None, split=1, seed=k)
        ma = np.isin(ua, list(keep))
        if ma.sum() < 2:
            return np.nan
        rs.append(rsa.compare_rdms(rsa.neural_rdm(A[:, ma]), rsa.neural_rdm(B[:, ma])))
    r = float(np.nanmean(rs))
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


if __name__ == "__main__":
    main()
