#!/usr/bin/env python
"""Stage 20 — single-unit image selectivity for every session.

Outputs (committed):
  data/manifests/unit_selectivity.csv   one row per unit (both tasks): F, p, pref image, DoS…
  data/manifests/unit_tuning.csv        one row per (unit, image): mean rate, z vs baseline
Sternberg sessions use encoding presentations only (enc1–enc3), never probes.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from conceptlens import neural
from conceptlens.dandi import list_assets
from conceptlens.log import get_logger
from conceptlens.paths import MANIFESTS

log = get_logger("selectivity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    assets = list_assets()
    if args.only:
        s, e = args.only.split("-")
        assets = [a for a in assets if a.subject == int(s) and a.session == int(e)]

    sel, tun = [], []
    for a in assets:
        t0 = time.time()
        roles = ("enc1", "enc2", "enc3") if a.task == "sternberg" else None
        u, t = neural.analyse_session(a.subject, a.session, n_perm=args.n_perm, roles=roles)
        u.insert(2, "task", a.task)
        t.insert(2, "task", a.task)
        sel.append(u)
        tun.append(t)
        log.info(f"{a.key} {a.task:9s} units={len(u):3d} concept={int(u.concept_cell.sum()):3d} ({time.time()-t0:.1f}s)")

    sel_df = pd.concat(sel, ignore_index=True)
    tun_df = pd.concat(tun, ignore_index=True)
    units = pd.read_csv(MANIFESTS / "units.csv")[["subject", "session", "unit", "area", "region", "hemisphere"]]
    sel_df = sel_df.merge(units, on=["subject", "session", "unit"], how="left")
    tun_df = tun_df.merge(units, on=["subject", "session", "unit"], how="left")
    if args.only:
        print(sel_df.head())
        return
    sel_df.to_csv(MANIFESTS / "unit_selectivity.csv", index=False)
    tun_df.to_csv(MANIFESTS / "unit_tuning.csv", index=False)
    summ = sel_df.groupby(["task", "region"]).concept_cell.agg(["sum", "count", "mean"])
    log.info("\n" + summ.to_string())


if __name__ == "__main__":
    main()
