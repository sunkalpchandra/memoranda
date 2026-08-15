#!/usr/bin/env python
"""Stage 03 — spike times and unit metadata for every session.

Outputs:
  data/neural/sub-XX_ses-Y_units.npz   spike times (object array) + unit ids   [git-ignored]
  data/manifests/units.csv             one row per unit: area, hemisphere, quality metrics,
                                       n_spikes, firing rate                       [committed]
  data/manifests/electrodes.csv        one row per electrode per session          [committed]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from memoranda import nwb
from memoranda.dandi import list_assets
from memoranda.log import get_logger
from memoranda.paths import MANIFESTS, NEURAL, ensure_dirs

log = get_logger("extract_units")


def units_npz_path(subject: int, session: int):
    return NEURAL / f"sub-{subject:02d}_ses-{session}_units.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    assets = list_assets()
    if args.only:
        s, e = args.only.split("-")
        assets = [a for a in assets if a.subject == int(s) and a.session == int(e)]

    unit_rows, elec_rows = [], []
    for a in assets:
        f = nwb.open_remote(a)
        elec = nwb.read_electrodes(f)
        units = nwb.read_units(f, elec)
        # session duration from events (first to last TTL) for firing-rate normalisation
        ev = nwb.read_events(f)
        t0, t1 = float(ev["time"].min()), float(ev["time"].max())
        dur = max(t1 - t0, 1e-6)
        f.close()

        out = units_npz_path(a.subject, a.session)
        if args.force or not out.exists():
            np.savez_compressed(
                out,
                unit=np.array([u.unit for u in units]),
                spike_times=np.array([u.spike_times for u in units], dtype=object),
                allow_pickle=True,
            )
        for u in units:
            unit_rows.append(
                {
                    "subject": a.subject,
                    "session": a.session,
                    "task": a.task,
                    "unit": u.unit,
                    "electrode": u.electrode,
                    "location": u.location,
                    "area": u.area,
                    "region": u.region,
                    "hemisphere": u.hemisphere,
                    "n_spikes": u.n_spikes,
                    "fr_hz": u.n_spikes / dur,
                    "isolation_distance": u.isolation_distance,
                    "mean_proj_dist": u.mean_proj_dist,
                    "mean_snr": u.mean_snr,
                    "peak_snr": u.peak_snr,
                    "cluster_id_orig": u.cluster_id_orig,
                }
            )
        e2 = elec.copy()
        e2.insert(0, "task", a.task)
        e2.insert(0, "session", a.session)
        e2.insert(0, "subject", a.subject)
        elec_rows.append(e2)
        log.info(f"{a.key} {a.task:9s} units={len(units):3d} electrodes={len(elec):2d} dur={dur/60:.1f} min")

    udf = pd.DataFrame(unit_rows)
    edf = pd.concat(elec_rows)
    if args.only:
        print(udf.head(), edf.head())
        return
    udf.to_csv(MANIFESTS / "units.csv", index=False)
    edf.to_csv(MANIFESTS / "electrodes.csv", index=False)
    log.info(f"wrote {len(udf)} units, {len(edf)} electrode rows")


if __name__ == "__main__":
    main()
