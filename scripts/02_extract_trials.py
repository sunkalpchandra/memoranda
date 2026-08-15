#!/usr/bin/env python
"""Stage 02 — trial tables, presentation logs and TTL events for every session.

Outputs (all small, committed under ``data/manifests``):
  trials_sternberg.csv   one row per Sternberg trial, all timestamp/response columns
  trials_screening.csv   one row per screening trial (start/stop only)
  presentations.csv      one row per stimulus presentation (both tasks) with onset
  events.csv             raw TTL stream (both tasks)
"""

from __future__ import annotations

import argparse

import pandas as pd

from memoranda import nwb
from memoranda.dandi import list_assets
from memoranda.log import get_logger
from memoranda.paths import MANIFESTS, ensure_dirs

log = get_logger("extract_trials")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    ensure_dirs()
    assets = list_assets()
    if args.only:
        s, e = args.only.split("-")
        assets = [a for a in assets if a.subject == int(s) and a.session == int(e)]

    sb, sc, pres, evs = [], [], [], []
    for a in assets:
        f = nwb.open_remote(a)
        tr = nwb.read_trials(f)
        tr.insert(0, "session", a.session)
        tr.insert(0, "subject", a.subject)
        (sb if a.task == "sternberg" else sc).append(tr)

        p = nwb.read_presentation(f)
        p.insert(0, "task", a.task)
        p.insert(0, "session", a.session)
        p.insert(0, "subject", a.subject)
        pres.append(p)

        e = nwb.read_events(f)
        e.insert(0, "task", a.task)
        e.insert(0, "session", a.session)
        e.insert(0, "subject", a.subject)
        evs.append(e)
        f.close()
        log.info(f"{a.key} {a.task:9s} trials={len(tr):3d} presentations={len(p):4d} events={len(e):5d}")

    if args.only:
        print(pd.concat(sb + sc).head())
        return
    pd.concat(sb).to_csv(MANIFESTS / "trials_sternberg.csv", index=False)
    pd.concat(sc).to_csv(MANIFESTS / "trials_screening.csv", index=False)
    pd.concat(pres).to_csv(MANIFESTS / "presentations.csv", index=False)
    pd.concat(evs).to_csv(MANIFESTS / "events.csv", index=False)
    log.info("wrote trials_sternberg / trials_screening / presentations / events")


if __name__ == "__main__":
    main()
