#!/usr/bin/env python
"""Stage 75 — influence of the two largest patients on the RSA result.

Patients 13 and 14 hold 42 % of MTL concept cells and the largest per-session ρ. Recompute the
MTL RSA summary (mean ρ, t-test, depth contrast last-vs-first layer) with each session left out
and with both left out. Output: results/tables/A3_influence.csv
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as sps

from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import TABLES

VIEW = {"vit": "cls", "dino": "cls", "clip": "cls", "cnn": "gap"}


def summarise(per: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for m in DEFAULT_MODELS:
        v = VIEW[get_spec(m).family]
        lays = [l for l in get_spec(m).layers if ((per.model == m) & (per.layer == l) & (per.view == v)).any()]
        first, last = lays[0], lays[-1]
        d_last = per[(per.model == m) & (per.layer == last) & (per.view == v)].set_index("subject").rho
        d_first = per[(per.model == m) & (per.layer == first) & (per.view == v)].set_index("subject").rho
        common = d_last.index.intersection(d_first.index)
        t_last, p_last = sps.ttest_1samp(d_last, 0)
        t_diff, p_diff = sps.ttest_rel(d_last[common], d_first[common])
        rows.append({"subset": label, "model": m, "last_layer": last, "n": len(d_last), "rho_last_mean": d_last.mean(), "rho_last_median": d_last.median(), "p_last_vs0": p_last, "rho_first_mean": d_first.mean(), "last_minus_first": float((d_last[common] - d_first[common]).mean()), "p_depth_paired": p_diff})
    return rows


def main() -> None:
    per = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    per = per[(per.region == "MTL") & (per.model != "baseline")]
    rows = summarise(per, "all sessions")
    for drop in ([13], [14], [13, 14]):
        rows += summarise(per[~per.subject.isin(drop)], f"without sub-{'/'.join(map(str, drop))}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A3_influence.csv", index=False)
    show = df[df.model.isin(["clip_vitb32", "resnet50", "alexnet"])]
    print(show[["subset", "model", "n", "rho_last_mean", "rho_last_median", "p_last_vs0", "last_minus_first", "p_depth_paired"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
