#!/usr/bin/env python
"""Stage 21 — replicate persistent activity of concept cells during WM maintenance.

For every Sternberg concept cell (selected on encoding presentations only) we compare
the firing rate in the maintenance period (0–2.5 s after maintenance onset, correct
trials) between trials in which its preferred image is held in memory and trials in
which it is not. Kyzar et al. report 4.76 ± 4.71 vs 2.67 ± 3.66 Hz (paired one-sided
t-test p = 1.4e−21) for MTL concept cells.
Output: results/tables/A0_maintenance_replication.csv, results/figures/A0_maintenance.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis.behavior import sternberg_trials_with_uids
from memoranda.paths import FIGURES, MANIFESTS, TABLES


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "sternberg") & sel.concept_cell]
    trials = sternberg_trials_with_uids()
    rows = []
    for s, g in cc.groupby("subject"):
        spikes = neural.load_spikes(s, 2)
        t = trials[(trials.subject == s) & (trials.correct == 1)]
        onsets = t.timestamps_Maintenance.to_numpy()
        enc = t[["enc1", "enc2", "enc3"]].to_numpy(object)
        base_on = t.timestamps_FixationCross.to_numpy()
        for r in g.itertuples():
            st = spikes[r.unit]
            maint = neural.count_in_windows(st, onsets, (0.0, 2.5)) / 2.5
            base = neural.count_in_windows(st, base_on, (0.0, 0.5)) / 0.5
            in_mem = np.array([r.pref_image in row for row in enc])
            if in_mem.sum() < 3 or (~in_mem).sum() < 3:
                continue
            rows.append(
                {
                    "subject": s,
                    "unit": r.unit,
                    "area": r.area,
                    "region": r.region,
                    "pref_image": r.pref_image,
                    "n_pref_trials": int(in_mem.sum()),
                    "maint_pref_hz": float(maint[in_mem].mean()),
                    "maint_nonpref_hz": float(maint[~in_mem].mean()),
                    "baseline_hz": float(base.mean()),
                    "p_unit": float(sps.mannwhitneyu(maint[in_mem], maint[~in_mem], alternative="greater").pvalue),
                }
            )
    df = pd.DataFrame(rows)
    df["persistent"] = df.p_unit < 0.05
    df.to_csv(TABLES / "A0_maintenance_replication.csv", index=False)
    for region in ("MTL", "MFC"):
        d = df[df.region == region]
        t, p = sps.ttest_rel(d.maint_pref_hz, d.maint_nonpref_hz, alternative="greater")
        print(f"{region}: n={len(d)} pref {d.maint_pref_hz.mean():.2f}±{d.maint_pref_hz.std():.2f} Hz vs non-pref {d.maint_nonpref_hz.mean():.2f}±{d.maint_nonpref_hz.std():.2f} Hz; paired one-sided t p={p:.2e}; {d.persistent.mean()*100:.1f}% units individually p<0.05")
    for area in ("amygdala", "hippocampus"):
        d = df[df.area == area]
        t, p = sps.ttest_rel(d.maint_pref_hz, d.maint_nonpref_hz, alternative="greater")
        print(f"  {area}: n={len(d)} {d.maint_pref_hz.mean():.2f} vs {d.maint_nonpref_hz.mean():.2f} Hz p={p:.2e}")

    fig, ax = plt.subplots(figsize=(5, 5))
    d = df[df.region == "MTL"]
    lim = max(d.maint_pref_hz.max(), d.maint_nonpref_hz.max()) * 1.05
    ax.scatter(d.maint_nonpref_hz, d.maint_pref_hz, s=14, c=["#C44E52" if p else "#4C72B0" for p in d.persistent], alpha=0.8)
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("maintenance rate, preferred NOT in memory (Hz)")
    ax.set_ylabel("maintenance rate, preferred in memory (Hz)")
    ax.set_title(f"MTL concept cells (n={len(d)}): persistent activity\nred = individually significant")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    fig.tight_layout()
    fig.savefig(FIGURES / "A0_maintenance.png", dpi=150)


if __name__ == "__main__":
    main()
