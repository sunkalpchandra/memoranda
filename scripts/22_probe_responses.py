#!/usr/bin/env python
"""Stage 22 — probe-period responses of concept cells: match vs non-match, and DNN similarity.

For each Sternberg MTL concept cell (correct trials):
  * probe response (200–1000 ms after probe onset) when the probe IS its preferred picture,
    split by whether that picture was in the memory set (match, IN) or not (lure, OUT)
    → repetition/match modulation
  * probe response to NON-preferred probes as a function of their late-layer similarity
    to the preferred picture (Spearman ρ across the 4 other pictures — coarse but many trials)
Also for MFC probe-like cells: probe response vs load and IN/OUT.
Output: results/tables/A0_probe_match.csv, results/figures/A0_probe_match.png
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
from memoranda.features import load_features
from memoranda.paths import FIGURES, MANIFESTS, TABLES


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    cc = sel[(sel.task == "sternberg") & sel.concept_cell]
    trials = sternberg_trials_with_uids()
    X, stored = load_features("resnet50", "avgpool", "gap")
    X = X.astype(np.float64)
    X = X - X.mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    pos = {u: i for i, u in enumerate(stored)}
    rows = []
    for s, g in cc.groupby("subject"):
        spikes = neural.load_spikes(s, 2)
        t = trials[(trials.subject == s) & (trials.correct == 1)]
        p_on = t.timestamps_Probe.to_numpy()
        enc = t[["enc1", "enc2", "enc3"]].to_numpy(object)
        probe = t.probe.to_numpy()
        for r in g.itertuples():
            st = spikes[r.unit]
            resp = neural.count_in_windows(st, p_on, (0.2, 1.0)) / 0.8
            base = neural.count_in_windows(st, t.timestamps_FixationCross.to_numpy(), (0.0, 0.5)) / 0.5
            is_pref = probe == r.pref_image
            in_mem = np.array([r.pref_image in row for row in enc])
            match = is_pref & in_mem
            lure = is_pref & ~in_mem
            if match.sum() < 3 or lure.sum() < 3:
                continue
            # similarity of non-preferred probes to the preferred picture
            others = [u for u in np.unique(probe) if u != r.pref_image and u in pos]
            sims = {u: float(X[pos[u]] @ X[pos[r.pref_image]]) for u in others} if r.pref_image in pos else {}
            mean_by_probe = {u: resp[probe == u].mean() for u in others}
            rho = sps.spearmanr([sims[u] for u in others], [mean_by_probe[u] for u in others]).statistic if len(others) >= 4 else np.nan
            rows.append(
                {
                    "subject": s,
                    "unit": r.unit,
                    "area": r.area,
                    "region": r.region,
                    "n_match": int(match.sum()),
                    "n_lure": int(lure.sum()),
                    "probe_pref_match_hz": float(resp[match].mean()),
                    "probe_pref_lure_hz": float(resp[lure].mean()),
                    "probe_nonpref_hz": float(resp[~is_pref].mean()),
                    "baseline_hz": float(base.mean()),
                    "p_match_vs_lure": float(sps.mannwhitneyu(resp[match], resp[lure]).pvalue),
                    "rho_sim_nonpref_probe": rho,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A0_probe_match.csv", index=False)
    for region in ("MTL", "MFC"):
        d = df[df.region == region]
        if len(d) < 3:
            continue
        w = sps.wilcoxon(d.probe_pref_match_hz - d.probe_pref_lure_hz)
        print(f"{region}: n={len(d)} preferred probe: match {d.probe_pref_match_hz.mean():.2f} Hz vs lure {d.probe_pref_lure_hz.mean():.2f} Hz (non-pref {d.probe_nonpref_hz.mean():.2f}); Wilcoxon p={w.pvalue:.3g}; sim-tuning of non-pref probes: mean ρ={d.rho_sim_nonpref_probe.mean():.3f}, Wilcoxon p={sps.wilcoxon(d.rho_sim_nonpref_probe.dropna()).pvalue:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    d = df[df.region == "MTL"]
    lim = max(d.probe_pref_match_hz.max(), d.probe_pref_lure_hz.max()) * 1.05
    axes[0].scatter(d.probe_pref_lure_hz, d.probe_pref_match_hz, s=14, color="#C44E52", alpha=0.8)
    axes[0].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[0].set_xlabel("probe = preferred, NOT in memory (lure) [Hz]")
    axes[0].set_ylabel("probe = preferred, in memory (match) [Hz]")
    axes[0].set_title(f"MTL concept cells (n={len(d)}): probe response, match vs lure")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[1].hist(d.rho_sim_nonpref_probe.dropna(), bins=np.linspace(-1, 1, 11), color="#4C72B0", edgecolor="w")
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_xlabel("ρ(probe response, ResNet-50 similarity to preferred) over the 4 non-preferred pictures")
    axes[1].set_title("Similarity tuning at probe (non-preferred probes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "A0_probe_match.png", dpi=150)


if __name__ == "__main__":
    main()
