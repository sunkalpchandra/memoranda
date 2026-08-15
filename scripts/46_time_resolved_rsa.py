#!/usr/bin/env python
"""Stage 46 — time-resolved RSA (screening task).

Sliding 200-ms windows (step 25 ms) from −300 to +1300 ms around image onset.
For each window and session we build the MTL (and MFC) population RDM and
correlate it with a few reference RDMs: an early layer (AlexNet conv1), a mid
layer (AlexNet conv4 / ResNet-50 layer2), late layers (ResNet-50 avgpool, CLIP
ln_post), the category RDM and the low-level RDM. Curves are averaged over
sessions with bootstrap CIs; a cluster-free per-window one-sample t-test marks
significant windows.
Output: results/tables/A3_time_resolved.csv, results/figures/A3_time_resolved.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.log import get_logger
from memoranda.paths import FIGURES, MANIFESTS, TABLES

log = get_logger("trsa")
REFS = {
    "alexnet conv1": ("alexnet", "conv1", "gap"),
    "alexnet conv4": ("alexnet", "conv4", "gap"),
    "resnet50 avgpool": ("resnet50", "avgpool", "gap"),
    "clip ln_post": ("clip_vitb32", "ln_post", "cls"),
}
WIN = 0.2
STEP = 0.025
CENTERS = np.arange(-0.3 + WIN / 2, 1.3 - WIN / 2 + 1e-9, STEP)


def main() -> None:
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    stats_df = pd.read_csv(MANIFESTS / "image_stats.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        spikes = neural.load_spikes(s, 1)
        pres = neural.load_presentations(s, 1)
        pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
        onsets = pres.onset.to_numpy()
        lab = pres.image_uid.to_numpy()
        uids = np.unique(lab)
        ut = neural.load_units_table(s, 1)
        refs = {k: rsa.model_rdm(*v, uids) for k, v in REFS.items()}
        refs["category"] = rsa.category_rdm(uids, labels)
        refs["low-level"] = rsa.lowlevel_rdm(uids, stats_df)
        for region in ("MTL", "MFC"):
            keep = set(ut.loc[ut.region == region, "unit"])
            sp = {u: st for u, st in spikes.items() if u in keep}
            if len(sp) < 2:
                continue
            for c in CENTERS:
                R, _ = neural.rate_matrix(sp, onsets, (c - WIN / 2, c + WIN / 2))
                M = np.stack([R[lab == u].mean(0) for u in uids])
                Dn = rsa.neural_rdm(M)
                for name, Dm in refs.items():
                    rows.append({"subject": s, "region": region, "t": round(float(c), 4), "ref": name, "rho": rsa.compare_rdms(Dn, Dm)})
        log.info(f"sub-{s:02d} done")
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_time_resolved_per_session.csv", index=False)

    summ = per.groupby(["region", "ref", "t"]).rho.agg(["mean", "sem", "count"]).reset_index()
    pv = per.groupby(["region", "ref", "t"]).rho.apply(lambda x: sps.ttest_1samp(x.dropna(), 0).pvalue if len(x.dropna()) > 2 else np.nan).rename("p").reset_index()
    summ = summ.merge(pv)
    summ.to_csv(TABLES / "A3_time_resolved.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    colors = {"alexnet conv1": "#9ecae1", "alexnet conv4": "#4292c6", "resnet50 avgpool": "#08306b", "clip ln_post": "#C44E52", "category": "grey", "low-level": "#bbbbbb"}
    for ax, region in zip(axes, ("MTL", "MFC")):
        d = summ[summ.region == region]
        for name in list(REFS) + ["category", "low-level"]:
            g = d[d.ref == name].sort_values("t")
            ax.plot(g.t * 1000, g["mean"], color=colors[name], label=name, lw=1.8 if name in ("clip ln_post", "resnet50 avgpool") else 1.2)
            ax.fill_between(g.t * 1000, g["mean"] - g["sem"], g["mean"] + g["sem"], color=colors[name], alpha=0.15)
            sig = g[g.p < 0.05]
            if len(sig) and name in ("clip ln_post", "alexnet conv1", "category"):
                yoff = {"clip ln_post": -0.012, "alexnet conv1": -0.016, "category": -0.02}[name]
                ax.scatter(sig.t * 1000, np.full(len(sig), yoff), s=6, color=colors[name], marker="s")
        ax.axvline(0, color="k", lw=0.8)
        ax.axvspan(0, 1000, color="k", alpha=0.04)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{region}: sliding-window RSA (200 ms windows)")
        ax.set_xlabel("time from image onset (ms)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Spearman ρ (neural RDM, reference RDM)\nmean ± sem over sessions")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_time_resolved.png", dpi=150)
    peak = summ[summ.region == "MTL"].sort_values("mean", ascending=False).groupby("ref").head(1)
    print(peak[["ref", "t", "mean", "sem", "p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
