#!/usr/bin/env python
"""Stage 63 — same architecture, different objective: ImageNet ResNet-50 vs CLIP ResNet-50.

Layer-matched comparison (layer1–4 + pooled) on (i) per-session MTL RSA and (ii) per-cell
similarity tuning of MTL concept cells; paired tests. Also ViT-B/16 (supervised) vs CLIP
ViT-B/32 vs DINOv2 at their last blocks as a looser transformer contrast.
Output: results/tables/A7_objective_contrast.csv, results/figures/A7_objective_contrast.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import load_space
from memoranda.paths import FIGURES, MANIFESTS, TABLES

PAIRS = [("layer1", "layer1"), ("layer2", "layer2"), ("layer3", "layer3"), ("layer4", "layer4"), ("avgpool", "attnpool")]


def main() -> None:
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    tun = tun[tun.task == "screening"]
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]

    rows = []
    # RSA
    for s in subjects:
        M, uids, _ = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        Dn = rsa.neural_rdm(M)
        for l_sup, l_clip in PAIRS:
            r_sup = rsa.compare_rdms(Dn, rsa.model_rdm("resnet50", l_sup, "gap", uids))
            r_clip = rsa.compare_rdms(Dn, rsa.model_rdm("clip_rn50", l_clip, "gap", uids))
            rows.append({"criterion": "RSA (MTL)", "unit_of_inference": s, "layer": l_sup, "supervised": r_sup, "clip": r_clip})
    # similarity tuning
    for l_sup, l_clip in PAIRS:
        Xs, ss = load_space("resnet50", l_sup, "gap", center=True, normalize=True)
        Xc, sc = load_space("clip_rn50", l_clip, "gap", center=True, normalize=True)
        ps, pc = {u: i for i, u in enumerate(ss)}, {u: i for i, u in enumerate(sc)}
        for r in cc.itertuples():
            t = tun[(tun.subject == r.subject) & (tun.unit == r.unit)]
            uids = t.image_uid.to_numpy()
            rates = t.rate.to_numpy()
            mask = uids != r.pref_image
            sim_s = Xs[[ps[u] for u in uids]] @ Xs[ps[r.pref_image]]
            sim_c = Xc[[pc[u] for u in uids]] @ Xc[pc[r.pref_image]]
            rows.append({"criterion": "similarity tuning (MTL concept cells)", "unit_of_inference": f"{r.subject}-{r.unit}", "layer": l_sup, "supervised": sps.spearmanr(rates[mask], sim_s[mask]).statistic, "clip": sps.spearmanr(rates[mask], sim_c[mask]).statistic})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A7_objective_contrast_per_unit.csv", index=False)
    out = []
    for (crit, layer), g in per.groupby(["criterion", "layer"]):
        d = (g["clip"] - g["supervised"]).dropna()
        p = sps.wilcoxon(d).pvalue if len(d) > 5 else np.nan
        out.append({"criterion": crit, "layer": layer, "n": len(d), "supervised_mean": g.supervised.mean(), "clip_mean": g["clip"].mean(), "clip_minus_sup": d.mean(), "sem": d.std() / np.sqrt(len(d)), "wilcoxon_p": p})
    res = pd.DataFrame(out)
    order = [p[0] for p in PAIRS]
    res["layer"] = pd.Categorical(res.layer, order)
    res = res.sort_values(["criterion", "layer"])
    res.to_csv(TABLES / "A7_objective_contrast.csv", index=False)
    print(res.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, crit in zip(axes, res.criterion.unique()):
        d = res[res.criterion == crit]
        x = np.arange(len(d))
        ax.plot(x, d.supervised_mean, "o-", color="#4C72B0", label="ResNet-50, ImageNet supervised")
        ax.plot(x, d.clip_mean, "o-", color="#C44E52", label="ResNet-50, CLIP (language)")
        for xi, (row) in zip(x, d.itertuples()):
            ax.text(xi, max(row.supervised_mean, row.clip_mean) + 0.003, f"p={row.wilcoxon_p:.2g}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_title(crit, fontsize=10)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Spearman ρ (mean)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Same architecture, different objective: layer-matched ResNet-50 vs CLIP-RN50")
    fig.tight_layout()
    fig.savefig(FIGURES / "A7_objective_contrast.png", dpi=150)


if __name__ == "__main__":
    main()
