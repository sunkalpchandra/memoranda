#!/usr/bin/env python
"""Stage 69 — model scale: small vs large network of the same family, depth-matched.

Contrasts CLIP ViT-B/32 (12 blocks, 88M) vs CLIP ViT-L/14 (24 blocks, 304M) and DINOv2-S/14
(22M) vs DINOv2-B/14 (86M) at matched *relative* depths (¼, ½, ¾, last block, final norm)
on (i) per-session MTL RSA and (ii) per-cell similarity tuning of MTL screening concept cells,
for both the ``cls`` and ``gap`` token views. Paired Wilcoxon (large − small) over sessions /
cells, as in scripts/63_objective_contrast.py; BH-FDR q over all contrasts.
Output: results/tables/A7_model_scale.csv (+ _per_unit.csv), results/figures/A7_model_scale.png
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

# family -> (small model, large model, [(depth label, small layer, large layer)])
CONTRASTS = {
    "CLIP ViT (B/32 → L/14)": (
        "clip_vitb32",
        "clip_vitl14",
        [
            ("¼ depth", "block2", "block5"),
            ("½ depth", "block5", "block11"),
            ("¾ depth", "block8", "block17"),
            ("last block", "block11", "block23"),
            ("final norm", "ln_post", "ln_post"),
        ],
    ),
    "DINOv2 (S/14 → B/14)": (
        "dinov2_small",
        "dinov2_base",
        [
            ("¼ depth", "block2", "block2"),
            ("½ depth", "block5", "block5"),
            ("¾ depth", "block8", "block8"),
            ("last block", "block11", "block11"),
            ("final norm", "norm", "norm"),
        ],
    ),
}
VIEWS = ("cls", "gap")
DEPTH_ORDER = ["¼ depth", "½ depth", "¾ depth", "last block", "final norm"]


def main() -> None:
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    tun = tun[tun.task == "screening"]
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]

    # neural RDMs once per session
    sess = []
    for s in subjects:
        M, uids, _ = rsa.session_response_matrix(s, 1, "MTL")
        if M is None:
            continue
        sess.append((s, uids, rsa.neural_rdm(M)))

    rows = []
    for family, (m_small, m_large, pairs) in CONTRASTS.items():
        for depth, l_small, l_large in pairs:
            for view in VIEWS:
                # (a) RSA
                for s, uids, Dn in sess:
                    r_s = rsa.compare_rdms(Dn, rsa.model_rdm(m_small, l_small, view, uids))
                    r_l = rsa.compare_rdms(Dn, rsa.model_rdm(m_large, l_large, view, uids))
                    rows.append({"criterion": "RSA (MTL)", "family": family, "depth": depth, "view": view, "layer_small": l_small, "layer_large": l_large, "unit_of_inference": s, "small": r_s, "large": r_l})
                # (b) similarity tuning
                Xs, ss = load_space(m_small, l_small, view, center=True, normalize=True)
                Xl, sl = load_space(m_large, l_large, view, center=True, normalize=True)
                ps, pl = {u: i for i, u in enumerate(ss)}, {u: i for i, u in enumerate(sl)}
                for r in cc.itertuples():
                    t = tun[(tun.subject == r.subject) & (tun.unit == r.unit)]
                    uids = t.image_uid.to_numpy()
                    rates = t.rate.to_numpy()
                    mask = uids != r.pref_image
                    sim_s = Xs[[ps[u] for u in uids]] @ Xs[ps[r.pref_image]]
                    sim_l = Xl[[pl[u] for u in uids]] @ Xl[pl[r.pref_image]]
                    rows.append({"criterion": "similarity tuning (MTL concept cells)", "family": family, "depth": depth, "view": view, "layer_small": l_small, "layer_large": l_large, "unit_of_inference": f"{r.subject}-{r.unit}", "small": sps.spearmanr(rates[mask], sim_s[mask]).statistic, "large": sps.spearmanr(rates[mask], sim_l[mask]).statistic})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A7_model_scale_per_unit.csv", index=False)

    out = []
    for (crit, family, depth, view), g in per.groupby(["criterion", "family", "depth", "view"], sort=False):
        d = (g["large"] - g["small"]).dropna()
        p = sps.wilcoxon(d).pvalue if len(d) > 5 and (d != 0).any() else np.nan
        out.append({"criterion": crit, "family": family, "depth": depth, "view": view, "layer_small": g.layer_small.iloc[0], "layer_large": g.layer_large.iloc[0], "n": len(d), "small_mean": g.small.mean(), "large_mean": g.large.mean(), "large_minus_small": d.mean(), "sem": d.std() / np.sqrt(len(d)), "frac_large_better": (d > 0).mean(), "wilcoxon_p": p})
    res = pd.DataFrame(out)
    m = res.wilcoxon_p.notna()
    res.loc[m, "fdr_q"] = sps.false_discovery_control(res.loc[m, "wilcoxon_p"].to_numpy(), method="bh")
    res["depth"] = pd.Categorical(res.depth, DEPTH_ORDER)
    res = res.sort_values(["criterion", "family", "view", "depth"])
    res.to_csv(TABLES / "A7_model_scale.csv", index=False)
    print(res.round(4).to_string(index=False))

    crits = list(res.criterion.unique())
    fams = list(CONTRASTS)
    fig, axes = plt.subplots(len(fams), len(crits), figsize=(11, 7.5), sharex=True)
    for i, family in enumerate(fams):
        for j, crit in enumerate(crits):
            ax = axes[i, j]
            x = np.arange(len(DEPTH_ORDER))
            for view, ls in zip(VIEWS, ("-", "--")):
                d = res[(res.criterion == crit) & (res.family == family) & (res.view == view)]
                ax.plot(x, d.small_mean, "o" + ls, color="#4C72B0", label=f"small ({view})")
                ax.plot(x, d.large_mean, "s" + ls, color="#C44E52", label=f"large ({view})")
                if view == "cls":
                    for xi, row in zip(x, d.itertuples()):
                        ax.text(xi, max(row.small_mean, row.large_mean) + 0.004, f"p={row.wilcoxon_p:.2g}", ha="center", fontsize=7)
            ax.set_xticks(x)
            ax.set_xticklabels(DEPTH_ORDER, fontsize=8)
            ax.set_title(f"{family} — {crit}", fontsize=9)
            ax.axhline(0, color="k", lw=0.6)
            ax.grid(alpha=0.25)
            if j == 0:
                ax.set_ylabel("Spearman ρ (mean)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Model scale at matched relative depth: small vs large (p = paired Wilcoxon, cls view)")
    fig.tight_layout()
    fig.savefig(FIGURES / "A7_model_scale.png", dpi=150)


if __name__ == "__main__":
    main()
