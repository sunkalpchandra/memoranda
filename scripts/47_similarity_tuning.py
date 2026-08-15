#!/usr/bin/env python
"""Stage 47 — similarity tuning of single concept cells.

For each screening MTL concept cell we take its preferred image and ask whether its
response to the *other* images falls off with the model's similarity to the preferred
image: Spearman ρ between (rate on image i) and (cosine sim(pref, i)) across the
non-preferred images, for each model layer. A positive ρ means the neuron
generalises along the model's similarity axis. Compared against a null obtained by
using a random other image as the anchor (same neuron), and summarised per layer
across cells (Wilcoxon), also for non-concept MTL cells and MFC concept cells.
Output: results/tables/A4_similarity_tuning.csv, results/figures/A4_similarity_tuning.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.features import list_layers, load_space
from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import FIGURES, MANIFESTS, TABLES

VIEW_PREF = {"vit": "cls", "dino": "cls", "clip": "cls", "cnn": "gap"}




def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    sel = sel[sel.task == "screening"]
    tun = tun[tun.task == "screening"]
    layers = [(m, l, VIEW_PREF[get_spec(m).family]) for m in DEFAULT_MODELS for (l, v) in list_layers(m) if v == VIEW_PREF[get_spec(m).family]]
    layers.append(("clip_vitb32", "embed", "gap"))
    groups = {
        "MTL concept": sel[(sel.region == "MTL") & sel.concept_cell],
        "MTL non-concept": sel[(sel.region == "MTL") & ~sel.concept_cell & (sel.mean_rate > 0.5)],
        "MFC concept": sel[(sel.region == "MFC") & sel.concept_cell],
    }
    rng = np.random.default_rng(0)
    rows = []
    for (m, l, v) in layers:
        X, stored = load_space(m, l, v, center=True, normalize=True)
        pos = {u: i for i, u in enumerate(stored)}
        for gname, g in groups.items():
            for r in g.itertuples():
                t = tun[(tun.subject == r.subject) & (tun.unit == r.unit)]
                uids = t.image_uid.to_numpy()
                rates = t.rate.to_numpy()
                if r.pref_image not in pos:
                    continue
                pref_i = np.where(uids == r.pref_image)[0][0]
                idx = np.array([pos[u] for u in uids])
                sims = X[idx] @ X[pos[r.pref_image]]
                mask = np.arange(len(uids)) != pref_i
                rho = sps.spearmanr(rates[mask], sims[mask]).statistic
                # null: anchor on a random non-preferred image
                j = rng.choice(np.where(mask)[0])
                sims_j = X[idx] @ X[idx[j]]
                mask_j = np.arange(len(uids)) != j
                rho_null = sps.spearmanr(rates[mask_j], sims_j[mask_j]).statistic
                rows.append({"group": gname, "subject": r.subject, "unit": r.unit, "area": r.area, "model": m, "layer": l, "view": v, "rho": rho, "rho_null": rho_null})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A4_similarity_tuning_per_unit.csv", index=False)

    def _summ(g):
        d = g.dropna(subset=["rho"])
        try:
            wp = sps.wilcoxon(d.rho).pvalue
            wp2 = sps.wilcoxon(d.rho - d.rho_null).pvalue
        except ValueError:
            wp = wp2 = np.nan
        return pd.Series({"n": len(d), "rho_mean": d.rho.mean(), "rho_sem": d.rho.std() / np.sqrt(max(len(d), 1)), "rho_null_mean": d.rho_null.mean(), "p_vs_zero": wp, "p_vs_null_anchor": wp2, "frac_pos": (d.rho > 0).mean()})

    summ = per.groupby(["group", "model", "layer", "view"]).apply(_summ, include_groups=False).reset_index()
    summ.to_csv(TABLES / "A4_similarity_tuning.csv", index=False)
    print(summ[summ.group == "MTL concept"].sort_values("rho_mean", ascending=False).head(15).round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    colors = {"MTL concept": "#C44E52", "MTL non-concept": "#DD8452", "MFC concept": "#4C72B0"}
    for ax, m in zip(axes.flat, DEFAULT_MODELS):
        lay = [l for l in get_spec(m).layers if ((summ.model == m) & (summ.layer == l)).any()]
        for gname in groups:
            d = summ[(summ.group == gname) & (summ.model == m)].set_index("layer").reindex(lay)
            ax.errorbar(range(len(lay)), d.rho_mean, d.rho_sem, fmt="o-", ms=4, color=colors[gname], label=gname, capsize=2)
        d0 = summ[(summ.group == "MTL concept") & (summ.model == m)].set_index("layer").reindex(lay)
        ax.plot(range(len(lay)), d0.rho_null_mean, "k:", lw=1, label="random-anchor null (MTL concept)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(lay)))
        ax.set_xticklabels(lay, rotation=60, fontsize=7)
        ax.set_title(m)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Spearman ρ(rate, sim to preferred image)\nover non-preferred images; mean ± sem over cells")
    axes[1, 0].set_ylabel("Spearman ρ")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Do single neurons generalise along DNN similarity to their preferred image?", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_similarity_tuning.png", dpi=150)


if __name__ == "__main__":
    main()
