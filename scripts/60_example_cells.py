#!/usr/bin/env python
"""Stage 60 — exemplar concept cells: preferred picture, tuning over the other pictures vs
their late-layer similarity to the preferred picture, and the six most / least similar pictures.

Picks the 6 MTL concept cells with the highest similarity-tuning ρ (CLIP ln_post) among cells
with ≥ 3 Hz preferred response, plus 2 with ρ ≈ 0 for contrast.
Output: results/figures/A4_example_cells.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats as sps

from memoranda.features import load_space
from memoranda.paths import FIGURES, MANIFESTS, ROOT, TABLES


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    st = st[(st.group == "MTL concept") & (st.model == "clip_vitb32") & (st.layer == "ln_post")]
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")].merge(st[["subject", "unit", "rho"]], on=["subject", "unit"])
    strong = cc[cc.pref_rate >= 3].sort_values("rho", ascending=False).head(6)
    flat = cc[cc.pref_rate >= 3].assign(a=lambda d: d.rho.abs()).sort_values("a").head(2)
    picks = pd.concat([strong, flat])
    img = pd.read_csv(MANIFESTS / "images.csv").drop_duplicates("image_uid").set_index("image_uid")
    X, stored = load_space("clip_vitb32", "ln_post", "cls", center=True, normalize=True)
    pos = {u: i for i, u in enumerate(stored)}

    fig = plt.figure(figsize=(16, 2.6 * len(picks)))
    gs = fig.add_gridspec(len(picks), 9, width_ratios=[1.3, 2.4] + [0.9] * 7, wspace=0.15, hspace=0.5)
    for i, r in enumerate(picks.itertuples()):
        t = tun[(tun.subject == r.subject) & (tun.unit == r.unit) & (tun.task == "screening")].set_index("image_uid")
        uids = t.index.to_numpy()
        pref = r.pref_image
        sims = X[[pos[u] for u in uids]] @ X[pos[pref]]
        rates = t.rate.to_numpy()
        mask = uids != pref
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(Image.open(ROOT / "data" / img.loc[pref, "path"]).convert("RGB"))
        ax0.set_title(f"sub-{r.subject} u{r.unit} {r.area}\npreferred: {t.loc[pref, 'rate']:.1f} Hz", fontsize=8)
        ax0.axis("off")
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.scatter(sims[mask], rates[mask], s=14, color="#4C72B0", alpha=0.8)
        rho = sps.spearmanr(rates[mask], sims[mask]).statistic
        ax1.set_title(f"other {mask.sum()} pictures: ρ = {rho:.2f}", fontsize=8)
        ax1.set_xlabel("CLIP similarity to preferred", fontsize=7)
        ax1.set_ylabel("rate (Hz)", fontsize=7)
        ax1.tick_params(labelsize=6)
        order = np.argsort(-sims[mask])
        others = uids[mask]
        top = others[order[:4]]
        bottom = others[order[-3:]]
        for j, u in enumerate(list(top) + list(bottom)):
            ax = fig.add_subplot(gs[i, 2 + j])
            ax.imshow(Image.open(ROOT / "data" / img.loc[u, "path"]).convert("RGB"))
            ax.set_title(f"{'sim' if j < 4 else 'dissim'} {sims[uids == u][0]:.2f}\n{t.loc[u, 'rate']:.1f} Hz", fontsize=6.5)
            ax.axis("off")
    fig.suptitle("Exemplar MTL concept cells: preferred picture, response vs CLIP similarity, and the 4 most / 3 least similar pictures (last two rows: ρ ≈ 0)", fontsize=10)
    fig.savefig(FIGURES / "A4_example_cells.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
