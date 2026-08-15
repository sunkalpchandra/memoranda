#!/usr/bin/env python
"""Stage 61 — sparse concept cells vs broadly tuned cells: who generalises along DNN similarity?

For MTL concept cells (screening): similarity-tuning ρ (CLIP ln_post, ResNet-50 avgpool)
against depth of selectivity (DoS), Treves–Rolls sparseness, and preferred/non-preferred
ratio; also split cells into sparse (DoS ≥ median) vs broad and compare ρ. Cells whose
preferred picture contains a face vs not.
Output: results/tables/A4_sparsity_vs_generalisation.csv, results/figures/A4_sparsity_vs_generalisation.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.paths import FIGURES, MANIFESTS, TABLES


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    faces = pd.read_csv(MANIFESTS / "faces.csv").set_index("image_uid")
    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    st = st[st.group == "MTL concept"]
    cc = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")].copy()
    for name, (m, l) in {"rho_clip": ("clip_vitb32", "ln_post"), "rho_resnet": ("resnet50", "avgpool"), "rho_conv1": ("alexnet", "conv1")}.items():
        s = st[(st.model == m) & (st.layer == l)][["subject", "unit", "rho"]].rename(columns={"rho": name})
        cc = cc.merge(s, on=["subject", "unit"], how="left")
    cc["pref_face"] = faces.loc[cc.pref_image, "face_found"].to_numpy()
    cc["ratio"] = cc.pref_rate / (cc.nonpref_rate + 0.1)
    rows = []
    for x in ("dos", "sparseness", "ratio", "pref_rate", "mean_rate"):
        for y in ("rho_clip", "rho_resnet", "rho_conv1"):
            r, p = sps.spearmanr(cc[x], cc[y], nan_policy="omit")
            rows.append({"x": x, "y": y, "spearman": r, "p": p, "n": int(cc[[x, y]].dropna().shape[0])})
    med = cc.dos.median()
    sparse, broad = cc[cc.dos >= med], cc[cc.dos < med]
    for y in ("rho_clip", "rho_resnet"):
        rows.append({"x": f"sparse (DoS≥{med:.2f}) vs broad", "y": y, "spearman": np.nan, "p": sps.mannwhitneyu(sparse[y].dropna(), broad[y].dropna()).pvalue, "n": len(cc), "mean_sparse": sparse[y].mean(), "mean_broad": broad[y].mean()})
        f1, f0 = cc[cc.pref_face == 1], cc[cc.pref_face == 0]
        rows.append({"x": "pref has face vs not", "y": y, "spearman": np.nan, "p": sps.mannwhitneyu(f1[y].dropna(), f0[y].dropna()).pvalue, "n": len(cc), "mean_sparse": f1[y].mean(), "mean_broad": f0[y].mean()})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A4_sparsity_vs_generalisation.csv", index=False)
    print(df.round(4).to_string(index=False))
    cc.to_csv(TABLES / "A4_concept_cells_with_generalisation.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(cc.dos, cc.rho_clip, s=16, c=np.where(cc.pref_face == 1, "#C44E52", "#4C72B0"), alpha=0.8)
    axes[0].set_xlabel("depth of selectivity (1 = fires to one picture only)")
    axes[0].set_ylabel("similarity-tuning ρ (CLIP ln_post)")
    r = df[(df.x == "dos") & (df.y == "rho_clip")].iloc[0]
    axes[0].set_title(f"Sparser cells generalise less: ρ_s = {r.spearman:.2f}, p = {r.p:.1e}\nred: preferred picture has a face")
    axes[1].scatter(cc.pref_rate, cc.rho_clip, s=16, c=np.where(cc.pref_face == 1, "#C44E52", "#4C72B0"), alpha=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("preferred-picture rate (Hz)")
    axes[1].set_ylabel("similarity-tuning ρ")
    axes[1].set_title("vs response strength")
    axes[2].hist([f1.rho_clip.dropna(), f0.rho_clip.dropna()], bins=np.linspace(-0.4, 0.9, 14), label=["preferred has face", "no face"], color=["#C44E52", "#4C72B0"], alpha=0.85)
    axes[2].set_xlabel("similarity-tuning ρ (CLIP)")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Face-preferring vs other concept cells")
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_sparsity_vs_generalisation.png", dpi=150)


if __name__ == "__main__":
    main()
