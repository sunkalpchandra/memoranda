#!/usr/bin/env python
"""Stage 49 — head-to-head model comparison (A7).

For each model we take its best layer under three criteria and compare models pairwise
with paired tests over the natural units of inference:
  * RSA (per screening session, MTL)                — paired t over sessions
  * similarity tuning (per MTL concept cell)         — Wilcoxon over cells
  * encoding r (per MTL concept cell), if available  — Wilcoxon over cells
Also asks whether *training objective* matters: supervised ImageNet (AlexNet…ViT) vs
self-supervised (DINOv2) vs language-aligned (CLIP), at the last layer.
Output: results/tables/A7_model_ranking.csv, A7_pairwise_*.csv, results/figures/A7_model_comparison.png
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.models.registry import DEFAULT_MODELS
from memoranda.paths import FIGURES, TABLES

OBJECTIVE = {"alexnet": "supervised", "vgg16": "supervised", "resnet18": "supervised", "resnet50": "supervised", "convnext_tiny": "supervised", "vit_b_16": "supervised", "dinov2_small": "self-supervised", "clip_vitb32": "language"}


def best_layer_table(per: pd.DataFrame, value: str, unit_cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Pick best (layer, view) per model by mean value; return summary and per-unit series."""
    rows, series = [], {}
    for m in DEFAULT_MODELS:
        d = per[per.model == m]
        if not len(d):
            continue
        g = d.groupby(["layer", "view"])[value].mean()
        (layer, view) = g.idxmax()
        dd = d[(d.layer == layer) & (d.view == view)].set_index(unit_cols)[value]
        series[m] = dd
        rows.append({"model": m, "objective": OBJECTIVE[m], "best_layer": layer, "view": view, "n": len(dd), "mean": dd.mean(), "sem": dd.std() / np.sqrt(len(dd))})
    return pd.DataFrame(rows), series


def pairwise(series: dict[str, pd.Series], test: str) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(series, 2):
        common = series[a].index.intersection(series[b].index)
        x, y = series[a].loc[common], series[b].loc[common]
        if len(common) < 5:
            continue
        p = sps.ttest_rel(x, y).pvalue if test == "t" else sps.wilcoxon(x - y).pvalue if (x - y).abs().sum() > 0 else np.nan
        rows.append({"model_a": a, "model_b": b, "n": len(common), "diff": float((x - y).mean()), "p": p})
    return pd.DataFrame(rows)


def main() -> None:
    out = []
    # RSA
    rsa = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    rsa = rsa[(rsa.region == "MTL") & (rsa.model != "baseline")]
    t_rsa, s_rsa = best_layer_table(rsa, "rho", ["subject"])
    t_rsa["criterion"] = "RSA (MTL, per session)"
    out.append(t_rsa)
    pairwise(s_rsa, "t").to_csv(TABLES / "A7_pairwise_rsa.csv", index=False)
    # similarity tuning
    st = pd.read_csv(TABLES / "A4_similarity_tuning_per_unit.csv")
    st = st[st.group == "MTL concept"]
    t_st, s_st = best_layer_table(st, "rho", ["subject", "unit"])
    t_st["criterion"] = "similarity tuning (MTL concept cells)"
    out.append(t_st)
    pairwise(s_st, "wilcoxon").to_csv(TABLES / "A7_pairwise_simtuning.csv", index=False)
    # encoding (optional)
    enc_path = TABLES / "A4_encoding_per_unit.csv"
    s_enc = None
    if enc_path.exists():
        enc = pd.read_csv(enc_path).dropna(subset=["r"])
        enc = enc[(enc.region == "MTL") & enc.concept_cell & (enc.model != "baseline")]
        t_enc, s_enc = best_layer_table(enc, "r", ["subject", "unit"])
        t_enc["criterion"] = "encoding r (MTL concept cells)"
        out.append(t_enc)
        pairwise(s_enc, "wilcoxon").to_csv(TABLES / "A7_pairwise_encoding.csv", index=False)
    rank = pd.concat(out, ignore_index=True)
    rank["rank_within_criterion"] = rank.groupby("criterion")["mean"].rank(ascending=False)
    rank.to_csv(TABLES / "A7_model_ranking.csv", index=False)
    print(rank.round(4).to_string(index=False))

    # objective contrast at the last layer of each model (similarity tuning + RSA)
    def last_layer(per, value, cols):
        rows = {}
        for m in DEFAULT_MODELS:
            d = per[per.model == m]
            layers = list(d.layer.unique())
            from memoranda.models.registry import get_spec

            spec_layers = list(get_spec(m).layers)
            last = [l for l in spec_layers if l in layers][-1]
            view = "cls" if (d[d.layer == last].view == "cls").any() else "gap"
            rows[m] = d[(d.layer == last) & (d.view == view)].set_index(cols)[value]
        return rows

    obj_rows = []
    for name, per, value, cols, test in [("RSA", rsa, "rho", ["subject"], "t"), ("simtuning", st, "rho", ["subject", "unit"], "wilcoxon")]:
        ll = last_layer(per, value, cols)
        for m, sr in ll.items():
            obj_rows.append({"criterion": name, "model": m, "objective": OBJECTIVE[m], "last_layer_mean": sr.mean(), "n": len(sr)})
    pd.DataFrame(obj_rows).to_csv(TABLES / "A7_objective_last_layer.csv", index=False)

    fig, axes = plt.subplots(1, len(out), figsize=(5 * len(out), 4.2))
    axes = np.atleast_1d(axes)
    colors = {"supervised": "#4C72B0", "self-supervised": "#55A868", "language": "#C44E52"}
    for ax, t in zip(axes, out):
        t = t.sort_values("mean")
        ax.barh(t.model, t["mean"], xerr=t.sem, color=[colors[o] for o in t.objective], capsize=2)
        ax.set_title(t.criterion.iloc[0], fontsize=10)
        ax.set_xlabel("mean at best layer")
        for y, (_, r) in enumerate(t.iterrows()):
            ax.text(r["mean"] + r.sem + 0.002, y, r.best_layer, va="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[0].legend(handles, colors.keys(), fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "A7_model_comparison.png", dpi=150)


if __name__ == "__main__":
    main()
