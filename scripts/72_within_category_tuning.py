#!/usr/bin/env python
"""Stage 72 — does similarity tuning survive *within* the preferred picture's category?

Script 47 showed that an MTL concept cell's response to the non-preferred pictures
correlates with their late-layer DNN similarity to the preferred picture. A trivial
explanation is category: a face-preferring cell fires for the other faces, and faces are
similar to each other in every DNN space. Here we recompute the per-cell similarity tuning
ρ (Spearman between rate and cosine similarity to the preferred picture) restricted to

  * ``same``     — non-preferred pictures of the SAME category as the preferred picture (≥ 6)
  * ``diff``     — pictures of DIFFERENT categories only
  * ``all``      — all non-preferred pictures (as in script 47)
  * ``partial``  — all non-preferred pictures, ranks of rate and of similarity both
                   residualised on a same-category indicator (category-partialled Spearman)
  * ``category`` — Spearman between rate and the same-category indicator alone (the
                   "fires for the category" baseline, for scale)

Random-anchor null (as in script 47): anchor on a random non-preferred picture, apply the
same restriction relative to that anchor (its category), exclude the preferred picture;
averaged over N_NULL anchors per cell. Summaries: Wilcoxon over cells vs 0 and vs the null,
split by preferred-picture category (face_person vs other) and by area.

Output: results/tables/A4_within_category_tuning.csv (+ _per_cell.csv),
        results/figures/A4_within_category_tuning.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.features import load_space
from memoranda.paths import FIGURES, MANIFESTS, TABLES

SPACES = {
    "clip ln_post": ("clip_vitb32", "ln_post", "cls"),
    "resnet50 avgpool": ("resnet50", "avgpool", "gap"),
    "alexnet conv1": ("alexnet", "conv1", "gap"),
}
SCOPES = ["all", "same", "diff", "partial", "category"]
MIN_N = 6
N_NULL = 20


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_N or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(sps.spearmanr(a, b).statistic)


def _partial(rate: np.ndarray, sim: np.ndarray, same: np.ndarray) -> float:
    """Spearman ρ(rate, sim) with a binary same-category indicator partialled out of both."""
    if len(rate) < MIN_N or same.all() or (~same).all():
        return np.nan
    D = np.column_stack([np.ones(len(rate)), same.astype(float)])
    rr = sps.rankdata(rate)
    rs = sps.rankdata(sim)
    rr = rr - D @ np.linalg.lstsq(D, rr, rcond=None)[0]
    rs = rs - D @ np.linalg.lstsq(D, rs, rcond=None)[0]
    if np.ptp(rr) == 0 or np.ptp(rs) == 0:
        return np.nan
    return float(sps.pearsonr(rr, rs).statistic)


def _tuning(rates: np.ndarray, sims: np.ndarray, cats: np.ndarray, anchor: int, exclude: np.ndarray) -> dict[str, float]:
    """All five ρ's for one anchor picture; ``exclude`` masks pictures never used."""
    keep = ~exclude
    keep[anchor] = False
    same = cats == cats[anchor]
    r, s = rates[keep], sims[keep]
    sm = same[keep]
    return {
        "all": _spearman(r, s),
        "same": _spearman(r[sm], s[sm]),
        "diff": _spearman(r[~sm], s[~sm]),
        "partial": _partial(r, s, sm),
        "category": _spearman(r, sm.astype(float)),
    }


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    lab = pd.read_csv(MANIFESTS / "image_labels.csv").set_index("image_uid").category
    sel = sel[(sel.task == "screening") & sel.concept_cell & (sel.region == "MTL")]
    tun = tun[tun.task == "screening"]
    spaces = {}
    for k, (m, l, v) in SPACES.items():
        X, stored = load_space(m, l, v, center=True, normalize=True)
        spaces[k] = (X, {u: i for i, u in enumerate(stored)})

    rng = np.random.default_rng(0)
    rows = []
    for r in sel.itertuples():
        t = tun[(tun.subject == r.subject) & (tun.unit == r.unit)]
        uids = t.image_uid.to_numpy()
        rates = t.rate.to_numpy(float)
        cats = lab.reindex(uids).to_numpy()
        pref_i = int(np.where(uids == r.pref_image)[0][0])
        pref_cat = cats[pref_i]
        n_same = int(((cats == pref_cat) & (np.arange(len(uids)) != pref_i)).sum())
        # null anchors: random non-preferred pictures whose category has ≥ MIN_N other non-pref members
        n_cat = pd.Series(cats).value_counts()
        cand = [i for i in range(len(uids)) if i != pref_i and n_cat[cats[i]] - 1 - int(cats[i] == pref_cat) >= MIN_N]
        anchors = rng.choice(cand, size=min(N_NULL, len(cand)), replace=False) if cand else []
        for sname, (X, pos) in spaces.items():
            F = X[[pos[u] for u in uids]]
            obs = _tuning(rates, F @ F[pref_i], cats, pref_i, np.zeros(len(uids), bool))
            excl = np.zeros(len(uids), bool)
            excl[pref_i] = True
            nulls = [_tuning(rates, F @ F[j], cats, int(j), excl) for j in anchors]
            for scope in SCOPES:
                nv = np.array([d[scope] for d in nulls], float)
                rows.append({
                    "subject": r.subject, "unit": r.unit, "area": r.area, "hemisphere": r.hemisphere,
                    "pref_image": r.pref_image, "pref_category": pref_cat, "pref_is_face": pref_cat == "face_person",
                    "n_same": n_same, "n_diff": len(uids) - 1 - n_same, "space": sname, "scope": scope,
                    "rho": obs[scope], "rho_null": np.nanmean(nv) if np.isfinite(nv).any() else np.nan,
                    "n_null": int(np.isfinite(nv).sum()),
                })
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A4_within_category_tuning_per_cell.csv", index=False)

    subsets = {
        "all cells": np.ones(len(per), bool),
        "pref face_person": per.pref_is_face.to_numpy(),
        "pref other": ~per.pref_is_face.to_numpy(),
        "amygdala": (per.area == "amygdala").to_numpy(),
        "hippocampus": (per.area == "hippocampus").to_numpy(),
    }

    def _w(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 5 or np.all(x == 0):
            return np.nan
        return sps.wilcoxon(x).pvalue

    srows = []
    for sub, mask in subsets.items():
        for sname in SPACES:
            for scope in SCOPES:
                d = per[mask & (per.space == sname) & (per.scope == scope)].dropna(subset=["rho"])
                if d.empty:
                    continue
                dn = d.dropna(subset=["rho_null"])
                srows.append({
                    "subset": sub, "space": sname, "scope": scope, "n": len(d),
                    "rho_mean": d.rho.mean(), "rho_sem": d.rho.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                    "rho_median": d.rho.median(), "frac_pos": (d.rho > 0).mean(),
                    "rho_null_mean": dn.rho_null.mean(), "p_vs_zero": _w(d.rho),
                    "p_vs_null_anchor": _w(dn.rho - dn.rho_null),
                })
    summ = pd.DataFrame(srows)

    # paired contrasts within cells (same − diff, partial − all) and between-cell contrasts (face vs other, amy vs hip)
    crows = []
    for sname in SPACES:
        p = per[per.space == sname].pivot_table(index=["subject", "unit"], columns="scope", values="rho")
        meta = per[(per.space == sname) & (per.scope == "all")].set_index(["subject", "unit"])[["pref_is_face", "area", "hemisphere"]]
        p = p.join(meta)
        for a, b in (("same", "diff"), ("same", "all"), ("partial", "all")):
            d = (p[a] - p[b]).dropna()
            crows.append({"space": sname, "contrast": f"{a} - {b} (paired)", "n": len(d), "diff_mean": d.mean(), "p": _w(d)})
        for scope in ("same", "partial", "all", "category"):
            f, o = p.loc[p.pref_is_face, scope].dropna(), p.loc[~p.pref_is_face, scope].dropna()
            crows.append({"space": sname, "contrast": f"{scope}: face - other (MW)", "n": len(f) + len(o), "diff_mean": f.mean() - o.mean(), "p": sps.mannwhitneyu(f, o).pvalue if len(f) > 2 and len(o) > 2 else np.nan})
            am, hp = p.loc[p.area == "amygdala", scope].dropna(), p.loc[p.area == "hippocampus", scope].dropna()
            crows.append({"space": sname, "contrast": f"{scope}: amygdala - hippocampus (MW)", "n": len(am) + len(hp), "diff_mean": am.mean() - hp.mean(), "p": sps.mannwhitneyu(am, hp).pvalue if len(am) > 2 and len(hp) > 2 else np.nan})
            amy = p[p.area == "amygdala"]
            R, L = amy.loc[amy.hemisphere == "R", scope].dropna(), amy.loc[amy.hemisphere == "L", scope].dropna()
            crows.append({"space": sname, "contrast": f"{scope}: right - left amygdala (MW)", "n": len(R) + len(L), "diff_mean": R.mean() - L.mean(), "p": sps.mannwhitneyu(R, L).pvalue if len(R) > 2 and len(L) > 2 else np.nan})
    contr = pd.DataFrame(crows)
    contr["subset"] = "contrast"
    out = pd.concat([summ, contr], ignore_index=True)
    out.to_csv(TABLES / "A4_within_category_tuning.csv", index=False)

    pd.set_option("display.width", 200)
    print(summ[summ.subset == "all cells"].round(4).to_string(index=False))
    print(summ[(summ.scope == "same")].round(4).to_string(index=False))
    print(summ[(summ.scope == "partial")].round(4).to_string(index=False))
    print(contr.round(4).to_string(index=False))

    # ---- figure: one panel per space, bars = scope × subset, null as black ticks
    plot_scopes = ["all", "same", "diff", "partial", "category"]
    plot_subsets = ["all cells", "pref face_person", "pref other", "amygdala", "hippocampus"]
    colors = {"all cells": "#4C4C4C", "pref face_person": "#C44E52", "pref other": "#DD8452", "amygdala": "#4C72B0", "hippocampus": "#55A868"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    w = 0.16
    for ax, sname in zip(axes, SPACES):
        for k, sub in enumerate(plot_subsets):
            d = summ[(summ.space == sname) & (summ.subset == sub)].set_index("scope").reindex(plot_scopes)
            x = np.arange(len(plot_scopes)) + (k - 2) * w
            ax.bar(x, d.rho_mean, w, yerr=d.rho_sem, color=colors[sub], label=sub, capsize=2, linewidth=0)
            ax.plot(x, d.rho_null_mean, "_", color="k", ms=8, mew=1.2, label="random-anchor null" if k == 0 else None)
            for xi, rm, se, pz in zip(x, d.rho_mean, d.rho_sem, d.p_vs_zero):
                if np.isfinite(pz) and pz < 0.05:
                    ax.text(xi, max(rm + se, 0) + 0.005, "*", ha="center", va="bottom", fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(plot_scopes)))
        ax.set_xticklabels(["all other\npictures", "same category\nonly", "different\ncategories", "category-\npartialled", "category\nindicator"], fontsize=8)
        ax.set_title(sname)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Spearman ρ(rate, sim to preferred picture)\nmean ± sem over MTL concept cells")
    axes[2].text(0.99, 0.02, "category indicator = ρ(rate, same-category flag); identical in every panel", transform=axes[2].transAxes, ha="right", fontsize=7, color="0.35")
    axes[0].legend(fontsize=7, loc="lower right", ncol=2, title="* p < 0.05 vs 0 (Wilcoxon)", title_fontsize=7)
    fig.suptitle("Similarity tuning within vs across the preferred picture's category (MTL screening concept cells)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "A4_within_category_tuning.png", dpi=150)


if __name__ == "__main__":
    main()
