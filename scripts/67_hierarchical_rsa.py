#!/usr/bin/env python
"""Stage 67 — hierarchical (mixed-effects) model of RSA ρ: layer depth × region.

Input: results/tables/A3_rsa_per_session.csv (one ρ per session × region × model × layer × view)
and the ordered layer list of each model (memoranda.models.registry) → relative depth =
index / (n_layers − 1). Preferred view: 'gap' for CNNs, 'cls' for ViT / DINOv2 / CLIP.
The 'baseline' pseudo-model (category / low-level RDMs) is excluded.

Models (statsmodels MixedLM, REML; sessions nested in subjects):
1a. ρ ~ rel_depth × region (MTL vs MFC), random intercept per subject + random slope for
    rel_depth if it converges (fallback: random intercept only). Also fitted within each model
    family (cnn / transformer) for the figure.
1b. Same with region ∈ {amygdala, hippocampus, MFC} (reference MFC) + amygdala − hippocampus
    contrasts.
2.  Per-model depth slope for MTL, random intercept per subject.
3.  Last layer only: ρ ~ region (MTL vs MFC) × family (cnn vs transformer), random intercept
    per subject.

Fitting detail: statsmodels flags "MLE may be on the boundary" whenever a random-effect variance
is < 0.01 in absolute terms; ρ values are ~0.03 so every fit trips it spuriously. Models are
therefore fitted on 100·ρ and the fixed effects / random-effect s.d. are divided back by 100
(z and p are scale-free). Estimates in the table are in ρ units.

Outputs: results/tables/A8_hierarchical_rsa.csv (fixed effects: estimate, se, z, p, plus the
random-effects structure that was actually used and its s.d.) and
results/figures/A8_hierarchical_rsa.png.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from memoranda.models.registry import get_spec
from memoranda.paths import FIGURES, TABLES

REGION_COLORS = {"MTL": "#C44E52", "MFC": "#4C72B0", "amygdala": "#DD8452", "hippocampus": "#55A868"}
FAMILY_LABEL = {"cnn": "cnn", "vit": "transformer", "dino": "transformer", "clip": "transformer"}
FIT_METHODS = ["lbfgs", "bfgs", "powell", "nm"]


def preferred_view(model: str) -> str:
    return "cls" if get_spec(model).family in ("vit", "dino", "clip") else "gap"


def load() -> pd.DataFrame:
    d = pd.read_csv(TABLES / "A3_rsa_per_session.csv")
    d = d[d.model != "baseline"].copy()
    d = d[d.apply(lambda r: r["view"] == preferred_view(r["model"]), axis=1)]
    depth, fam = [], []
    for r in d.itertuples():
        lay = list(get_spec(r.model).layers)
        depth.append(lay.index(r.layer) / max(len(lay) - 1, 1) if r.layer in lay else np.nan)
        fam.append(FAMILY_LABEL[get_spec(r.model).family])
    d["rel_depth"] = depth
    d["family"] = fam
    d = d.dropna(subset=["rho", "rel_depth"]).reset_index(drop=True)
    d["subject"] = d.subject.astype(str)
    d["rho_pct"] = 100 * d.rho
    return d


def fit_mixed(formula: str, data: pd.DataFrame, structures: list[tuple[str, str]]):
    """Try random-effects structures in order; return (result, label, note).

    `formula` is written in ρ units ("rho ~ ..."); it is fitted on rho_pct = 100·ρ.
    A structure counts as failed if fitting raises, statsmodels reports non-convergence /
    a boundary solution (ConvergenceWarning), or the final random-effects covariance is
    singular / a variance collapses to ~0.
    """
    formula = formula.replace("rho ~", "rho_pct ~", 1)
    notes = []
    for label, re_formula in structures:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                res = smf.mixedlm(formula, data, groups=data["subject"], re_formula=re_formula).fit(reml=True, method=FIT_METHODS)
            except Exception as e:  # LinAlgError, ValueError, ...
                notes.append(f"{label}: raised {type(e).__name__}")
                continue
        bad = [str(x.message)[:60] for x in w if issubclass(x.category, ConvergenceWarning)]
        # judge singularity on the *final* covariance (statsmodels also warns transiently
        # during optimisation, which is harmless)
        cov = np.atleast_2d(np.asarray(res.cov_re))
        ev = np.linalg.eigvalsh(cov)
        collapsed = bool(ev.min() < 1e-8 or ev.min() / ev.max() < 1e-4)
        if res.converged and not bad and not collapsed:
            return res, label, "; ".join(notes) or "ok"
        why = "; ".join(bad) or ("not converged" if not res.converged else "RE covariance singular / variance ~0")
        notes.append(f"{label}: {why}")
    # last resort: whatever the simplest structure gives, flagged
    res = smf.mixedlm(formula, data, groups=data["subject"], re_formula="1").fit(reml=True, method=FIT_METHODS)
    return res, structures[-1][0] + " (unconverged)", "; ".join(notes)


def re_note(res) -> str:
    """Random-effect standard deviations in ρ units."""
    sd = np.sqrt(np.diag(np.asarray(res.cov_re))) / 100
    names = list(res.cov_re.index) if hasattr(res.cov_re, "index") else [f"re{i}" for i in range(len(sd))]
    parts = [f"sd({n})={v:.4f}" for n, v in zip(names, sd)]
    parts.append(f"sd(resid)={np.sqrt(res.scale) / 100:.4f}")
    return "RE " + ", ".join(parts)


def collect(res, analysis: str, label: str, note: str, data: pd.DataFrame, rows: list) -> None:
    note = f"{re_note(res)}; fits: {note}"
    for term in res.fe_params.index:
        rows.append({"analysis": analysis, "term": term, "estimate": res.fe_params[term] / 100, "se": res.bse_fe[term] / 100, "z": res.tvalues[term], "p": res.pvalues[term], "n_obs": int(res.nobs), "n_subjects": data.subject.nunique(), "random_structure": label, "note": note})


def contrast(res, analysis: str, name: str, weights: dict, label: str, note: str, data: pd.DataFrame, rows: list) -> None:
    """Linear contrast of fixed effects (weights over fe_params names)."""
    names = list(res.fe_params.index)
    L = np.zeros(len(res.params))
    for k, v in weights.items():
        L[names.index(k)] = v
    try:
        t = res.t_test(L)
        est, se, z, p = float(np.squeeze(t.effect)), float(np.squeeze(t.sd)), float(np.squeeze(t.tvalue)), float(np.squeeze(t.pvalue))
    except Exception:
        est = float(sum(v * res.fe_params[k] for k, v in weights.items()))
        cov = res.cov_params().loc[list(weights), list(weights)].to_numpy()
        w = np.array(list(weights.values()))
        se = float(np.sqrt(w @ cov @ w))
        z = est / se
        from scipy import stats as sps

        p = float(2 * sps.norm.sf(abs(z)))
    rows.append({"analysis": analysis, "term": name, "estimate": est / 100, "se": se / 100, "z": z, "p": p, "n_obs": int(res.nobs), "n_subjects": data.subject.nunique(), "random_structure": label, "note": f"{re_note(res)}; fits: {note}"})


def main() -> None:
    d = load()
    rows: list[dict] = []
    fits: dict[str, tuple] = {}
    slope_structs = [("RI subject + RS rel_depth", "1 + rel_depth"), ("RI subject", "1")]

    # ---- 1a: depth × region, MTL vs MFC (all models pooled, then per family)
    two = d[d.region.isin(["MTL", "MFC"])].copy()
    f1 = "rho ~ rel_depth * C(region, Treatment('MFC'))"
    res, label, note = fit_mixed(f1, two, slope_structs)
    collect(res, "1a depth x region (MTL vs MFC), all models", label, note, two, rows)
    contrast(res, "1a depth x region (MTL vs MFC), all models", "rel_depth slope within MTL", {"rel_depth": 1, "rel_depth:C(region, Treatment('MFC'))[T.MTL]": 1}, label, note, two, rows)
    fits["all"] = (res, label)
    for fam in ("cnn", "transformer"):
        sub = two[two.family == fam]
        res_f, label_f, note_f = fit_mixed(f1, sub, slope_structs)
        collect(res_f, f"1a depth x region (MTL vs MFC), {fam} models", label_f, note_f, sub, rows)
        contrast(res_f, f"1a depth x region (MTL vs MFC), {fam} models", "rel_depth slope within MTL", {"rel_depth": 1, "rel_depth:C(region, Treatment('MFC'))[T.MTL]": 1}, label_f, note_f, sub, rows)
        fits[fam] = (res_f, label_f)

    # ---- 1b: depth × region, amygdala vs hippocampus vs MFC
    three = d[d.region.isin(["amygdala", "hippocampus", "MFC"])].copy()
    f1b = "rho ~ rel_depth * C(region, Treatment('MFC'))"
    res3, label3, note3 = fit_mixed(f1b, three, slope_structs)
    a3 = "1b depth x region (amygdala / hippocampus / MFC)"
    collect(res3, a3, label3, note3, three, rows)
    contrast(res3, a3, "amygdala - hippocampus (intercept)", {"C(region, Treatment('MFC'))[T.amygdala]": 1, "C(region, Treatment('MFC'))[T.hippocampus]": -1}, label3, note3, three, rows)
    contrast(res3, a3, "amygdala - hippocampus (depth slope)", {"rel_depth:C(region, Treatment('MFC'))[T.amygdala]": 1, "rel_depth:C(region, Treatment('MFC'))[T.hippocampus]": -1}, label3, note3, three, rows)
    for reg in ("amygdala", "hippocampus"):
        contrast(res3, a3, f"rel_depth slope within {reg}", {"rel_depth": 1, f"rel_depth:C(region, Treatment('MFC'))[T.{reg}]": 1}, label3, note3, three, rows)
    fits["three"] = (res3, label3)

    # ---- 2: per-model depth slope, MTL, RI subject
    mtl = d[d.region == "MTL"]
    for m in sorted(mtl.model.unique(), key=lambda x: (FAMILY_LABEL[get_spec(x).family], x)):
        sub = mtl[mtl.model == m]
        res_m, label_m, note_m = fit_mixed("rho ~ rel_depth", sub, [("RI subject", "1")])
        collect(res_m, f"2 MTL depth slope, {m}", label_m, note_m, sub, rows)

    # ---- 3: last layer, region × family
    last = d.loc[d.groupby(["model", "region"]).rel_depth.idxmax()]
    last_layers = last.groupby("model").layer.first().to_dict()
    last = d.merge(last[["model", "layer"]].drop_duplicates(), on=["model", "layer"])
    last = last[last.region.isin(["MTL", "MFC"])].copy()
    f3 = "rho ~ C(region, Treatment('MFC')) * C(family, Treatment('cnn'))"
    res_l, label_l, note_l = fit_mixed(f3, last, [("RI subject", "1")])
    a3l = "3 last layer: region (MTL vs MFC) x family (cnn vs transformer)"
    note_l = f"{note_l}; last layers: " + ", ".join(f"{k}={v}" for k, v in sorted(last_layers.items()))
    collect(res_l, a3l, label_l, note_l, last, rows)
    contrast(res_l, a3l, "MTL - MFC within transformer", {"C(region, Treatment('MFC'))[T.MTL]": 1, "C(region, Treatment('MFC'))[T.MTL]:C(family, Treatment('cnn'))[T.transformer]": 1}, label_l, note_l, last, rows)
    contrast(res_l, a3l, "transformer - cnn within MTL", {"C(family, Treatment('cnn'))[T.transformer]": 1, "C(region, Treatment('MFC'))[T.MTL]:C(family, Treatment('cnn'))[T.transformer]": 1}, label_l, note_l, last, rows)

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "A8_hierarchical_rsa.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(out.drop(columns=["note"]).round(4).to_string(index=False))
    print("\nrandom-effects notes:")
    for a, n in out.drop_duplicates("analysis")[["analysis", "note"]].itertuples(index=False):
        print(f"  {a}: {n}")

    # ---- figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    xs = np.linspace(0, 1, 50)
    rng = np.random.default_rng(0)
    for ax, fam in zip(axes[:2], ("cnn", "transformer")):
        res_f, label_f = fits[fam]
        sub = two[two.family == fam]
        for reg in ("MFC", "MTL"):
            s = sub[sub.region == reg]
            ax.scatter(s.rel_depth + rng.uniform(-0.015, 0.015, len(s)), s.rho, s=9, alpha=0.3, color=REGION_COLORS[reg], lw=0)
            b = res_f.fe_params / 100
            y = b["Intercept"] + b["rel_depth"] * xs
            if reg == "MTL":
                y = y + b["C(region, Treatment('MFC'))[T.MTL]"] + b["rel_depth:C(region, Treatment('MFC'))[T.MTL]"] * xs
            ax.plot(xs, y, color=REGION_COLORS[reg], lw=2.2, label=reg)
        slope = res_f.fe_params["rel_depth:C(region, Treatment('MFC'))[T.MTL]"] / 100
        p = res_f.pvalues["rel_depth:C(region, Treatment('MFC'))[T.MTL]"]
        n_models = sub.model.nunique()
        ax.set_title(f"{fam} models (n = {n_models})\ndepth × MTL: {slope:+.3f}, p = {p:.2g}  [{label_f}]", fontsize=9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("relative layer depth")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    res3, label3 = fits["three"]
    b = res3.fe_params / 100
    for reg in ("MFC", "hippocampus", "amygdala"):
        s = three[three.region == reg]
        ax.scatter(s.rel_depth + rng.uniform(-0.015, 0.015, len(s)), s.rho, s=9, alpha=0.3, color=REGION_COLORS[reg], lw=0)
        y = b["Intercept"] + b["rel_depth"] * xs
        if reg != "MFC":
            y = y + b[f"C(region, Treatment('MFC'))[T.{reg}]"] + b[f"rel_depth:C(region, Treatment('MFC'))[T.{reg}]"] * xs
        ax.plot(xs, y, color=REGION_COLORS[reg], lw=2.2, label=reg)
    pa = res3.pvalues["rel_depth:C(region, Treatment('MFC'))[T.amygdala]"]
    ph = res3.pvalues["rel_depth:C(region, Treatment('MFC'))[T.hippocampus]"]
    ax.set_title(f"all 8 models: subregions\ndepth × amygdala p = {pa:.2g}, × hippocampus p = {ph:.2g}  [{label3}]", fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("relative layer depth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Spearman ρ (session RDM, model RDM)")
    ymax = 0.22
    n_clip = int((d[d.region.isin(["MTL", "MFC", "amygdala", "hippocampus"])].rho > ymax).sum())
    axes[0].set_ylim(-0.08, ymax)
    fig.suptitle(f"Mixed-effects RSA: ρ ~ layer depth × region, random intercept (+ slope) per patient; points = sessions × layers (y clipped, {n_clip} points > {ymax})", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "A8_hierarchical_rsa.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
