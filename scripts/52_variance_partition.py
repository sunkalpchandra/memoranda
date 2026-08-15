#!/usr/bin/env python
"""Stage 52 — variance partitioning of the MTL RDM between CLIP, ResNet-50 and category.

Per screening session (MTL, MTL concept cells): rank-RDM regressions with all subsets of
{CLIP ln_post, ResNet-50 avgpool, category} give R² for each subset; unique and shared
portions follow by inclusion–exclusion (commonality analysis). Group-level means ± sem and
Wilcoxon tests of each unique component.
Output: results/tables/A3_variance_partition.csv, results/figures/A3_variance_partition.png
"""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.paths import FIGURES, MANIFESTS, TABLES

PRED = {"clip": ("clip_vitb32", "ln_post", "cls"), "resnet50": ("resnet50", "avgpool", "gap"), "category": None}


def r2_subset(Dn, preds: dict[str, np.ndarray], keys) -> float:
    if not keys:
        return 0.0
    return rsa.rdm_regression(Dn, {k: preds[k] for k in keys})["r2"]


def commonality(r2: dict[frozenset, float], names) -> dict[str, float]:
    """Unique + shared components via inclusion–exclusion over the 3 predictors."""
    A, B, C = names
    f = lambda *k: r2[frozenset(k)]  # noqa: E731
    out = {
        f"unique_{A}": f(A, B, C) - f(B, C),
        f"unique_{B}": f(A, B, C) - f(A, C),
        f"unique_{C}": f(A, B, C) - f(A, B),
        f"shared_{A}_{B}": f(A, C) + f(B, C) - f(C) - f(A, B, C),
        f"shared_{A}_{C}": f(A, B) + f(B, C) - f(B) - f(A, B, C),
        f"shared_{B}_{C}": f(A, B) + f(A, C) - f(A) - f(A, B, C),
        "total": f(A, B, C),
    }
    out[f"shared_{A}_{B}_{C}"] = out["total"] - sum(v for k, v in out.items() if k != "total")
    return out


def main() -> None:
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    names = list(PRED)
    rows = []
    for s in subjects:
        _, uids, _ = rsa.session_response_matrix(s, 1, None)
        preds = {"clip": rsa.model_rdm(*PRED["clip"], uids), "resnet50": rsa.model_rdm(*PRED["resnet50"], uids), "category": rsa.category_rdm(uids, labels)}
        for region in ("MTL", "MTL_concept"):
            if region == "MTL":
                M, u2, _ = rsa.session_response_matrix(s, 1, "MTL")
            else:
                M, u2, units = rsa.session_response_matrix(s, 1, "MTL")
                if M is not None:
                    keep = sel[(sel.subject == s) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
                    mask = np.isin(units, keep)
                    M = M[:, mask] if mask.sum() >= 2 else None
            if M is None:
                continue
            Dn = rsa.neural_rdm(M)
            r2 = {}
            for k in range(0, 4):
                for sub in itertools.combinations(names, k):
                    r2[frozenset(sub)] = r2_subset(Dn, preds, sub)
            rows.append({"subject": s, "region": region, **commonality(r2, names)})
    per = pd.DataFrame(rows)
    per.to_csv(TABLES / "A3_variance_partition_per_session.csv", index=False)
    comps = [c for c in per.columns if c.startswith(("unique_", "shared_", "total"))]
    out = []
    for region, g in per.groupby("region"):
        for c in comps:
            x = g[c].dropna()
            try:
                p = sps.wilcoxon(x).pvalue
            except ValueError:
                p = np.nan
            out.append({"region": region, "component": c, "n": len(x), "mean": x.mean(), "sem": x.std() / np.sqrt(len(x)), "wilcoxon_p": p})
    summ = pd.DataFrame(out)
    summ.to_csv(TABLES / "A3_variance_partition.csv", index=False)
    print(summ.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, region in zip(axes, ("MTL", "MTL_concept")):
        d = summ[(summ.region == region) & (summ.component != "total")].set_index("component").reindex([c for c in comps if c != "total"])
        ax.bar(range(len(d)), d["mean"] * 100, yerr=d["sem"] * 100, color=["#C44E52" if c.startswith("unique") else "#8da0cb" for c in d.index], capsize=2)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([c.replace("_", "\n") for c in d.index], fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{region}: total R² = {summ[(summ.region == region) & (summ.component == 'total')]['mean'].iloc[0]*100:.2f}%")
    axes[0].set_ylabel("share of MTL RDM variance (rank R², %)")
    fig.suptitle("Commonality analysis: CLIP ln_post vs ResNet-50 avgpool vs category RDM")
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_variance_partition.png", dpi=150)


if __name__ == "__main__":
    main()
