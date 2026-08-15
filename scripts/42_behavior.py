#!/usr/bin/env python
"""Stage 42 — does representational similarity among the memoranda predict WM behaviour?

Tests (per feature space; per-subject slopes → group t-test / Wilcoxon):
  OUT trials  : RT and accuracy vs max similarity between lure probe and encoded set
  IN  trials  : RT and accuracy vs max similarity between probe and the *other* encoded items
  load ≥ 2    : RT and accuracy vs mean within-set similarity
Outputs: results/tables/A6_behavior_similarity.csv, results/figures/A6_behavior.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memoranda.analysis import behavior as B
from memoranda.paths import FIGURES, TABLES

SPACES = {
    "clip": ("clip_vitb32", "embed", "gap"),
    "resnet50": ("resnet50", "avgpool", "gap"),
    "dinov2": ("dinov2_small", "norm", "cls"),
    "alexnet_conv2": ("alexnet", "conv2", "gap"),
    "vgg16_conv5": ("vgg16", "conv5_3", "gap"),
}


def main() -> None:
    t = B.sternberg_trials_with_uids()
    for name, (m, l, v) in SPACES.items():
        t = B.add_similarity(t, m, l, v, name)
    t.to_csv(TABLES / "A6_trials_with_similarity.csv", index=False)

    rows = []
    for name in SPACES:
        tests = [
            ("OUT", "rt", f"{name}_sim_probe_max", (t.probe_in == 0) & (t.correct == 1)),
            ("OUT", "correct", f"{name}_sim_probe_max", t.probe_in == 0),
            ("IN", "rt", f"{name}_sim_probe_other", (t.probe_in == 1) & (t.correct == 1) & (t.loads >= 2)),
            ("IN", "correct", f"{name}_sim_probe_other", (t.probe_in == 1) & (t.loads >= 2)),
            ("ALL", "rt", f"{name}_sim_within", (t.correct == 1) & (t.loads >= 2)),
            ("ALL", "correct", f"{name}_sim_within", t.loads >= 2),
        ]
        for cond, y, x, mask in tests:
            sl = B.per_subject_slopes(t, x, y, subset=mask)
            g = B.group_test(sl)
            rows.append({"space": name, "condition": cond, "outcome": y, "predictor": x.split("_", 1)[1], **g})
    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "A6_behavior_similarity.csv", index=False)
    print(res.round(4).to_string(index=False))

    # figure: OUT-trial RT vs lure similarity (CLIP), pooled after per-subject z-scoring
    d = t[(t.probe_in == 0) & (t.correct == 1)].copy()
    d["rt_z"] = d.groupby("subject").rt.transform(lambda x: (x - x.mean()) / x.std())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axes, ["clip", "resnet50", "dinov2"]):
        x = d[f"{name}_sim_probe_max"]
        bins = np.quantile(x.dropna(), np.linspace(0, 1, 7))
        cat = pd.cut(x, bins, include_lowest=True)
        m = d.groupby(cat, observed=True).rt_z.agg(["mean", "sem", "size"])
        centers = [iv.mid for iv in m.index]
        ax.errorbar(centers, m["mean"], m["sem"], fmt="o-", color="#C44E52")
        r = res[(res.space == name) & (res.condition == "OUT") & (res.outcome == "rt")].iloc[0]
        ax.set_title(f"{name}: lure similarity vs RT (OUT, correct)\nmean slope={r.mean_slope:.3f}, p={r.p:.3f}")
        ax.set_xlabel("max cosine sim(probe, encoded set)")
        ax.set_ylabel("RT (z within subject)")
    fig.tight_layout()
    fig.savefig(FIGURES / "A6_behavior.png", dpi=150)


if __name__ == "__main__":
    main()
