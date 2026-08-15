#!/usr/bin/env python
"""Stage 24 — does the working-memory *maintenance* code inherit visual similarity structure?

Per Sternberg session (MTL units): 5 memoranda → RDM from (i) encoding responses
(200–1000 ms after each encoding onset, all loads) and (ii) maintenance responses on load-1
correct trials (0–2.5 s after maintenance onset; only one item in memory). Each RDM has 10
pairs; we rank pairs within subject and pool across subjects (210 pairs), then Spearman
against DNN pair distances (also ranked within subject). Significance by shuffling picture
identity within subject (2000×). Also encoding-vs-maintenance RDM agreement.
Output: results/tables/A3_maintenance_geometry.csv, results/figures/A3_maintenance_geometry.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import rankdata

from memoranda import neural
from memoranda.analysis import rsa
from memoranda.analysis.behavior import sternberg_trials_with_uids
from memoranda.dandi import list_assets
from memoranda.paths import FIGURES, MANIFESTS, TABLES

SPACES = {"alexnet conv1": ("alexnet", "conv1", "gap"), "resnet50 avgpool": ("resnet50", "avgpool", "gap"), "clip ln_post": ("clip_vitb32", "ln_post", "cls"), "dinov2 norm": ("dinov2_small", "norm", "gap")}


def main() -> None:
    trials = sternberg_trials_with_uids()
    subjects = sorted({a.subject for a in list_assets() if a.session == 2})
    ut_all = pd.read_csv(MANIFESTS / "units.csv")
    labels = pd.read_csv(MANIFESTS / "image_labels.csv")
    recs = []
    for s in subjects:
        spikes = neural.load_spikes(s, 2)
        ut = ut_all[(ut_all.subject == s) & (ut_all.session == 2)]
        mtl = set(ut.loc[ut.region == "MTL", "unit"])
        sp = {u: st for u, st in spikes.items() if u in mtl}
        if len(sp) < 3:
            continue
        t = trials[(trials.subject == s) & (trials.correct == 1)]
        # encoding: every encoding presentation
        enc_on, enc_img = [], []
        for col_t, col_i in (("timestamps_Encoding1", "enc1"), ("timestamps_Encoding2", "enc2"), ("timestamps_Encoding3", "enc3")):
            m = t[col_t] > 0
            enc_on += t.loc[m, col_t].tolist()
            enc_img += t.loc[m, col_i].tolist()
        enc_on, enc_img = np.array(enc_on), np.array(enc_img, dtype=object)
        R_enc, units = neural.rate_matrix(sp, enc_on, neural.RESP_WIN)
        # maintenance: load-1 trials
        t1 = t[t.loads == 1]
        R_m, _ = neural.rate_matrix(sp, t1.timestamps_Maintenance.to_numpy(), (0.0, 2.5))
        m_img = t1.enc1.to_numpy(dtype=object)
        uids = np.array(sorted(set(enc_img)))
        if len(uids) < 4:
            continue
        M_enc = np.stack([R_enc[enc_img == u].mean(0) for u in uids])
        M_m = np.stack([R_m[m_img == u].mean(0) for u in uids])
        D_enc = rsa.upper(rsa.neural_rdm(M_enc))
        D_m = rsa.upper(rsa.neural_rdm(M_m))
        rec = {"subject": s, "n_units": len(units), "n_images": len(uids), "enc": D_enc, "maint": D_m, "uids": uids}
        for name, (m, l, v) in SPACES.items():
            rec[name] = rsa.upper(rsa.model_rdm(m, l, v, uids))
        rec["category"] = rsa.upper(rsa.category_rdm(uids, labels))
        recs.append(rec)

    def pooled(key_a, key_b, n_perm=2000, seed=0):
        rng = np.random.default_rng(seed)
        A = np.concatenate([rankdata(r[key_a]) for r in recs])
        B = np.concatenate([rankdata(r[key_b]) for r in recs])
        obs = sps.spearmanr(A, B).statistic
        null = np.zeros(n_perm)
        for k in range(n_perm):
            Bp = []
            for r in recs:
                n = r["n_images"]
                D = np.zeros((n, n))
                D[np.triu_indices(n, 1)] = r[key_b]
                D = D + D.T
                perm = rng.permutation(n)
                Bp.append(rankdata(rsa.upper(D[np.ix_(perm, perm)])))
            null[k] = sps.spearmanr(A, np.concatenate(Bp)).statistic
        return obs, (np.sum(null >= obs) + 1) / (n_perm + 1), float(null.std())

    rows = []
    for neural_key in ("enc", "maint"):
        for ref in list(SPACES) + ["category"]:
            rho, p, sd = pooled(neural_key, ref)
            rows.append({"neural": neural_key, "reference": ref, "n_subjects": len(recs), "n_pairs": sum(len(r["enc"]) for r in recs), "rho_pooled": rho, "p_perm": p, "null_sd": sd})
    rho, p, sd = pooled("maint", "enc")
    rows.append({"neural": "maint", "reference": "encoding RDM", "n_subjects": len(recs), "n_pairs": sum(len(r["enc"]) for r in recs), "rho_pooled": rho, "p_perm": p, "null_sd": sd})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "A3_maintenance_geometry.csv", index=False)
    print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    refs = list(SPACES) + ["category"]
    w = 0.38
    for i, (nk, col) in enumerate((("enc", "#C44E52"), ("maint", "#4C72B0"))):
        d = df[(df.neural == nk) & df.reference.isin(refs)].set_index("reference").reindex(refs)
        ax.bar(np.arange(len(refs)) + (i - 0.5) * w, d.rho_pooled, w, yerr=d.null_sd, color=col, label={"enc": "encoding (200–1000 ms)", "maint": "maintenance, load 1 (0–2.5 s)"}[nk], capsize=2)
        for x, (rr, pv) in enumerate(zip(d.rho_pooled, d.p_perm)):
            ax.text(x + (i - 0.5) * w, rr + 0.01, f"p={pv:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(refs)))
    ax.set_xticklabels(refs, fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("pooled Spearman ρ (within-subject ranked pairs)")
    ax.set_title(f"Sternberg MTL geometry of the 5 memoranda vs DNN similarity ({len(recs)} subjects, {sum(len(r['enc']) for r in recs)} pairs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "A3_maintenance_geometry.png", dpi=150)


if __name__ == "__main__":
    main()
