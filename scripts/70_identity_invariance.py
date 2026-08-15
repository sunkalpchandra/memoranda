#!/usr/bin/env python
"""Stage 70 — second-rater famous labels and identity invariance.

(a) Inter-rater agreement for the hand famous-person labels: a second rater labelled a random
    120-picture subset (seed 0 over the 342 real pictures) blind to configs/famous_labels.csv
    (configs/famous_labels_rater2.csv: image_uid, famous, identity).  Cohen's κ for `famous`
    (with a bootstrap CI), raw agreement, and identity agreement among the pictures both raters
    named (names normalised: lower-case, parenthetical role removed, multi-person lists sorted).
    Rater 1 coded non-human fictional characters (Yoda, Vault Boy) as famous = 0 /
    fictional_character = 1; rater 2 coded them famous = 1, so κ is also given for
    "famous OR fictional_character".
    -> results/tables/A5_famous_interrater.csv

(b) Identity invariance.  Identities in configs/famous_labels.csv that appear in >= 2 pictures
    (Tom Cruise x3; Buzz Aldrin, Jack Nicholson, Anne Hathaway, Pink Floyd, Beyonce, Meryl Streep,
    Johnny Depp x2).  Three levels:
      * DNN: CLIP ViT-B/32 ln_post cls (centred, L2-normalised) cosine similarity of every
        same-identity pair and its percentile rank among each member's similarities to all other
        340 pictures (0.5 = chance, 1 = nearest neighbour).
      * single unit: for every screening MTL concept cell whose preferred picture belongs to such
        an identity AND whose session also showed another picture of the same identity — the
        z-score / percentile of the same-identity picture within the cell's non-preferred rate
        distribution, and top-3 / top-10 % counts.  (Only two sessions ever co-showed an identity
        pair — sub-01 Jack Nicholson, sub-03 Pink Floyd — and no concept cell there prefers one
        of the pair pictures, so this set is empty; the table therefore also carries an
        exploratory row for every MTL unit in those sessions: percentile of |z_A - z_B| among that
        unit's |z_A - z_k|, |z_B - z_k| distances, and the percentile rank of each pair picture
        given the other as anchor.)
      * population: correlation-distance RDM over all MTL units of the co-showing sessions
        (rsa.session_response_matrix / rsa.neural_rdm); percentile of d(A,B) among all pairs and
        among pairs involving A or B (small = the pair is neurally close).
    -> results/tables/A5_identity_invariance.csv (one row per unit x pair / population x pair /
       DNN x pair), results/tables/A5_identity_invariance_summary.csv,
       results/figures/A5_identity_invariance.png
"""

from __future__ import annotations

import itertools
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

from memoranda.analysis import rsa
from memoranda.features import load_space
from memoranda.paths import CONFIGS, FIGURES, MANIFESTS, TABLES

DNN = ("clip_vitb32", "ln_post", "cls")


# ------------------------------------------------------------------ (a) inter-rater
def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def _norm_identity(s: str) -> str:
    s = re.sub(r"\(.*?\)", "", str(s)).lower()
    s = s.replace("é", "e")
    parts = sorted(p.strip() for p in s.split(";") if p.strip())
    return "; ".join(parts)


def interrater() -> pd.DataFrame:
    r1 = pd.read_csv(CONFIGS / "famous_labels.csv")
    r2 = pd.read_csv(CONFIGS / "famous_labels_rater2.csv")
    m = r2.merge(r1, on="image_uid", suffixes=("_r2", "_r1"), how="left")
    n = len(m)
    f1, f2 = m.famous_r1.to_numpy(int), m.famous_r2.to_numpy(int)
    kappa = cohen_kappa(f1, f2)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boots.append(cohen_kappa(f1[idx], f2[idx]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    f1x = (m.famous_r1.astype(int) | m.fictional_character.fillna(0).astype(int)).to_numpy(int)
    kappa_x = cohen_kappa(f1x, f2)
    dis = m[f1 != f2].image_uid.tolist()
    both = m.dropna(subset=["identity_r1", "identity_r2"])
    same = both.identity_r1.map(_norm_identity) == both.identity_r2.map(_norm_identity)
    id_dis = both.loc[~same, ["image_uid", "identity_r1", "identity_r2"]]
    only1 = m[m.identity_r1.notna() & m.identity_r2.isna()].image_uid.tolist()
    only2 = m[m.identity_r1.isna() & m.identity_r2.notna()].image_uid.tolist()
    rows = [
        ("n_pictures", n, ""),
        ("famous_pos_rater1", int(f1.sum()), ""),
        ("famous_pos_rater2", int(f2.sum()), ""),
        ("famous_raw_agreement", float((f1 == f2).mean()), ""),
        ("famous_cohen_kappa", kappa, f"bootstrap 95% CI {lo:.3f}-{hi:.3f}"),
        ("famous_kappa_ci_lo", lo, ""),
        ("famous_kappa_ci_hi", hi, ""),
        ("famous_n_disagree", len(dis), "; ".join(dis)),
        ("famous_or_fictional_kappa", kappa_x, "rater1 famous|fictional_character vs rater2 famous"),
        ("famous_or_fictional_n_disagree", int((f1x != f2).sum()), "; ".join(m[f1x != f2].image_uid)),
        ("identity_n_both_named", len(both), ""),
        ("identity_agreement", float(same.mean()), "normalised names, multi-person lists sorted"),
        ("identity_n_disagree", int((~same).sum()), "; ".join(f"{r.image_uid}: {r.identity_r1} vs {r.identity_r2}" for r in id_dis.itertuples())),
        ("identity_named_by_rater1_only", len(only1), "; ".join(only1)),
        ("identity_named_by_rater2_only", len(only2), "; ".join(only2)),
    ]
    out = pd.DataFrame(rows, columns=["metric", "value", "detail"])
    out.to_csv(TABLES / "A5_famous_interrater.csv", index=False)
    print(out.to_string(index=False))
    return out


# ------------------------------------------------------------------ (b) identity invariance
def _pct(x: float, ref: np.ndarray, higher_is_better: bool = True) -> float:
    """Fraction of reference values that x beats (ties half). 1 = extreme in the 'good' direction."""
    ref = np.asarray(ref, float)
    if higher_is_better:
        return float(((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref))
    return float(((ref > x).sum() + 0.5 * (ref == x).sum()) / len(ref))


def identity_pairs(labels: pd.DataFrame) -> pd.DataFrame:
    vc = labels.identity.dropna().value_counts()
    ids = vc[vc >= 2].index
    rows = []
    for ident in ids:
        pics = sorted(labels.loc[labels.identity == ident, "image_uid"])
        for a, b in itertools.combinations(pics, 2):
            rows.append({"identity": ident, "uid_a": a, "uid_b": b})
    return pd.DataFrame(rows)


def dnn_level(pairs: pd.DataFrame) -> pd.DataFrame:
    X, stored = load_space(*DNN, center=True, normalize=True)
    pos = {u: i for i, u in enumerate(stored)}
    S = X @ X.T
    n = len(stored)
    rows = []
    for p in pairs.itertuples():
        i, j = pos[p.uid_a], pos[p.uid_b]
        others_i = np.delete(S[i], [i, j])
        others_j = np.delete(S[j], [i, j])
        rows.append({
            "level": "dnn", "model": "/".join(DNN), "identity": p.identity, "uid_a": p.uid_a, "uid_b": p.uid_b,
            "sim_ab": float(S[i, j]),
            "pct_b_given_a": _pct(S[i, j], others_i), "pct_a_given_b": _pct(S[j, i], others_j),
            "rank_b_given_a": int((others_i > S[i, j]).sum() + 1), "rank_a_given_b": int((others_j > S[i, j]).sum() + 1),
            "n_reference": n - 2,
        })
    return pd.DataFrame(rows)


def coshown_sessions(pairs: pd.DataFrame, tun: pd.DataFrame) -> pd.DataFrame:
    shown = tun.drop_duplicates(["subject", "image_uid"]).groupby("subject").image_uid.apply(set)
    rows = []
    for p in pairs.itertuples():
        for s, imgs in shown.items():
            if p.uid_a in imgs and p.uid_b in imgs:
                rows.append({"subject": int(s), "identity": p.identity, "uid_a": p.uid_a, "uid_b": p.uid_b})
    return pd.DataFrame(rows)


def unit_level(co: pd.DataFrame, sel: pd.DataFrame, tun: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict rows: MTL concept cells preferring a pair picture in a co-showing session.
    Exploratory rows: every MTL unit of the co-showing sessions."""
    strict, explore = [], []
    for c in co.itertuples():
        units = sel[(sel.subject == c.subject) & (sel.region == "MTL")]
        for u in units.itertuples():
            t = tun[(tun.subject == u.subject) & (tun.unit == u.unit)].set_index("image_uid")
            if c.uid_a not in t.index or c.uid_b not in t.index:
                continue
            r = t.rate
            z = (r - r.mean()) / (r.std(ddof=1) or 1.0)
            others = [k for k in t.index if k not in (c.uid_a, c.uid_b)]
            d_ab = abs(z[c.uid_a] - z[c.uid_b])
            d_ref = np.concatenate([np.abs(z[c.uid_a] - z[others]), np.abs(z[c.uid_b] - z[others])])
            base = {
                "subject": u.subject, "unit": u.unit, "area": u.area, "hemisphere": u.hemisphere,
                "concept_cell": bool(u.concept_cell), "pref_image": u.pref_image,
                "identity": c.identity, "uid_a": c.uid_a, "uid_b": c.uid_b,
                "rate_a": float(r[c.uid_a]), "rate_b": float(r[c.uid_b]), "z_a": float(z[c.uid_a]), "z_b": float(z[c.uid_b]),
                "pct_pair_distance": _pct(d_ab, d_ref, higher_is_better=False),
                "pct_b_given_a": _pct(r[c.uid_b], r[others].to_numpy()),
                "pct_a_given_b": _pct(r[c.uid_a], r[others].to_numpy()),
                "n_other_pictures": len(others),
            }
            explore.append({"level": "unit_exploratory", **base})
            if u.concept_cell and u.pref_image in (c.uid_a, c.uid_b):
                pref, other = (c.uid_a, c.uid_b) if u.pref_image == c.uid_a else (c.uid_b, c.uid_a)
                nonpref = r.drop(pref)
                zo = (r[other] - nonpref.mean()) / (nonpref.std(ddof=1) or 1.0)
                pct = _pct(r[other], nonpref.drop(other).to_numpy())
                rank = int((nonpref.drop(other) > r[other]).sum() + 1)
                strict.append({
                    "level": "concept_cell", **base, "same_identity_z_in_nonpref": float(zo),
                    "same_identity_pct_in_nonpref": pct, "same_identity_rank_in_nonpref": rank,
                    "top3": rank <= 3, "top10pct": pct >= 0.9,
                })
    return pd.DataFrame(strict), pd.DataFrame(explore)


def population_level(co: pd.DataFrame, region: str = "MTL") -> pd.DataFrame:
    rows = []
    for c in co.itertuples():
        M, uids, units = rsa.session_response_matrix(c.subject, 1, region)
        if M is None:
            continue
        D = rsa.neural_rdm(M)
        pos = {u: i for i, u in enumerate(uids)}
        i, j = pos[c.uid_a], pos[c.uid_b]
        allp = rsa.upper(D)
        iu = np.triu_indices(D.shape[0], k=1)
        ref = allp[~((iu[0] == min(i, j)) & (iu[1] == max(i, j)))]  # every pair except (A, B)
        d_ab = D[i, j]
        others_i = np.delete(D[i], [i, j])
        others_j = np.delete(D[j], [i, j])
        rows.append({
            "level": f"population_{region}", "subject": c.subject, "n_units": len(units), "identity": c.identity,
            "uid_a": c.uid_a, "uid_b": c.uid_b, "dist_ab": float(d_ab),
            "pct_all_pairs_closer": _pct(d_ab, ref, higher_is_better=False),
            "pct_b_given_a": _pct(d_ab, others_i, higher_is_better=False),
            "pct_a_given_b": _pct(d_ab, others_j, higher_is_better=False),
            "rank_b_given_a": int((others_i < d_ab).sum() + 1), "rank_a_given_b": int((others_j < d_ab).sum() + 1),
            "n_other_pictures": len(others_i), "n_pairs": len(ref),
        })
    return pd.DataFrame(rows)


def dnn_within_session(co: pd.DataFrame, tun: pd.DataFrame) -> pd.DataFrame:
    """CLIP rank of the pair restricted to the pictures the co-showing session actually used."""
    X, stored = load_space(*DNN, center=True, normalize=True)
    pos = {u: i for i, u in enumerate(stored)}
    rows = []
    for c in co.itertuples():
        uids = sorted(tun.loc[tun.subject == c.subject, "image_uid"].unique())
        idx = np.array([pos[u] for u in uids])
        S = X[idx] @ X[idx].T
        i, j = uids.index(c.uid_a), uids.index(c.uid_b)
        oi, oj = np.delete(S[i], [i, j]), np.delete(S[j], [i, j])
        rows.append({
            "level": "dnn_within_session", "model": "/".join(DNN), "subject": c.subject, "identity": c.identity,
            "uid_a": c.uid_a, "uid_b": c.uid_b, "sim_ab": float(S[i, j]),
            "pct_b_given_a": _pct(S[i, j], oi), "pct_a_given_b": _pct(S[i, j], oj),
            "rank_b_given_a": int((oi > S[i, j]).sum() + 1), "rank_a_given_b": int((oj > S[i, j]).sum() + 1),
            "n_reference": len(uids) - 2,
        })
    return pd.DataFrame(rows)


def main() -> None:
    interrater()

    labels = pd.read_csv(CONFIGS / "famous_labels.csv")
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    sel = sel[sel.task == "screening"]
    tun = tun[tun.task == "screening"]

    pairs = identity_pairs(labels)
    dnn = dnn_level(pairs)
    co = coshown_sessions(pairs, tun)
    strict, explore = unit_level(co, sel, tun)
    pop_mtl = population_level(co, "MTL")
    pop_all = population_level(co, None).assign(level="population_all_units")
    dnn_sess = dnn_within_session(co, tun)

    per = pd.concat([strict, explore, pop_mtl, pop_all, dnn_sess, dnn], ignore_index=True)
    lead = ["level", "identity", "uid_a", "uid_b", "subject", "unit", "area", "concept_cell", "pref_image"]
    per = per[[c for c in lead if c in per.columns] + [c for c in per.columns if c not in lead]]
    per.to_csv(TABLES / "A5_identity_invariance.csv", index=False)

    # concept cells preferring a pair picture anywhere (partner never shown → untestable)
    pair_pics = set(pairs.uid_a) | set(pairs.uid_b)
    cc_any = sel[(sel.region == "MTL") & sel.concept_cell & sel.pref_image.isin(pair_pics)]

    summ = []
    summ.append({"level": "design", "metric": "n_identities_with_ge2_pictures", "n": pairs.identity.nunique(), "value": len(pairs), "detail": "identities / same-identity pairs (Tom Cruise gives 3)"})
    summ.append({"level": "design", "metric": "n_sessions_coshowing_a_pair", "n": len(co), "value": co.subject.nunique(), "detail": "; ".join(f"sub-{c.subject:02d}: {c.identity} ({c.uid_a},{c.uid_b})" for c in co.itertuples())})
    summ.append({"level": "concept_cell", "metric": "n_MTL_concept_cells_pref_pair_picture_any_session", "n": len(cc_any), "value": cc_any.subject.nunique(), "detail": "partner picture not shown in their sessions: " + "; ".join(f"sub-{r.subject:02d} u{r.unit} {r.pref_image}" for r in cc_any.itertuples())})
    summ.append({"level": "concept_cell", "metric": "n_MTL_concept_cells_testable", "n": len(strict), "value": np.nan, "detail": "pref picture in a pair AND partner shown in same session"})
    if len(strict):
        summ.append({"level": "concept_cell", "metric": "mean_same_identity_pct_in_nonpref", "n": len(strict), "value": strict.same_identity_pct_in_nonpref.mean(), "detail": f"top3 {int(strict.top3.sum())}, top10% {int(strict.top10pct.sum())}"})
    if len(explore):
        e = explore
        try:
            wp = sps.wilcoxon(e.pct_pair_distance - 0.5).pvalue
        except ValueError:
            wp = np.nan
        summ.append({"level": "unit_exploratory", "metric": "n_MTL_units_in_coshowing_sessions", "n": len(e), "value": e.subject.nunique(), "detail": f"concept cells among them: {int(e.concept_cell.sum())}"})
        summ.append({"level": "unit_exploratory", "metric": "mean_pct_pair_distance_smaller_than_ref", "n": len(e), "value": e.pct_pair_distance.mean(), "detail": f"1 = pair closest of all; chance 0.5; Wilcoxon vs 0.5 p = {wp:.3f}; median {e.pct_pair_distance.median():.3f}"})
        summ.append({"level": "unit_exploratory", "metric": "mean_pct_partner_given_anchor", "n": len(e), "value": float(np.mean(np.r_[e.pct_b_given_a, e.pct_a_given_b])), "detail": "percentile of the partner's rate among the other pictures, both anchors; chance 0.5"})
        summ.append({"level": "unit_exploratory", "metric": "n_units_partner_top3_either_anchor", "n": len(e), "value": int(((e.n_other_pictures - e.pct_b_given_a * e.n_other_pictures) < 3).sum() + ((e.n_other_pictures - e.pct_a_given_b * e.n_other_pictures) < 3).sum()), "detail": "unit x anchor count (2 anchors per unit); chance ≈ 2 * n * 3/61"})
    for name, pop in (("population_MTL", pop_mtl), ("population_all_units", pop_all)):
        for r in pop.itertuples():
            summ.append({"level": name, "metric": f"sub-{r.subject:02d} {r.identity}", "n": r.n_units, "value": r.pct_all_pairs_closer, "detail": f"fraction of all {r.n_pairs} pairs farther than the pair (1 = closest); given A: rank {r.rank_b_given_a}/{r.n_other_pictures + 1}, given B: rank {r.rank_a_given_b}/{r.n_other_pictures + 1}"})
    for r in dnn_sess.itertuples():
        summ.append({"level": "dnn_within_session", "metric": f"sub-{r.subject:02d} {r.identity}", "n": r.n_reference, "value": float(np.mean([r.pct_b_given_a, r.pct_a_given_b])), "detail": f"CLIP cos {r.sim_ab:.3f}; rank {r.rank_b_given_a} / {r.rank_a_given_b} of {r.n_reference + 1}"})
    summ.append({"level": "dnn", "metric": "mean_pct_partner_given_anchor_all_pairs", "n": len(dnn), "value": float(np.mean(np.r_[dnn.pct_b_given_a, dnn.pct_a_given_b])), "detail": f"CLIP {DNN[1]}/{DNN[2]}, dataset-wide (340 references); median rank {np.median(np.r_[dnn.rank_b_given_a, dnn.rank_a_given_b]):.0f}; pairs top-3 in both directions: {int(((dnn.rank_b_given_a <= 3) & (dnn.rank_a_given_b <= 3)).sum())}/{len(dnn)}"})
    for r in dnn.itertuples():
        summ.append({"level": "dnn", "metric": f"{r.identity} ({r.uid_a},{r.uid_b})", "n": r.n_reference, "value": float(np.mean([r.pct_b_given_a, r.pct_a_given_b])), "detail": f"CLIP cos {r.sim_ab:.3f}; rank {r.rank_b_given_a} / {r.rank_a_given_b} of {r.n_reference + 1}"})
    summ = pd.DataFrame(summ)
    summ.to_csv(TABLES / "A5_identity_invariance_summary.csv", index=False)
    print(summ.to_string(index=False))

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    if len(explore):
        for k, (s, g) in enumerate(explore.groupby("subject")):
            jitter = np.random.default_rng(k).uniform(-0.12, 0.12, len(g))
            ax.scatter(np.full(len(g), k) + jitter, g.pct_pair_distance, s=45, c=["#C44E52" if c else "#4C72B0" for c in g.concept_cell], edgecolor="k", lw=0.5, zorder=3)
        ax.axhline(0.5, color="k", ls=":", lw=1)
        ax.set_xticks(range(explore.subject.nunique()))
        ax.set_xticklabels([f"sub-{s:02d}\n{g.identity.iloc[0]}\n{len(g)} MTL units" for s, g in explore.groupby("subject")], fontsize=8)
        ax.set_xlim(-0.6, explore.subject.nunique() - 0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("percentile of |z(A) − z(B)| among |z(A/B) − z(other)|\n(1 = pair closest for this unit)")
        ax.set_title("Single MTL units, exploratory (red = concept cell)\nstrictly testable concept cells: n = %d" % len(strict), fontsize=9)
    ax = axes[1]
    for k, r in enumerate(pop_mtl.itertuples()):
        M, uids, _ = rsa.session_response_matrix(r.subject, 1, "MTL")
        D = rsa.neural_rdm(M)
        ax.hist(rsa.upper(D), bins=30, alpha=0.45, label=f"sub-{r.subject:02d} all pairs ({r.n_units} units)")
        ax.axvline(r.dist_ab, color=f"C{k}", lw=2, label=f"sub-{r.subject:02d} {r.identity} pair (pct {r.pct_all_pairs_closer:.2f})")
    ax.set_xlabel("MTL population correlation distance")
    ax.set_ylabel("# picture pairs")
    ax.set_title("Population RDM: same-identity pair vs all pairs", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    lab = [f"{r.identity}\n{r.uid_a[-4:]}/{r.uid_b[-4:]}" for r in dnn.itertuples()]
    y = np.arange(len(dnn))
    ax.scatter(dnn.pct_b_given_a, y, marker=">", color="#4C72B0", label="B given A")
    ax.scatter(dnn.pct_a_given_b, y, marker="<", color="#DD8452", label="A given B")
    ax.axvline(0.5, color="k", ls=":", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("percentile of same-identity cosine among 340 other pictures\n(1 = nearest neighbour)")
    ax.set_title(f"CLIP {DNN[1]}/{DNN[2]}: same-identity pairs (dataset-wide)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.invert_yaxis()
    fig.suptitle("Identity invariance: are two pictures of the same person close for neurons and for CLIP?", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "A5_identity_invariance.png", dpi=150)


if __name__ == "__main__":
    main()
