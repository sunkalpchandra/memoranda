#!/usr/bin/env python
"""Stage 71 — semantic (text-space) vs visual (image-space) similarity tuning.

Every picture gets a one-line description — ``a photo of {identity}`` when the hand
label (configs/famous_labels.csv) names the person, else ``a photo of a {fine}`` from the
CLIP fine label — encoded with the CLIP ViT-B/32 (OpenAI) *text* tower and L2-normalised.
That gives a text-space embedding per picture that knows who / what is depicted but has
never seen the pixels. We ask whether concept cells generalise along this semantic
similarity, along image-space similarity (CLIP image embedding / ln_post, ResNet-50
avgpool), and along each *controlling for the other*:

* single cells (screening MTL / MFC concept cells): Spearman ρ between the rate on the
  non-preferred pictures and their similarity to the preferred picture, in text space and in
  image space, plus partial ρ(rate, text | image) and ρ(rate, image | text) obtained by
  residualising ranks (vector analogue of ``rsa.partial_compare``); random-anchor null for
  the text ρ as in script 47;
* population RSA per screening session (MTL, MTL concept cells, MFC): neural RDM vs text
  RDM vs image RDM and the two partials (``rsa.partial_compare`` on upper triangles).

A second text variant (``fine-only``: everyone described by the fine label, no names)
shows how much of any text effect is carried by identity. Text embeddings are *centred*
(mean over the 342 pictures subtracted, then re-normalised — as ``load_space(center=True)``
does for the image spaces, script 47): raw CLIP text embeddings are strongly anisotropic
(pairwise cosine 0.80 ± 0.07 across all descriptions), and the shared "a photo of …" prompt
component dominates raw similarities. The raw (uncentred) text similarities are kept as
``text_centering = raw`` rows in every table for comparison; they roughly halve the text
effect. Caveat: the fine label is CLIP's zero-shot pick from a 56-label list (script 12),
so the fine-only description is not fully independent of the picture's CLIP image embedding;
identities are hand labels.
Outputs: results/tables/A4_semantic_vs_visual.csv (+ _per_cell.csv, _per_session.csv),
results/figures/A4_semantic_vs_visual.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.spatial.distance import pdist, squareform

from memoranda.analysis import rsa
from memoranda.dandi import list_assets
from memoranda.features import load_space
from memoranda.paths import CONFIGS, FIGURES, MANIFESTS, TABLES

IMAGE_SPACES = {
    "clip_embed": ("clip_vitb32", "embed", "gap", True),  # (model, layer, view, center) — as in script 47
    "clip_ln_post": ("clip_vitb32", "ln_post", "cls", True),
    "resnet50_avgpool": ("resnet50", "avgpool", "gap", True),
}
TEXT_VARIANTS = ["identity+fine", "fine-only"]
TEXT_CENTERING = ["centred", "raw"]  # centred = default (figure / headline); raw = uncentred cosine
REGIONS = ["MTL", "MTL_concept", "MFC"]
COLORS = {"text": "#2a78d6", "image": "#eb6834", "text|image": "#1baf7a", "image|text": "#4a3aa7"}


# ------------------------------------------------------------------ descriptions
def _article(noun: str) -> str:
    return "an" if noun[:1].lower() in "aeiou" else "a"


def describe(fine: str, identity: str | None) -> str:
    if isinstance(identity, str) and identity.strip():
        return f"a photo of {identity.strip()}"
    return fine if fine.startswith("a ") else f"a photo of {_article(fine)} {fine}"


def build_descriptions() -> pd.DataFrame:
    lab = pd.read_csv(MANIFESTS / "image_labels.csv")[["image_uid", "fine", "category"]]
    fam = pd.read_csv(CONFIGS / "famous_labels.csv")[["image_uid", "identity"]]
    d = lab.merge(fam, on="image_uid", how="left")
    d["desc"] = [describe(f, i) for f, i in zip(d.fine, d.identity)]
    d["desc_fine"] = [describe(f, None) for f in d.fine]
    d["desc_type"] = np.where(d.identity.notna(), "identity", "fine")
    return d


def _unit(T: np.ndarray) -> np.ndarray:
    return T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)


def text_embeddings(desc: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """{(variant, centering): images × dims unit-norm text embeddings}."""
    from memoranda.models.zeroshot import ClipZeroShot

    zs = ClipZeroShot()
    out = {}
    for var, col in [("identity+fine", "desc"), ("fine-only", "desc_fine")]:
        embs = []
        for i in range(0, len(desc), 64):
            embs.append(zs.text_embed(desc[col].tolist()[i : i + 64]).cpu().numpy())
        T = _unit(np.concatenate(embs, 0).astype(np.float64))
        out[(var, "raw")] = T
        out[(var, "centred")] = _unit(T - T.mean(0))
    return out


# ------------------------------------------------------------------ statistics
def partial_spearman(x: np.ndarray, y: np.ndarray, controls: list[np.ndarray]) -> float:
    """Partial Spearman ρ(x, y | controls): rank everything, residualise x and y on the
    ranked controls (with intercept), Pearson-correlate the residuals."""
    rx, ry = sps.rankdata(x), sps.rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [sps.rankdata(c) for c in controls])
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    ex, ey = rx - C @ bx, ry - C @ by
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def _spear(x, y) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(sps.spearmanr(x, y).statistic)


def _wilcoxon_p(x: pd.Series) -> float:
    x = x.dropna()
    x = x[x != 0]
    if len(x) < 5:
        return np.nan
    try:
        return float(sps.wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def _ttest_p(x: pd.Series) -> float:
    x = x.dropna()
    return float(sps.ttest_1samp(x, 0).pvalue) if len(x) >= 3 else np.nan


QUANTS = ["rho_text", "rho_img", "rho_text_given_img", "rho_img_given_text"]


def summarise(per: pd.DataFrame, keys: list[str], extra_null: bool) -> pd.DataFrame:
    out = []
    for k, g in per.groupby(keys):
        rec = dict(zip(keys, k))
        rec["n"] = int(g.rho_text.notna().sum())
        for q in QUANTS:
            x = g[q].dropna()
            rec[f"{q}_mean"] = x.mean()
            rec[f"{q}_sem"] = x.std() / np.sqrt(max(len(x), 1))
            rec[f"{q}_frac_pos"] = (x > 0).mean()
            rec[f"{q}_p_wilcoxon"] = _wilcoxon_p(x)
            rec[f"{q}_p_t"] = _ttest_p(x)
        rec["text_minus_img_mean"] = (g.rho_text - g.rho_img).mean()
        rec["text_minus_img_p_wilcoxon"] = _wilcoxon_p(g.rho_text - g.rho_img)
        rec["rho_text_img_mean"] = g.rho_text_img.mean()  # collinearity of the two predictors
        if extra_null:
            rec["rho_text_null_mean"] = g.rho_text_null.mean()
            rec["text_vs_null_p_wilcoxon"] = _wilcoxon_p(g.rho_text - g.rho_text_null)
        out.append(rec)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ single cells
def cell_rows(sel: pd.DataFrame, tun: pd.DataFrame, T: dict[tuple[str, str], np.ndarray], tpos: dict, X: dict, desc: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "MTL concept": sel[(sel.region == "MTL") & sel.concept_cell],
        "MFC concept": sel[(sel.region == "MFC") & sel.concept_cell],
        "MTL non-concept": sel[(sel.region == "MTL") & ~sel.concept_cell & (sel.mean_rate > 0.5)],
    }
    dtype = desc.set_index("image_uid")["desc_type"]
    dtext = desc.set_index("image_uid")["desc"]
    rng = np.random.default_rng(0)
    rows = []
    for gname, g in groups.items():
        for r in g.itertuples():
            t = tun[(tun.subject == r.subject) & (tun.unit == r.unit)]
            uids = t.image_uid.to_numpy()
            rates = t.rate.to_numpy()
            if r.pref_image not in tpos or not all(u in tpos for u in uids):
                continue
            pref_i = int(np.where(uids == r.pref_image)[0][0])
            mask = np.arange(len(uids)) != pref_i
            j = int(rng.choice(np.where(mask)[0]))  # random anchor for the text null (shared across spaces)
            mask_j = np.arange(len(uids)) != j
            idx = np.array([tpos[u] for u in uids])
            for (var, cent), Tv in T.items():
                tsim = Tv[idx] @ Tv[tpos[r.pref_image]]
                tsim_j = Tv[idx] @ Tv[idx[j]]
                rho_text = _spear(rates[mask], tsim[mask])
                rho_text_null = _spear(rates[mask_j], tsim_j[mask_j])
                for sname, (Xs, spos) in X.items():
                    sidx = np.array([spos[u] for u in uids])
                    isim = Xs[sidx] @ Xs[spos[r.pref_image]]
                    rows.append(
                        {
                            "group": gname,
                            "subject": r.subject,
                            "unit": r.unit,
                            "area": r.area,
                            "hemisphere": r.hemisphere,
                            "pref_image": r.pref_image,
                            "pref_desc": dtext[r.pref_image],
                            "pref_desc_type": dtype[r.pref_image],
                            "n_images": int(mask.sum()),
                            "text_variant": var,
                            "text_centering": cent,
                            "image_space": sname,
                            "rho_text": rho_text,
                            "rho_text_null": rho_text_null,
                            "rho_img": _spear(rates[mask], isim[mask]),
                            "rho_text_given_img": partial_spearman(rates[mask], tsim[mask], [isim[mask]]),
                            "rho_img_given_text": partial_spearman(rates[mask], isim[mask], [tsim[mask]]),
                            "rho_text_img": _spear(tsim[mask], isim[mask]),
                        }
                    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ population RSA
def region_matrix(subject: int, region: str, sel: pd.DataFrame):
    if region == "MFC":
        M, uids, _ = rsa.session_response_matrix(subject, 1, "MFC")
        return M, uids
    M, uids, units = rsa.session_response_matrix(subject, 1, "MTL")
    if M is None:
        return None, None
    if region == "MTL":
        return M, uids
    keep = sel[(sel.subject == subject) & (sel.session == 1) & sel.concept_cell & (sel.region == "MTL")].unit.to_numpy()
    mask = np.isin(units, keep)
    return (M[:, mask], uids) if mask.sum() >= 2 else (None, None)


def rsa_rows(sel: pd.DataFrame, T: dict[tuple[str, str], np.ndarray], tpos: dict) -> pd.DataFrame:
    subjects = sorted({a.subject for a in list_assets() if a.session == 1})
    rows = []
    for s in subjects:
        _, uids, _ = rsa.session_response_matrix(s, 1, None)
        if uids is None or not all(u in tpos for u in uids):
            continue
        idx = np.array([tpos[u] for u in uids])
        Dtext = {key: squareform(pdist(Tv[idx], metric="correlation")) for key, Tv in T.items()}
        Dimg = {name: rsa.model_rdm(m, l, v, uids) for name, (m, l, v, _) in IMAGE_SPACES.items()}
        for region in REGIONS:
            M, u2 = region_matrix(s, region, sel)
            if M is None:
                continue
            Dn = rsa.neural_rdm(M)
            for (var, cent), Dt in Dtext.items():
                for sname, Di in Dimg.items():
                    rows.append(
                        {
                            "subject": s,
                            "region": region,
                            "n_units": M.shape[1],
                            "n_images": len(uids),
                            "text_variant": var,
                            "text_centering": cent,
                            "image_space": sname,
                            "rho_text": rsa.compare_rdms(Dn, Dt),
                            "rho_img": rsa.compare_rdms(Dn, Di),
                            "rho_text_given_img": rsa.partial_compare(Dn, Dt, [Di]),
                            "rho_img_given_text": rsa.partial_compare(Dn, Di, [Dt]),
                            "rho_text_img": rsa.compare_rdms(Dt, Di),
                        }
                    )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure
def _bar_panel(ax, summ: pd.DataFrame, title: str, null_col: str | None = None):
    spaces = list(IMAGE_SPACES)
    d = summ.set_index("image_space").reindex(spaces)
    x = np.arange(len(spaces))
    w = 0.2
    for k, (label, q) in enumerate(zip(COLORS, QUANTS)):
        ax.bar(x + (k - 1.5) * w, d[f"{q}_mean"], w * 0.9, yerr=d[f"{q}_sem"], color=COLORS[label], label=label, capsize=2, linewidth=0)
    if null_col is not None:
        ax.axhline(d[null_col].mean(), color="k", ls=":", lw=1, label="text, random-anchor null")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["CLIP image\nembed", "CLIP\nln_post", "ResNet-50\navgpool"], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(cs: pd.DataFrame, rs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    cs, rs = cs[cs.text_centering == "centred"], rs[rs.text_centering == "centred"]
    for row, var in enumerate(TEXT_VARIANTS):
        a = cs[(cs.group == "MTL concept") & (cs.text_variant == var)]
        _bar_panel(axes[row, 0], a, f"MTL concept cells (n = {int(a.n.max())}) — text: {var}", "rho_text_null_mean")
        b = cs[(cs.group == "MFC concept") & (cs.text_variant == var)]
        _bar_panel(axes[row, 1], b, f"MFC concept cells (n = {int(b.n.max())}) — text: {var}", "rho_text_null_mean")
        c = rs[(rs.region == "MTL") & (rs.text_variant == var)]
        _bar_panel(axes[row, 2], c, f"Population RSA, MTL sessions (n = {int(c.n.max())}) — text: {var}")
        axes[row, 0].set_ylabel("Spearman ρ(rate, similarity to preferred picture)\nnon-preferred pictures; mean ± sem over cells", fontsize=8.5)
        axes[row, 2].set_ylabel("Spearman ρ(neural RDM, model RDM)\nmean ± sem over sessions", fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5, frameon=False, title="similarity space (x-axis: image space used for 'image')", title_fontsize=8.5)
    fig.suptitle(
        "Semantic (CLIP text description of the picture) vs visual (image features) similarity: which one do concept cells follow?\n"
        "top: 'a photo of {identity}' when a person is named, else 'a photo of a {fine label}'; bottom: fine label for every picture (text embeddings centred over pictures)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURES / "A4_semantic_vs_visual.png", dpi=150)


# ------------------------------------------------------------------ main
def main() -> None:
    desc = build_descriptions()
    T = text_embeddings(desc)
    tpos = {u: i for i, u in enumerate(desc.image_uid)}
    print(desc.desc_type.value_counts().to_dict(), "descriptions; examples:")
    print(desc.sample(8, random_state=1)[["image_uid", "desc"]].to_string(index=False))

    X = {}
    for name, (m, l, v, center) in IMAGE_SPACES.items():
        Xs, stored = load_space(m, l, v, center=center, normalize=True)
        X[name] = (Xs, {u: i for i, u in enumerate(stored)})

    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    tun = pd.read_csv(MANIFESTS / "unit_tuning.csv")
    sel = sel[sel.task == "screening"]
    tun = tun[tun.task == "screening"]

    per_cell = cell_rows(sel, tun, T, tpos, X, desc)
    per_cell.to_csv(TABLES / "A4_semantic_vs_visual_per_cell.csv", index=False)
    cell_summ = summarise(per_cell, ["group", "text_variant", "text_centering", "image_space"], extra_null=True)
    cell_summ.insert(0, "analysis", "cell")

    per_sess = rsa_rows(sel, T, tpos)
    per_sess.to_csv(TABLES / "A4_semantic_vs_visual_per_session.csv", index=False)
    rsa_summ = summarise(per_sess, ["region", "text_variant", "text_centering", "image_space"], extra_null=False)
    rsa_summ.insert(0, "analysis", "rsa")

    by_pref = summarise(per_cell[per_cell.group == "MTL concept"], ["group", "pref_desc_type", "text_variant", "text_centering", "image_space"], extra_null=True)
    by_pref.insert(0, "analysis", "cell_by_pref_desc_type")

    summ = pd.concat([cell_summ, rsa_summ, by_pref], ignore_index=True)
    summ.to_csv(TABLES / "A4_semantic_vs_visual.csv", index=False)

    cols = ["image_space", "n", "rho_text_mean", "rho_text_p_wilcoxon", "rho_img_mean", "rho_text_given_img_mean", "rho_text_given_img_p_wilcoxon", "rho_img_given_text_mean", "rho_img_given_text_p_wilcoxon", "text_minus_img_mean", "text_minus_img_p_wilcoxon", "rho_text_img_mean"]
    for cent in TEXT_CENTERING:
        print(f"\n######## text embeddings: {cent}")
        for grp in ["MTL concept", "MFC concept", "MTL non-concept"]:
            for var in TEXT_VARIANTS:
                print(f"\n== cells: {grp} / text = {var} ({cent})")
                d = cell_summ[(cell_summ.group == grp) & (cell_summ.text_variant == var) & (cell_summ.text_centering == cent)]
                print(d[cols + ["rho_text_null_mean", "text_vs_null_p_wilcoxon"]].round(4).to_string(index=False))
        for pt in ["identity", "fine"]:
            d = by_pref[(by_pref.pref_desc_type == pt) & (by_pref.text_variant == "identity+fine") & (by_pref.text_centering == cent)]
            print(f"\n== MTL concept cells whose preferred picture is described by {pt} / text = identity+fine ({cent})")
            print(d[cols].round(4).to_string(index=False))
        for region in REGIONS:
            for var in TEXT_VARIANTS:
                print(f"\n== RSA: {region} / text = {var} ({cent})")
                d = rsa_summ[(rsa_summ.region == region) & (rsa_summ.text_variant == var) & (rsa_summ.text_centering == cent)]
                print(d[cols].round(4).to_string(index=False))

    make_figure(cell_summ, rsa_summ)


if __name__ == "__main__":
    main()
