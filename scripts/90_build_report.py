#!/usr/bin/env python
"""Stage 90 — assemble the HTML report from result tables and figures.

``--embed`` inlines figures as data URIs (for publishing as a self-contained page);
without it the page references results/figures/ relatively (small file, committed).
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from memoranda.paths import FIGURES, MANIFESTS, RESULTS, TABLES

FIGS = {
    "montage": ("montage_all_unique.png", 1400),
    "sternberg": ("montage_sternberg_by_subject.png", 900),
    "maint": ("A0_maintenance.png", 900),
    "comp": ("A1_composition.png", 1400),
    "enrich": ("A2_enrichment_contrasts.png", 1400),
    "pref": ("A5_concept_cell_preferences.png", 1400),
    "topbot": ("A5_top_bottom_images.png", 1400),
    "animacy": ("A5_animacy.png", 1400),
    "drive": ("A5_image_drive.png", 1400),
    "layers": ("A3_rsa_layer_curves.png", 1600),
    "best": ("A3_rsa_best_layer.png", 1400),
    "partial": ("A3_partial_rsa.png", 1400),
    "time": ("A3_time_resolved.png", 1500),
    "mds": ("A3_pooled_mds.png", 800),
    "simtune": ("A4_similarity_tuning.png", 1600),
    "depth": ("A4_area_depth.png", 800),
    "enc_layers": ("A4_encoding_layer_curves.png", 1600),
    "enc_best": ("A4_encoding_best.png", 1400),
    "behav": ("A6_behavior.png", 1400),
    "models": ("A7_model_comparison.png", 1400),
    "probe": ("A0_probe_match.png", 1100),
    "within": ("A3_within_category.png", 1400),
    "varpart": ("A3_variance_partition.png", 1200),
    "reliab": ("A3_reliability.png", 1200),
    "tsim": ("A4_time_resolved_similarity.png", 1000),
    "facespace": ("A3_face_space.png", 1300),
    "xpat": ("A5_cross_patient_tuning.png", 800),
    "catdec": ("A5_category_decoding.png", 800),
    "gallery": ("A2_memoranda_gallery.png", 900),
    "examples": ("A4_example_cells.png", 1300),
    "sparsity": ("A4_sparsity_vs_generalisation.png", 1300),
    "objective": ("A7_objective_contrast.png", 1200),
    "debiased": ("A4_debiased_depth.png", 1200),
}


def fig_src(key: str, embed: bool) -> str:
    name, maxw = FIGS[key]
    p = FIGURES / name
    if not p.exists():
        return ""
    if not embed:
        return f"figures/{name}"
    im = Image.open(p).convert("RGB")
    if im.width > maxw:
        im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if "montage" in name or "top_bottom" in name or "sternberg" in name:
        im.save(buf, format="JPEG", quality=80, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def figure(key: str, caption: str, embed: bool, wide: bool = True) -> str:
    src = fig_src(key, embed)
    if not src:
        return ""
    cls = "fig wide" if wide else "fig"
    return f'<figure class="{cls}"><img src="{src}" alt="{html.escape(caption)}" loading="lazy"><figcaption>{caption}</figcaption></figure>'


def table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, fmt: dict | None = None, caption: str = "") -> str:
    fmt = fmt or {}
    headers = headers or cols
    out = ['<div class="tablewrap"><table>']
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in fmt:
                v = fmt[c].format(v)
            elif isinstance(v, float):
                v = f"{v:.3f}"
            cells.append(f"<td>{html.escape(str(v))}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def load_numbers() -> dict:
    n = {}
    units = pd.read_csv(MANIFESTS / "units.csv")
    n["n_units"] = len(units)
    img = pd.read_csv(MANIFESTS / "images.csv")
    n["n_unique"] = img[img.image_uid != "img_null"].image_uid.nunique()
    n["n_subjects"] = img.subject.nunique()
    sel = pd.read_csv(MANIFESTS / "unit_selectivity.csv")
    scr = sel[sel.task == "screening"]
    n["cc_mtl"] = scr[scr.region == "MTL"].concept_cell.mean() * 100
    n["cc_mfc"] = scr[scr.region == "MFC"].concept_cell.mean() * 100
    n["n_cc_mtl"] = int(scr[scr.region == "MTL"].concept_cell.sum())
    t = pd.read_csv(MANIFESTS / "trials_sternberg.csv")
    n["acc"] = t.response_accuracy.mean() * 100
    rsa = pd.read_csv(TABLES / "A3_rsa_summary.csv")
    m = rsa[(rsa.region == "MTL") & (rsa.model != "baseline")]
    b = m.loc[m.rho_mean.idxmax()]
    n["rsa_best"] = f"{b.model} {b.layer}"
    n["rsa_best_rho"] = b.rho_mean
    n["rsa_best_p"] = b.p
    n["rsa_cat"] = rsa[(rsa.region == "MTL") & (rsa.layer == "category")].rho_mean.iloc[0]
    n["rsa_mfc"] = rsa[(rsa.region == "MFC") & (rsa.model != "baseline")].rho_mean.max()
    pooled = pd.read_csv(TABLES / "A3_pooled_rsa.csv")
    pm = pooled[(pooled.region == "MTL") & (pooled.model != "baseline")]
    n["pooled_best_rho"] = pm.rho.max()
    n["pooled_cat"] = pooled[(pooled.region == "MTL") & (pooled.layer == "category")].rho.iloc[0]
    st = pd.read_csv(TABLES / "A4_similarity_tuning.csv")
    s = st[st.group == "MTL concept"]
    bs = s.loc[s.rho_mean.idxmax()]
    n["sim_best"] = f"{bs.model} {bs.layer}"
    n["sim_rho"] = bs.rho_mean
    n["sim_frac"] = bs.frac_pos * 100
    pr = pd.read_csv(TABLES / "A3_partial_rsa.csv")
    pr = pr[pr.region == "MTL"]
    for key, (m, l) in {"clip": ("clip_vitb32", "ln_post"), "rn": ("resnet50", "avgpool")}.items():
        row = pr[(pr.model == m) & (pr.layer == l)].iloc[0]
        n[f"part_{key}_rho"] = row.rho_partial_cat_low_mean
        n[f"part_{key}_p"] = row.rho_partial_cat_low_p
        n[f"catpart_{key}_p"] = row.rho_cat_partial_model_p
    tr = pd.read_csv(TABLES / "A3_time_resolved.csv")
    tm = tr[(tr.region == "MTL") & (tr.ref == "clip ln_post")]
    n["t_peak"] = int(tm.loc[tm["mean"].idxmax(), "t"] * 1000)
    enc = TABLES / "A4_encoding_best_layer.csv"
    if enc.exists():
        e = pd.read_csv(enc)
        e = e[(e.group == "MTL concept") & (e.model != "baseline")]
        be = e.loc[e.r_mean.idxmax()]
        n["enc_best"] = f"{be.model} {be.best_layer}"
        n["enc_r"] = be.r_mean
        n["enc_frac"] = be.frac_r_pos * 100
        n["enc_ceiling"] = be.r_over_ceiling
        eb = pd.read_csv(enc)
        n["enc_cat"] = eb[(eb.group == "MTL concept") & (eb.model == "baseline")].r_mean.max()
    return n


CSS = """
:root{--paper:#F4F5F1;--ink:#1C1F1E;--muted:#626864;--rule:#D9DCD6;--card:#FBFBF9;--accent:#B4433E;--accent-soft:#F1DDDB;--second:#3E63A0;--second-soft:#DCE4F1;--code:#ECEEE9}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#141716;--ink:#E4E7E1;--muted:#9AA09B;--rule:#2A2F2C;--card:#1B1F1D;--accent:#E37A74;--accent-soft:#3A2321;--second:#7EA1DA;--second-soft:#1E2A3D;--code:#202522}}
:root[data-theme="dark"]{--paper:#141716;--ink:#E4E7E1;--muted:#9AA09B;--rule:#2A2F2C;--card:#1B1F1D;--accent:#E37A74;--accent-soft:#3A2321;--second:#7EA1DA;--second-soft:#1E2A3D;--code:#202522}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Segoe UI",system-ui,-apple-system,sans-serif;font-size:17px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:76rem;margin:0 auto;padding:2.5rem 1.5rem 5rem}
.prose{max-width:68ch}
h1,h2,h3,.display{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;text-wrap:balance;letter-spacing:-.005em}
h1{font-size:clamp(2.2rem,4vw,3.4rem);line-height:1.08;margin:.2rem 0 .8rem;font-weight:500}
h2{font-size:1.75rem;line-height:1.2;margin:0 0 .6rem;font-weight:500}
h3{font-size:1.2rem;margin:1.6rem 0 .4rem;font-weight:600}
p{margin:0 0 1rem}
a{color:var(--second);text-decoration-thickness:1px;text-underline-offset:2px}
.mast{border-bottom:2px solid var(--ink);padding-bottom:1.6rem;margin-bottom:2rem}
.eyebrow{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)}
.thesis{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-size:1.35rem;line-height:1.4;color:var(--ink);max-width:60ch}
.thesis em{color:var(--accent);font-style:normal}
.meta{color:var(--muted);font-size:.9rem;margin-top:.8rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.8rem;margin:1.8rem 0 2.6rem}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:4px;padding:.9rem 1rem}
.stat b{display:block;font-family:"SF Mono",Menlo,Consolas,monospace;font-size:1.55rem;font-weight:500;font-variant-numeric:tabular-nums;color:var(--accent);line-height:1.1}
.stat span{font-size:.8rem;color:var(--muted)}
.toc{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;font-size:.9rem;margin:0 0 2.5rem;padding:0;list-style:none}
.toc a{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--rule)}
.toc a:hover,.toc a:focus{color:var(--accent);border-color:var(--accent);outline:none}
section{padding:2.2rem 0;border-top:1px solid var(--rule)}
section .head{display:grid;grid-template-columns:5rem 1fr;gap:1rem;align-items:baseline;margin-bottom:.8rem}
section .head .tag{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.85rem;color:var(--accent);letter-spacing:.06em}
.fig{margin:1.4rem 0;background:var(--card);border:1px solid var(--rule);border-radius:4px;padding:.8rem}
.fig img{display:block;width:100%;height:auto;max-width:100%}
.fig figcaption{font-size:.86rem;color:var(--muted);margin-top:.6rem;line-height:1.45}
.fig:not(.wide){max-width:52rem}
.tablewrap{overflow-x:auto;margin:1rem 0 1.4rem;max-width:100%}
table{border-collapse:collapse;font-size:.86rem;font-variant-numeric:tabular-nums;min-width:32rem}
caption{text-align:left;font-size:.82rem;color:var(--muted);padding:0 0 .4rem}
th,td{padding:.35rem .7rem;border-bottom:1px solid var(--rule);text-align:left;white-space:nowrap}
th{font-weight:600;font-size:.78rem;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
td:first-child,th:first-child{padding-left:0}
.callout{border-left:3px solid var(--accent);background:var(--accent-soft);padding:.8rem 1rem;border-radius:0 4px 4px 0;margin:1rem 0 1.4rem;max-width:68ch}
.callout.blue{border-color:var(--second);background:var(--second-soft)}
code,.mono{font-family:"SF Mono",Menlo,Consolas,monospace;font-size:.85em;background:var(--code);padding:.1em .35em;border-radius:3px}
ul.findings{padding-left:1.1rem;margin:0 0 1rem}
ul.findings li{margin:.35rem 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem;align-items:start}
.small{font-size:.9rem;color:var(--muted)}
footer{margin-top:3rem;padding-top:1.2rem;border-top:2px solid var(--ink);font-size:.86rem;color:var(--muted)}
@media (prefers-reduced-motion:no-preference){a{transition:color .15s,border-color .15s}}
"""


def build(embed: bool) -> str:
    n = load_numbers()
    F = lambda k, cap, wide=True: figure(k, cap, embed, wide)  # noqa: E731
    enr = pd.read_csv(TABLES / "A2_category_enrichment.csv")
    bl = pd.read_csv(TABLES / "A3_rsa_best_layer.csv")
    bl_mtl = bl[bl.region.isin(["MTL", "MFC"])].copy()
    partial = pd.read_csv(TABLES / "A3_partial_rsa.csv")
    partial = partial[(partial.region == "MTL") & partial.layer.isin(["conv1", "avgpool", "ln_post", "ln", "norm", "logits"])]
    anim = pd.read_csv(TABLES / "A5_animacy_by_area.csv")
    rank = pd.read_csv(TABLES / "A7_model_ranking.csv")
    behav = pd.read_csv(TABLES / "A6_behavior_similarity.csv")
    behav = behav[behav.space.isin(["clip", "resnet50", "dinov2"]) & (behav.outcome == "rt")]
    enc_tbl = ""
    enc_txt = ""
    if "enc_r" in n:
        eb = pd.read_csv(TABLES / "A4_encoding_best_layer.csv")
        e = eb[eb.group == "MTL concept"]
        enc_tbl = table(e, ["model", "best_layer", "n_units", "r_mean", "r_sem", "frac_r_pos", "r_over_ceiling"], ["model", "best layer", "cells", "mean CV r", "sem", "frac r>0", "r / ceiling"], caption="MTL concept cells: encoding performance at each model's best layer (6-fold CV ridge, in-fold PCA-20). Baseline rows: category one-hot, low-level statistics.")
        enc_txt = f"""<p>Ridge regression from layer features to each cell's 54–63-image tuning curve reaches a mean cross-validated <i>r</i> of <b>{n['enc_r']:.2f}</b> for MTL concept cells at the best layer ({n['enc_best']}; {n['enc_frac']:.0f}% of cells above zero; ≈{n['enc_ceiling']:.2f} of the split-half ceiling), versus {n['enc_cat']:.2f} for a category one-hot baseline. Raw CV <i>r</i> is <em>negatively</em> biased for signal-free targets with ~60 samples, so we also computed a per-cell picture-label-shuffle null (100 shuffles). Against that null the picture changes: debiased <i>r</i> for MTL concept cells is <b>0.29</b> for CLIP's last layer (47% of cells individually significant), 0.26 for DINOv2, ≈0.20 for AlexNet fc6 / ResNet-50 avgpool / VGG fc7 and 0.21 for a category one-hot — CLIP is the best single-cell predictor once the bias is removed, and MFC concept cells are predicted about half as well (0.21).</p>"""

    parts = [f"<title>Memoranda</title><style>{CSS}</style>", '<div class="wrap">']
    parts.append(f"""
<header class="mast">
<div class="eyebrow">memoranda · DANDI 000469 · Kyzar, Kamiński & Rutishauser</div>
<h1>What makes a human concept cell fire?</h1>
<p class="thesis">The 342 pictures shown to 21 epilepsy patients, pushed through eight vision networks and held up against {n['n_units']} single neurons. <em>Late-layer geometry tracks the medial temporal lobe; nothing about a picture by itself predicts which one a patient's neurons will pick.</em></p>
<p class="meta">Sunkalp Chandra · August 2026 · code &amp; tables: <a href="https://github.com/sunkalpchandra/memoranda">github.com/sunkalpchandra/memoranda</a></p>
</header>
<div class="stats">
<div class="stat"><b>{n['n_units']}</b><span>single units, {n['n_subjects']} patients (matches paper)</span></div>
<div class="stat"><b>{n['n_unique']}</b><span>unique pictures behind 1341 stored templates</span></div>
<div class="stat"><b>{n['cc_mtl']:.1f}%</b><span>MTL concept cells in screening (paper 32.8%)</span></div>
<div class="stat"><b>{n['acc']:.1f}%</b><span>working-memory accuracy (paper 88.7%)</span></div>
<div class="stat"><b>ρ = {n['pooled_best_rho']:.2f}</b><span>pooled MTL RDM vs best DNN layer (category {n['pooled_cat']:.2f})</span></div>
<div class="stat"><b>{n['t_peak']} ms</b><span>peak of MTL–DNN correspondence after onset</span></div>
</div>
<ul class="toc">
<li><a href="#a0">A0 Data & replication</a></li><li><a href="#a1">A1–A2 The stimuli</a></li><li><a href="#a5">A5 What concept cells like</a></li><li><a href="#a3">A3 Geometry (RSA)</a></li><li><a href="#a4">A4 Single neurons</a></li><li><a href="#a6">A6 Behaviour</a></li><li><a href="#a7">A7 Models compared</a></li><li><a href="#next">Caveats & next</a></li>
</ul>
""")

    parts.append(f"""
<section id="a0"><div class="head"><span class="tag">A0</span><h2>Data and replication</h2></div>
<div class="prose">
<p>All 41 NWB files were streamed over HTTP; only images, trial tables and spike times were pulled out. Unit counts per area reproduce Kyzar et al. exactly, behaviour matches ({n['acc']:.1f}% correct; RT rises with load), and re-running the concept-cell criteria (permuted one-way ANOVA + max-vs-rest permutation <i>t</i>, 200–1000 ms) yields <b>{n['cc_mtl']:.1f}%</b> concept cells in the MTL and {n['cc_mfc']:.1f}% in the MFC during screening (paper: 32.8% / 5.4%). MTL concept cells keep firing during the maintenance period when their preferred picture is held in memory (paired <i>p</i> = 7×10<sup>−9</sup>; 36% individually significant), and MFC cells do not — the persistent-activity result of Kamiński et al. 2017 falls out of the same pipeline.</p>
</div>
<div class="grid2">
{F('maint', 'Maintenance-period firing of every MTL concept cell (Sternberg task, correct trials): preferred picture held in memory vs not. Red: individually significant.', wide=False)}
{F('probe', 'Probe period: the preferred picture evokes a slightly weaker response when it was already in memory (match) than as a fresh lure (Wilcoxon p = 4e-4) — match suppression.', wide=False)}
</div>
</section>""")

    parts.append(f"""
<section id="a1"><div class="head"><span class="tag">A1–A2</span><h2>The stimuli — and which ones became memoranda</h2></div>
<div class="prose">
<p>Only {n['n_unique']} distinct pictures exist across all patients; the pool is heavily reused (one picture reached 16 patients). CLIP zero-shot labels put 37% at faces/people, 14% animals, 11% landmarks, 10% vehicles, and the remainder objects, scenes with people, nature, food and cartoons.</p>
<p>The five Sternberg memoranda per patient were chosen online as the pictures that drove the most selective neurons. Almost nothing about a picture <em>by itself</em> distinguishes them: no category is enriched (faces OR 1.4, n.s.), no CLIP attribute, no low-level statistic and no DNN-space descriptor (typicality, isolation, distance to centroid) separates memoranda from the rest of a patient's set (all FDR q &gt; 0.2). The one exception is a <em>detected</em> face (MTCNN): 62% of memoranda contain one vs 49% of the rest (AUC 0.57, q = 0.02), and their faces are larger — matching the amygdala concept cells' face preference below. Our own replicated neural selectivity does separate them (within-subject AUC 0.67–0.71) — but a picture that "won" in <em>other</em> patients is barely more likely to win in this one (leave-one-subject-out AUC 0.54).</p>
<div class="callout">Beyond "has a face", being a concept-cell picture is a <b>patient × picture</b> property, not a picture property.</div>
</div>
{F('montage', 'The 342 unique pictures (dataset-wide image_uid in each cell).')}
{F('enrich', 'Left: category enrichment among memoranda vs the rest of each patient’s screening set (none survives FDR). Right: the 20 strongest feature contrasts, all near AUC 0.5.')}
{table(enr.sort_values('odds_ratio', ascending=False), ['category','n_sternberg','n_rest','odds_ratio','p','q'], ['category','memoranda','rest','odds ratio','p','q'], caption='Category enrichment among Sternberg memoranda (Fisher exact, BH-FDR).')}
</section>""")

    parts.append(f"""
<section id="a5"><div class="head"><span class="tag">A5</span><h2>What concept cells like</h2></div>
<div class="prose">
<p>Against the composition each cell actually saw, MTL concept cells lean toward famous faces (58% prefer a person picture vs 47% shown; CLIP "famous" AUC 0.565, q = 0.047) and away from pictures containing text or logos (q = 0.007). Per shown picture, animals recruit the most cells (0.19 per patient), then faces (0.14), vehicles, places, food and finally objects (0.01). Right-amygdala concept cells prefer animals four times as often as left-amygdala cells (21% vs 5%, Fisher p = 0.04) — the right-amygdala animal preference of Mormann et al. 2011, rediscovered in an independent dataset — and, as shown below, right-amygdala cells are also the ones that generalise along DNN similarity (ρ 0.30 vs 0.04 on the left; hemispheres are partly confounded with patients).</p>
<p>Overall MTL population drive (mean z over all MTL units) is category-structured (Kruskal–Wallis p = 0.0006: people highest, food and objects lowest); pictures that sit centrally in DNN space drive the MTL more (ρ ≈ −0.28 with distance-to-centroid), but that effect is almost entirely carried by category.</p>
</div>
{F('pref', 'Left: preferred-image category of MTL concept cells vs what they were shown. Right: CLIP attributes of preferred pictures vs all shown pictures.')}
{F('animacy', 'Concept cells per shown picture, animate vs inanimate, by area; and per category × area.')}
{table(anim[anim.area.isin(['amygdala','hippocampus','dACC','preSMA'])], ['area','n_concept_cells','cells_per_animate_image','cells_per_inanimate_image','frac_cells_animate','frac_images_animate_shown','wilcoxon_p'], ['area','concept cells','cells / animate pic','cells / inanimate pic','frac cells animate','frac shown animate','Wilcoxon p'], caption='Animate vs inanimate recruitment by area (screening).')}
{F('topbot', 'The 24 pictures that recruited the most concept cells per patient shown, and the 24 fewest.')}
<div class="prose"><p>Two further checks make the "patient × picture" point. Concept cells in different patients that prefer the <em>same</em> picture agree on the rest of their tuning no better (ρ = 0.04, n.s.) than cells preferring different pictures of the same category (0.045), both above different-category pairs (0.008): what transfers across people is category structure. And a leave-one-subject-out classifier trained on other patients cannot predict which pictures a held-out patient's concept cells will prefer from any feature set (AUC 0.44–0.49). Category itself is decodable from single-trial MTL population activity, but only weakly (+4.5% over chance; MFC and hippocampus at chance).</p></div>
<div class="grid2">
{F('xpat', 'Cross-patient agreement of concept-cell tuning curves.', wide=False)}
{F('catdec', 'Leave-one-picture-out category decoding per region.', wide=False)}
</div>
</section>""")

    parts.append(f"""
<section id="a3"><div class="head"><span class="tag">A3</span><h2>Population geometry: RSA against eight networks</h2></div>
<div class="prose">
<p>For each screening session we built the MTL and MFC population RDM (correlation distance over z-scored unit responses, 54–63 pictures) and compared it with layer-wise RDMs from AlexNet, VGG-16, ResNet-18/50, ConvNeXt-T, ViT-B/16, DINOv2-S and CLIP ViT-B/32. In every architecture the MTL correlation climbs monotonically with depth and peaks at the last layers (best: {n['rsa_best']}, ρ = {n['rsa_best_rho']:.3f}, p = {n['rsa_best_p']:.3f} across 18 sessions; category RDM alone {n['rsa_cat']:.3f} with hand-corrected labels; low-level 0.014). The MFC stays flat near {n['rsa_mfc']:.3f}. Split-half reliability of an MTL RDM is only ≈0.06, so ceiling-normalised ρ is ≈0.2–0.25. Amygdala (0.05–0.06) ≫ hippocampus (0.015); hippocampal geometry is not categorical (ρ = 0.003 with the category RDM) but is weakly captured by late layers.</p>
<p>Because the picture pool is shared, per-session RDMs can be rank-normalised and pooled over co-shown pairs into a consensus 342×342 MTL RDM. Against it the best layer reaches ρ = {n['pooled_best_rho']:.3f} (permutation p = 0.005, ≈10 null SDs), category {n['pooled_cat']:.3f}. Partialling out the category and low-level RDMs leaves CLIP's last layer significant (partial ρ = {n['part_clip_rho']:.3f}, p = {n['part_clip_p']:.3f}; ResNet-50 avgpool {n['part_rn_rho']:.3f}, p = {n['part_rn_p']:.3f}), while category is no longer significant once CLIP is controlled (p = {n['catpart_clip_p']:.2f}) — late layers largely subsume the categorical structure and add a little to it. Restricted to <em>within</em>-category pairs, CLIP still tracks the MTL (ρ = 0.052, p = 0.015) whereas ResNet-50 does not (−0.006). In time, the correspondence rises from ~200 ms and peaks at <b>{n['t_peak']} ms</b> after picture onset, with late layers &gt; category &gt; early layers &gt; low-level throughout.</p>
</div>
{F('layers', 'RSA layer-depth curves: mean ± sem Spearman ρ over sessions for MTL, MFC and MTL concept cells; dashed/dotted lines are the MTL correlation with the category and low-level RDMs.')}
<div class="grid2">
{F('time', 'Sliding 200-ms windows: MTL correspondence peaks ~350 ms; MFC small and sustained. Squares mark windows with p < 0.05.', wide=False)}
{F('partial', 'Raw vs partial ρ (controlling category + low-level) for a curated set of layers, MTL and MTL concept cells.', wide=False)}
</div>
{table(bl_mtl, ['region','model','best_layer','rho_mean','rho_sem','p','rho_norm_mean','n_sessions'], ['region','model','best layer','mean ρ','sem','p','ρ / √reliability','sessions'], caption='Best layer per model, MTL vs MFC (screening sessions as random effects). A session-wise picture-label permutation gives z ≈ 10–11 for MTL vs late layers and z ≈ 6 for MTL − MFC.')}
<div class="grid2">
{F('within', 'RSA restricted to between-category pairs, within-category pairs, and single categories.', wide=False)}
{F('varpart', 'Commonality analysis of the MTL RDM: CLIP carries the largest unique share.', wide=False)}
</div>
{F('reliab', 'Split-half reliability of neural RDMs vs population size, and observed ρ against the √reliability ceiling per session.', wide=False)}
</section>""")

    parts.append(f"""
<section id="a4"><div class="head"><span class="tag">A4</span><h2>Single neurons generalise along DNN similarity</h2></div>
<div class="prose">
<p>Take a concept cell, drop its preferred picture, and ask whether its firing to the remaining 53–62 pictures follows their similarity to that preferred picture in a given layer. It does: mean Spearman ρ reaches <b>{n['sim_rho']:.2f}</b> at {n['sim_best']} with {n['sim_frac']:.0f}% of MTL concept cells positive, against a random-anchor null of ≈0. The effect grows with layer depth (amygdala late &gt; early, p = 0.0006); MFC concept cells show it only at late layers; non-selective MTL cells barely at all.</p>
{enc_txt}
</div>
{F('simtune', 'Similarity tuning per model layer for MTL concept cells (red), MTL non-concept cells (orange), MFC concept cells (blue); dotted: random-anchor null.')}
<div class="prose"><p>Who carries this generalisation? The strongest cases are broadly tuned "celebrity face" cells whose response ramps smoothly with CLIP similarity across other famous faces; the classic sparse concept cells (a horse cell, an astronaut cell) sit at ρ ≈ 0. Across cells, sparser tuning goes with weaker generalisation (DoS vs ρ −0.2, p ≤ 0.03) and face-preferring cells generalise most (ρ 0.22 vs 0.07, p = 0.001) — but even the sparse half of the population averages ρ ≈ 0.14 &gt; 0.</p></div>
{F('examples', 'Exemplar MTL concept cells: preferred picture, response vs CLIP similarity over the other pictures, and the most / least similar pictures with their rates. Last two rows: ρ ≈ 0.')}
{F('sparsity', 'Similarity tuning vs depth of selectivity and response strength; face-preferring cells in red.', wide=False)}
<div class="grid2">
{F('depth', 'Layer-depth preference by area, pooling layers of all eight models into five relative-depth bins.', wide=False)}
{F('enc_best', 'Encoding models: best-predicting model per MTL concept cell, relative depth of its best layer, and per-cell CV r against the split-half ceiling.', wide=False)}
</div>
<div class="grid2">
{F('tsim', 'Time course of similarity tuning (100-ms windows): onset ~225 ms, peak 325 ms; late layers ≫ conv1.', wide=False)}
{F('facespace', 'Face pictures only: RSA vs a VGGFace2 identity space and object models; face-preferring cells generalise across other faces along object-model similarity more than identity similarity.', wide=False)}
</div>
{F('debiased', 'Shuffle-debiased encoding r vs layer depth: the same early→late gradient as RSA and similarity tuning.', wide=False)}
{F('enc_layers', 'Raw encoding-model layer curves (negatively biased under the null; see the debiased panel above).')}
{enc_tbl}
</section>""")

    parts.append(f"""
<section id="a6"><div class="head"><span class="tag">A6</span><h2>Behaviour</h2></div>
<div class="prose">
<p>With only five memoranda per patient there is little similarity variance to work with. On OUT trials, a lure probe that is CLIP-similar to the encoded set slows correct responses (mean within-subject slope +0.055 s per SD, p = 0.04 uncorrected); other spaces and outcomes are null, and within-set similarity trends, if anything, toward faster responses.</p>
</div>
{table(behav, ['space','condition','predictor','n_subjects','mean_slope','sem','p','wilcoxon_p'], ['space','trials','predictor','subjects','mean slope','sem','p','Wilcoxon p'], caption='Per-subject OLS slopes of RT on similarity (load as covariate), group-level tests.')}
{F('behav', 'OUT-trial RT (z within subject) as a function of maximum lure–set similarity, binned by sextile.')}
</section>""")

    parts.append(f"""
<section id="a7"><div class="head"><span class="tag">A7</span><h2>Do architecture or training objective matter?</h2></div>
<div class="prose">
<p>Not much. Ranking models by their best layer under RSA (paired over sessions) or similarity tuning (paired over cells), the eight networks are statistically close: CLIP and ResNet-50 lead by a hair, only CLIP &gt; AlexNet reaches p = 0.03 in RSA, and the self-supervised DINOv2 sits mid-pack. The cleanest test holds architecture fixed: OpenAI's CLIP ResNet-50 against the ImageNet-trained ResNet-50, layer by layer, differs by at most 0.007 on either criterion (all p ≥ 0.16). Any modern network's late layers capture roughly the same part of MTL selectivity, and the training objective is not what matters.</p>
</div>
{F('models', 'Model ranking at best layer under each criterion (colour: training objective).')}
{F('objective', 'Layer-matched ResNet-50 (ImageNet) vs CLIP-RN50 (language): no difference at any depth.', wide=False)}
{table(rank, ['criterion','model','objective','best_layer','n','mean','sem'], ['criterion','model','objective','best layer','n','mean','sem'], caption='Best-layer scores per model and criterion.')}
</section>""")

    parts.append("""
<section id="next"><div class="head"><span class="tag">—</span><h2>Caveats and what's next</h2></div>
<div class="prose">
<ul class="findings">
<li>Population RDMs from ~20–60 sparse units are noisy (split-half ≈ 0.06); pooling across patients and restricting to concept cells help, but absolute ρ values are small and should be read against the ceiling.</li>
<li>Concept-cell selection reuses the same responses that enter the similarity-tuning and encoding analyses (the preferred picture is excluded, and the random-anchor null is ≈ 0, but selection bias could still inflate the non-preferred slope slightly).</li>
<li>Sternberg concept cells were selected on all encoding presentations (paper: encoding 1 only), giving 29% rather than 21% MTL concept cells in that task.</li>
<li>CLIP zero-shot labels are accurate for animals/vehicles/food but blur "face" vs "scene with people"; a hand-checked label pass would tighten A5.</li>
<li>Cross-validated encoding r is negatively biased under the null; the shuffle-null-debiased values are the ones to quote (script 51 covers six predictors — extending it to every layer is on the list).</li>
<li>Next: variance partitioning between models, a face-trained network, per-cell noise-corrected encoding at each layer, and time-resolved encoding models.</li>
</ul>
</div>
</section>
<footer>Built from <code>results/tables/*.csv</code> by <code>scripts/90_build_report.py</code>. Data: DANDI 000469 (CC-BY-4.0). Kyzar et al., Sci Data 2024; Kamiński et al., Nat Neurosci 2017.</footer>
</div>""")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    html_text = build(args.embed)
    out = args.out or (RESULTS / ("report_embedded.html" if args.embed else "report.html"))
    out.write_text(html_text)
    print(out, f"{out.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
