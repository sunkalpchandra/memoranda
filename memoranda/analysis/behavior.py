"""Sternberg behaviour vs. representational similarity of the memoranda.

For each trial we know the 1–3 encoded images and the probe. In a feature space
(CLIP, ResNet, …) we compute

* ``sim_probe_max``  – max cosine similarity between the probe and the encoded set
                       (for OUT trials: how confusable is the lure? for IN trials
                       this is 1 by construction, so we use the max over the *other*
                       encoded items: ``sim_probe_other``)
* ``sim_within``     – mean pairwise similarity among the encoded items (load ≥ 2)

and relate them to reaction time (correct trials) and accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..features import load_space
from ..paths import MANIFESTS


def _embed(model: str, layer: str, view: str) -> tuple[np.ndarray, dict[str, int]]:
    X, stored = load_space(model, layer, view, normalize=True)
    return X, {u: i for i, u in enumerate(stored)}


def sternberg_trials_with_uids() -> pd.DataFrame:
    """Trial table with image_uid columns for enc1..3 and probe."""
    t = pd.read_csv(MANIFESTS / "trials_sternberg.csv")
    img = pd.read_csv(MANIFESTS / "images.csv")
    img = img[(img.task == "sternberg")][["subject", "stim_index", "stim_name", "image_uid", "is_null"]]
    # PicIDs are 1..5 -> template position 0..4 (position 5 is the null image)
    lut = {(r.subject, r.stim_index): r.image_uid for r in img.itertuples()}
    for col, new in [("loadsEnc1_PicIDs", "enc1"), ("loadsEnc2_PicIDs", "enc2"), ("loadsEnc3_PicIDs", "enc3"), ("loadsProbe_PicIDs", "probe")]:
        t[new] = [lut.get((s, int(p) - 1)) if p > 0 else None for s, p in zip(t.subject, t[col])]
    t["rt"] = t.timestamps_Response - t.timestamps_Probe
    t["correct"] = t.response_accuracy.astype(int)
    t["probe_in"] = t.probe_in_out.astype(int)
    return t


def add_similarity(t: pd.DataFrame, model: str, layer: str, view: str, prefix: str) -> pd.DataFrame:
    X, pos = _embed(model, layer, view)
    t = t.copy()
    s_max, s_other, s_within = [], [], []
    for r in t.itertuples():
        enc = [u for u in (r.enc1, r.enc2, r.enc3) if isinstance(u, str)]
        p = r.probe
        if p not in pos or any(u not in pos for u in enc):
            s_max.append(np.nan)
            s_other.append(np.nan)
            s_within.append(np.nan)
            continue
        e = X[[pos[u] for u in enc]]
        sims = e @ X[pos[p]]
        s_max.append(float(sims.max()))
        others = [u for u in enc if u != p]
        s_other.append(float((X[[pos[u] for u in others]] @ X[pos[p]]).max()) if others else np.nan)
        if len(enc) >= 2:
            S = e @ e.T
            iu = np.triu_indices(len(enc), 1)
            s_within.append(float(S[iu].mean()))
        else:
            s_within.append(np.nan)
    t[f"{prefix}_sim_probe_max"] = s_max
    t[f"{prefix}_sim_probe_other"] = s_other
    t[f"{prefix}_sim_within"] = s_within
    return t


def per_subject_slopes(t: pd.DataFrame, x: str, y: str, subset: pd.Series | None = None, min_n: int = 15, controls: tuple[str, ...] = ("loads",)) -> pd.DataFrame:
    """OLS slope of y on x (with controls) per subject; then group-level t-test."""
    d = t if subset is None else t[subset]
    rows = []
    for s, g in d.groupby("subject"):
        g = g.dropna(subset=[x, y, *controls])
        if len(g) < min_n or g[x].std() == 0:
            continue
        A = np.column_stack([np.ones(len(g)), (g[x] - g[x].mean()) / g[x].std(), *[g[c] for c in controls]])
        beta, *_ = np.linalg.lstsq(A, g[y].to_numpy(float), rcond=None)
        rows.append({"subject": s, "n": len(g), "slope": beta[1]})
    df = pd.DataFrame(rows)
    return df


def group_test(slopes: pd.DataFrame) -> dict:
    if len(slopes) < 3:
        return {"n_subjects": len(slopes), "mean_slope": np.nan, "t": np.nan, "p": np.nan, "wilcoxon_p": np.nan}
    t, p = sps.ttest_1samp(slopes.slope, 0)
    try:
        wp = sps.wilcoxon(slopes.slope).pvalue
    except ValueError:
        wp = np.nan
    return {"n_subjects": len(slopes), "mean_slope": float(slopes.slope.mean()), "sem": float(slopes.slope.std() / np.sqrt(len(slopes))), "t": float(t), "p": float(p), "wilcoxon_p": float(wp)}
