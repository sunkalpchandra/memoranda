"""Neural response extraction and single-unit selectivity statistics.

Replicates the concept-cell criteria of Kamiński et al. 2017 / Kyzar et al. 2024:

* response window 200–1000 ms after stimulus onset
* permuted one-way ANOVA with image identity as factor (p < 0.05)
* post-hoc permutation t-test: max-response image vs. all other images (p < 0.05)

Everything is vectorised over permutations so a session runs in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .paths import MANIFESTS, NEURAL

RESP_WIN = (0.2, 1.0)  # s after onset (paper)
BASE_WIN = (-0.2, 0.0)  # s before onset (screening ISI is short; see docs/decisions.md)


# ---------------------------------------------------------------------- loading
def load_spikes(subject: int, session: int) -> dict[int, np.ndarray]:
    z = np.load(NEURAL / f"sub-{subject:02d}_ses-{session}_units.npz", allow_pickle=True)
    return {int(u): np.asarray(st, dtype=float) for u, st in zip(z["unit"], z["spike_times"], strict=True)}


def load_presentations(subject: int, session: int) -> pd.DataFrame:
    p = pd.read_csv(MANIFESTS / "presentations.csv")
    p = p[(p.subject == subject) & (p.session == session)].reset_index(drop=True)
    img = pd.read_csv(MANIFESTS / "images.csv")
    img = img[(img.subject == subject) & (img.session == session)][["stim_name", "image_uid", "is_null"]]
    return p.merge(img, on="stim_name", how="left")


def load_units_table(subject: int | None = None, session: int | None = None) -> pd.DataFrame:
    u = pd.read_csv(MANIFESTS / "units.csv")
    if subject is not None:
        u = u[u.subject == subject]
    if session is not None:
        u = u[u.session == session]
    return u.reset_index(drop=True)


# ---------------------------------------------------------------------- counting
def count_in_windows(spikes: np.ndarray, onsets: np.ndarray, win: tuple[float, float]) -> np.ndarray:
    """Spike counts in ``[onset+win0, onset+win1)`` for every onset (vectorised)."""
    spikes = np.sort(spikes)
    lo = np.searchsorted(spikes, onsets + win[0], side="left")
    hi = np.searchsorted(spikes, onsets + win[1], side="left")
    return (hi - lo).astype(float)


def rate_matrix(spike_dict: dict[int, np.ndarray], onsets: np.ndarray, win=RESP_WIN) -> tuple[np.ndarray, list[int]]:
    """(n_events, n_units) firing rates in Hz."""
    units = sorted(spike_dict)
    dur = win[1] - win[0]
    R = np.stack([count_in_windows(spike_dict[u], onsets, win) / dur for u in units], axis=1)
    return R, units


# ---------------------------------------------------------------- statistics
def _group_onehot(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cats, inv = np.unique(labels, return_inverse=True)
    G = np.zeros((len(labels), len(cats)))
    G[np.arange(len(labels)), inv] = 1.0
    return G, cats


def anova_f(R: np.ndarray, G: np.ndarray) -> np.ndarray:
    """One-way ANOVA F for each column of R (events × units) given one-hot groups G."""
    n, k = G.shape
    counts = G.sum(0)  # (k,)
    grand = R.mean(0)  # (u,)
    sums = G.T @ R  # (k, u)
    means = sums / counts[:, None]
    ss_between = (counts[:, None] * (means - grand) ** 2).sum(0)
    ss_total = ((R - grand) ** 2).sum(0)
    ss_within = ss_total - ss_between
    df_b, df_w = k - 1, n - k
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ss_between / df_b) / (ss_within / df_w)
    return np.nan_to_num(F, nan=0.0, posinf=0.0)


def permuted_anova(R: np.ndarray, labels: np.ndarray, n_perm: int = 1000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (F_obs, p_perm) per unit; labels permuted across events."""
    rng = np.random.default_rng(seed)
    G, _ = _group_onehot(labels)
    F_obs = anova_f(R, G)
    count = np.zeros(R.shape[1])
    for _ in range(n_perm):
        perm = rng.permutation(len(labels))
        count += anova_f(R[perm], G) >= F_obs
    p = (count + 1) / (n_perm + 1)
    return F_obs, p


def _welch_t(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Welch t between column-sets a (na × u) and b (nb × u)."""
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0, ddof=1), b.var(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ma - mb) / np.sqrt(va / len(a) + vb / len(b))
    return np.nan_to_num(t, nan=0.0)


def permuted_max_vs_rest(R: np.ndarray, labels: np.ndarray, n_perm: int = 1000, seed: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each unit: preferred image (max mean), t(pref vs rest), one-sided permutation p."""
    rng = np.random.default_rng(seed + 7)
    G, cats = _group_onehot(labels)
    counts = G.sum(0)
    means = (G.T @ R) / counts[:, None]  # (k, u)
    best = means.argmax(0)  # (u,)
    n_units = R.shape[1]
    t_obs = np.zeros(n_units)
    p = np.zeros(n_units)
    for u in range(n_units):
        m = G[:, best[u]] == 1
        a, b = R[m, u][:, None], R[~m, u][:, None]
        t_obs[u] = _welch_t(a, b)[0]
        pooled = R[:, u]
        na = m.sum()
        cnt = 0
        for _ in range(n_perm):
            perm = rng.permutation(len(pooled))
            aa, bb = pooled[perm[:na]][:, None], pooled[perm[na:]][:, None]
            cnt += _welch_t(aa, bb)[0] >= t_obs[u]
        p[u] = (cnt + 1) / (n_perm + 1)
    return cats[best], t_obs, p


def depth_of_selectivity(means: np.ndarray) -> float:
    """Rainer/Miller depth-of-selectivity: (n − Σ r_i / r_max) / (n − 1)."""
    n = len(means)
    rmax = means.max()
    if rmax <= 0 or n < 2:
        return 0.0
    return float((n - means.sum() / rmax) / (n - 1))


def sparseness(means: np.ndarray) -> float:
    """Treves–Rolls lifetime sparseness a = (Σr/n)² / (Σr²/n); low = sparse."""
    r = np.clip(means, 0, None)
    n = len(r)
    if r.sum() == 0:
        return 1.0
    return float((r.sum() / n) ** 2 / ((r**2).sum() / n))


# ------------------------------------------------------------------ session run
@dataclass
class UnitSelectivity:
    subject: int
    session: int
    unit: int
    n_events: int
    n_images: int
    mean_rate: float
    F: float
    p_anova: float
    pref_image: str
    t_post: float
    p_post: float
    concept_cell: bool
    dos: float
    sparseness: float
    pref_rate: float
    nonpref_rate: float
    baseline_rate: float


def analyse_session(subject: int, session: int, n_perm: int = 1000, min_spikes: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the concept-cell pipeline on one session.

    Returns (units_df, tuning_df) where tuning_df has one row per (unit, image)
    with mean/sd rate and baseline-normalised z, plus n reps.
    """
    spikes = load_spikes(subject, session)
    pres = load_presentations(subject, session)
    pres = pres[~pres.is_null.fillna(False)].reset_index(drop=True)
    onsets = pres["onset"].to_numpy()
    labels = pres["image_uid"].to_numpy()

    R, units = rate_matrix(spikes, onsets, RESP_WIN)
    B, _ = rate_matrix(spikes, onsets, BASE_WIN)
    keep = np.array([spikes[u].size >= min_spikes for u in units])

    F, p_anova = permuted_anova(R, labels, n_perm=n_perm)
    pref, t_post, p_post = permuted_max_vs_rest(R, labels, n_perm=n_perm)

    G, cats = _group_onehot(labels)
    counts = G.sum(0)
    means = (G.T @ R) / counts[:, None]
    sq = (G.T @ R**2) / counts[:, None]
    sds = np.sqrt(np.clip(sq - means**2, 0, None))
    base_mean = B.mean(0)
    base_sd = B.std(0) + 1e-9

    rows, tune = [], []
    for j, u in enumerate(units):
        best = np.where(cats == pref[j])[0][0]
        others = np.delete(means[:, j], best)
        rows.append(
            UnitSelectivity(
                subject=subject,
                session=session,
                unit=u,
                n_events=len(labels),
                n_images=len(cats),
                mean_rate=float(R[:, j].mean()),
                F=float(F[j]),
                p_anova=float(p_anova[j]),
                pref_image=str(pref[j]),
                t_post=float(t_post[j]),
                p_post=float(p_post[j]),
                concept_cell=bool(keep[j] and p_anova[j] < 0.05 and p_post[j] < 0.05),
                dos=depth_of_selectivity(means[:, j]),
                sparseness=sparseness(means[:, j]),
                pref_rate=float(means[best, j]),
                nonpref_rate=float(others.mean()) if len(others) else np.nan,
                baseline_rate=float(base_mean[j]),
            ).__dict__
        )
        for k, c in enumerate(cats):
            tune.append(
                {
                    "subject": subject,
                    "session": session,
                    "unit": u,
                    "image_uid": c,
                    "n_reps": int(counts[k]),
                    "rate": float(means[k, j]),
                    "rate_sd": float(sds[k, j]),
                    "z_base": float((means[k, j] - base_mean[j]) / base_sd[j]),
                    "is_pref": bool(k == best),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(tune)
