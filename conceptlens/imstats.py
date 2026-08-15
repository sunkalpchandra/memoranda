"""Low-level image statistics (model-free descriptors of each stimulus).

All functions take an RGB uint8 array (H, W, 3) and return floats. These are
the "nuisance" covariates one wants to control for when asking whether a DNN
representation explains neural selectivity beyond simple image properties.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def to_gray(arr: np.ndarray) -> np.ndarray:
    """ITU-R 601 luma in [0, 1]."""
    a = arr.astype(np.float32) / 255.0
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def mean_luminance(arr) -> float:
    return float(to_gray(arr).mean())


def rms_contrast(arr) -> float:
    return float(to_gray(arr).std())


def michelson_contrast(arr) -> float:
    g = to_gray(arr)
    lo, hi = np.percentile(g, [1, 99])
    return float((hi - lo) / (hi + lo + 1e-8))


def colorfulness(arr) -> float:
    """Hasler & Süsstrunk (2003) colorfulness metric."""
    a = arr.astype(np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def saturation_mean(arr) -> float:
    hsv = np.asarray(Image.fromarray(arr).convert("HSV"), dtype=np.float32) / 255.0
    return float(hsv[..., 1].mean())


def hue_entropy(arr, bins: int = 36) -> float:
    hsv = np.asarray(Image.fromarray(arr).convert("HSV"), dtype=np.float32) / 255.0
    w = hsv[..., 1] * hsv[..., 2]  # weight by chroma so grey pixels don't count
    h, _ = np.histogram(hsv[..., 0], bins=bins, range=(0, 1), weights=w)
    p = h / (h.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gray_entropy(arr, bins: int = 64) -> float:
    g = to_gray(arr)
    h, _ = np.histogram(g, bins=bins, range=(0, 1))
    p = h / h.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def edge_density(arr, sigma: float = 1.0, thresh: float = 0.1) -> float:
    g = to_gray(arr)
    gx = ndimage.gaussian_filter(g, sigma, order=(0, 1))
    gy = ndimage.gaussian_filter(g, sigma, order=(1, 0))
    mag = np.hypot(gx, gy)
    return float((mag > thresh).mean())


def spectral_slope(arr) -> float:
    """Slope of log power vs log spatial frequency (natural images ≈ -2)."""
    g = to_gray(arr)
    g = g - g.mean()
    n = min(g.shape)
    g = g[:n, :n]
    win = np.outer(np.hanning(n), np.hanning(n))
    F = np.fft.fftshift(np.fft.fft2(g * win))
    P = np.abs(F) ** 2
    cy, cx = n // 2, n // 2
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - cy, xx - cx).astype(int)
    radial = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    f = np.arange(len(radial))
    m = (f >= 2) & (f <= n // 4) & (radial > 0)
    if m.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(f[m]), np.log(radial[m]), 1)
    return float(slope)


def high_freq_energy(arr, cutoff_frac: float = 0.25) -> float:
    """Fraction of spectral power above ``cutoff_frac`` of Nyquist."""
    g = to_gray(arr) - to_gray(arr).mean()
    n = min(g.shape)
    g = g[:n, :n]
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    cy, cx = n // 2, n // 2
    yy, xx = np.indices(P.shape)
    r = np.hypot(yy - cy, xx - cx) / (n / 2)
    return float(P[r > cutoff_frac].sum() / (P.sum() + 1e-12))


def spatial_center_of_mass(arr) -> tuple[float, float]:
    """Centre of mass of edge magnitude, normalised to [0,1] (x, y)."""
    g = to_gray(arr)
    gx = ndimage.gaussian_filter(g, 1.0, order=(0, 1))
    gy = ndimage.gaussian_filter(g, 1.0, order=(1, 0))
    mag = np.hypot(gx, gy) + 1e-8
    cy, cx = ndimage.center_of_mass(mag)
    return float(cx / arr.shape[1]), float(cy / arr.shape[0])


def compute_all(arr: np.ndarray) -> dict[str, float]:
    cx, cy = spatial_center_of_mass(arr)
    return {
        "luminance": mean_luminance(arr),
        "rms_contrast": rms_contrast(arr),
        "michelson_contrast": michelson_contrast(arr),
        "colorfulness": colorfulness(arr),
        "saturation": saturation_mean(arr),
        "hue_entropy": hue_entropy(arr),
        "gray_entropy": gray_entropy(arr),
        "edge_density": edge_density(arr),
        "spectral_slope": spectral_slope(arr),
        "high_freq_energy": high_freq_energy(arr),
        "edge_com_x": cx,
        "edge_com_y": cy,
        "aspect": arr.shape[1] / arr.shape[0],
    }


STAT_NAMES = [
    "luminance",
    "rms_contrast",
    "michelson_contrast",
    "colorfulness",
    "saturation",
    "hue_entropy",
    "gray_entropy",
    "edge_density",
    "spectral_slope",
    "high_freq_energy",
    "edge_com_x",
    "edge_com_y",
    "aspect",
]
