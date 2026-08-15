"""Layer-wise feature extraction with forward hooks.

For every hooked module we keep two pooled views of the activation:

* ``gap``  — global average pool over spatial / token positions → (C,)
* ``rp``   — a fixed sparse random projection of the *flattened* activation
             → (RP_DIM,), which preserves spatial layout information that
             pooling throws away (useful for early layers in RSA).

Token-based models (ViT / DINOv2 / CLIP) additionally get ``cls`` (first token).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

from .registry import ModelSpec, device

RP_DIM = 2048
RP_SEED = 1234


def resolve_module(model: nn.Module, dotted: str) -> nn.Module:
    m = model
    for part in dotted.split("."):
        m = m[int(part)] if part.isdigit() and not hasattr(m, part) else getattr(m, part)
    return m


class _Projector:
    """Cache of sparse random projections keyed by input dimensionality."""

    def __init__(self, out_dim: int = RP_DIM, seed: int = RP_SEED):
        self.out_dim = out_dim
        self.seed = seed
        self._cache: dict[int, torch.Tensor] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)
        d = x.shape[1]
        if d <= self.out_dim:
            return x
        if d not in self._cache:
            g = torch.Generator().manual_seed(self.seed + d)
            # Achlioptas sparse projection: entries in {-1,0,+1} with p=1/6,2/3,1/6
            r = torch.rand((d, self.out_dim), generator=g)
            w = torch.zeros((d, self.out_dim))
            w[r < 1 / 6] = -1.0
            w[r > 5 / 6] = 1.0
            w *= np.sqrt(3.0 / self.out_dim)
            self._cache[d] = w
        if len(self._cache) > 6:  # bound memory: drop the oldest projection
            self._cache.pop(next(iter(self._cache)))
        w = self._cache[d].to(x.device, x.dtype)
        return x @ w


MAX_FLAT = 32768  # cap on flattened dims before random projection (memory)


def _spatial_cap(act: torch.Tensor) -> torch.Tensor:
    """Adaptive-avg-pool a (B, C, H, W) map so that C*h*w <= MAX_FLAT."""
    b, c, h, w = act.shape
    g = min(h, w)
    while g > 1 and c * g * g > MAX_FLAT:
        g //= 2
    if g < min(h, w):
        act = torch.nn.functional.adaptive_avg_pool2d(act, g)
    return act


def _token_grid(tokens: torch.Tensor) -> torch.Tensor | None:
    """(B, N, D) patch tokens → (B, D, s, s) if N is a perfect square."""
    b, n, d = tokens.shape
    s = int(round(n**0.5))
    if s * s != n:
        return None
    return tokens.transpose(1, 2).reshape(b, d, s, s)


def _pool(act: torch.Tensor, is_token_model: bool) -> dict[str, torch.Tensor]:
    """Return pooled views of one activation tensor (batch first)."""
    out: dict[str, torch.Tensor] = {}
    if act.ndim == 4:  # B, C, H, W
        out["gap"] = act.mean(dim=(2, 3))
        out["flat"] = _spatial_cap(act).flatten(1)
    elif act.ndim == 3:  # B, N, D  (tokens) — CLIP resblocks give N, B, D
        if is_token_model and act.shape[0] > act.shape[1]:
            act = act.transpose(0, 1)
        out["cls"] = act[:, 0]
        patches = act[:, 1:] if act.shape[1] > 1 else act
        out["gap"] = patches.mean(dim=1)
        grid = _token_grid(patches)
        if grid is not None:
            out["flat"] = _spatial_cap(grid).flatten(1)
        else:
            out["flat"] = act.flatten(1)[:, :MAX_FLAT]
    elif act.ndim == 2:  # B, D
        out["gap"] = act
        out["flat"] = act
    else:
        out["flat"] = act.flatten(1)[:, :MAX_FLAT]
        out["gap"] = out["flat"]
    return out


@dataclass
class Extracted:
    model: str
    layer: str
    view: str  # gap | rp | cls
    features: np.ndarray  # (n_images, D) float32


class FeatureExtractor:
    def __init__(self, spec: ModelSpec, batch_size: int = 16, rp_dim: int = RP_DIM):
        self.spec = spec
        self.batch_size = batch_size
        self.dev = device()
        self.model, self.preprocess = spec.build()
        self.model.to(self.dev)
        self.projector = _Projector(rp_dim)
        self._acts: dict[str, torch.Tensor] = {}
        self._handles = []
        for label, path in spec.layers.items():
            mod = resolve_module(self.model, path)
            self._handles.append(mod.register_forward_hook(self._make_hook(label)))

    def _make_hook(self, label):
        def hook(_m, _i, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            self._acts[label] = out.detach()

        return hook

    def close(self):
        for h in self._handles:
            h.remove()

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) -> None:
        self._acts.clear()
        if self.spec.family == "clip":
            self.model.encode_image(x)
        else:
            self.model(x)

    def run(self, images: Iterable[Image.Image], n: int | None = None, progress=None) -> dict[tuple[str, str], np.ndarray]:
        """Extract features for an iterable of PIL images.

        Returns ``{(layer, view): array (n, D)}``.
        """
        is_token = self.spec.family in ("vit", "dino", "clip")
        buffers: dict[tuple[str, str], list[np.ndarray]] = {}
        batch: list[torch.Tensor] = []

        def flush():
            if not batch:
                return
            x = torch.stack(batch).to(self.dev)
            self._forward(x)
            for label, act in self._acts.items():
                pooled = _pool(act.float(), is_token)
                flat = pooled.pop("flat")
                pooled["rp"] = self.projector(flat)
                for view, t in pooled.items():
                    buffers.setdefault((label, view), []).append(t.cpu().numpy().astype(np.float32))
            batch.clear()

        it = images if progress is None else progress(images, total=n)
        for im in it:
            batch.append(self.preprocess(im))
            if len(batch) >= self.batch_size:
                flush()
        flush()
        return {k: np.concatenate(v, axis=0) for k, v in buffers.items()}
