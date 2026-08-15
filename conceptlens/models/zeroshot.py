"""CLIP zero-shot labelling of stimuli against the taxonomy in configs/taxonomy.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ..paths import CONFIGS
from .registry import device


def load_taxonomy(path: Path | None = None) -> dict:
    path = path or CONFIGS / "taxonomy.yaml"
    return yaml.safe_load(path.read_text())


class ClipZeroShot:
    def __init__(self, arch: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip

        self.dev = device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = self.model.eval().to(self.dev)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self.logit_scale = float(self.model.logit_scale.exp().item())

    @torch.no_grad()
    def text_embed(self, prompts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(prompts).to(self.dev)
        e = self.model.encode_text(tok).float()
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def class_embeds(self, classes: dict[str, list[str]]) -> tuple[list[str], torch.Tensor]:
        names, embs = [], []
        for name, prompts in classes.items():
            e = self.text_embed(prompts).mean(dim=0)
            embs.append(e / e.norm())
            names.append(name)
        return names, torch.stack(embs)

    @torch.no_grad()
    def image_embed(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.stack([self.preprocess(im) for im in images[i : i + batch_size]]).to(self.dev)
            e = self.model.encode_image(x).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out, 0)

    def probs(self, img_emb: np.ndarray, txt_emb: torch.Tensor) -> np.ndarray:
        """Softmax over classes for each image (n_images, n_classes)."""
        logits = self.logit_scale * torch.from_numpy(img_emb).to(self.dev) @ txt_emb.T
        return logits.softmax(dim=-1).cpu().numpy()


def label_images(zs: ClipZeroShot, img_emb: np.ndarray, taxonomy: dict) -> dict[str, np.ndarray]:
    """Return a dict of arrays keyed by column name.

    * ``category`` (str), ``category_p`` (float), ``p_cat_<name>`` (float) for each class
    * ``attr_<name>`` : probability of the *first* prompt of each attribute contrast
    * ``fine`` (str), ``fine_p``
    """
    out: dict[str, np.ndarray] = {}
    names, te = zs.class_embeds(taxonomy["category"])
    p = zs.probs(img_emb, te)
    out["category"] = np.array(names)[p.argmax(1)]
    out["category_p"] = p.max(1)
    for j, n in enumerate(names):
        out[f"p_cat_{n}"] = p[:, j]

    for attr, pair in taxonomy["attributes"].items():
        te = zs.text_embed(pair)
        p = zs.probs(img_emb, te)
        out[f"attr_{attr}"] = p[:, 0]

    fine = taxonomy["fine"]
    te = zs.text_embed([f"a photo of a {f}" if not f.startswith("a ") else f for f in fine])
    p = zs.probs(img_emb, te)
    out["fine"] = np.array(fine)[p.argmax(1)]
    out["fine_p"] = p.max(1)
    return out
