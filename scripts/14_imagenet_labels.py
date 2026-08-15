#!/usr/bin/env python
"""Stage 14 — ImageNet-1k top-5 labels per unique image, from stored ResNet-50 / ConvNeXt logits.

Output: data/manifests/imagenet_labels.csv (image_uid, top1..top5 with probabilities, entropy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torchvision.models import ResNet50_Weights

from memoranda.features import load_features
from memoranda.paths import MANIFESTS


def main() -> None:
    cats = ResNet50_Weights.DEFAULT.meta["categories"]
    rows = []
    for model in ("resnet50", "convnext_tiny"):
        logits, uids = load_features(model, "logits", "gap")
        p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        top = np.argsort(-p, axis=1)[:, :5]
        ent = -(p * np.log(p + 1e-12)).sum(1)
        for i, u in enumerate(uids):
            rows.append(
                {
                    "image_uid": u,
                    "model": model,
                    **{f"top{k+1}": cats[top[i, k]] for k in range(5)},
                    **{f"p{k+1}": float(p[i, top[i, k]]) for k in range(5)},
                    "entropy": float(ent[i]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(MANIFESTS / "imagenet_labels.csv", index=False)
    r = df[df.model == "resnet50"]
    print(r[["image_uid", "top1", "p1", "top2", "entropy"]].head(20).to_string(index=False))
    print("mean top1 prob", r.p1.mean(), " median entropy", r.entropy.median())
    print(r.top1.value_counts().head(15))


if __name__ == "__main__":
    main()
