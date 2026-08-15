#!/usr/bin/env python
"""Stage 10 — run every registered vision model over the unique image pool.

Writes ``data/features/<model>.npz`` (layer × view arrays) and a summary of
shapes to ``data/manifests/feature_index.csv``. Resumable per model.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from PIL import Image
from tqdm import tqdm

from memoranda.features import (
    feature_path,
    list_layers,
    load_features,
    save_features,
    unique_image_table,
)
from memoranda.log import get_logger
from memoranda.models.extract import FeatureExtractor
from memoranda.models.registry import DEFAULT_MODELS, get_spec
from memoranda.paths import MANIFESTS, ensure_dirs

log = get_logger("features")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    table = unique_image_table()
    uids = table["image_uid"].tolist()
    paths = table["abs_path"].tolist()
    log.info(f"{len(uids)} unique images")

    def images():
        for p in paths:
            yield Image.open(p).convert("RGB")

    for name in args.models:
        out = feature_path(name)
        if out.exists() and not args.force:
            log.info(f"[skip] {name} exists")
            continue
        spec = get_spec(name)
        t0 = time.time()
        fx = FeatureExtractor(spec, batch_size=args.batch_size)
        feats = fx.run(images(), n=len(uids), progress=tqdm)
        fx.close()
        save_features(name, uids, feats)
        log.info(f"{name}: {len(feats)} arrays in {time.time()-t0:.0f}s -> {out}")

    rows = []
    # index every model that has a feature file, not only the ones extracted in this call
    all_models = sorted(p.stem for p in feature_path("x").parent.glob("*.npz") if not p.stem.endswith("_embed"))
    for name in all_models:
        for layer, view in list_layers(name):
            X, _ = load_features(name, layer, view)
            rows.append({"model": name, "layer": layer, "view": view, "n_images": X.shape[0], "dim": X.shape[1]})
    pd.DataFrame(rows).to_csv(MANIFESTS / "feature_index.csv", index=False)
    log.info("wrote feature_index.csv")


if __name__ == "__main__":
    main()
