#!/usr/bin/env python
"""Stage 01 — pull every stimulus image out of every NWB file.

Writes ``data/images/sub-XX/ses-Y/<name>.png`` (upright RGB) and the manifest
``data/manifests/images.csv`` with one row per (subject, session, template).
Also writes ``data/manifests/sessions.csv`` with per-session metadata.
Resumable: sessions whose PNGs already exist are skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from memoranda import images as I
from memoranda import nwb
from memoranda.dandi import list_assets
from memoranda.log import get_logger
from memoranda.paths import MANIFESTS, ensure_dirs

log = get_logger("extract_images")


def process(asset, force: bool) -> tuple[list[dict], dict]:
    f = nwb.open_remote(asset)
    meta = nwb.read_meta(f)
    names = nwb.stimulus_names(f)
    rows = []
    for idx, name in enumerate(names):
        out = I.image_path(asset.subject, asset.session, name)
        raw = nwb.read_image(f, name)
        arr = I.fix_orientation(raw)
        blank = I.is_blank(arr) or name == nwb.NULL_IMAGE
        if force or not out.exists():
            I.save_png(arr, out)
        rows.append(
            {
                "subject": asset.subject,
                "session": asset.session,
                "task": asset.task,
                "stim_index": idx,
                "stim_name": name,
                "stim_num": int(name.split("_")[-1]) if name.split("_")[-1].isdigit() else -1,
                "is_null": bool(blank),
                "height": arr.shape[0],
                "width": arr.shape[1],
                "sha1": I.sha1_of_array(arr),
                "dhash": I.dhash(arr),
                "path": str(out.relative_to(out.parents[3])),
            }
        )
    sess = {
        "subject": asset.subject,
        "session": asset.session,
        "task": asset.task,
        "asset_id": asset.asset_id,
        "size_mb": round(asset.size / 1e6, 1),
        **meta.__dict__,
    }
    f.close()
    return rows, sess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", type=str, default=None, help="e.g. 19-2 to run one session")
    args = ap.parse_args()
    ensure_dirs()

    assets = list_assets()
    if args.only:
        s, e = args.only.split("-")
        assets = [a for a in assets if a.subject == int(s) and a.session == int(e)]

    all_rows, sess_rows = [], []
    for a in assets:
        t0 = time.time()
        rows, sess = process(a, args.force)
        all_rows += rows
        sess_rows.append(sess)
        log.info(f"{a.key} {a.task:9s} {len(rows):3d} images, {sess['n_units']:3d} units  ({time.time()-t0:.1f}s)")

    img_df = pd.DataFrame(all_rows)
    sess_df = pd.DataFrame(sess_rows)
    if not args.only:
        img_df.to_csv(MANIFESTS / "images.csv", index=False)
        sess_df.to_csv(MANIFESTS / "sessions.csv", index=False)
        log.info(f"wrote {len(img_df)} image rows, {len(sess_df)} sessions")
    else:
        print(img_df.head(), sess_df.T)


if __name__ == "__main__":
    main()
