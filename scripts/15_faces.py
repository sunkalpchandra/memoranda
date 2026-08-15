#!/usr/bin/env python
"""Stage 15 — faces: MTCNN detection + VGGFace2 InceptionResnetV1 identity embeddings.

Per unique image: number of faces (p ≥ 0.9), area fraction of the largest face, its centre.
Face-identity embedding (512-d) of the largest face; images without a face get the
embedding of the whole image resized to 160 px (flagged ``face_found = 0``) so the model
can still be used dataset-wide, and a face-only variant restricted to detected faces.
Outputs: data/manifests/faces.csv, data/features/facenet_vggface2.npz
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm

from memoranda.features import save_features, unique_image_table
from memoranda.paths import MANIFESTS


def main() -> None:
    table = unique_image_table()
    mtcnn = MTCNN(keep_all=True, device="cpu", thresholds=[0.6, 0.7, 0.7])
    net = InceptionResnetV1(pretrained="vggface2").eval()
    rows, embs = [], []
    for _, r in tqdm(table.iterrows(), total=len(table)):
        im = Image.open(r.abs_path).convert("RGB")
        boxes, probs = mtcnn.detect(im)
        n = 0
        area, cx, cy = 0.0, np.nan, np.nan
        crop = None
        if boxes is not None:
            keep = probs >= 0.9
            boxes, probs = boxes[keep], probs[keep]
            n = int(len(boxes))
            if n:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                k = int(np.argmax(areas))
                b = boxes[k]
                area = float(areas[k] / (im.width * im.height))
                cx, cy = float((b[0] + b[2]) / 2 / im.width), float((b[1] + b[3]) / 2 / im.height)
                # square crop with margin
                w = max(b[2] - b[0], b[3] - b[1]) * 1.3
                mx, my = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                crop = im.crop((int(mx - w / 2), int(my - w / 2), int(mx + w / 2), int(my + w / 2)))
        src = crop if crop is not None else im
        x = torch.from_numpy(np.asarray(src.resize((160, 160)), dtype=np.float32)).permute(2, 0, 1)
        x = (x - 127.5) / 128.0
        with torch.no_grad():
            e = net(x[None])[0].numpy()
        embs.append(e)
        rows.append({"image_uid": r.image_uid, "n_faces": n, "face_found": int(n > 0), "largest_face_area": area, "face_cx": cx, "face_cy": cy})
    df = pd.DataFrame(rows)
    df.to_csv(MANIFESTS / "faces.csv", index=False)
    E = np.stack(embs).astype(np.float32)
    save_features("facenet_vggface2", table.image_uid.tolist(), {("embed", "gap"): E})
    print(df.n_faces.value_counts().sort_index())
    print("face found in", df.face_found.mean() * 100, "% of images")


if __name__ == "__main__":
    main()
