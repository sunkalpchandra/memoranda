"""Canonical filesystem locations for the project.

Everything is resolved relative to the repository root so scripts can be run
from anywhere. Large artefacts (raw NWB slices, images, features) live under
``data/`` and are git-ignored; small manifests under ``data/manifests`` and
all of ``results/`` are committed.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CONCEPTLENS_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
MANIFESTS = DATA / "manifests"
IMAGES = DATA / "images"
FEATURES = DATA / "features"
CACHE = DATA / "cache"
NEURAL = DATA / "neural"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

CONFIGS = ROOT / "configs"
DOCS = ROOT / "docs"


def ensure_dirs() -> None:
    """Create every directory this project writes to."""
    for p in (MANIFESTS, IMAGES, FEATURES, CACHE, NEURAL, FIGURES, TABLES):
        p.mkdir(parents=True, exist_ok=True)
