"""DANDI archive access for dandiset 000469 (Kyzar et al. 2024).

We talk to the public DANDI REST API directly (no ``dandi`` CLI needed) to
enumerate assets and build direct-download URLs that ``remfile`` can stream.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from .paths import MANIFESTS

DANDISET_ID = "000469"
DANDISET_VERSION = "0.240123.1806"
API = "https://api.dandiarchive.org/api"

# 1: screening (SC), 2: Sternberg (SB) — see Kyzar et al. "Data Records"
TASK_BY_SESSION = {1: "screening", 2: "sternberg"}

_PATH_RE = re.compile(r"sub-(\d+)/sub-\d+_ses-(\d+)_ecephys\+image\.nwb")


@dataclass(frozen=True)
class Asset:
    subject: int
    session: int
    task: str
    path: str
    asset_id: str
    size: int

    @property
    def download_url(self) -> str:
        return f"{API}/assets/{self.asset_id}/download/"

    @property
    def key(self) -> str:
        return f"sub-{self.subject:02d}_ses-{self.session}"


def _fetch_assets_from_api() -> list[Asset]:
    url = f"{API}/dandisets/{DANDISET_ID}/versions/{DANDISET_VERSION}/assets/?page_size=200"
    out: list[Asset] = []
    while url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        payload = r.json()
        for a in payload["results"]:
            m = _PATH_RE.fullmatch(a["path"])
            if not m:
                continue
            sub, ses = int(m.group(1)), int(m.group(2))
            out.append(
                Asset(
                    subject=sub,
                    session=ses,
                    task=TASK_BY_SESSION[ses],
                    path=a["path"],
                    asset_id=a["asset_id"],
                    size=a["size"],
                )
            )
        url = payload.get("next")
    return sorted(out, key=lambda a: (a.subject, a.session))


def list_assets(refresh: bool = False, cache: Path | None = None) -> list[Asset]:
    """Return all NWB assets, using a committed JSON manifest as cache."""
    cache = cache or MANIFESTS / "dandi_assets.json"
    if cache.exists() and not refresh:
        return [Asset(**a) for a in json.loads(cache.read_text())]
    assets = _fetch_assets_from_api()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(a) for a in assets], indent=1))
    return assets


def get_asset(subject: int, session: int) -> Asset:
    for a in list_assets():
        if a.subject == subject and a.session == session:
            return a
    raise KeyError(f"no asset for sub-{subject} ses-{session}")
