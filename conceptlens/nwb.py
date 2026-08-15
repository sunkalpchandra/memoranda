"""Streaming readers for the NWB files of DANDI 000469.

Files are opened over HTTP with :mod:`remfile` so that only the byte ranges we
touch are downloaded. Nothing here needs ``pynwb`` — we read the HDF5 groups
directly, which is both faster and more transparent for the handful of
datasets we care about (images, presentation order, trials, units).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
import remfile

from .dandi import Asset

NULL_IMAGE = "image_999"  # blank placeholder referenced for loads < 3 in Sternberg files


def open_remote(asset: Asset) -> h5py.File:
    """Open an NWB asset over HTTP (read-only, lazily fetched)."""
    return h5py.File(remfile.File(asset.download_url), "r")


def _decode(x) -> str:
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray):
        return _decode(x.item()) if x.size == 1 else str(x)
    return str(x)


# --------------------------------------------------------------------------- meta
@dataclass
class SessionMeta:
    identifier: str
    session_description: str
    subject_id: str
    native_subject_id: str
    age: str
    sex: str
    session_start_time: str
    n_electrodes: int
    n_units: int
    n_stimuli: int
    n_presentations: int
    n_trials: int


def read_meta(f: h5py.File) -> SessionMeta:
    ident = _decode(f["identifier"][()])
    # identifier is e.g. 'SBID_19_P47HMH' or 'SCID_20_P49CS'
    parts = ident.split("_")
    native = parts[-1] if len(parts) >= 3 else ""
    return SessionMeta(
        identifier=ident,
        session_description=_decode(f["session_description"][()]),
        subject_id=_decode(f["general/subject/subject_id"][()]),
        native_subject_id=native,
        age=_decode(f["general/subject/age"][()]),
        sex=_decode(f["general/subject/sex"][()]),
        session_start_time=_decode(f["session_start_time"][()]),
        n_electrodes=int(f["general/extracellular_ephys/electrodes/id"].shape[0]),
        n_units=int(f["units/id"].shape[0]),
        n_stimuli=len(stimulus_names(f)),
        n_presentations=int(f["stimulus/presentation/StimulusPresentation/data"].shape[0]),
        n_trials=int(f["intervals/trials/id"].shape[0]),
    )


# ------------------------------------------------------------------------ stimuli
def stimulus_names(f: h5py.File) -> list[str]:
    """Names of stimulus templates in *presentation index order*.

    ``StimulusPresentation/data`` indexes into ``order_of_images`` (an array of
    HDF5 object references), *not* into the alphabetically sorted group keys.
    """
    grp = f["stimulus/templates/StimulusTemplates"]
    refs = grp["order_of_images"][:]
    return [f[r].name.rsplit("/", 1)[-1] for r in refs]


def read_image(f: h5py.File, name: str) -> np.ndarray:
    """Return the raw stored array for one template (H, W, 3) uint8."""
    return f[f"stimulus/templates/StimulusTemplates/{name}"][:]


def read_all_images(f: h5py.File, skip_null: bool = True) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in stimulus_names(f):
        if skip_null and name == NULL_IMAGE:
            continue
        out[name] = read_image(f, name)
    return out


def read_presentation(f: h5py.File) -> pd.DataFrame:
    names = stimulus_names(f)
    grp = f["stimulus/presentation/StimulusPresentation"]
    idx = grp["data"][:].astype(int)
    ts = grp["timestamps"][:]
    return pd.DataFrame(
        {
            "presentation": np.arange(len(idx)),
            "stim_index": idx,
            "stim_name": [names[i] for i in idx],
            "onset": ts,
        }
    )


# ------------------------------------------------------------------------- trials
def read_trials(f: h5py.File) -> pd.DataFrame:
    grp = f["intervals/trials"]
    cols = {}
    for k in grp.keys():
        ds = grp[k]
        if isinstance(ds, h5py.Dataset) and ds.ndim == 1:
            cols[k] = ds[:]
    df = pd.DataFrame(cols)
    if "id" in df:
        df = df.rename(columns={"id": "trial"})
    return df


# ---------------------------------------------------------------------- electrodes
def read_electrodes(f: h5py.File) -> pd.DataFrame:
    grp = f["general/extracellular_ephys/electrodes"]
    df = pd.DataFrame(
        {
            "electrode": grp["id"][:],
            "location": [_decode(x) for x in grp["location"][:]],
            "group_name": [_decode(x) for x in grp["group_name"][:]],
            "x": grp["x"][:],
            "y": grp["y"][:],
            "z": grp["z"][:],
            "orig_channel": grp["origChannel"][:],
        }
    )
    df["area"] = df["location"].map(area_from_location)
    df["hemisphere"] = df["location"].map(hemisphere_from_location)
    return df


_AREA_MAP = {
    "amygdala": "amygdala",
    "hippocampus": "hippocampus",
    "dorsal_anterior_cingulate_cortex": "dACC",
    "pre_supplementary_motor_area": "preSMA",
    "ventral_medial_prefrontal_cortex": "vmPFC",
}


def area_from_location(loc: str) -> str:
    loc = loc.lower()
    for k, v in _AREA_MAP.items():
        if loc.startswith(k):
            return v
    return "other"


def hemisphere_from_location(loc: str) -> str:
    loc = loc.lower()
    if loc.endswith("_left"):
        return "L"
    if loc.endswith("_right"):
        return "R"
    return "?"


def region_of(area: str) -> str:
    return "MTL" if area in ("amygdala", "hippocampus") else "MFC"


# -------------------------------------------------------------------------- units
@dataclass
class Unit:
    unit: int
    electrode: int
    location: str
    area: str
    hemisphere: str
    spike_times: np.ndarray
    isolation_distance: float = np.nan
    mean_proj_dist: float = np.nan
    mean_snr: float = np.nan
    peak_snr: float = np.nan
    cluster_id_orig: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def region(self) -> str:
        return region_of(self.area)

    @property
    def n_spikes(self) -> int:
        return int(self.spike_times.size)


def read_units(f: h5py.File, electrodes: pd.DataFrame | None = None) -> list[Unit]:
    """Read every unit's spike times and quality metrics (waveforms are skipped)."""
    if electrodes is None:
        electrodes = read_electrodes(f)
    grp = f["units"]
    ids = grp["id"][:]
    st_index = grp["spike_times_index"][:]
    st_all = grp["spike_times"][:]
    elec_idx = grp["electrodes"][:]  # row index into electrodes table
    starts = np.concatenate([[0], st_index[:-1]])

    def _col(name):
        return grp[name][:] if name in grp else np.full(len(ids), np.nan)

    iso, mpd, msnr, psnr = (
        _col("waveforms_isolation_distance"),
        _col("waveforms_mean_proj_dist"),
        _col("waveforms_mean_snr"),
        _col("waveforms_peak_snr"),
    )
    cid = grp["clusterID_orig"][:] if "clusterID_orig" in grp else np.full(len(ids), -1)

    units = []
    for i, uid in enumerate(ids):
        e = electrodes.iloc[int(elec_idx[i])]
        units.append(
            Unit(
                unit=int(uid),
                electrode=int(e["electrode"]),
                location=str(e["location"]),
                area=str(e["area"]),
                hemisphere=str(e["hemisphere"]),
                spike_times=st_all[int(starts[i]) : int(st_index[i])].astype(float),
                isolation_distance=float(np.ravel(iso)[i]),
                mean_proj_dist=float(np.ravel(mpd)[i]),
                mean_snr=float(np.ravel(msnr)[i]),
                peak_snr=float(np.ravel(psnr)[i]),
                cluster_id_orig=int(np.ravel(cid)[i]),
            )
        )
    return units


# ------------------------------------------------------------------------- events
def read_events(f: h5py.File) -> pd.DataFrame:
    grp = f["acquisition/events"]
    return pd.DataFrame({"ttl": grp["data"][:], "time": grp["timestamps"][:]})
