"""Download and load the KW51 tracked-modes file (Maes & Lombaert, Zenodo 3745914).

The file holds hourly modal parameters identified by the dataset authors with
stochastic subspace identification. We do NOT re-identify modes from raw
acceleration; we use their published natural frequencies as measurements.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ZENODO_URL = "https://zenodo.org/api/records/3745914/files/trackedmodes.zip/content"
ZENODO_MD5 = "5cdb446519d1150228fab51b3a002239"  # published by Zenodo for trackedmodes.zip

# Period boundaries. Retrofit window 15 May - 27 Sep 2019: confirmed in the dataset authors'
# own plotting code (matlab-functions/plot_trackedmodes.m, Zenodo 3745914).
MONITOR_START = pd.Timestamp("2018-10-02")
RETROFIT_START = pd.Timestamp("2019-05-15")
RETROFIT_END = pd.Timestamp("2019-09-28")  # first day *after* retrofit

ENV_COLUMNS = {
    "tBD31A": "T_steel",   # steel (deck) surface temperature [degC]
    "rhBD31A": "RH_deck",  # relative humidity at deck [%]
    "tVL": "T_air",        # air temperature, nearby weather station [degC]
    "rhVL": "RH_air",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: Path) -> Path:
    """Download trackedmodes.zip (13 MB), verify MD5, extract. Returns .mat path."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zpath = data_dir / "trackedmodes.zip"
    if not zpath.exists():
        print(f"Downloading {ZENODO_URL} ...")
        urllib.request.urlretrieve(ZENODO_URL, zpath)
    got = _md5(zpath)
    if got != ZENODO_MD5:
        raise RuntimeError(f"MD5 mismatch for {zpath}: {got} != {ZENODO_MD5}")
    mat = data_dir / "trackedmodes" / "trackedmodes.mat"
    if not mat.exists():
        with zipfile.ZipFile(zpath) as zf:
            zf.extract("trackedmodes/trackedmodes.mat", data_dir)
    return mat


def _datenum_to_datetime(d: float) -> datetime:
    """MATLAB datenum -> Python datetime (rounded to the minute)."""
    dt = datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366)
    return (pd.Timestamp(dt).round("min")).to_pydatetime()


def load(mat_path: Path) -> pd.DataFrame:
    """Return an hourly DataFrame: columns f1..f14 [Hz], xi1..xi14 [%], env vars, period."""
    M = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["modes"]
    t = pd.DatetimeIndex([_datenum_to_datetime(x) for x in M.sdn], name="time")
    df = pd.DataFrame(index=t)
    for j in range(M.f.shape[1]):
        df[f"f{j + 1}"] = M.f[:, j]
        df[f"xi{j + 1}"] = M.xi[:, j]
    labels = [str(s) for s in M.labels_env]
    for k, name in ENV_COLUMNS.items():
        df[name] = M.env[:, labels.index(k)]
    df["period"] = np.select(
        [df.index < RETROFIT_START, df.index < RETROFIT_END],
        ["pre", "retrofit"], default="post")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Covariates for the data-driven parts. Lagged temperature captures thermal
    inertia (the steel and ballast lag the surface sensor)."""
    out = df.copy()
    T = out["T_steel"]
    out["T_mean24"] = T.rolling("24h", min_periods=12).mean()
    out["dT_3h"] = T - T.shift(freq="3h").reindex(out.index)
    hour = out.index.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
