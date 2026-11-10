"""Load recordings made by tools/log_serial.py (or the simulator) into physical units.

Targets (all have a known true value while the board sits still):
  gx, gy, gz [deg/s]  true value 0 (Earth's rotation, 0.004 deg/s, is below the noise)
  anorm [mg]          |acceleration| - 1000 mg; the true magnitude is local gravity,
                      a constant, whatever the tilt of the board
Single accelerometer axes are NOT used: moving the box between fridge and table
changes its tilt, and 1 degree of tilt shifts a horizontal axis by 17 mg.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ACC_LSB_PER_G, GYRO_LSB_PER_DPS = 16384.0, 131.0
CHANNELS = ["gx", "gy", "gz", "anorm"]
UNITS_OF = {"gx": "deg/s", "gy": "deg/s", "gz": "deg/s", "anorm": "mg"}


def temp_c(raw):
    return raw / 340.0 + 36.53          # MPU6050 datasheet


def load_runs(data_dir, block_s: int = 10, warmup_s: int = 600, max_gyro_std_dps: float = 0.5) -> pd.DataFrame:
    """One row per `block_s` seconds per run and unit.
    - drops the first `warmup_s` seconds of each run (self-heating after power-on)
    - drops seconds where the gyro noise is high (someone moved or bumped the box)
    - averages into blocks, which also reduces autocorrelation
    """
    frames = []
    for csv in sorted(Path(data_dir).glob("*.csv")):
        meta_p = csv.with_suffix(".json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        d = pd.read_csv(csv)
        if d.empty:
            continue
        d["simulated"] = bool(meta.get("simulated", False))
        for unit, u in d.groupby("unit"):
            u = u.copy()
            u["t_s"] = (u["ms"] - u["ms"].iloc[0]) / 1000.0
            u["T"] = temp_c(u["temp"])
            for a in "xyz":
                u[f"g{a}"] = u[f"g{a}"] / GYRO_LSB_PER_DPS
            u["anorm"] = np.sqrt(u.ax ** 2 + u.ay ** 2 + u.az ** 2) / ACC_LSB_PER_G * 1000.0 - 1000.0
            still = (u[["sgx", "sgy", "sgz"]].max(axis=1) / GYRO_LSB_PER_DPS) < max_gyro_std_dps
            u = u[(u.t_s >= warmup_s) & still]
            if len(u) < 10 * block_s:
                continue
            u["block"] = (u.t_s // block_s).astype(int)
            b = u.groupby("block").agg(t_s=("t_s", "mean"), T=("T", "mean"), gx=("gx", "mean"), gy=("gy", "mean"),
                                       gz=("gz", "mean"), anorm=("anorm", "mean"), n=("T", "size")).reset_index(drop=True)
            b = b[b.n >= 0.8 * block_s]
            # heating/cooling rate [degC/min], from a 2-minute centred difference
            k = max(1, int(60 / block_s))
            b["dTdt"] = (b["T"].shift(-k) - b["T"].shift(k)) / ((b["t_s"].shift(-k) - b["t_s"].shift(k)) / 60.0)
            b = b.dropna(subset=["dTdt"])
            b["run"], b["unit"], b["orientation"] = u["run"].iloc[0], unit, u["orientation"].iloc[0]
            b["simulated"] = u["simulated"].iloc[0]
            frames.append(b)
    if not frames:
        raise SystemExit(f"No usable recordings found in {data_dir}")
    return pd.concat(frames, ignore_index=True)
