"""FAKE data generator, in the exact file format of tools/log_serial.py.

Purpose: test the analysis code before any real measurement exists.
The numbers are INVENTED (loosely the size of typical MEMS drift). Nothing
produced from these files is a result, and it must never appear in a paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import ACC_LSB_PER_G, GYRO_LSB_PER_DPS

# (start temperature, ambient temperature the box drifts toward) for six runs
PROFILES = [(-15, 23), (4, 23), (50, 23), (-15, 24), (48, 22), (4, 24)]


def _ar1(n, sd, rho, rng):
    e = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n); x = np.empty(n); x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x


def make(out_dir, n_units: int = 2, hours: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = int(hours * 3600)
    t = np.arange(n, dtype=float)
    coef = {}
    for u in range(n_units):
        unit = "AB"[u] if u < 2 else f"U{u}"
        coef[unit] = dict(
            g={a: (rng.normal(0, 0.02), rng.normal(0, 2e-4), rng.normal(0, 3e-6), rng.normal(0, 0.01)) for a in "xyz"},
            g0={a: rng.normal(0, 1.0) for a in "xyz"},
            a=(rng.normal(0.3, 0.1), rng.normal(0, 4e-3), rng.normal(0, 5e-5), rng.normal(0, 0.15)), a0=rng.normal(0, 8))
    for r, (T0, Tamb) in enumerate(PROFILES):
        run = f"SIM_run{r + 1}"
        rows = []
        for unit, c in coef.items():
            tau = rng.uniform(35, 50) * 60
            T = Tamb + (T0 - Tamb) * np.exp(-t / tau) + 1.5 * (1 - np.exp(-t / 300)) + _ar1(n, 0.05, 0.999, rng)
            rate = np.gradient(T) * 60                      # degC/min
            dT = T - 25
            d = dict(ms=(t * 1000).astype(int), addr="0x68" if unit == "A" else "0x69", n=100)
            for a in "xyz":
                c1, c2, c3, h = c["g"][a]
                bias = c["g0"][a] + rng.normal(0, 0.03) + c1 * dT + c2 * dT ** 2 + c3 * dT ** 3 + h * rate
                d[f"g{a}"] = (bias + _ar1(n, 0.004, 0.999, rng) + rng.normal(0, 0.006, n)) * GYRO_LSB_PER_DPS
                d[f"sg{a}"] = np.abs(rng.normal(0.05, 0.005, n)) * GYRO_LSB_PER_DPS
            c1, c2, c3, h = c["a"]
            err_mg = c["a0"] + rng.normal(0, 0.3) + c1 * dT + c2 * dT ** 2 + c3 * dT ** 3 + h * rate \
                + _ar1(n, 0.15, 0.999, rng) + rng.normal(0, 0.25, n)
            tilt = np.deg2rad(rng.uniform(0, 3)); az = (1000 + err_mg) / 1000 * ACC_LSB_PER_G
            d["ax"], d["ay"], d["az"] = az * np.sin(tilt), 0.0 * t, az * np.cos(tilt)
            for a in "xyz":
                d[f"sa{a}"] = np.abs(rng.normal(40, 3, n))
            d["temp"] = (T - 36.53) * 340
            f = pd.DataFrame(d); f["unit"] = unit
            rows.append(f)
        f = pd.concat(rows); f["run"], f["orientation"], f["utc"] = run, "z_up", "SIMULATED"
        cols = ["utc", "run", "unit", "orientation", "ms", "addr", "n", "ax", "ay", "az", "gx", "gy", "gz", "temp",
                "sax", "say", "saz", "sgx", "sgy", "sgz"]
        f[cols].round(2).to_csv(out / f"{run}.csv", index=False)
        (out / f"{run}.json").write_text(json.dumps(dict(run=run, simulated=True,
                                                         note="FAKE DATA from thermcal.simulate"), indent=2))
