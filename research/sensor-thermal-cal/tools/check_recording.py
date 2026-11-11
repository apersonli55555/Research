"""Check a recording before you rely on it.

  python tools/check_recording.py data/raw/test1.csv

Prints one block per sensor with PASS / CHECK next to each test. "CHECK" means
look into it before collecting more data; it does not always mean something is broken.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ACC, GYRO = 16384.0, 131.0


def line(ok: bool, text: str) -> bool:
    print(f"  [{'PASS ' if ok else 'CHECK'}] {text}")
    return ok


def check(path: str) -> bool:
    d = pd.read_csv(path)
    all_ok = True
    print(f"{path}: {len(d)} lines, sensors {sorted(d['unit'].astype(str).unique())}")
    for unit, u in d.groupby("unit"):
        print(f"\nSensor {unit}:")
        dur_min = (u.ms.iloc[-1] - u.ms.iloc[0]) / 60000
        gaps = np.diff(u.ms.to_numpy())
        T = u.temp / 340 + 36.53
        g = np.sqrt(u.ax ** 2 + u.ay ** 2 + u.az ** 2) / ACC
        gyro_mean = u[["gx", "gy", "gz"]].abs().mean().max() / GYRO
        moving = (u[["sgx", "sgy", "sgz"]].max(axis=1) / GYRO > 0.5).mean()
        oks = [
            line(dur_min >= 1, f"length {dur_min:.1f} min"),
            line(np.mean(np.abs(gaps - 1000) <= 20) > 0.99, f"one line per second ({np.mean(np.abs(gaps - 1000) > 20):.1%} of gaps off; largest {gaps.max() / 1000:.1f} s)"),
            line((u.n == 100).mean() > 0.99, f"100 readings per line ({(u.n == 100).mean():.1%} of lines)"),
            line(0.95 < g.median() < 1.05, f"gravity magnitude {g.median():.4f} g (expect close to 1)"),
            line(gyro_mean < 10, f"largest average gyro reading {gyro_mean:.2f} deg/s (expect a few at most when still)"),
            line(moving < 0.02, f"board was still ({moving:.1%} of seconds look like movement or vibration)"),
            line(-45 < T.min() and T.max() < 90 and T.std() > 0, f"chip temperature {T.min():.1f} to {T.max():.1f} degC and changing"),
            line(u[["ax", "ay", "az", "gx", "gy", "gz"]].std().min() > 0, "no channel is stuck at a constant value"),
        ]
        all_ok &= all(oks)
    print("\nOverall:", "all checks passed." if all_ok else "at least one item needs a look (see CHECK lines).")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(0 if check(sys.argv[1]) else 1)
