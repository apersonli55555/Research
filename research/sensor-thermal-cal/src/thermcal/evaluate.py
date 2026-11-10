"""Leave-one-run-out evaluation of calibration models and their intervals.

For each unit and channel, and for each held-out TEST run:
  The other runs are the DEVELOPMENT runs. Calibration errors are collected by
  cross-validation over runs: each development run in turn is predicted by a
  model fitted to the remaining development runs, and all those errors are
  pooled. (One calibration run is not enough: every power-on has its own slow
  drift, so the errors must be sampled across several runs.) The final model is
  then fitted to all development runs and scored on the test run.
  "restricted" design: development data use only temperatures inside
      [t_lo, t_hi] (a room-temperature calibration, as is common in practice);
  "full" design: development data use every temperature (the reference).
  The test run is scored at ALL its temperatures, split into
      inside [t_lo, t_hi], cold (< t_lo) and warm (> t_hi).
Whole runs are held out because rows within a run are strongly correlated.
A per-run constant offset (turn-on bias) is removed using each run's own
readings inside the reference window, so only the temperature-driven change is modelled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import conformal, models
from .io import CHANNELS


def remove_run_offset(d: pd.DataFrame, ch: str, ref=(20.0, 30.0)) -> pd.Series:
    """y minus the run's median in the reference temperature window (NaN if the run never visits it)."""
    in_ref = d["T"].between(*ref)
    off = d[ch].where(in_ref).groupby(d["run"]).transform("median")
    return d[ch] - off


def evaluate(df: pd.DataFrame, t_lo: float = 15.0, t_hi: float = 30.0, alpha: float = 0.1) -> pd.DataFrame:
    rows = []
    for unit, du in df.groupby("unit"):
        runs = sorted(du["run"].unique())
        if len(runs) < 3:
            raise SystemExit(f"Unit {unit}: need at least 3 runs, found {len(runs)}")
        for ch in CHANNELS:
            d = du.copy(); d["y"] = remove_run_offset(d, ch); d = d.dropna(subset=["y"])
            runs_ok = sorted(d["run"].unique())
            for test_run in runs_ok:
                te = d[d.run == test_run]
                region = np.where(te["T"] < t_lo, "cold", np.where(te["T"] > t_hi, "warm", "inside"))
                for design in ("restricted", "full"):
                    keep = d["T"].between(t_lo, t_hi) if design == "restricted" else pd.Series(True, index=d.index)
                    dev = d[(d.run != test_run) & keep]
                    dev_runs = [r for r in sorted(dev["run"].unique()) if (dev.run == r).sum() >= 30]
                    if len(dev_runs) < 3:
                        continue
                    dev = dev[dev.run.isin(dev_runs)]
                    for m in models.all_models():
                        s_parts, T_parts = [], []
                        for j in dev_runs:                      # cross-validation over runs
                            fit_j, out_j = dev[dev.run != j], dev[dev.run == j]
                            m.fit(fit_j, fit_j["y"].to_numpy())
                            s_parts.append(np.abs(out_j["y"].to_numpy() - m.predict(out_j)))
                            T_parts.append(out_j["T"].to_numpy())
                        s_cal, T_cal = np.concatenate(s_parts), np.concatenate(T_parts)
                        m.fit(dev, dev["y"].to_numpy())
                        res = te["y"].to_numpy() - m.predict(te)
                        q = conformal.quantile(s_cal, alpha)
                        w_cal, w_te = conformal.density_ratio(T_cal, te["T"].to_numpy())
                        qw = conformal.weighted_halfwidths(s_cal, w_cal, w_te, alpha)
                        for reg in ("inside", "cold", "warm"):
                            mk = region == reg
                            if mk.sum() < 20:
                                continue
                            fin = np.isfinite(qw[mk])
                            rows.append(dict(
                                unit=unit, channel=ch, test_run=test_run, design=design, model=m.name, region=reg,
                                n=int(mk.sum()), raw_rms=float(np.sqrt(np.mean(te["y"].to_numpy()[mk] ** 2))),
                                resid_rms=float(np.sqrt(np.mean(res[mk] ** 2))),
                                cov_split=float(np.mean(np.abs(res[mk]) <= q)), halfwidth_split=q,
                                cov_weighted_given_finite=float(np.mean(np.abs(res[mk][fin]) <= qw[mk][fin])) if fin.any() else np.nan,
                                weighted_refused=float(1 - fin.mean())))
    return pd.DataFrame(rows)
