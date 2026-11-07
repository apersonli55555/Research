"""Evaluation metrics for interval predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(y, lo, hi) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def mean_width(lo, hi) -> float:
    w = np.asarray(hi) - np.asarray(lo)
    return float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else np.inf


def interval_score(y, lo, hi, alpha: float) -> float:
    """Winkler / interval score (Gneiting & Raftery 2007): width plus a penalty
    of 2/alpha times the miss distance. Lower is better; it rewards intervals
    that are narrow AND cover."""
    y, lo, hi = map(np.asarray, (y, lo, hi))
    s = (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(s))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def block_bootstrap_ci(hits: pd.Series, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """CI for coverage that respects autocorrelation: resample whole DAYS.
    `hits` is a boolean Series indexed by timestamp."""
    rng = np.random.default_rng(seed)
    daily = hits.groupby(hits.index.floor("D")).agg(["sum", "count"])
    s, c = daily["sum"].to_numpy(), daily["count"].to_numpy()
    idx = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(boots, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
