"""Split conformal prediction, plain and weighted for covariate shift.

Plain (Angelopoulos & Bates 2021, arXiv:2107.07511): half-width = the
ceil((n+1)(1-alpha))-th smallest calibration error. Valid if calibration and
test data are exchangeable.

Weighted (Tibshirani, Barber, Candes & Ramdas 2019, arXiv:1904.06019): if test
temperatures are distributed differently from calibration temperatures, weight
each calibration error by w(T) = p_test(T) / p_cal(T). Valid if w is known and
finite. For a temperature never seen in calibration, w is effectively infinite
and the method (correctly) returns an infinitely wide interval: it refuses to
extrapolate. The fraction of such refusals is reported.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def quantile(scores, alpha: float) -> float:
    s = np.sort(np.asarray(scores)); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(s[k - 1])


def density_ratio(T_cal, T_test, support_frac: float = 1e-3):
    """w(T) = p_test(T)/p_cal(T) by kernel density estimates.
    Returns (w at calibration points, w at test points). A test temperature where the
    calibration density is below `support_frac` of its peak has no calibration data
    nearby: its weight is set to infinity, which makes its interval infinite."""
    T_cal, T_test = np.asarray(T_cal, float), np.asarray(T_test, float)
    pc, pt = gaussian_kde(T_cal), gaussian_kde(T_test)
    pc_cal, pc_te = pc(T_cal), pc(T_test)
    floor = support_frac * pc_cal.max()
    w_cal = pt(T_cal) / np.maximum(pc_cal, floor)
    w_te = np.where(pc_te < floor, np.inf, pt(T_test) / np.maximum(pc_te, floor))
    return w_cal, w_te


def weighted_halfwidths(scores, w_cal, w_test, alpha: float):
    """Half-width for every test point; inf where the method cannot support the test point."""
    order = np.argsort(scores)
    s, w = np.asarray(scores)[order], np.asarray(w_cal)[order]
    w_test = np.asarray(w_test, float)
    cum, W = np.cumsum(w), np.sum(w)
    finite = np.isfinite(w_test)
    need = (1 - alpha) * (W + np.where(finite, w_test, 0.0))   # weight that must lie at or below the quantile
    idx = np.searchsorted(cum, need, side="left")
    q = np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)
    return np.where(finite, q, np.inf)
