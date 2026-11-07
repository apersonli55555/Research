"""Conformal prediction intervals.

Split conformal (Angelopoulos & Bates 2021, arXiv:2107.07511, Sec. 1-2):
  1. Fit the model on a TRAINING set.
  2. On a separate CALIBRATION set, compute scores s_i = |y_i - f_hat(x_i)|.
  3. q = the ceil((n+1)(1-alpha))/n empirical quantile of the scores.
  4. Interval for a new point: [f_hat(x) - q, f_hat(x) + q].
Guarantee: if calibration and test points are EXCHANGEABLE, P(y in interval)
>= 1 - alpha. Hourly bridge data are autocorrelated and seasonal, so this
assumption is violated -- testing how much coverage degrades is the point.

Adaptive conformal inference (ACI; Gibbs & Candes, NeurIPS 2021,
arXiv:2106.00170): after each observation, update the working miscoverage
level alpha_t <- alpha_t + gamma * (alpha - err_t), where err_t = 1 if the
point fell outside. It restores long-run coverage under drift -- but it will
also "learn away" a real structural change, so it is evaluated for coverage
only, never used for detection.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores)[~np.isnan(scores)])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


def split_conformal(y_cal, yhat_cal, yhat_test, alpha: float):
    q = conformal_quantile(np.abs(np.asarray(y_cal) - np.asarray(yhat_cal)), alpha)
    return yhat_test - q, yhat_test + q, q


def aci(y_cal, yhat_cal, y_test, yhat_test, alpha: float, gamma: float = 0.005):
    """Online ACI over the test sequence (in time order). Calibration scores are
    fixed; only the quantile level adapts. Returns lo, hi arrays."""
    scores = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
    scores = np.sort(scores[~np.isnan(scores)])
    n = len(scores)
    a_t = alpha
    lo = np.empty(len(y_test)); hi = np.empty(len(y_test))
    for t, (y, f) in enumerate(zip(y_test, yhat_test)):
        if a_t <= 0:
            q = np.inf
        elif a_t >= 1:
            q = 0.0
        else:
            k = int(np.ceil((n + 1) * (1 - a_t)))
            q = np.inf if k > n else scores[max(k, 1) - 1]
        lo[t], hi[t] = f - q, f + q
        if not np.isnan(y):
            err = float(not (lo[t] <= y <= hi[t]))
            a_t = a_t + gamma * (alpha - err)
    return lo, hi
