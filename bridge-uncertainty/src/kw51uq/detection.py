"""Day-level change detection from residuals of a temperature-compensated model.

Daily statistic D_d = median over the day's hours of r_t / q, where r_t is the
residual y - f_hat and q is the split-conformal half-width at level 1-alpha
fitted on the calibration period. An alarm is raised on day d if |D_d| > 1,
i.e. the TYPICAL hour of that day lies outside the calibrated band. Using the
daily median (not single hours) reduces the effect of hourly autocorrelation
and of occasional mis-tracked modes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def daily_stat(resid: pd.Series, q: float, min_hours: int = 12) -> pd.Series:
    g = (resid / q).groupby(resid.index.floor("D"))
    d = g.median()
    return d[g.count() >= min_hours]


def alarm_rate(daily: pd.Series) -> float:
    return float(np.mean(np.abs(daily) > 1)) if len(daily) else np.nan


class PCABaseline:
    """Output-only baseline in the spirit of PCA-based environmental removal
    (as used on KW51 by Maes et al. 2022 -- this is a simplified
    re-implementation, not their exact method). Standardize the multi-mode
    frequency vector, treat the first k principal components as 'environment',
    and use the norm of what is left as a novelty index. No temperature input."""

    def __init__(self, k: int = 1):
        self.k = k

    def fit(self, F: np.ndarray):
        self.sc = StandardScaler().fit(F)
        self.p = PCA(self.k).fit(self.sc.transform(F))
        return self

    def score(self, F: np.ndarray) -> np.ndarray:
        Z = self.sc.transform(F)
        R = Z - self.p.inverse_transform(self.p.transform(Z))
        return np.sqrt((R ** 2).sum(1))
