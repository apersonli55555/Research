"""Calibration models: predict a sensor's error from its own temperature reading.

Each model has fit(df, y) and predict(df). `df` has columns T [degC] and dTdt [degC/min].
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

T_REF = 25.0


class Poly:
    """Polynomial in temperature. Degree 1-3 polynomials are the standard way
    low-cost inertial sensors are temperature-compensated."""

    def __init__(self, degree: int):
        self.degree = degree
        self.name = {1: "Linear", 2: "Quadratic", 3: "Cubic"}[degree]

    def _X(self, df):
        d = df["T"].to_numpy() - T_REF
        return np.column_stack([d ** k for k in range(self.degree + 1)])

    def fit(self, df, y):
        self.beta, *_ = np.linalg.lstsq(self._X(df), np.asarray(y), rcond=None)
        return self

    def predict(self, df):
        return self._X(df) @ self.beta


class QuadraticRate(Poly):
    """Quadratic in temperature plus a term proportional to the heating rate.
    Motivation: the thermometer and the sensing element sit at slightly different
    places on the chip, so while the temperature is changing they disagree by an
    amount roughly proportional to dT/dt (thermal lag, which shows up as hysteresis)."""

    def __init__(self):
        super().__init__(2)
        self.name = "Quadratic + rate"

    def _X(self, df):
        return np.column_stack([super()._X(df), df["dTdt"].to_numpy()])


class GBM:
    """Gradient-boosted trees on (T, dT/dt): flexible, but predicts a constant
    outside the temperatures it was trained on."""
    name = "Boosted trees"

    def fit(self, df, y):
        self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                               l2_regularization=1.0, random_state=0).fit(df[["T", "dTdt"]], y)
        return self

    def predict(self, df):
        return self.m.predict(df[["T", "dTdt"]])


def all_models():
    return [Poly(1), Poly(2), Poly(3), QuadraticRate(), GBM()]
