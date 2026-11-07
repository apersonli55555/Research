"""Point-prediction models f_hat(x) for one mode's natural frequency.

Every model exposes fit(df_train, y) and predict(df) -> np.ndarray.
Uncertainty is added afterwards by conformal prediction (conformal.py), so all
models are compared with the *same* uncertainty wrapper. The GP additionally
exposes its own Bayesian standard deviation for comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

T_REF = 10.0  # reference temperature [degC] for the physics model
RESIDUAL_FEATURES = ["T_steel", "T_mean24", "dT_3h", "RH_deck", "T_air", "hour_sin", "hour_cos"]


class LinearT:
    """Baseline 1: f = a + b*T. The simplest temperature compensation."""
    name = "Linear(T)"

    def fit(self, X: pd.DataFrame, y):
        self.m = LinearRegression().fit(X[["T_steel"]].to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X[["T_steel"]].to_numpy())


class CubicT:
    """Baseline 2: cubic polynomial in T (polynomial regression is a common
    compensation choice in the SHM literature). Can behave badly when
    extrapolating to temperatures not seen in training."""
    name = "Cubic(T)"

    def fit(self, X, y):
        self.m = make_pipeline(StandardScaler(), PolynomialFeatures(3), LinearRegression())
        self.m.fit(X[["T_steel"]].to_numpy(), y)
        return self

    def predict(self, X):
        return self.m.predict(X[["T_steel"]].to_numpy())


class GP:
    """Baseline 3: Gaussian-process regression on (T, 24 h mean T, RH).
    A GP gives its own Bayesian predictive std; we report both its native
    intervals and conformalized ones. Training is subsampled for speed
    (exact GP cost grows as n^3)."""
    name = "GP"
    features = ["T_steel", "T_mean24", "RH_deck"]

    def __init__(self, n_max: int = 1500, seed: int = 0):
        self.n_max, self.seed = n_max, seed

    def fit(self, X, y):
        Z = X[self.features].to_numpy()
        ok = ~np.isnan(Z).any(1)
        Z, y = Z[ok], np.asarray(y)[ok]
        rng = np.random.default_rng(self.seed)
        if len(y) > self.n_max:
            idx = rng.choice(len(y), self.n_max, replace=False)
            Z, y = Z[idx], y[idx]
        self.sx = StandardScaler().fit(Z)
        self.ym, self.ys = y.mean(), y.std()
        k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Z.shape[1])) + WhiteKernel(0.1)
        self.m = GaussianProcessRegressor(k, normalize_y=False, random_state=self.seed)
        self.m.fit(self.sx.transform(Z), (y - self.ym) / self.ys)
        return self

    def predict(self, X, return_std: bool = False):
        Z = X[self.features].to_numpy()
        # fall back to instantaneous T where the 24 h mean / RH is missing
        Z = np.where(np.isnan(Z), X[["T_steel"]].to_numpy(), Z)
        mu, sd = self.m.predict(self.sx.transform(Z), return_std=True)
        mu, sd = mu * self.ys + self.ym, sd * self.ys
        return (mu, sd) if return_std else mu


class BlackBoxGBM:
    """Baseline 4: gradient-boosted trees on the SAME features the hybrid's
    residual model uses, but with no physics. Isolates the value of the physics
    prior. Trees predict a constant outside the training range of T, so they
    cannot extrapolate a temperature trend."""
    name = "GBM (no physics)"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        self.m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                               l2_regularization=1.0, random_state=self.seed)
        self.m.fit(X[RESIDUAL_FEATURES], y)
        return self

    def predict(self, X):
        return self.m.predict(X[RESIDUAL_FEATURES])


def physics_curve(theta, T, s: float = 1.0):
    """Thermo-elastic stiffness model with a cold-stiffening term.

    f(T) = f0 * sqrt(1 + a (T - T_REF)) * (1 + b * softplus_s(Tc - T))

    - sqrt(1 + a dT): natural frequency ~ sqrt(k/m), and k ~ E(T); steel's
      Young's modulus falls roughly linearly with T, so `a` (1/degC) should be
      small and negative. Compare the fitted value with handbook dE/dT / E.
    - softplus term: extra stiffness below a critical temperature Tc (e.g. a
      freezing ballast/bearing mechanism; the exact cause is NOT established by
      this data). softplus_s(z) = s*log(1+exp(z/s)) is a smooth hinge.
    """
    f0, a, b, Tc = theta
    hinge = s * np.logaddexp(0.0, (Tc - T) / s)
    return f0 * np.sqrt(np.clip(1 + a * (T - T_REF), 1e-6, None)) * (1 + b * hinge)


class PhysicsOnly:
    """Proposed model, stage 1 alone: 4 interpretable parameters, fitted by
    nonlinear least squares with a robust (soft-L1) loss."""
    name = "Physics only"

    def fit(self, X, y):
        T, y = X["T_steel"].to_numpy(), np.asarray(y)
        f0 = np.median(y)
        res = least_squares(lambda th: physics_curve(th, T) - y,
                            x0=[f0, -5e-4, 1e-3, 1.0],
                            bounds=([0.5 * f0, -0.02, 0.0, -5.0], [1.5 * f0, 0.02, 0.1, 8.0]),
                            loss="soft_l1", f_scale=np.std(y))
        self.theta = res.x
        return self

    def predict(self, X):
        return physics_curve(self.theta, X["T_steel"].to_numpy())

    def params(self) -> dict:
        f0, a, b, Tc = self.theta
        return {"f0_Hz": f0, "a_per_degC": a, "b": b, "Tc_degC": Tc}


class PhysicsHybrid(PhysicsOnly):
    """Proposed model: physics curve + gradient-boosted residual model.

    f_hat = physics(T) + g(T, T_mean24, dT_3h, RH, ...)
    The physics term carries the temperature trend (and extrapolates it);
    the residual learner only corrects what the physics misses (thermal lag,
    humidity, daily cycles). The residual learner is deliberately small and
    regularized so it cannot override the physics."""
    name = "Physics + GBM (proposed)"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, X, y):
        super().fit(X, y)
        r = np.asarray(y) - super().predict(X)
        self.g = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                               l2_regularization=1.0, random_state=self.seed)
        self.g.fit(X[RESIDUAL_FEATURES], r)
        return self

    def predict(self, X):
        return super().predict(X) + self.g.predict(X[RESIDUAL_FEATURES])


def all_models(seed: int = 0):
    return [LinearT(), CubicT(), GP(seed=seed), BlackBoxGBM(seed), PhysicsOnly(), PhysicsHybrid(seed)]
