"""Sanity tests: run with  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kw51uq import conformal, metrics, models  # noqa: E402


def test_split_conformal_covers_on_iid_data():
    """With exchangeable (iid) data, split conformal must reach >= 1-alpha on average."""
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(200):
        y_cal, y_te = rng.standard_t(3, 500), rng.standard_t(3, 500)
        lo, hi, _ = conformal.split_conformal(y_cal, np.zeros(500), np.zeros(500), 0.1)
        covs.append(metrics.coverage(y_te, lo, hi))
    assert 0.895 < np.mean(covs) < 0.915


def test_quantile_infinite_when_too_few_points():
    assert conformal.conformal_quantile(np.arange(5.0), 0.01) == np.inf


def test_physics_curve_decreases_with_temperature_above_tc():
    th = [3.0, -3e-4, 3e-3, 1.0]
    f = models.physics_curve(th, np.array([5.0, 15.0, 25.0]))
    assert np.all(np.diff(f) < 0)


def test_physics_curve_cold_stiffening():
    th = [3.0, -3e-4, 3e-3, 1.0]
    assert models.physics_curve(th, np.array([-3.0]))[0] > models.physics_curve(th, np.array([5.0]))[0]


def test_aci_reaches_target_under_shift():
    """ACI should recover ~1-alpha long-run coverage even when test noise grows."""
    rng = np.random.default_rng(1)
    y_cal = rng.normal(0, 1, 1000)
    y_te = rng.normal(0, 2, 5000)  # variance doubled: split conformal under-covers
    lo, hi = conformal.aci(y_cal, np.zeros(1000), y_te, np.zeros(5000), 0.1, gamma=0.01)
    assert abs(metrics.coverage(y_te, lo, hi) - 0.9) < 0.02


def test_all_models_fit_and_predict_on_synthetic_data():
    """Smoke test: every model fits and predicts finite values on a toy dataset."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 400
    T = rng.uniform(-3, 25, n)
    X = pd.DataFrame({"T_steel": T, "T_mean24": T + rng.normal(0, 1, n), "dT_3h": rng.normal(0, 1, n),
                      "RH_air": rng.uniform(40, 100, n), "T_air": T + rng.normal(0, 2, n),
                      "hour_sin": rng.uniform(-1, 1, n), "hour_cos": rng.uniform(-1, 1, n)})
    y = models.physics_curve([3.0, -3e-4, 3e-3, 1.0], T) + rng.normal(0, 0.003, n)
    for m in models.all_models(0):
        p = m.fit(X, y).predict(X)
        assert np.all(np.isfinite(p)), m.name
        assert metrics.rmse(y, p) < 0.02, m.name
