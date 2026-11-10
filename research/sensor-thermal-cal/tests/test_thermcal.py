"""Run with: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from thermcal import conformal, evaluate, io, models, simulate  # noqa: E402


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cov = [np.mean(np.abs(rng.normal(size=400)) <= conformal.quantile(np.abs(rng.normal(size=400)), 0.1)) for _ in range(300)]
    assert 0.89 < np.mean(cov) < 0.915


def test_weighted_conformal_fixes_a_known_shift_and_refuses_extrapolation():
    """Noise grows with T. Calibration is mostly cool, test mostly warm: plain conformal under-covers,
    weighted conformal recovers; a test point far outside calibration gets an infinite interval."""
    rng = np.random.default_rng(1)
    T_cal = rng.normal(20, 4, 4000); T_te = rng.normal(27, 3, 4000)
    s_cal = np.abs(rng.normal(0, 1 + 0.15 * np.clip(T_cal - 10, 0, None)))
    s_te = np.abs(rng.normal(0, 1 + 0.15 * np.clip(T_te - 10, 0, None)))
    plain = np.mean(s_te <= conformal.quantile(s_cal, 0.1))
    w_cal, w_te = conformal.density_ratio(T_cal, T_te)
    q = conformal.weighted_halfwidths(s_cal, w_cal, w_te, 0.1)
    fin = np.isfinite(q)
    assert plain < 0.87
    assert fin.mean() > 0.9 and abs(np.mean(s_te[fin] <= q[fin]) - 0.9) < 0.03
    _, w_far = conformal.density_ratio(T_cal, np.append(T_te, 80.0))
    assert np.isinf(conformal.weighted_halfwidths(s_cal, w_cal, w_far, 0.1)[-1])


def test_unit_conversions():
    assert abs(io.temp_c(0) - 36.53) < 1e-9 and abs(io.temp_c(-521) - 34.9976) < 1e-3


def test_models_recover_a_quadratic_drift():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"T": rng.uniform(0, 45, 600), "dTdt": rng.normal(0, 0.3, 600)})
    y = 0.02 * (df["T"] - 25) + 3e-4 * (df["T"] - 25) ** 2 + 0.05 * df["dTdt"]
    assert np.max(np.abs(models.QuadraticRate().fit(df, y).predict(df) - y)) < 1e-9
    assert np.sqrt(np.mean((models.Poly(2).fit(df, y).predict(df) - y) ** 2)) < 0.03


def test_pipeline_runs_end_to_end_on_fake_data(tmp_path):
    simulate.make(tmp_path, n_units=1, hours=1.5, seed=3)
    df = io.load_runs(tmp_path)
    assert df["simulated"].all() and set(io.CHANNELS) <= set(df.columns)
    r = evaluate.evaluate(df[df.run.isin(sorted(df.run.unique()))])
    assert {"inside"} <= set(r.region) and r.cov_split.between(0, 1).all()
