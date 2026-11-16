"""Two reviewer checks.

(A) Hyperparameter sensitivity. Was the proposed model quietly tuned? We vary
    its settings AND give the strongest physics-free competitor (GBM) the same
    sweep, then re-score both on the 11 forward rolling windows (warm-side
    out-of-range hours and inside-range hours, all 5 modes).
      GBM part (both models): max_depth in {2, 3, 5} x learning_rate in {0.02, 0.05, 0.1}
      physics part (proposed only): softplus width s in {0.5, 1, 2} degC
(B) Uncertainty of the fitted physics parameters (a, Tc, b) on the main-split
    training data, by a day-block bootstrap (resample whole days, refit).

Outputs: results/sensitivity.csv, results/physics_parameter_ci.csv
Usage:   python scripts/run_sensitivity.py   (~10-20 min)
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from kw51uq import conformal, data, models  # noqa: E402
from run_rolling import MODES, sl, windows  # noqa: E402

ALPHA = 0.1


def configs():
    for depth, lr in itertools.product([2, 3, 5], [0.02, 0.05, 0.1]):
        yield "GBM (no physics)", dict(max_depth=depth, learning_rate=lr), \
            lambda d=depth, l=lr: models.BlackBoxGBM(0, max_depth=d, learning_rate=l)
        yield "Physics + GBM (proposed)", dict(max_depth=depth, learning_rate=lr, s=1.0), \
            lambda d=depth, l=lr: models.PhysicsHybrid(0, max_depth=d, learning_rate=l)
    for s in [0.5, 2.0]:
        yield "Physics + GBM (proposed)", dict(max_depth=3, learning_rate=0.05, s=s), \
            lambda s=s: models.PhysicsHybrid(0, s=s)


def sensitivity(df):
    rows = []
    for wid, direction, trw, caw, tew in windows():
        if direction != "forward":
            continue
        for mode in MODES:
            y = f"f{mode}"
            d = df.dropna(subset=[y, "T_steel"])
            tr, ca, te = sl(d, trw), sl(d, caw), sl(d, tew)
            if min(len(tr), len(ca), len(te)) < 100:
                continue
            side = np.where(te["T_steel"] > tr["T_steel"].max(), "warm",
                            np.where(te["T_steel"] < tr["T_steel"].min(), "cold", "inside"))
            for name, cfg, make in configs():
                m = make().fit(tr, tr[y].to_numpy())
                lo, hi, q = conformal.split_conformal(ca[y], m.predict(ca), m.predict(te), ALPHA)
                hit = ((te[y] >= lo) & (te[y] <= hi)).to_numpy()
                for region in ("warm", "inside"):
                    mk = side == region
                    if mk.sum():
                        rows.append(dict(window=wid, mode=mode, model=name, **cfg, region=region,
                                         n=int(mk.sum()), hits=int(hit[mk].sum()), width_Hz=2 * q))
        print(wid, "done", flush=True)
    r = pd.DataFrame(rows)
    keys = ["model", "max_depth", "learning_rate", "s", "region"]
    summ = r.groupby(keys, dropna=False).agg(n=("n", "sum"), hits=("hits", "sum"),
                                             median_width_Hz=("width_Hz", "median")).reset_index()
    summ["pooled_coverage"] = summ.hits / summ.n
    return summ


def physics_ci(df, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mode in MODES:
        y = f"f{mode}"
        tr = df.dropna(subset=[y, "T_steel"]).loc["2018-11-01":"2019-01-31"]
        days = tr.index.floor("D")
        ud = np.unique(days)
        groups = {d: np.where(days == d)[0] for d in ud}
        point = models.PhysicsOnly().fit(tr, tr[y].to_numpy()).theta
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(ud, len(ud), replace=True)
            idx = np.concatenate([groups[d] for d in pick])
            b = tr.iloc[idx]
            boots.append(models.PhysicsOnly().fit(b, b[y].to_numpy()).theta)
        B = np.array(boots)
        for j, name in [(1, "a_per_degC"), (2, "b"), (3, "Tc_degC")]:
            rows.append(dict(mode=mode, parameter=name, estimate=point[j],
                             ci_lo=np.quantile(B[:, j], 0.025), ci_hi=np.quantile(B[:, j], 0.975)))
        half = B[:, 1] / 2
        rows.append(dict(mode=mode, parameter="df_over_f_per_degC (= a/2)", estimate=point[1] / 2,
                         ci_lo=np.quantile(half, 0.025), ci_hi=np.quantile(half, 0.975)))
    return pd.DataFrame(rows)


def main():
    df = data.add_features(data.load(data.download(ROOT / "data")))
    out = ROOT / "results"
    physics_ci(df).to_csv(out / "physics_parameter_ci.csv", index=False)
    print("physics CIs done", flush=True)
    sensitivity(df).to_csv(out / "sensitivity.csv", index=False)
    print("finished")


if __name__ == "__main__":
    main()
