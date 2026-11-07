"""Robustness study: does the main finding survive other train/calibrate/test splits?

Five folds, each a contiguous train / calibration / test block:
  F1 forward   train Nov-Dec 2018, cal Jan,  test Feb-Mar 2019
  F2 forward   train Nov-Jan,      cal Feb,  test Mar-May 14   (the main split)
  F3 forward   train Nov-Feb,      cal Mar,  test Apr-May 14
  B1 backward  train Apr-May 14,   cal Mar,  test Nov 2018-Feb 2019  (warm -> cold; includes
               sub-zero hours, so this is the first OUT-OF-SAMPLE test of the cold term)
  Q1 post-retrofit, warm -> cold: train 28 Sep-31 Oct 2019, cal Nov, test Dec 2019-15 Jan 2020
               (a different structural state of the same bridge)
"Backward" folds are not forecasts; they are a legitimate test of how a model
extrapolates to temperatures outside its training range, which is the property
that matters for conformal coverage.

For every test hour we record whether its temperature lies INSIDE or OUTSIDE the
range of training temperatures, and report coverage separately for the two.
Usage: python scripts/run_robustness.py   (~10 min on 4 cores)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import conformal, data, metrics, models  # noqa: E402

MODES = [3, 5, 6, 9, 13]
ALPHA = 0.1
FOLDS = {
    "F1 fwd": (("2018-11-01", "2018-12-31"), ("2019-01-01", "2019-01-31"), ("2019-02-01", "2019-03-31")),
    "F2 fwd (main)": (("2018-11-01", "2019-01-31"), ("2019-02-01", "2019-02-28"), ("2019-03-01", "2019-05-14")),
    "F3 fwd": (("2018-11-01", "2019-02-28"), ("2019-03-01", "2019-03-31"), ("2019-04-01", "2019-05-14")),
    "B1 warm→cold": (("2019-04-01", "2019-05-14"), ("2019-03-01", "2019-03-31"), ("2018-11-01", "2019-02-28")),
    "Q1 post, warm→cold": (("2019-09-28", "2019-10-31"), ("2019-11-01", "2019-11-30"), ("2019-12-01", "2020-01-15")),
}


def main():
    df = data.add_features(data.load(data.download(ROOT / "data")))
    rows, cold_rows = [], []
    for fold, (trw, caw, tew) in FOLDS.items():
        for mode in MODES:
            y = f"f{mode}"
            d = df.dropna(subset=[y, "T_steel"])
            tr, ca, te = d.loc[trw[0]:trw[1]], d.loc[caw[0]:caw[1]], d.loc[tew[0]:tew[1]]
            tmin, tmax = tr["T_steel"].min(), tr["T_steel"].max()
            outside = (te["T_steel"] < tmin) | (te["T_steel"] > tmax)
            print(f"{fold:20s} mode {mode:2d}: train T [{tmin:5.1f},{tmax:5.1f}]  "
                  f"n_tr={len(tr)} n_cal={len(ca)} n_te={len(te)} outside={outside.mean():.2f}")
            for m in models.all_models(0):
                m.fit(tr, tr[y].to_numpy())
                lo, hi, q = conformal.split_conformal(ca[y], m.predict(ca), m.predict(te), ALPHA)
                hit = pd.Series((te[y] >= lo) & (te[y] <= hi), index=te.index)
                for region, mask in [("all", np.ones(len(te), bool)), ("inside", ~outside.to_numpy()),
                                     ("outside", outside.to_numpy())]:
                    h = hit[mask]
                    if len(h) == 0:
                        continue
                    clo, chi = metrics.block_bootstrap_ci(h)
                    rows.append(dict(fold=fold, mode=mode, model=m.name, region=region, n=len(h),
                                     n_days=h.index.floor("D").nunique(), coverage=h.mean(),
                                     ci_lo=clo, ci_hi=chi, width_Hz=2 * q,
                                     interval_score=metrics.interval_score(
                                         te[y].to_numpy()[mask], lo[mask], hi[mask], ALPHA),
                                     train_Tmin=tmin, train_Tmax=tmax))
                if fold == "B1 warm→cold":   # coverage below 0 degC specifically
                    cm = (te["T_steel"] < 0).to_numpy()
                    if cm.any():
                        cold_rows.append(dict(mode=mode, model=m.name, n=int(cm.sum()),
                                              n_days=hit[cm].index.floor("D").nunique(),
                                              coverage_below_0C=hit[cm].mean()))
    out = ROOT / "results"
    pd.DataFrame(rows).to_csv(out / "robustness_folds.csv", index=False)
    pd.DataFrame(cold_rows).to_csv(out / "robustness_subzero.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
