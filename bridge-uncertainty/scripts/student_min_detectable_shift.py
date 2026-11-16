"""YOUR EXPERIMENT: minimum detectable shift (MDS).

QUESTION: How small a change in a natural frequency can each model detect,
using its calibrated band, without too many false alarms?

WHY: The real 2019 retrofit was so large (+2 % on mode 6) that every model
caught it, so it can't tell the models apart. A deliberately small,
CLEARLY-LABELLED synthetic shift can. This is standard practice ("damage
injection"). You must always say in the paper that the shifts are synthetic.

HOW IT WORKS
  1. Fit each model on the main split (train Nov-Jan, calibrate Feb).
  2. Take the REAL pre-retrofit test data (Mar 1 - May 14).
  3. From a chosen date onward, multiply the measured frequency by (1 - delta),
     i.e. a stiffness loss. delta = 0.1 %, 0.25 %, 0.5 %, 1 %, 2 %.
  4. Compute the daily alarm statistic (already written: detection.daily_stat).
  5. Detection rate = fraction of SHIFTED days that alarm.
     False-alarm rate = fraction of UNSHIFTED days that alarm.
  6. Plot detection rate vs delta, one line per model.

Run:  python scripts/student_min_detectable_shift.py
The parts marked TODO are yours. Everything else already works.
Tip: start with ONE mode (6) and TWO models, get it working, then expand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import conformal, data, detection, models  # noqa: E402

TRAIN = ("2018-11-01", "2019-01-31")
CAL = ("2019-02-01", "2019-02-28")
TEST = ("2019-03-01", "2019-05-14")
SHIFT_START = "2019-04-01"          # shift begins here; Mar 1-31 stays unshifted
DELTAS = [0.001, 0.0025, 0.005, 0.01, 0.02]   # fractional frequency drops
ALPHA = 0.1
MODES = [6]                          # TODO (later): try [3, 5, 6, 9, 13]


def fitted_model_and_band(model, d, y):
    """Already done for you: fit on TRAIN, calibrate on CAL, return model and half-width q."""
    tr, ca = d.loc[TRAIN[0]:TRAIN[1]], d.loc[CAL[0]:CAL[1]]
    model.fit(tr, tr[y].to_numpy())
    q = conformal.conformal_quantile(np.abs(ca[y].to_numpy() - model.predict(ca)), ALPHA)
    return model, q


def inject_shift(y_test: pd.Series, delta: float, start: str) -> pd.Series:
    """TODO 1: return a COPY of y_test where every value at or after `start`
    is multiplied by (1 - delta). Values before `start` stay unchanged.
    Hint: y = y_test.copy(); y.loc[start:] = y.loc[start:] * (1 - delta)
    Think: why multiply instead of subtracting a fixed number of Hz?
    """
    raise NotImplementedError("TODO 1: write inject_shift")


def detection_and_false_alarm(daily: pd.Series, start: str) -> tuple[float, float]:
    """TODO 2: `daily` is the daily statistic D_d (one value per day).
    An alarm on day d means |D_d| > 1.
    Return (detection_rate, false_alarm_rate):
      detection_rate   = fraction of days ON/AFTER `start` with an alarm
      false_alarm_rate = fraction of days BEFORE `start` with an alarm
    Hint: alarm = daily.abs() > 1; split using daily.index >= pd.Timestamp(start)
    """
    raise NotImplementedError("TODO 2: write detection_and_false_alarm")


def main():
    df = data.add_features(data.load(data.download(ROOT / "data")))
    rows = []
    for mode in MODES:
        y = f"f{mode}"
        d = df.dropna(subset=[y, "T_steel"])
        te = d.loc[TEST[0]:TEST[1]]
        # TODO 3 (later): add more models from models.all_models()
        for model in [models.BlackBoxGBM(0), models.PhysicsHybrid(0)]:
            model, q = fitted_model_and_band(model, d, y)
            yhat = model.predict(te)
            for delta in DELTAS:
                y_shifted = inject_shift(te[y], delta, SHIFT_START)
                resid = pd.Series(y_shifted.to_numpy() - yhat, index=te.index)
                daily = detection.daily_stat(resid, q)
                det, fa = detection_and_false_alarm(daily, SHIFT_START)
                rows.append(dict(mode=mode, model=model.name, delta_pct=100 * delta,
                                 detection_rate=det, false_alarm_rate=fa, q_Hz=q))
                print(f"mode {mode} {model.name:28s} delta={100*delta:5.2f}%  detect={det:.2f}  false alarm={fa:.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "student_min_detectable_shift.csv", index=False)

    # TODO 4: make a figure: x = delta_pct, y = detection_rate, one line per model.
    #   Save it to figures/fig9_min_detectable_shift.png. Copy the style from
    #   scripts/make_figures.py (look at fig_calibration for a line plot).
    # TODO 5: write in your notebook, in your own words:
    #   - the smallest delta each model detects on >= 80 % of shifted days
    #   - how that compares with the real retrofit (+2 % on mode 6)
    #   - WHY a narrower band (smaller q) detects smaller shifts, and why that only
    #     matters if the band's coverage is also honest (link to your main result!)


if __name__ == "__main__":
    main()
