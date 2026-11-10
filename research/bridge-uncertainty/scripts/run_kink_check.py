"""Reviewer check: thermo-elastic form vs a plain two-regime ('kink') line.

sqrt(1 + a*dT) is almost linear for the tiny fitted a (~5e-4 per degC), so the
proposed model's trend is close to a straight line plus a kink. This script
asks whether a physics-free model of the same SHAPE (models.KinkGBM) matches
it, on all 22 rolling windows, with the same paired day-block bootstrap.

Output: results/kink_check_hits.csv.gz, results/kink_check.csv
Usage:  python scripts/run_kink_check.py   (~3 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from kw51uq import conformal, data, models  # noqa: E402
from paired_tests import paired  # noqa: E402
from run_rolling import ALPHA, MODES, sl, windows  # noqa: E402


def main():
    df = data.add_features(data.load(data.download(ROOT / "data")))
    hits, par = [], []
    for wid, direction, trw, caw, tew in windows():
        for mode in MODES:
            y = f"f{mode}"
            d = df.dropna(subset=[y, "T_steel"])
            tr, ca, te = sl(d, trw), sl(d, caw), sl(d, tew)
            if min(len(tr), len(ca), len(te)) < 100:
                continue
            side = np.where(te["T_steel"] > tr["T_steel"].max(), "warm",
                            np.where(te["T_steel"] < tr["T_steel"].min(), "cold", "inside"))
            for m in (models.PhysicsHybrid(0), models.KinkGBM(0)):
                m.fit(tr, tr[y].to_numpy())
                lo, hi, q = conformal.split_conformal(ca[y], m.predict(ca), m.predict(te), ALPHA)
                hits.append(pd.DataFrame({"window": wid, "direction": direction, "mode": mode, "model": m.name,
                                          "time": te.index, "side": side, "width_Hz": 2 * q,
                                          "hit": ((te[y] >= lo) & (te[y] <= hi)).to_numpy()}))
                # warm-side slope implied by each trend, as fractional change per degC at 15 degC
                if isinstance(m, models.KinkGBM):
                    slope = m.theta[1] / models.kink_curve(m.theta, np.array([15.0]))[0]
                else:
                    slope = m.theta[1] / 2 / (1 + m.theta[1] * 5)
                par.append(dict(window=wid, direction=direction, mode=mode, model=m.name,
                                slope_per_degC=slope, Tc=m.theta[3]))
    h = pd.concat(hits)
    h.to_csv(ROOT / "results" / "kink_check_hits.csv.gz", index=False)
    rows = []
    for name, sub in {"Forward windows, warm-side out-of-range": h[(h.direction == "forward") & (h.side == "warm")],
                      "Backward windows, cold-side out-of-range": h[(h.direction == "backward") & (h.side == "cold")],
                      "All windows, inside training range": h[h.side == "inside"]}.items():
        r = paired(sub, ["window", "mode"]); r.insert(0, "setting", name); rows.append(r)
    out = pd.concat(rows).drop(columns=["p_holm"])
    out.to_csv(ROOT / "results" / "kink_check.csv", index=False)
    pd.DataFrame(par).to_csv(ROOT / "results" / "kink_check_params.csv", index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))
        p = pd.DataFrame(par)
        print(p[p.direction == "forward"].groupby(["mode", "model"])[["slope_per_degC", "Tc"]].median().to_string())


if __name__ == "__main__":
    main()
