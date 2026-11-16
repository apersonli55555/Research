"""Systematic rolling-origin evaluation (replaces hand-picked folds).

Windows: TRAIN 8 weeks, CALIBRATE 3 weeks, TEST 4 weeks, contiguous, stepped
by 1 week across the pre-retrofit period (20 Nov 2018 - 14 May 2019; steel
temperature is missing before ~20 Nov). Two directions:
  forward : train -> cal -> test   (later in time; mostly warm-side extrapolation)
  backward: test  <- cal <- train  (earlier in time; mostly cold-side extrapolation)
Every window position is used; nothing is chosen by hand.

Output: results/rolling_cells.csv (one row per window x mode x model x region)
        results/rolling_hits.csv.gz (per-hour hits, for paired tests)
Usage:  python scripts/run_rolling.py              (all windows, one process)
        python scripts/run_rolling.py SHARD NSHARDS  (run 1/NSHARDS of the windows;
        set OMP_NUM_THREADS=1 and run NSHARDS copies in parallel, then
        python scripts/run_rolling.py merge NSHARDS)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import conformal, data, models  # noqa: E402

MODES = [3, 5, 6, 9, 13]
ALPHA = 0.1
START, END = pd.Timestamp("2018-11-20"), pd.Timestamp("2019-05-15")  # END exclusive
W_TR, W_CA, W_TE = pd.Timedelta(weeks=8), pd.Timedelta(weeks=3), pd.Timedelta(weeks=4)
STEP = pd.Timedelta(weeks=1)


def windows():
    total = W_TR + W_CA + W_TE
    t0 = START
    k = 0
    while t0 + total <= END:
        a, b, c, d = t0, t0 + W_TR, t0 + W_TR + W_CA, t0 + total
        yield f"fwd{k:02d}", "forward", (a, b), (b, c), (c, d)
        # backward, same window lengths: test (4 wk) <- cal (3 wk) <- train (8 wk)
        e, f = t0 + W_TE, t0 + W_TE + W_CA
        yield f"bwd{k:02d}", "backward", (f, d), (e, f), (a, e)
        t0 += STEP
        k += 1


def sl(df, w):
    return df[(df.index >= w[0]) & (df.index < w[1])]


def main(shard: int = 0, nshards: int = 1):
    df = data.add_features(data.load(data.download(ROOT / "data")))
    cells, hits = [], []
    for i, (wid, direction, trw, caw, tew) in enumerate(windows()):
        if i % nshards != shard:
            continue
        for mode in MODES:
            y = f"f{mode}"
            d = df.dropna(subset=[y, "T_steel"])
            tr, ca, te = sl(d, trw), sl(d, caw), sl(d, tew)
            if min(len(tr), len(ca), len(te)) < 100:
                continue
            tmin, tmax = tr["T_steel"].min(), tr["T_steel"].max()
            outside = ((te["T_steel"] < tmin) | (te["T_steel"] > tmax)).to_numpy()
            side = np.where(te["T_steel"] > tmax, "warm", np.where(te["T_steel"] < tmin, "cold", "inside"))
            for m in models.all_models(0):
                m.fit(tr, tr[y].to_numpy())
                lo, hi, q = conformal.split_conformal(ca[y], m.predict(ca), m.predict(te), ALPHA)
                hit = ((te[y] >= lo) & (te[y] <= hi)).to_numpy()
                hits.append(pd.DataFrame({"window": wid, "direction": direction, "mode": mode,
                                          "model": m.name, "time": te.index, "side": side, "hit": hit}))
                for region in ("inside", "warm", "cold"):
                    mk = side == region
                    if mk.sum():
                        cells.append(dict(window=wid, direction=direction, mode=mode, model=m.name,
                                          region=region, n=int(mk.sum()), coverage=float(hit[mk].mean()),
                                          width_Hz=2 * q, train_start=str(trw[0].date()),
                                          test_start=str(tew[0].date())))
        print(wid, "done", flush=True)
    out = ROOT / "results"
    suffix = "" if nshards == 1 else f".shard{shard}"
    pd.DataFrame(cells).to_csv(out / f"rolling_cells{suffix}.csv", index=False)
    pd.concat(hits).to_csv(out / f"rolling_hits{suffix}.csv.gz", index=False)
    print("finished")


def merge(nshards: int):
    out = ROOT / "results"
    for name, ext in (("rolling_cells", ".csv"), ("rolling_hits", ".csv.gz")):
        parts = [out / f"{name}.shard{k}{ext}" for k in range(nshards)]
        pd.concat(pd.read_csv(p) for p in parts).to_csv(out / f"{name}{ext}", index=False)
        for p in parts:
            p.unlink()
    print("merged")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "merge":
        merge(int(sys.argv[2]))
    elif len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        main()
