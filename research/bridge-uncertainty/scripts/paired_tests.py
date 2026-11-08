"""Paired comparison of coverage: proposed model vs every other model.

For a set of test hours, delta = coverage(proposed) - coverage(other), computed
on EXACTLY the same hours (paired). Uncertainty: bootstrap that resamples
whole CALENDAR DAYS with replacement; all rows sharing a day (all modes, all
overlapping windows) are kept together, so within-day and cross-mode
dependence is preserved. We report the 95% percentile CI and the bootstrap
fraction of delta <= 0 (an approximate one-sided p-value). With 8 comparisons
per setting, a Holm correction is applied to those p-values.

Usage: python scripts/paired_tests.py   (needs robustness_hits.csv.gz, rolling_hits.csv.gz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROPOSED = "Physics + GBM (proposed)"
N_BOOT = 4000


def paired(h: pd.DataFrame, keys: list[str], seed: int = 0) -> pd.DataFrame:
    """h: rows (keys..., time, model, hit). Returns one row per comparator."""
    wide = h.pivot_table(index=keys + ["time"], columns="model", values="hit", aggfunc="first")
    wide = wide.dropna()
    day = pd.to_datetime(wide.index.get_level_values("time")).floor("D")
    days, inv = np.unique(day, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(N_BOOT, len(days)))
    counts = np.stack([np.bincount(d, minlength=len(days)) for d in draws])  # N_BOOT x n_days
    n_day = np.bincount(inv, minlength=len(days)).astype(float)
    P = wide[PROPOSED].to_numpy(float)
    out = []
    for other in [c for c in wide.columns if c != PROPOSED]:
        dlt = P - wide[other].to_numpy(float)
        s_day = np.bincount(inv, weights=dlt, minlength=len(days))
        boot = (counts @ s_day) / (counts @ n_day)
        out.append(dict(comparator=other, n_hours=len(dlt), n_days=len(days),
                        cov_proposed=P.mean(), cov_other=wide[other].mean(),
                        delta=dlt.mean(), ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
                        p_one_sided=float(np.mean(boot <= 0))))
    df = pd.DataFrame(out).sort_values("p_one_sided")
    # Holm step-down correction
    m = len(df)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(df.p_one_sided)])
    df["p_holm"] = np.minimum(adj, 1.0)
    return df.sort_values("delta")


def main():
    rows = []
    rob = pd.read_csv(RES / "robustness_hits.csv.gz", parse_dates=["time"])
    settings = {
        "Folds F2+F3, warm-side out-of-range": rob[rob.fold.isin(["F2 fwd (main)", "F3 fwd"]) & rob.outside],
        # (F2/F3 test periods never go below their training minimum, so "outside" = warmer)
        "Fold B1, cold-side out-of-range": rob[(rob.fold == "B1 warm→cold") & rob.outside],
        "Fold Q1 (post-retrofit), out-of-range": rob[(rob.fold == "Q1 post, warm→cold") & rob.outside],
        "All 5 folds, inside training range": rob[~rob.outside],
    }
    for name, h in settings.items():
        r = paired(h, ["fold", "mode"]); r.insert(0, "setting", name); rows.append(r)
    roll_path = RES / "rolling_hits.csv.gz"
    if roll_path.exists():
        rl = pd.read_csv(roll_path, parse_dates=["time"])
        for name, h in {
            "Rolling forward windows, warm-side out-of-range": rl[(rl.direction == "forward") & (rl.side == "warm")],
            "Rolling backward windows, cold-side out-of-range": rl[(rl.direction == "backward") & (rl.side == "cold")],
            "Rolling (all windows), inside training range": rl[rl.side == "inside"],
        }.items():
            if len(h):
                r = paired(h, ["window", "mode"]); r.insert(0, "setting", name); rows.append(r)
    out = pd.concat(rows)
    out.to_csv(RES / "paired_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
