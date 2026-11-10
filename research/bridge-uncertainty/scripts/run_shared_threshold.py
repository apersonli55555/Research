"""Two follow-ups from docs/PROFESSOR_REVIEW.md.

(A) Shared freezing threshold. Fit modes 3, 5, 6, 9, 13 together with ONE
    common T_c (each mode keeps its own f0, a, b). Compare the day-block
    bootstrap CI of the shared T_c with the per-mode CIs.
    Data: main-split training period (1 Nov 2018 - 31 Jan 2019), as before.
(B) Temperature slope by mode type. The dataset authors classify the modes in
    their plotting code (matlab-functions/plot_trackedmodes.m):
      arch 1,2,4,7,8,14 | lateral deck 3,11 | global vertical 5,6,10,12,13 | torsional 9
    Fit the physics curve to every mode with enough data on the whole
    pre-retrofit period and report the warm-side slope with a bootstrap CI.
    Mode 11 (lateral) was never analysed before: it is an out-of-sample test
    of "lateral modes stiffen as they warm".

Outputs: results/shared_threshold.csv, results/mode_type_slopes.csv
Usage:   python scripts/run_shared_threshold.py   (~3 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import data, models  # noqa: E402

MODES = [3, 5, 6, 9, 13]
TYPE = {**{m: "arch" for m in (1, 2, 4, 7, 8, 14)}, 3: "lateral", 11: "lateral",
        **{m: "vertical" for m in (5, 6, 10, 12, 13)}, 9: "torsional"}
N_BOOT = 300


def fit_shared(parts):
    """parts: list of (T, y) per mode. theta = [Tc, (f0, a, b) per mode]."""
    scales = [np.std(y) for _, y in parts]
    x0, lo, hi = [1.0], [-5.0], [8.0]
    for T, y in parts:
        f0 = np.median(y)
        x0 += [f0, -5e-4, 1e-3]; lo += [0.5 * f0, -0.02, 0.0]; hi += [1.5 * f0, 0.02, 0.1]

    def resid(th):
        return np.concatenate([(models.physics_curve([th[1 + 3 * k], th[2 + 3 * k], th[3 + 3 * k], th[0]], T) - y) / sc
                               for k, ((T, y), sc) in enumerate(zip(parts, scales))])
    return least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0).x


def day_groups(index):
    days = index.floor("D")
    ud = np.unique(days)
    return ud, {d: np.where(days == d)[0] for d in ud}


def main():
    df = data.add_features(data.load(data.download(ROOT / "data")))
    rng = np.random.default_rng(0)
    out = ROOT / "results"

    # ---------------- (A) shared threshold
    tr = df.loc["2018-11-01":"2019-01-31"].dropna(subset=["T_steel"])
    ud, groups = day_groups(tr.index)

    def parts_of(frame):
        return [(frame.dropna(subset=[f"f{m}"])["T_steel"].to_numpy(), frame[f"f{m}"].dropna().to_numpy()) for m in MODES]

    point = fit_shared(parts_of(tr))
    sep_point = {m: models.PhysicsOnly().fit(tr.dropna(subset=[f"f{m}"]), tr[f"f{m}"].dropna().to_numpy()).theta[3] for m in MODES}
    shared_b, sep_b = [], {m: [] for m in MODES}
    for _ in range(N_BOOT):
        b = tr.iloc[np.concatenate([groups[d] for d in rng.choice(ud, len(ud), replace=True)])]
        shared_b.append(fit_shared(parts_of(b))[0])
        for m in MODES:
            bm = b.dropna(subset=[f"f{m}"])
            sep_b[m].append(models.PhysicsOnly().fit(bm, bm[f"f{m}"].to_numpy()).theta[3])
    rows = [dict(fit="shared (5 modes)", Tc=point[0], ci_lo=np.quantile(shared_b, .025), ci_hi=np.quantile(shared_b, .975))]
    rows += [dict(fit=f"separate, mode {m}", Tc=sep_point[m], ci_lo=np.quantile(sep_b[m], .025),
                  ci_hi=np.quantile(sep_b[m], .975)) for m in MODES]
    A = pd.DataFrame(rows); A["ci_width"] = A.ci_hi - A.ci_lo
    A.to_csv(out / "shared_threshold.csv", index=False)
    print(A.round(2).to_string(index=False))

    # ---------------- (B) slope by mode type
    pre = df.loc["2018-11-01":"2019-05-14"].dropna(subset=["T_steel"])
    rows = []
    for m in range(1, 15):
        d = pre.dropna(subset=[f"f{m}"])
        if len(d) < 800:
            rows.append(dict(mode=m, type=TYPE[m], n_hours=len(d))); continue
        est = models.PhysicsOnly().fit(d, d[f"f{m}"].to_numpy()).theta
        ud_m, g_m = day_groups(d.index)
        bs = []
        for _ in range(200):
            b = d.iloc[np.concatenate([g_m[x] for x in rng.choice(ud_m, len(ud_m), replace=True)])]
            bs.append(models.PhysicsOnly().fit(b, b[f"f{m}"].to_numpy()).theta[1] / 2)
        # Sturdier cross-check that needs no freezing term: straight line through hours above 5 degC
        w = d[d["T_steel"] > 5]
        f0w = float(np.median(w[f"f{m}"]))
        lin = lambda fr: np.polyfit(fr["T_steel"].to_numpy(), fr[f"f{m}"].to_numpy(), 1)[0] / f0w  # noqa: E731
        udw, gw = day_groups(w.index)
        bl = [lin(w.iloc[np.concatenate([gw[x] for x in rng.choice(udw, len(udw), replace=True)])]) for _ in range(500)]
        rows.append(dict(mode=m, type=TYPE[m], n_hours=len(d), f_Hz=float(np.median(d[f"f{m}"])),
                         slope_per_degC=est[1] / 2, ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975),
                         Tc=est[3], warm_linear_slope=lin(w), warm_ci_lo=np.quantile(bl, .025),
                         warm_ci_hi=np.quantile(bl, .975), n_warm_hours=len(w)))
    B = pd.DataFrame(rows).sort_values(["type", "mode"])
    B.to_csv(out / "mode_type_slopes.csv", index=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.3g}"):
        print(B.to_string(index=False))


if __name__ == "__main__":
    main()
