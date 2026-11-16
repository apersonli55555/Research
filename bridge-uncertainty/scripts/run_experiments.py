"""Run every experiment and write tables to results/.

Usage (from the project root):  python scripts/run_experiments.py
Deterministic: all randomness is seeded. Runtime: a few minutes on a laptop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kw51uq import conformal, data, detection, metrics, models  # noqa: E402

MODES = [3, 5, 6, 9, 13]          # modes with <15% missing frequencies before retrofit
LEVELS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]  # miscoverage alpha values
MAIN_ALPHA = 0.1                  # headline: 90% intervals
T_BINS = [-np.inf, 0, 5, 10, 15, 20, np.inf]

# Chronological split, all BEFORE the retrofit (Oct 2018 has no steel temperature)
TRAIN = ("2018-11-01", "2019-01-31")
CAL = ("2019-02-01", "2019-02-28")
TEST = ("2019-03-01", "2019-05-14")
POST = ("2019-09-28", "2020-01-15")


def window(df, a, b):
    return df.loc[a:b]


def usable(df, mode):
    cols = [f"f{mode}", "T_steel"]
    return df.dropna(subset=cols)


def random_split(df, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(len(df))
    return df[u < 0.5], df[(u >= 0.5) & (u < 0.7)], df[u >= 0.7]


def evaluate(model, tr, ca, te, mode, split, seed, rows, extra=None):
    y = f"f{mode}"
    model.fit(tr, tr[y].to_numpy())
    pc, pt = model.predict(ca), model.predict(te)
    for a in LEVELS:
        lo, hi, q = conformal.split_conformal(ca[y], pc, pt, a)
        rows.append(dict(mode=mode, split=split, seed=seed, model=model.name, method="split conformal",
                         alpha=a, coverage=metrics.coverage(te[y], lo, hi),
                         width_Hz=metrics.mean_width(lo, hi),
                         interval_score=metrics.interval_score(te[y], lo, hi, a),
                         rmse_Hz=metrics.rmse(te[y], pt), n_test=len(te)))
    if isinstance(model, models.GP):
        mu, sd = model.predict(te, return_std=True)
        for a in LEVELS:
            z = norm.ppf(1 - a / 2)
            lo, hi = mu - z * sd, mu + z * sd
            rows.append(dict(mode=mode, split=split, seed=seed, model=model.name, method="GP Bayesian",
                             alpha=a, coverage=metrics.coverage(te[y], lo, hi),
                             width_Hz=metrics.mean_width(lo, hi),
                             interval_score=metrics.interval_score(te[y], lo, hi, a),
                             rmse_Hz=metrics.rmse(te[y], mu), n_test=len(te)))
    return pc, pt


def main():
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    df = data.add_features(data.load(data.download(ROOT / "data")))

    cov_rows, cond_rows, det_rows, phys_rows, aci_rows = [], [], [], [], []
    preds = {}  # saved for figures

    for mode in MODES:
        y = f"f{mode}"
        d = usable(df, mode)
        tr, ca, te = window(d, *TRAIN), window(d, *CAL), window(d, *TEST)
        post = window(d, *POST)
        print(f"mode {mode}: train {len(tr)}, cal {len(ca)}, test {len(te)}, post {len(post)}")

        # --- Experiment A: random (exchangeable) split, 5 seeds -------------
        pre = window(d, TRAIN[0], TEST[1])
        for seed in range(5):
            rtr, rca, rte = random_split(pre, seed)
            for m in models.all_models(seed):
                evaluate(m, rtr, rca, rte, mode, "random", seed, cov_rows)

        # --- Experiment B: chronological (seasonal extrapolation) split -----
        for m in models.all_models(0):
            pc, pt = evaluate(m, tr, ca, te, mode, "chronological", 0, cov_rows)
            lo, hi, q = conformal.split_conformal(ca[y], pc, pt, MAIN_ALPHA)
            hits = pd.Series((te[y] >= lo) & (te[y] <= hi), index=te.index)
            blo, bhi = metrics.block_bootstrap_ci(hits)  # day-block bootstrap 95% CI
            # conditional coverage by temperature bin
            bins = pd.cut(te["T_steel"], T_BINS)
            for b, h in hits.groupby(bins, observed=True):
                clo, chi = metrics.block_bootstrap_ci(h)
                cond_rows.append(dict(mode=mode, model=m.name, T_bin=str(b), n=len(h),
                                      n_days=h.index.floor("D").nunique(),
                                      coverage=h.mean(), ci_lo=clo, ci_hi=chi))
            cond_rows.append(dict(mode=mode, model=m.name, T_bin="ALL", n=len(hits),
                                  coverage=hits.mean(), ci_lo=blo, ci_hi=bhi))
            # ACI (coverage only)
            alo, ahi = conformal.aci(ca[y], pc, te[y].to_numpy(), pt, MAIN_ALPHA)
            aci_rows.append(dict(mode=mode, model=m.name,
                                 coverage_split=hits.mean(),
                                 coverage_aci=metrics.coverage(te[y], alo, ahi),
                                 width_split=2 * q, width_aci=metrics.mean_width(alo, ahi)))
            # --- Experiment C: retrofit detection (frozen split-conformal band)
            r_te = te[y] - pt
            r_post = post[y] - m.predict(post)
            d_te, d_post = detection.daily_stat(r_te, q), detection.daily_stat(r_post, q)
            det_rows.append(dict(mode=mode, model=m.name,
                                 false_alarm_rate_pre=detection.alarm_rate(d_te),
                                 detection_rate_post=detection.alarm_rate(d_post),
                                 n_days_pre=len(d_te), n_days_post=len(d_post),
                                 median_shift_post_Hz=float(np.median(r_post)),
                                 median_shift_pre_test_Hz=float(np.median(r_te)),
                                 half_width_q_Hz=q))
            if isinstance(m, models.PhysicsOnly):
                phys_rows.append(dict(mode=mode, model=m.name, **m.params(),
                                      df_over_f_per_degC=m.params()["a_per_degC"] / 2))
            preds[(mode, m.name)] = dict(
                te=pd.DataFrame({"y": te[y], "yhat": pt, "T": te["T_steel"]}),
                post=pd.DataFrame({"y": post[y], "yhat": m.predict(post), "T": post["T_steel"]}),
                q=q)
        preds[(mode, "train")] = tr[[y, "T_steel"]].rename(columns={y: "y", "T_steel": "T"})
        preds[(mode, "cal")] = ca[[y, "T_steel"]].rename(columns={y: "y", "T_steel": "T"})

    # --- PCA output-only baseline for detection (all modes jointly) ---------
    F = df[[f"f{m}" for m in MODES]].dropna()
    ftr, fca, fte, fpo = (window(F, *w) for w in (TRAIN, CAL, TEST, POST))
    for k in (1, 2):
        p = detection.PCABaseline(k).fit(ftr.to_numpy())
        q = conformal.conformal_quantile(p.score(fca.to_numpy()), MAIN_ALPHA)
        s_te = pd.Series(p.score(fte.to_numpy()), index=fte.index)
        s_po = pd.Series(p.score(fpo.to_numpy()), index=fpo.index)
        d_te, d_po = detection.daily_stat(s_te, q), detection.daily_stat(s_po, q)
        det_rows.append(dict(mode="all", model=f"PCA residual (k={k}, no T)",
                             false_alarm_rate_pre=detection.alarm_rate(d_te),
                             detection_rate_post=detection.alarm_rate(d_po),
                             n_days_pre=len(d_te), n_days_post=len(d_po)))

    pd.DataFrame(cov_rows).to_csv(out / "coverage_all.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(out / "coverage_by_temperature.csv", index=False)
    pd.DataFrame(det_rows).to_csv(out / "detection.csv", index=False)
    pd.DataFrame(phys_rows).to_csv(out / "physics_parameters.csv", index=False)
    pd.DataFrame(aci_rows).to_csv(out / "aci.csv", index=False)
    pd.to_pickle(preds, out / "predictions.pkl")
    json.dump(dict(modes=MODES, train=TRAIN, cal=CAL, test=TEST, post=POST,
                   retrofit=[str(data.RETROFIT_START.date()), str(data.RETROFIT_END.date())],
                   main_alpha=MAIN_ALPHA), open(out / "config.json", "w"), indent=2)
    print("done ->", out)


if __name__ == "__main__":
    main()
