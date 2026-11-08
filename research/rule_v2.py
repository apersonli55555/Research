#!/usr/bin/env python3
"""
Proposed detector v2: impact -> settle -> hold, with an optional posture-change gate.

    1. impact:  |a| > T_PEAK
    2. settle:  within SETTLE_S seconds of the impact a still period begins
                (|a| within TOL of the calibrated baseline)
    3. hold:    that still period lasts HOLD_S seconds without interruption
    4. posture: the gravity direction during the hold differs from the direction
                in the second before impact by more than ANGLE_DEG degrees
                (orientation-agnostic: compares the device with itself, so it
                works whatever way the phone sits in a pocket)

Everything is implementable with DeviceMotionEvent.accelerationIncludingGravity.

Protocol (no test-set peeking):
  * DEV subjects : SA01-SA12, SE01-SE05, SE07   -> grid search
  * TEST subjects: SA13-SA23, SE06, SE08-SE15  -> reported once
  Selection criterion on DEV: maximise sensitivity subject to specificity >= 0.95.

Usage: python3 research/rule_v2.py /path/to/SisFall_dataset
"""
import itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from benchmark_sisfall import iter_dataset, cluster_boot, G  # noqa: E402

FS = 50
HOLD_S = 5.0  # same 5 s confirmation the app already uses
DEV = {f"SA{i:02d}" for i in range(1, 13)} | {"SE01", "SE02", "SE03", "SE04", "SE05", "SE07"}


def load_vec(path):
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            p = line.strip().rstrip(";").split(",")
            if len(p) >= 3 and p[0].strip():
                rows.append([float(v) for v in p[:3]])
    a = np.asarray(rows) * (32.0 / 8192.0) * G
    k = 200 // FS
    n = (len(a) // k) * k
    return a[:n].reshape(-1, k, 3).mean(1)


def run_lengths(mask):
    """r[i] = number of consecutive True starting at i."""
    m = np.asarray(mask, int)
    n = len(m)
    # index of the next False at or after i
    idx = np.where(m == 0, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]
    return nxt - np.arange(n)


_CACHE = {}


def detect(acc, T, settle_s, tol, angle, calib=2 * FS):
    key = (id(acc), tol)
    if key not in _CACHE:
        mag = np.linalg.norm(acc, axis=1)
        base = mag[:calib].mean()
        _CACHE[key] = (mag, run_lengths(np.abs(mag - base) <= tol))
    mag, rl = _CACHE[key]
    hold = int(HOLD_S * FS)
    settle = int(settle_s * FS)
    peaks = np.where(mag[calib:] > T)[0] + calib
    i_last = -1
    for p in peaks:
        if p <= i_last:
            continue
        seg = rl[p:p + settle + 1]
        ok = np.where(seg >= hold)[0]
        if not len(ok):
            continue
        s = p + ok[0]
        if angle > 0:
            pre = acc[max(0, p - FS):max(1, p - FS // 5)].mean(0)
            dur = acc[s:s + hold].mean(0)
            c = np.dot(pre, dur) / (np.linalg.norm(pre) * np.linalg.norm(dur) + 1e-9)
            if np.degrees(np.arccos(np.clip(c, -1, 1))) < angle:
                i_last = s + hold
                continue
        return True
    return False


def evaluate(trials, params):
    per_f, per_a, per_act = {}, {}, {}
    for t in trials:
        hit = detect(t["acc"], *params)
        d = per_f if t["fall"] else per_a
        k, n = d.get(t["subj"], (0, 0))
        d[t["subj"]] = (k + (hit if t["fall"] else (not hit)), n + 1)
        k, n = per_act.get(t["act"], (0, 0))
        per_act[t["act"]] = (k + hit, n + 1)
    return per_f, per_a, per_act


def pooled(d):
    k = sum(v[0] for v in d.values()); n = sum(v[1] for v in d.values())
    return k / n if n else float("nan")


def main():
    root = sys.argv[1]
    trials = [dict(id=os.path.basename(p)[:-4], act=a, subj=s, fall=a[0] == "F", acc=load_vec(p))
              for a, s, r, p in iter_dataset(root)]
    trials = [t for t in trials if len(t["acc"]) > 3 * FS]
    dev = [t for t in trials if t["subj"] in DEV]
    test = [t for t in trials if t["subj"] not in DEV]
    print(f"dev {len(dev)} trials / test {len(test)} trials", file=sys.stderr)

    grid = list(itertools.product([16, 20, 24, 28, 32],   # T_PEAK m/s^2
                                  [1.0, 2.0, 3.0],         # SETTLE_S
                                  [1.0, 1.5, 2.0, 3.0],    # TOL m/s^2
                                  [0, 30, 45, 60]))        # ANGLE_DEG (0 = gate off)
    rows = []
    for g in grid:
        f, a, _ = evaluate(dev, g)
        rows.append((g, pooled(f), pooled(a)))
    feas = [r for r in rows if r[2] >= 0.95]
    best = max(feas, key=lambda r: (r[1], r[2]))
    nogate = [r for r in rows if r[0][3] == 0]
    nogate_feas = [r for r in nogate if r[2] >= 0.95]
    # magnitude-only rules may never reach 95% spec; fall back to Youden's J on dev
    best_nogate = (max(nogate_feas, key=lambda r: (r[1], r[2])) if nogate_feas
                   else max(nogate, key=lambda r: r[1] + r[2]))
    max_nogate_spec = max(r[2] for r in nogate)
    print("DEV best:", best, "\nDEV best w/o posture gate:", best_nogate, file=sys.stderr)

    out = {"protocol": {"dev_subjects": sorted(DEV), "criterion": "max sens s.t. spec>=0.95 on dev",
                        "grid_size": len(grid), "fs_hz": FS, "hold_s": HOLD_S,
                        "no_posture_reaches_spec95_on_dev": bool(nogate_feas),
                        "no_posture_max_dev_spec": max_nogate_spec}}
    # the shipped rule's analogue in this framework for reference: T=24, settle=0, tol=1.5, no gate
    for name, g in [("v2_posture", best[0]), ("v2_no_posture", best_nogate[0])]:
        f, a, act = evaluate(test, g)
        out[name] = dict(params=dict(T_PEAK=g[0], SETTLE_S=g[1], TOL=g[2], ANGLE_DEG=g[3]),
                         dev_sens=[r for r in rows if r[0] == g][0][1],
                         dev_spec=[r for r in rows if r[0] == g][0][2],
                         test_sensitivity=cluster_boot(f), test_specificity=cluster_boot(a),
                         test_elderly_adl_specificity=cluster_boot({k: v for k, v in a.items() if k.startswith("SE")}),
                         test_per_activity_positive_rate={k: v[0] / v[1] for k, v in sorted(act.items())})
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    # per-trial decisions of the selected rule, for checking the JavaScript port against
    g = best[0]
    with open(os.path.join(os.path.dirname(__file__), "results", "rule_v2_trials.json"), "w") as fh:
        json.dump({t["id"]: bool(detect(t["acc"], *g)) for t in trials}, fh)
    with open(os.path.join(os.path.dirname(__file__), "results", "rule_v2.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for name in ("v2_posture", "v2_no_posture"):
        r = out[name]
        s, sp, se = r["test_sensitivity"], r["test_specificity"], r["test_elderly_adl_specificity"]
        print(f"{name:14s} {r['params']}\n  TEST sens {s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}]  spec {sp[0]:.3f} [{sp[1]:.3f},{sp[2]:.3f}]"
              f"  elderly-ADL spec {se[0]:.3f} [{se[1]:.3f},{se[2]:.3f}]")


if __name__ == "__main__":
    main()
