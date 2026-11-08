#!/usr/bin/env python3
"""
Benchmark the Motion Sense fall detector (index.html) on the public SisFall dataset.

This is a faithful port of the in-browser pipeline:
  * 1.5 s rolling window of |accelerationIncludingGravity| and |rotationRate|
  * computeFeatures()            -> 9 features (sound / endFlat unavailable, see notes)
  * thresholdDetect()            -> peak > 24 m/s^2 AND post-peak variance < 2.5
  * startFallConfirmation()      -> 5 s with |mag - baseline| <= 1.5 m/s^2, else cancel
  * generateSyntheticSamples() + trainMLP() -> the app's bootstrap "AI model" (p > 0.5)

plus a learned reference model (same accel/gyro features, trained on real data,
leave-one-subject-out) to show what training on real falls buys.

Dataset: SisFall (Sucerquia, Lopez & Vargas-Bonilla, Sensors 2017), CC-BY-4.0,
Zenodo record 22212281. Usage:

    python3 research/benchmark_sisfall.py /path/to/SisFall_dataset  [--fs 50]

Writes research/results/*.json and prints a summary table. Deterministic (seeded).
"""
import argparse, json, os, re, sys, time
from collections import defaultdict
import numpy as np

G = 9.80665
# --- constants copied verbatim from index.html ---
WINDOW_MS = 1500
PEAK_THRESHOLD_BASE = 24.0
STILLNESS_THRESHOLD = 2.5
CONFIRM_DURATION_MS = 5000
STILLNESS_TOLERANCE = 1.5
HIDDEN_UNITS = 6
CALIB_MS = 2000

FNAME = re.compile(r"^([DF]\d\d)_(S[AE]\d\d)_R(\d\d)\.txt$")


# ----------------------------------------------------------------------------- data
def load_trial(path, fs_out):
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(";")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            rows.append([float(p) for p in parts[:6]])
    a = np.asarray(rows)
    if len(a) == 0:
        return None
    # SisFall conversions (dataset readme / original paper):
    # ADXL345: +-16 g, 13-bit  -> g = (2*16/2^13) * raw
    # ITG3200: +-2000 deg/s, 16-bit -> deg/s = (2*2000/2^16) * raw
    acc = a[:, 0:3] * (32.0 / 8192.0) * G          # m/s^2, includes gravity
    gyr = a[:, 3:6] * (4000.0 / 65536.0)           # deg/s
    mag = np.linalg.norm(acc, axis=1)
    rot = np.linalg.norm(gyr, axis=1)
    # Resample 200 Hz -> fs_out (phones deliver DeviceMotion at ~50-60 Hz).
    # Block-average as an anti-alias filter, then decimate.
    k = int(round(200 / fs_out))
    n = (len(mag) // k) * k
    mag = mag[:n].reshape(-1, k).mean(1)
    rot = rot[:n].reshape(-1, k).mean(1)
    return mag, rot


def iter_dataset(root):
    for subj in sorted(os.listdir(root)):
        d = os.path.join(root, subj)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            m = FNAME.match(fn)
            if m:
                yield m.group(1), m.group(2), int(m.group(3)), os.path.join(d, fn)


# ------------------------------------------------------------------- feature port
def window_features(mag, rot, L, baseline):
    """Vectorised computeFeatures() for every full window ending at sample i (i >= L-1).
    Returns (n_windows, 9) in extractVector() order. Sound and endFlat are 0."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    W = swv(mag, L)
    R = swv(rot, L)
    peak_idx = W.argmax(1)
    peak = W[np.arange(len(W)), peak_idx]
    var_mag = W.var(1)
    idx = np.arange(L)[None, :]
    after = idx > peak_idx[:, None]
    n_after = after.sum(1)
    safe = np.maximum(n_after, 1)
    am = (W * after).sum(1) / safe
    still = (((W - am[:, None]) ** 2) * after).sum(1) / safe
    still = np.where(n_after >= 3, still, var_mag)
    base_dev = np.abs(peak - baseline)
    peak_rot = R.max(1)
    rot_var = R.var(1)
    z = np.zeros_like(peak)
    return np.stack([peak, var_mag, still, base_dev, peak_rot, rot_var, z, z, z], 1)


def simulate_stream(mag, trig, fs, baseline, rearm=True):
    """Replay the onMotion() state machine. trig[i] = detector fired on window ending at i
    (aligned to sample index). Returns list of alert sample indices."""
    confirm_n = int(round(CONFIRM_DURATION_MS / 1000 * fs))
    alerts, confirming, start = [], False, 0
    i = 0
    n = len(mag)
    while i < n:
        if confirming:
            if abs(mag[i] - baseline) > STILLNESS_TOLERANCE:
                confirming = False
            elif i - start >= confirm_n:
                alerts.append(i)
                confirming = False
                if not rearm:
                    break
                # after an alert the user dismisses the check-in; buffer refills
                i += int(round(WINDOW_MS / 1000 * fs))
                continue
        elif trig[i]:
            confirming, start = True, i
        i += 1
    return alerts


# ---------------------------------------------------------------- MLP (JS port)
def gen_synthetic(rng):
    r = lambda lo, hi: rng.uniform(lo, hi)
    spec = [
        # falls
        (lambda: [r(22,32), r(6,11), r(.2,.6), r(10,18), r(160,280), r(30,60), r(.3,.6), r(.03,.07), 1], 1, 6),
        (lambda: [r(18,26), r(4,8), r(.3,.8), r(7,14), r(100,180), r(20,40), r(.2,.5), r(.02,.06), 1], 1, 5),
        (lambda: [r(20,30), r(5,10), r(.2,.6), r(9,17), r(140,260), r(28,55), r(.03,.12), r(.01,.03), 1], 1, 6),
        (lambda: [r(18,28), r(5,9), r(.3,.7), r(8,15), r(120,220), r(25,48), r(.5,.9), r(.08,.15), 1], 1, 5),
        # normals
        (lambda: [r(10,14), r(.3,1), r(.3,1), r(1,4), r(15,40), r(3,10), r(.05,.15), r(.01,.03), 0], 0, 8),
        (lambda: [r(12,18), r(.8,2), r(.4,1.2), r(2,7), r(30,70), r(8,18), r(.08,.2), r(.01,.03), 0], 0, 6),
        (lambda: [r(10,20), r(.4,1.2), r(.2,.6), r(4,12), r(5,20), r(2,6), r(.03,.14), r(.01,.03), 0], 0, 8),
        (lambda: [r(16,26), r(.6,1.6), r(.2,.7), r(8,16), r(8,25), r(3,9), r(.08,.2), r(.02,.05), 0], 0, 6),
        (lambda: [r(11,17), r(.4,1), r(.3,.9), r(1,5), r(20,45), r(5,12), r(.05,.15), r(.01,.03), 0], 0, 6),
        (lambda: [r(19,25), r(5,9), r(4,9), r(9,15), r(50,90), r(12,25), r(.08,.18), r(.02,.04), 0], 0, 8),
        (lambda: [r(13,19), r(1.5,3), r(1,2.5), r(3,9), r(25,55), r(6,15), r(.1,.25), r(.02,.04), 0], 0, 6),
    ]
    X, y = [], []
    for f, lab, times in spec:
        for _ in range(times):
            X.append(f()); y.append(lab)
    return np.array(X), np.array(y, float)


def train_mlp(X, y, rng, nh=HIDDEN_UNITS, epochs=2000, lr=0.3, lam=0.002):
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1
    Xn = (X - mean) / std
    n, d = Xn.shape
    W1 = rng.uniform(-.5, .5, (d, nh)); b1 = np.zeros(nh)
    W2 = rng.uniform(-.5, .5, nh); b2 = 0.0
    for _ in range(epochs):
        a1 = np.tanh(Xn @ W1 + b1)
        yh = 1 / (1 + np.exp(-(a1 @ W2 + b2)))
        dz2 = yh - y
        gW2 = a1.T @ dz2; gb2 = dz2.sum()
        dz1 = dz2[:, None] * W2[None, :] * (1 - a1 ** 2)
        gW1 = Xn.T @ dz1; gb1 = dz1.sum(0)
        W2 -= lr * (gW2 / n + lam * W2)
        b1 -= lr * (gb1 / n)
        W1 -= lr * (gW1 / n + lam * W1)
        b2 -= lr * (gb2 / n)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, mean=mean, std=std)


def mlp_predict(m, X):
    Xn = (X - m["mean"]) / m["std"]
    a1 = np.tanh(Xn @ m["W1"] + m["b1"])
    return 1 / (1 + np.exp(-(a1 @ m["W2"] + m["b2"])))


# ------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def cluster_boot(per_subject, n_boot=2000, seed=0):
    """per_subject: dict subj -> (k, n). Subject-level bootstrap CI for pooled k/n."""
    rng = np.random.default_rng(seed)
    subs = list(per_subject)
    kn = np.array([per_subject[s] for s in subs], float)
    if kn[:, 1].sum() == 0:
        return (float("nan"),) * 3
    est = kn[:, 0].sum() / kn[:, 1].sum()
    bs = []
    for _ in range(n_boot):
        pick = kn[rng.integers(0, len(subs), len(subs))]
        if pick[:, 1].sum() > 0:
            bs.append(pick[:, 0].sum() / pick[:, 1].sum())
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fs = a.fs
    L = int(round(WINDOW_MS / 1000 * fs))
    calib = int(round(CALIB_MS / 1000 * fs))
    t0 = time.time()

    # Bootstrap MLPs exactly as the app trains them (synthetic-only), several seeds.
    mlps = []
    for s in range(a.seeds):
        rng = np.random.default_rng(1000 + s)
        Xs, ys = gen_synthetic(rng)
        mlps.append(train_mlp(Xs, ys, rng))

    trials = []          # per-trial records
    trial_feats = []     # features at peak window for learned model
    for act, subj, rep, path in iter_dataset(a.root):
        lr_ = load_trial(path, fs)
        if lr_ is None:
            continue
        mag, rot = lr_
        if len(mag) < calib + L + 5:
            continue
        # finishCalibration(): mean |a| over the first 2 s (subject standing/sitting still at start)
        baseline = float(mag[:calib].mean())
        F = window_features(mag, rot, L, baseline)
        trig_rule = np.zeros(len(mag), bool)
        rule = (F[:, 0] > PEAK_THRESHOLD_BASE) & (F[:, 2] < STILLNESS_THRESHOLD)
        trig_rule[L - 1:] = rule
        trig_rule[:calib] = False  # app ignores motion during calibration
        alerts_rule = simulate_stream(mag, trig_rule, fs, baseline)

        # app's bootstrap MLP. endFlat is a phone-orientation flag that a waist sensor
        # cannot reproduce; evaluate both bounds (0 = never flat, 1 = always flat).
        mlp_res = {}
        for flat in (0, 1):
            Ff = F.copy(); Ff[:, 8] = flat
            hits, any_trig = [], []
            for m in mlps:
                p = mlp_predict(m, Ff)
                tr = np.zeros(len(mag), bool); tr[L - 1:] = p > 0.5; tr[:calib] = False
                # app: AI OR rule starts confirmation
                hits.append(len(simulate_stream(mag, tr | trig_rule, fs, baseline)) > 0)
                any_trig.append(bool(tr.any()))
            mlp_res[flat] = (float(np.mean(hits)), float(np.mean(any_trig)))

        # peak-centred window for the learned (LOSO) reference model
        gi = int(np.argmax(mag[calib:])) + calib
        end = min(len(mag) - 1, gi + int(0.75 * fs))
        wi = max(0, end - (L - 1))
        feat_peak = F[min(wi, len(F) - 1), :6]
        # post-impact descriptors (2-4 s after peak): mean |a - baseline| and var
        post = mag[min(gi + 2 * fs, len(mag) - 1): min(gi + 4 * fs, len(mag))]
        post_dev = float(np.abs(post - baseline).mean()) if len(post) else 0.0
        post_var = float(post.var()) if len(post) else 0.0

        dur_s = len(mag) / fs
        trials.append(dict(act=act, subj=subj, rep=rep, fall=act[0] == "F",
                           elderly=subj.startswith("SE"), dur_s=dur_s,
                           rule_alerts=len(alerts_rule),
                           rule_trig=bool(trig_rule.any()),
                           mlp_flat0=mlp_res[0][0], mlp_flat1=mlp_res[1][0],
                           mlp_trig_flat0=mlp_res[0][1], mlp_trig_flat1=mlp_res[1][1],
                           peak=float(mag.max()),
                           post_peak_still=bool(np.all(np.abs(post - baseline) <= STILLNESS_TOLERANCE)) if len(post) else False))
        trial_feats.append(list(feat_peak) + [post_dev, post_var])

    print(f"loaded {len(trials)} trials in {time.time()-t0:.0f}s", file=sys.stderr)
    X = np.array(trial_feats)
    y = np.array([t["fall"] for t in trials], int)
    groups = np.array([t["subj"] for t in trials])

    # ---- learned reference model: LOSO, trial-level ----
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    learned = {}
    for name, mk in [("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
                     ("gbm", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0))]:
        p = np.zeros(len(y))
        for g in np.unique(groups):
            te = groups == g
            m = mk(); m.fit(X[~te], y[~te])
            p[te] = m.predict_proba(X[te])[:, 1]
        learned[name] = p
        for t, pi in zip(trials, p):
            t[f"{name}_p"] = float(pi)

    # ---- summarise ----
    def summarise(pred_key, thresh=None, label=""):
        per_s_f, per_s_a = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        per_act = defaultdict(lambda: [0, 0])
        tp = fn = fp = tn = 0
        for t in trials:
            v = t[pred_key]
            hit = (v > thresh) if thresh is not None else (v > 0.5 if isinstance(v, float) else bool(v))
            if pred_key == "rule_alerts":
                hit = t["rule_alerts"] > 0
            if t["fall"]:
                per_s_f[t["subj"]][1] += 1; per_s_f[t["subj"]][0] += hit
                tp += hit; fn += not hit
            else:
                per_s_a[t["subj"]][1] += 1; per_s_a[t["subj"]][0] += (not hit)
                fp += hit; tn += not hit
            per_act[t["act"]][1] += 1; per_act[t["act"]][0] += hit
        sens = cluster_boot({k: tuple(v) for k, v in per_s_f.items()})
        spec = cluster_boot({k: tuple(v) for k, v in per_s_a.items()})
        return dict(label=label, tp=tp, fn=fn, fp=fp, tn=tn,
                    sensitivity=sens, specificity=spec,
                    per_activity_positive_rate={k: v[0] / v[1] for k, v in sorted(per_act.items())})

    res = {}
    res["rule_streaming"] = summarise("rule_alerts", label="App rule + 5 s confirmation (as shipped)")
    res["rule_trigger_only"] = summarise("rule_trig", label="App rule, trigger only (no confirmation)")
    res["mlp_flat0"] = summarise("mlp_flat0", label="App rule OR bootstrap MLP, endFlat=0")
    res["mlp_flat1"] = summarise("mlp_flat1", label="App rule OR bootstrap MLP, endFlat=1")
    res["mlp_trig_flat0"] = summarise("mlp_trig_flat0", label="Bootstrap MLP alone fires (endFlat=0)")
    res["mlp_trig_flat1"] = summarise("mlp_trig_flat1", label="Bootstrap MLP alone fires (endFlat=1)")
    res["logreg_loso"] = summarise("logreg_p", label="Learned logistic regression (LOSO, trial-level)")
    res["gbm_loso"] = summarise("gbm_p", label="Learned gradient boosting (LOSO, trial-level)")
    for k in ("logreg", "gbm"):
        res[f"{k}_loso"]["auc"] = float(roc_auc_score(y, learned[k]))

    # false alarms per hour of ADL for the shipped rule (re-arming after each alert)
    adl = [t for t in trials if not t["fall"]]
    hrs = sum(t["dur_s"] for t in adl) / 3600
    res["rule_streaming"]["false_alarms_per_hour_adl"] = sum(t["rule_alerts"] for t in adl) / hrs
    res["rule_streaming"]["adl_hours"] = hrs

    # diagnostics: why falls are missed
    falls = [t for t in trials if t["fall"]]
    res["diagnostics"] = dict(
        n_trials=len(trials), n_falls=len(falls), n_adl=len(adl),
        n_subjects=len(set(groups)),
        falls_peak_over_24=float(np.mean([t["peak"] > PEAK_THRESHOLD_BASE for t in falls])),
        falls_rule_triggered=float(np.mean([t["rule_trig"] for t in falls])),
        falls_still_2to4s_post_peak=float(np.mean([t["post_peak_still"] for t in falls])),
        median_fall_peak=float(np.median([t["peak"] for t in falls])),
        fs_hz=fs,
    )

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(a.out, "trials.json"), "w") as fh:
        json.dump(trials, fh)

    print(f"\nSisFall: {len(trials)} trials, {len(falls)} falls, {len(adl)} ADLs, {len(set(groups))} subjects, fs={fs} Hz\n")
    print(f"{'pipeline':58s} {'sensitivity [95% CI]':24s} {'specificity [95% CI]':24s}")
    for k, r in res.items():
        if k == "diagnostics":
            continue
        s, sp = r["sensitivity"], r["specificity"]
        extra = f"  AUC={r['auc']:.3f}" if "auc" in r else ""
        print(f"{r['label']:58s} {s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}]   {sp[0]:.3f} [{sp[1]:.3f},{sp[2]:.3f}]{extra}")
    print("\nfalse alarms / ADL hour (shipped rule):", f"{res['rule_streaming']['false_alarms_per_hour_adl']:.2f}")
    print("diagnostics:", json.dumps(res["diagnostics"], indent=1))


if __name__ == "__main__":
    main()
