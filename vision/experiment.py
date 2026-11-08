#!/usr/bin/env python3
"""Staged-to-real experiment: do fall detectors trained on lab falls work on real falls?

Inputs
  --staged   urfall.npz from urfall_prep.py                       (staged lab falls + ADLs)
  --oops     directory from oops_mine.py, with labels.csv          (real in-the-wild clips)

Real-world TEST set = OOPS! clips in the hash-based random sample that a human labelled:
  positives: event clips labelled "fall"
  negatives: event clips labelled "not_fall" (other accidents: hard negatives)
             + pre-event clips of those same videos (ordinary real-world activity)
"unsure" clips are excluded. The real TRAINING pool (for real-trained probes) is the
labelled top-K candidates, which never overlap the random sample.

Clip representations (from the same frozen CLIP encoder, 8 ordered frames per clip)
  appearance : mean frame embedding (what is in the clip)
  temporal   : [mean, late-minus-early frames, mean |frame-to-frame change|]
               (how the clip changes over time: a fall is a motion, not a pose)

Methods: zero-shot CLIP prompts, and logistic-regression probes trained on staged, real,
or staged+real data. Metrics: AUROC, average precision, recall at 5% false-positive rate,
with 95% CIs from a video-level bootstrap. Writes JSON, a markdown table, an ROC figure,
and the real-test clips each staged-trained probe gets most wrong.
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labeler import load_labels  # noqa: E402
from oops_mine import load_frames  # noqa: E402


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def featurize(frames, kind):
    """frames: (n, T, d) ordered frame embeddings."""
    f = l2n(frames.astype(np.float32))
    mean = l2n(f.mean(1))
    if kind == "appearance":
        return mean
    k = max(1, f.shape[1] // 3)
    delta = f[:, -k:].mean(1) - f[:, :k].mean(1)
    change = np.abs(np.diff(f, axis=1)).mean(1)
    return np.concatenate([mean, delta, change], 1)


def metrics(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    if len(set(y)) < 2:
        return dict(auroc=float("nan"), ap=float("nan"), recall_at_fpr5=float("nan"))
    fpr, tpr, _ = roc_curve(y, s)
    return dict(auroc=float(roc_auc_score(y, s)), ap=float(average_precision_score(y, s)),
                recall_at_fpr5=float(np.interp(0.05, fpr, tpr)))


def boot(y, s, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    est = metrics(y, s)
    samples = {k: [] for k in est}
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        m = metrics(y[pick], s[pick])
        for k in m:
            if not np.isnan(m[k]):
                samples[k].append(m[k])
    return {k: [est[k], *(np.percentile(samples[k], [2.5, 97.5]).tolist() if samples[k] else [np.nan, np.nan])]
            for k in est}


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True)
    ap.add_argument("--oops", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-test-pos", type=int, default=20)
    a = ap.parse_args()
    out = a.out or os.path.join(a.oops, "results")
    os.makedirs(out, exist_ok=True)

    S = np.load(a.staged)
    Fs, ys, gs, ms = S["frames"], S["label"], S["group"], S["margin"]

    labels = load_labels(a.oops)
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    frames = load_frames(a.oops)
    test, train = [], []
    for (name, seg), fr in frames.items():
        lab = labels.get(name)
        if lab not in ("fall", "not_fall") or name not in meta:
            continue
        r = meta[name]
        y = 1 if (seg == "event" and lab == "fall") else 0
        m = float(r["event_margin"] if seg == "event" else r["pre_margin"])
        (test if r["random_sample"] == "1" else train).append((fr, y, name, seg, m))
    if not test:
        sys.exit("no labelled test clips yet: label the random-sample (test set) clips first")
    Ft = np.array([t[0] for t in test]); yt = np.array([t[1] for t in test])
    gt = np.array([t[2] for t in test]); segt = [t[3] for t in test]; mt = np.array([t[4] for t in test])
    print(f"real test set: {len(test)} clips from {len(set(gt))} videos, {yt.sum()} falls")
    if yt.sum() < a.min_test_pos:
        print(f"WARNING: only {yt.sum()} real falls in the test set; results will be very noisy")

    res = {"n": {"staged_clips": int(len(ys)), "staged_falls": int(ys.sum()),
                 "real_test_clips": int(len(yt)), "real_test_falls": int(yt.sum()),
                 "real_test_videos": int(len(set(gt))), "real_train_clips": len(train),
                 "real_train_falls": int(sum(t[1] for t in train))},
           "model": str(S["model"]) if "model" in S.files else "", "staged": {}, "real": {}}
    scores = {}

    # ---- in-domain staged performance: 5-fold cross-validation grouped by sequence ----
    from sklearn.model_selection import GroupKFold
    res["staged"]["zero_shot"] = boot(ys, ms, gs)
    for kind in ("appearance", "temporal"):
        X = featurize(Fs, kind)
        p = np.zeros(len(ys))
        for tr, te in GroupKFold(5).split(X, ys, gs):
            p[te] = probe().fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
        res["staged"][f"staged_probe_{kind}_cv"] = boot(ys, p, gs)

    # ---- staged -> real ----
    res["real"]["zero_shot"] = boot(yt, mt, gt); scores["zero_shot"] = mt
    have_real = bool(train) and len({t[1] for t in train}) == 2
    if not have_real:
        print("no labelled training-pool clips with both classes: skipping real-trained probes")
    for kind in ("appearance", "temporal"):
        Xs, Xt = featurize(Fs, kind), featurize(Ft, kind)
        s = probe().fit(Xs, ys).predict_proba(Xt)[:, 1]
        res["real"][f"staged_probe_{kind}"] = boot(yt, s, gt); scores[f"staged_probe_{kind}"] = s
        if have_real:
            Xr = featurize(np.array([t[0] for t in train]), kind); yr = np.array([t[1] for t in train])
            s = probe().fit(Xr, yr).predict_proba(Xt)[:, 1]
            res["real"][f"real_probe_{kind}"] = boot(yt, s, gt); scores[f"real_probe_{kind}"] = s
            s = probe().fit(np.vstack([Xs, Xr]), np.concatenate([ys, yr])).predict_proba(Xt)[:, 1]
            res["real"][f"staged_plus_real_probe_{kind}"] = boot(yt, s, gt)
            scores[f"staged_plus_real_probe_{kind}"] = s

    # ---- outputs ----
    with open(os.path.join(out, "staged_to_real.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    lines = ["| Evaluated on | Method | AUROC | AP | Recall @ 5% FPR |", "|---|---|---|---|---|"]
    for dom in ("staged", "real"):
        for k, v in res[dom].items():
            f = lambda m: f"{v[m][0]:.3f} [{v[m][1]:.3f}, {v[m][2]:.3f}]"
            lines.append(f"| {dom} | {k} | {f('auroc')} | {f('ap')} | {f('recall_at_fpr5')} |")
    table = "\n".join(lines)
    with open(os.path.join(out, "staged_to_real.md"), "w") as fh:
        fh.write("Counts: " + json.dumps(res["n"]) + "\n\n" + table + "\n")

    # the real-test clips a staged-trained probe gets most wrong (for qualitative analysis)
    with open(os.path.join(out, "worst_errors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "error", "score", "name", "segment", "label"])
        for k in ("staged_probe_appearance", "staged_probe_temporal"):
            s = scores[k]
            fn = [i for i in np.argsort(s) if yt[i] == 1][:15]
            fp = [i for i in np.argsort(-s) if yt[i] == 0][:15]
            for err, ids in (("missed fall", fn), ("false alarm", fp)):
                for i in ids:
                    w.writerow([k, err, f"{s[i]:.4f}", gt[i], segt[i], int(yt[i])])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for k, s in scores.items():
            fpr, tpr, _ = roc_curve(yt, s)
            ax.plot(fpr, tpr, label=f"{k} ({res['real'][k]['auroc'][0]:.2f})")
        ax.plot([0, 1], [0, 1], ":", color="grey")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate (recall)")
        ax.set_title("Real-world falls (OOPS! test set)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(out, "roc_real.png"), dpi=200)
    except ImportError:
        pass

    print("Counts:", json.dumps(res["n"]))
    print(table)
    print(f"\nwrote {out}/staged_to_real.md, .json, roc_real.png, worst_errors.csv")


if __name__ == "__main__":
    main()
