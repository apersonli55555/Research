#!/usr/bin/env python3
"""Staged-to-real experiment: do fall detectors trained on lab falls work on real falls?

Inputs
  --staged   urfall.npz from urfall_prep.py           (staged lab falls + ADLs)
  --oops     directory from oops_mine.py + labels.csv  (real in-the-wild clips)

Real-world TEST set = OOPS! clips in the hash-based random sample that a human labelled:
  positives: event clips labelled "fall"
  negatives: event clips labelled "not_fall" (other accidents: hard negatives)
             + pre-event clips of those same videos (ordinary real-world activity)
"unsure" clips are excluded. The real TRAINING pool (for the in-domain reference) is the
labelled top-K candidates, which never overlap the random sample.

Methods (all on frozen CLIP clip embeddings)
  zero-shot       : CLIP text-prompt margin (no training)
  staged probe    : logistic regression trained on UR Fall
  real probe      : logistic regression trained on the real training pool (in-domain reference)
  staged+real     : both
Metrics: AUROC, average precision, recall at 5% false-positive rate; 95% CIs from a
video-level bootstrap.
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labeler import load_labels  # noqa: E402


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
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")


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
    Xs, ys, gs, ms = S["emb"], S["label"], S["group"], S["margin"]

    labels = load_labels(a.oops)
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    Z = np.load(os.path.join(a.oops, "embeddings.npz"))

    test, train = [], []
    for name, seg, e in zip(Z["name"], Z["seg"], Z["emb"].astype(np.float32)):
        lab = labels.get(name)
        if lab not in ("fall", "not_fall"):
            continue
        r = meta[name]
        y = 1 if (seg == "event" and lab == "fall") else 0
        m = float(r["event_margin"] if seg == "event" else r["pre_margin"])
        item = (e, y, name, m)
        (test if r["random_sample"] == "1" else train).append(item)
    if not test:
        sys.exit("no labelled test clips yet: label the random-sample clips first")
    Xt = np.array([t[0] for t in test]); yt = np.array([t[1] for t in test])
    gt = np.array([t[2] for t in test]); mt = np.array([t[3] for t in test])
    print(f"real test set: {len(test)} clips from {len(set(gt))} videos, {yt.sum()} falls")
    if yt.sum() < a.min_test_pos:
        print(f"WARNING: only {yt.sum()} real falls in the test set; results will be very noisy")

    res = {"n": {"staged_clips": int(len(ys)), "staged_falls": int(ys.sum()),
                 "real_test_clips": int(len(yt)), "real_test_falls": int(yt.sum()),
                 "real_test_videos": int(len(set(gt))), "real_train_clips": len(train),
                 "real_train_falls": int(sum(t[1] for t in train))},
           "model": str(S["model"]) if "model" in S.files else "", "staged": {}, "real": {}}

    # ---- in-domain staged performance: 5-fold cross-validation grouped by sequence ----
    from sklearn.model_selection import GroupKFold
    p = np.zeros(len(ys))
    for tr, te in GroupKFold(5).split(Xs, ys, gs):
        p[te] = probe().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
    res["staged"]["staged_probe_cv"] = boot(ys, p, gs)
    res["staged"]["zero_shot"] = boot(ys, ms, gs)

    # ---- staged -> real ----
    res["real"]["zero_shot"] = boot(yt, mt, gt)
    res["real"]["staged_probe"] = boot(yt, probe().fit(Xs, ys).predict_proba(Xt)[:, 1], gt)
    if train and len({t[1] for t in train}) == 2:
        Xr = np.array([t[0] for t in train]); yr = np.array([t[1] for t in train])
        res["real"]["real_probe"] = boot(yt, probe().fit(Xr, yr).predict_proba(Xt)[:, 1], gt)
        res["real"]["staged_plus_real_probe"] = boot(
            yt, probe().fit(np.vstack([Xs, Xr]), np.concatenate([ys, yr])).predict_proba(Xt)[:, 1], gt)
    else:
        print("no labelled training-pool clips with both classes: skipping real-trained probes")

    with open(os.path.join(out, "staged_to_real.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    lines = ["| Evaluated on | Method | AUROC | AP | Recall @ 5% FPR |", "|---|---|---|---|---|"]
    for dom in ("staged", "real"):
        for k, v in res[dom].items():
            f = lambda m: f"{v[m][0]:.3f} [{v[m][1]:.3f}, {v[m][2]:.3f}]"
            lines.append(f"| {dom} | {k} | {f('auroc')} | {f('ap')} | {f('recall_at_fpr5')} |")
    table = "\n".join(lines)
    with open(os.path.join(out, "staged_to_real.md"), "w") as fh:
        fh.write(json.dumps(res["n"]) + "\n\n" + table + "\n")
    print(json.dumps(res["n"]))
    print(table)


if __name__ == "__main__":
    main()
