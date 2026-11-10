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

Clip representations (all from frozen, pretrained encoders)
  clip_appearance : CLIP image model, mean of 8 frame embeddings (what is in the clip)
  clip_temporal   : CLIP, [mean, late-minus-early frames, mean |frame-to-frame change|]
  videomae        : VideoMAE video model over 16 frames (attends across time: motion)
  clip+videomae   : clip_temporal and videomae concatenated

Methods: zero-shot CLIP prompts, and logistic-regression probes trained on staged, real,
or staged+real data. Metrics: AUROC, average precision, recall at 5% false-positive rate,
with 95% CIs from a video-level bootstrap. Writes JSON, a markdown table, an ROC figure,
and the real-test clips each staged-trained probe gets most wrong.
"""
import argparse, csv, json, os, re, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from labeler import load_labels  # noqa: E402
from oops_mine import load_frames  # noqa: E402


REPRESENTATIONS = ["clip_appearance", "clip_temporal", "videomae", "clip+videomae"]


def compilation(name):
    """OOPS! video names are '<compilation title><clip index>'; strip the index."""
    return re.sub(r"\d+$", "", str(name)).strip()


def l2n(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def featurize(frames, kind, video=None):
    """kind in REPRESENTATIONS. frames: (n, 8, d) CLIP frame embeddings; video: (n, d') VideoMAE."""
    if kind == "videomae":
        return l2n(video.astype(np.float32))
    if kind == "clip+videomae":
        return np.concatenate([featurize(frames, "clip_temporal"), l2n(video.astype(np.float32))], 1)
    kind = kind.replace("clip_", "")
    return _featurize_clip(frames, kind)


def _featurize_clip(frames, kind):
    """frames: (n, T, d) ordered frame embeddings."""
    f = l2n(frames.astype(np.float32))
    mean = l2n(f.mean(1))
    if kind == "appearance":
        return mean
    k = max(1, f.shape[1] // 3)
    delta = f[:, -k:].mean(1) - f[:, :k].mean(1)
    change = np.abs(np.diff(f, axis=1)).mean(1)
    return np.concatenate([mean, delta, change], 1)


def _auroc(y, s):
    """AUROC via the rank-sum (Mann-Whitney) statistic, ties averaged. Equals sklearn's roc_auc_score."""
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    ranks = np.empty(len(s))
    _, start_idx, counts = np.unique(ss, return_index=True, return_counts=True)
    avg = np.repeat(start_idx + (counts + 1) / 2.0, counts)
    ranks[order] = avg
    npos = y.sum(); nneg = len(y) - npos
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def _roc_points(y, s):
    """Distinct-threshold ROC points (fpr, tpr) and AP, as sklearn computes them."""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]       # last index of each tie group
    tps = np.cumsum(ys)[last].astype(float)
    fps = (last + 1 - tps)
    P, N = tps[-1], fps[-1]
    tpr = np.r_[0.0, tps / P]; fpr = np.r_[0.0, fps / N]
    precision = tps / (tps + fps)
    recall = tps / P
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return fpr, tpr, ap


def metrics(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return dict(auroc=float("nan"), ap=float("nan"), recall_at_fpr5=float("nan"))
    fpr, tpr, ap = _roc_points(y, s)
    return dict(auroc=float(_auroc(y, s)), ap=ap, recall_at_fpr5=float(np.interp(0.05, fpr, tpr)))


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


def paired(y, s1, s2, groups, n=2000, seed=0):
    """AUROC(s1) - AUROC(s2) on the same clips, with a video-level paired bootstrap.
    Returns [diff, ci_low, ci_high, p] where p is a two-sided bootstrap p-value."""
    roc_auc_score = _auroc
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    est = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    d = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(set(y[pick])) == 2:
            d.append(roc_auc_score(y[pick], s1[pick]) - roc_auc_score(y[pick], s2[pick]))
    d = np.array(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return [float(est), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)]


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
    ap.add_argument("--group-by", choices=["video", "compilation"], default="video",
                    help="resampling unit for FallOrNot confidence intervals and paired tests")
    ap.add_argument("--synthetic", default="", help="ofsyn.npz from ofsyn_prep.py (optional)")
    ap.add_argument("--external", default="", help="hmdb.npz from hmdb_prep.py: external real-world test (optional)")
    ap.add_argument("--finetuned", nargs="*", default=[], help="scores_*.npz from finetune.py score (optional)")
    ap.add_argument("--test-labels", default="", help="file in --oops (e.g. labels_friend.csv): take TEST labels from "
                    "this annotator instead (test set = clips they labelled); training labels unchanged")
    a = ap.parse_args()
    out = a.out or os.path.join(a.oops, "results")
    os.makedirs(out, exist_ok=True)

    S = np.load(a.staged)
    Fs, ys, gs, ms = S["frames"], S["label"], S["group"], S["margin"]
    Vs = S["video"] if "video" in S.files else None

    labels = load_labels(a.oops)
    test_labels = None
    if a.test_labels:
        test_labels = {}
        with open(os.path.join(a.oops, a.test_labels)) as fh:
            for r_ in csv.DictReader(fh):
                test_labels[r_["name"]] = r_["label"]
        test_labels = {k: v for k, v in test_labels.items() if v}
        print(f"TEST LABELS FROM {a.test_labels}: {sum(v in ('fall', 'not_fall') for v in test_labels.values())} clips")
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    frames = load_frames(a.oops)
    videos = load_frames(a.oops, "video")
    reps = [r for r in REPRESENTATIONS if "videomae" not in r or (Vs is not None and videos)]
    test, train = [], []
    for (name, seg), fr in frames.items():
        if name not in meta:
            continue
        r = meta[name]
        lab = test_labels.get(name) if (test_labels is not None and r["random_sample"] == "1") else labels.get(name)
        if lab not in ("fall", "not_fall"):
            continue
        y = 1 if (seg == "event" and lab == "fall") else 0
        m = float(r["event_margin"] if seg == "event" else r["pre_margin"])
        v = videos.get((name, seg))
        if videos and v is None:
            continue
        (test if r["random_sample"] == "1" else train).append((fr, y, name, seg, m, v))
    if not test:
        sys.exit("no labelled test clips yet: label the random-sample (test set) clips first")
    Ft = np.array([t[0] for t in test]); yt = np.array([t[1] for t in test])
    Vt = np.array([t[5] for t in test]) if videos else None
    gt = np.array([t[2] for t in test]); segt = [t[3] for t in test]; mt = np.array([t[4] for t in test])
    # resampling unit for FallOrNot confidence intervals: the video (default) or the source
    # compilation (OOPS! clips are cut from YouTube compilations; clips of one compilation are
    # correlated, so --group-by compilation gives more conservative intervals)
    gb = np.array([compilation(n) for n in gt]) if a.group_by == "compilation" else gt
    test_comps = {compilation(n) for n in gt}
    print(f"real test set: {len(test)} clips from {len(set(gt))} videos, {yt.sum()} falls")
    if yt.sum() < a.min_test_pos:
        print(f"WARNING: only {yt.sum()} real falls in the test set; results will be very noisy")

    res = {"n": {"staged_clips": int(len(ys)), "staged_falls": int(ys.sum()),
                 "real_test_clips": int(len(yt)), "real_test_falls": int(yt.sum()),
                 "real_test_videos": int(len(set(gt))), "real_train_clips": len(train),
                 "real_train_falls": int(sum(t[1] for t in train))},
           "model": str(S["model"]) if "model" in S.files else "", "staged": {}, "real": {}, "real_hard": {}}
    scores = {}

    # ---- in-domain staged performance: 5-fold cross-validation grouped by sequence ----
    from sklearn.model_selection import GroupKFold
    res["staged"]["zero_shot"] = boot(ys, ms, gs)
    for kind in reps:
        X = featurize(Fs, kind, Vs)
        p = np.zeros(len(ys))
        for tr, te in GroupKFold(5).split(X, ys, gs):
            p[te] = probe().fit(X[tr], ys[tr]).predict_proba(X[te])[:, 1]
        res["staged"][f"staged_probe_{kind}_cv"] = boot(ys, p, gs)

    # ---- staged -> real ----
    res["real"]["zero_shot"] = boot(yt, mt, gb); scores["zero_shot"] = mt
    have_real = bool(train) and len({t[1] for t in train}) == 2
    if not have_real:
        print("no labelled training-pool clips with both classes: skipping real-trained probes")
    for kind in reps:
        Xs, Xt = featurize(Fs, kind, Vs), featurize(Ft, kind, Vt)
        s = probe().fit(Xs, ys).predict_proba(Xt)[:, 1]
        res["real"][f"staged_probe_{kind}"] = boot(yt, s, gb); scores[f"staged_probe_{kind}"] = s
        if have_real:
            Xr = featurize(np.array([t[0] for t in train]), kind,
                           np.array([t[5] for t in train]) if videos else None)
            yr = np.array([t[1] for t in train])
            s = probe().fit(Xr, yr).predict_proba(Xt)[:, 1]
            res["real"][f"real_probe_{kind}"] = boot(yt, s, gb); scores[f"real_probe_{kind}"] = s
            s = probe().fit(np.vstack([Xs, Xr]), np.concatenate([ys, yr])).predict_proba(Xt)[:, 1]
            res["real"][f"staged_plus_real_probe_{kind}"] = boot(yt, s, gb)
            scores[f"staged_plus_real_probe_{kind}"] = s

    # ---- leakage check: real training clips from compilations that also appear in the test set ----
    if have_real:
        from sklearn.metrics import roc_auc_score
        tr_comp = np.array([compilation(t[2]) for t in train])
        disj = np.array([c not in test_comps for c in tr_comp])
        yr_all = np.array([t[1] for t in train])
        res["leakage"] = {"train_clips": int(len(train)), "train_clips_sharing_compilation_with_test": int((~disj).sum()),
                          "disjoint_clips": int(disj.sum()), "disjoint_falls": int(yr_all[disj].sum()), "by_rep": {}}
        print(f"LEAKAGE CHECK: {int((~disj).sum())}/{len(train)} real training clips come from a compilation "
              f"that also appears in the test set; {int(disj.sum())} clips ({int(yr_all[disj].sum())} falls) do not")
        if disj.sum() >= 10 and len(set(yr_all[disj])) == 2:
            rng_l = np.random.default_rng(7)
            for kind in reps:
                Xr = featurize(np.array([t[0] for t in train]), kind, np.array([t[5] for t in train]) if videos else None)
                Xt_l = featurize(Ft, kind, Vt)
                auc = lambda m: roc_auc_score(yt, probe().fit(Xr[m], yr_all[m]).predict_proba(Xt_l)[:, 1])
                d_auc = auc(disj)
                rnd = []
                for _ in range(20):                    # same number of clips and falls, drawn from ALL training clips
                    pi = rng_l.choice(np.where(yr_all == 1)[0], int(yr_all[disj].sum()), replace=False)
                    ni = rng_l.choice(np.where(yr_all == 0)[0], int((1 - yr_all[disj]).sum()), replace=False)
                    m = np.zeros(len(yr_all), bool); m[pi] = True; m[ni] = True
                    rnd.append(auc(m))
                res["leakage"]["by_rep"][kind] = {"all_train": res["real"][f"real_probe_{kind}"]["auroc"][0],
                                                  "disjoint": float(d_auc), "random_same_size_mean": float(np.mean(rnd)),
                                                  "random_same_size_sd": float(np.std(rnd)),
                                                  "staged": res["real"][f"staged_probe_{kind}"]["auroc"][0]}

    # ---- synthetic (OmniFall OF-Syn) -> real ----
    Y = np.load(a.synthetic) if a.synthetic else None
    if Y is not None:
        Fy, yy = Y["frames"], Y["label"]
        Vy = Y["video"] if "video" in Y.files else None
        res["n"]["synthetic_clips"] = int(len(yy)); res["n"]["synthetic_falls"] = int(yy.sum())
        for kind in reps:
            if "videomae" in kind and Vy is None:
                continue
            Xy, Xt = featurize(Fy, kind, Vy), featurize(Ft, kind, Vt)
            s = probe().fit(Xy, yy).predict_proba(Xt)[:, 1]
            res["real"][f"synthetic_probe_{kind}"] = boot(yt, s, gb); scores[f"synthetic_probe_{kind}"] = s
            Xs = featurize(Fs, kind, Vs)
            s = probe().fit(np.vstack([Xs, Xy]), np.concatenate([ys, yy])).predict_proba(Xt)[:, 1]
            res["real"][f"staged_plus_synthetic_probe_{kind}"] = boot(yt, s, gb)
            scores[f"staged_plus_synthetic_probe_{kind}"] = s
            if have_real:
                Xr = featurize(np.array([t[0] for t in train]), kind,
                               np.array([t[5] for t in train]) if videos else None)
                yr = np.array([t[1] for t in train])
                s = probe().fit(np.vstack([Xy, Xr]), np.concatenate([yy, yr])).predict_proba(Xt)[:, 1]
                res["real"][f"synthetic_plus_real_probe_{kind}"] = boot(yt, s, gb)
                scores[f"synthetic_plus_real_probe_{kind}"] = s

    # ---- external real-world test (HMDB51): no training on it, clip score = max over windows ----
    ext_scores, ye, ne, hard = {}, None, None, None
    if a.external:
        E = np.load(a.external)
        Fe, wc = E["frames"], E["win_clip"]
        Ve = E["video"] if "video" in E.files else None
        ye, he, ne = E["clip_label"], E["clip_hard"].astype(bool), E["clip_name"]
        nclip = len(ye)

        def clip_max(w):
            out_ = np.full(nclip, -np.inf)
            np.maximum.at(out_, wc, w)
            return out_
        hard = he | (ye == 1)
        res["external"], res["external_hard"], ext_scores = {}, {}, {}
        res["n"].update(external_clips=int(nclip), external_falls=int(ye.sum()),
                        external_hard_clips=int(hard.sum()))
        ext_scores["zero_shot"] = clip_max(E["margin"])
        train_sets = {"staged": (Fs, Vs, ys)}
        if Y is not None:
            train_sets["synthetic"] = (Fy, Vy, yy)
        if have_real:
            train_sets["real"] = (np.array([t[0] for t in train]),
                                  np.array([t[5] for t in train]) if videos else None,
                                  np.array([t[1] for t in train]))
        if Y is not None and have_real:
            r_ = train_sets["real"]
            train_sets["synthetic_plus_real"] = (np.concatenate([Fy, r_[0]]),
                                                 np.concatenate([Vy, r_[1]]) if (Vy is not None and r_[1] is not None) else None,
                                                 np.concatenate([yy, r_[2]]))
        for kind in reps:
            if "videomae" in kind and Ve is None:
                continue
            Xe = featurize(Fe, kind, Ve)
            for src, (F_, V_, y_) in train_sets.items():
                if "videomae" in kind and V_ is None:
                    continue
                w = probe().fit(featurize(F_, kind, V_), y_).predict_proba(Xe)[:, 1]
                ext_scores[f"{src}_probe_{kind}"] = clip_max(w)
        for k, sc in ext_scores.items():
            res["external"][k] = boot(ye, sc, ne)
            res["external_hard"][k] = boot(ye[hard], sc[hard], ne[hard])
        res["paired_external"] = {}
        for kind in reps:
            for a_, b_ in (("real", "synthetic"), ("real", "staged"), ("synthetic", "staged"),
                           ("synthetic_plus_real", "real")):
                ka, kb = f"{a_}_probe_{kind}", f"{b_}_probe_{kind}"
                if ka in ext_scores and kb in ext_scores:
                    res["paired_external"][f"{ka} vs {kb}"] = paired(ye, ext_scores[ka], ext_scores[kb], ne)

    # ---- fine-tuned VideoMAE (finetune.py); pre-registered test: does fine-tuning on
    #      synthetic+real beat the best probe on BOTH test sets? ----
    res["preregistered"] = {}
    for fpath in a.finetuned:
        Z = np.load(fpath)
        seed = int(Z["seed"]) if "seed" in Z.files else 0
        tag = "finetuned_" + str(Z["train_on"]).replace("+", "_plus_") + ("" if seed == 0 else f"_s{seed}")
        fmap = {(str(n), str(sg)): float(v) for n, sg, v in zip(Z["fon_name"], Z["fon_seg"], Z["fon_score"])}
        keys = [(str(n), str(sg)) for n, sg in zip(gt, segt)]
        have = np.array([k in fmap for k in keys])
        if have.all():
            scores[tag] = np.array([fmap[k] for k in keys])
            res["real"][tag] = boot(yt, scores[tag], gb)
        else:
            print(f"WARNING {tag}: {int((~have).sum())} FallOrNot test clips missing; skipped")
        if a.external and "hmdb_name" in Z.files:
            hmap = dict(zip(map(str, Z["hmdb_name"]), Z["hmdb_score"].astype(float)))
            if all(str(n) in hmap for n in ne):
                ext_scores[tag] = np.array([hmap[str(n)] for n in ne])
                res["external"][tag] = boot(ye, ext_scores[tag], ne)
                res["external_hard"][tag] = boot(ye[hard], ext_scores[tag][hard], ne[hard])
            else:
                print(f"WARNING {tag}: some HMDB51 clips missing; skipped")
    ft = "finetuned_synthetic_plus_real"
    if ft in scores and a.external and ft in ext_scores:
        best_fon = max((k for k in scores if "_probe_" in k), key=lambda k: res["real"][k]["auroc"][0])
        best_ext = max((k for k in ext_scores if "_probe_" in k), key=lambda k: res["external"][k]["auroc"][0])
        p1 = paired(yt, scores[ft], scores[best_fon], gb)
        p2 = paired(ye, ext_scores[ft], ext_scores[best_ext], ne)
        res["preregistered"] = {
            "rule": "fine-tuned (synthetic+real) beats the best frozen probe on BOTH test sets, paired 95% CI above zero",
            "fallornot": {"vs": best_fon, "diff_ci_p": p1},
            "hmdb51": {"vs": best_ext, "diff_ci_p": p2},
            "success": bool(p1[1] > 0 and p2[1] > 0)}
        for src in ("synthetic", "real", "synthetic_plus_real"):
            k = "finetuned_" + src
            for dom_scores, yy_, gg_, key in ((scores, yt, gb, "paired"), (ext_scores, ye, ne, "paired_external")):
                b = f"{src}_probe_clip+videomae"
                if k in dom_scores and b in dom_scores:
                    res.setdefault(key + "_ft", {})[f"{k} vs {b}"] = paired(yy_, dom_scores[k], dom_scores[b], gg_)

    # ---- head-to-head comparisons on the same test clips ----
    res["paired"] = {}
    pairs = []
    for kind in reps:
        pairs += [(f"real_probe_{kind}", f"synthetic_probe_{kind}"),
                  (f"synthetic_probe_{kind}", f"staged_probe_{kind}"),
                  (f"synthetic_plus_real_probe_{kind}", f"real_probe_{kind}")]
    for kind in reps:
        for a_, b_ in ((f"real_probe_{kind}", f"staged_probe_{kind}"),
                       (f"staged_plus_real_probe_{kind}", f"staged_probe_{kind}")):
            pairs.append((a_, b_))
    pairs += [(f"staged_probe_{r}", "staged_probe_clip_appearance") for r in reps if r != "clip_appearance"]
    for a_, b_ in pairs:
        if a_ in scores and b_ in scores:
            res["paired"][f"{a_} vs {b_}"] = paired(yt, scores[a_], scores[b_], gb)

    # ---- learning curve: real training videos -> test AUROC ----
    if have_real:
        from sklearn.metrics import roc_auc_score
        kind = "clip+videomae" if "clip+videomae" in reps else "clip_temporal"
        tr_names = sorted({t[2] for t in train})
        Xr_all = featurize(np.array([t[0] for t in train]), kind,
                           np.array([t[5] for t in train]) if videos else None)
        yr_all = np.array([t[1] for t in train]); nr_all = np.array([t[2] for t in train])
        Xs_k, Xt_k = featurize(Fs, kind, Vs), featurize(Ft, kind, Vt)
        sizes = sorted({n for n in (5, 10, 20, 40, 80, 160, 320) if n < len(tr_names)} | {len(tr_names)})
        curve = []
        rng = np.random.default_rng(0)
        for n_v in sizes:
            real_auc, both_auc, falls = [], [], []
            for _ in range(10 if n_v < len(tr_names) else 1):
                pick = set(rng.choice(tr_names, n_v, replace=False))
                m = np.array([nm in pick for nm in nr_all])
                if len(set(yr_all[m])) < 2:
                    continue
                falls.append(int(yr_all[m].sum()))
                real_auc.append(roc_auc_score(yt, probe().fit(Xr_all[m], yr_all[m]).predict_proba(Xt_k)[:, 1]))
                both_auc.append(roc_auc_score(yt, probe().fit(np.vstack([Xs_k, Xr_all[m]]),
                                np.concatenate([ys, yr_all[m]])).predict_proba(Xt_k)[:, 1]))
            if real_auc:
                curve.append(dict(videos=n_v, falls=float(np.mean(falls)), real=float(np.mean(real_auc)),
                                  real_sd=float(np.std(real_auc)), both=float(np.mean(both_auc)),
                                  both_sd=float(np.std(both_auc))))
        res["learning_curve"] = {"representation": kind, "staged_only": res["real"][f"staged_probe_{kind}"][ "auroc"][0],
                                 "points": curve}

    # ---- equal budget: same number of training falls from each source ----
    if Y is not None:
        from sklearn.metrics import roc_auc_score
        kind = "clip+videomae" if ("clip+videomae" in reps and Vy is not None) else "clip_temporal"
        Xt_k = featurize(Ft, kind, Vt)
        sources = {"staged": (featurize(Fs, kind, Vs), ys), "synthetic": (featurize(Fy, kind, Vy), yy)}
        if have_real:
            sources["real"] = (featurize(np.array([t[0] for t in train]), kind,
                                         np.array([t[5] for t in train]) if videos else None),
                               np.array([t[1] for t in train]))
        budgets = (4, 8, 15, 30, 60, 125, 250, 500, 1000, 2000)
        rng = np.random.default_rng(1)
        eb = {}
        for src, (X, y) in sources.items():
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            pts = []
            for k in budgets + (len(pos),):
                if k > len(pos) or (pts and pts[-1]["falls"] == k):
                    continue
                aucs = []
                for _ in range((10 if k <= 60 else 3) if k < len(pos) else 1):
                    pi = rng.choice(pos, k, replace=False)
                    ni = rng.choice(neg, min(k, len(neg)), replace=False)  # balanced: as many non-falls
                    idx = np.concatenate([pi, ni])
                    aucs.append(roc_auc_score(yt, probe().fit(X[idx], y[idx]).predict_proba(Xt_k)[:, 1]))
                pts.append(dict(falls=int(k), auroc=float(np.mean(aucs)), sd=float(np.std(aucs))))
            eb[src] = pts
        best_real = max((p_["auroc"] for p_ in eb.get("real", [])), default=None)
        match = None
        if best_real is not None:
            hit = [p_["falls"] for p_ in eb["synthetic"] if p_["auroc"] >= best_real]
            match = hit[0] if hit else None
        res["equal_budget"] = {"representation": kind, "points": eb,
                               "real_max_falls": max((p_["falls"] for p_ in eb.get("real", [])), default=None),
                               "synthetic_falls_to_match_all_real": match}

    # ---- harder test: falls vs other accidents only (accident-moment clips; the easy
    #      "normal activity before the accident" negatives are left out) ----
    ev = np.array([sg == "event" for sg in segt])
    res["n"]["real_hard_clips"] = int(ev.sum()); res["n"]["real_hard_falls"] = int(yt[ev].sum())
    for k, sc in scores.items():
        res["real_hard"][k] = boot(yt[ev], sc[ev], gb[ev])

    # ---- robustness across fine-tuning seeds (the pre-registered test above is seed 0) ----
    import re
    bases = sorted({re.sub(r"_s\d+$", "", k) for k in list(scores) + list(ext_scores) if k.startswith("finetuned_")})
    if any(re.search(r"_s\d+$", k) for k in list(scores) + list(ext_scores)):
        res["seeds"] = {}
        for b in bases:
            for dom, dd in (("real", res["real"]), ("real_hard", res["real_hard"]), ("external", res.get("external", {})),
                            ("external_hard", res.get("external_hard", {}))):
                if dd is None:
                    continue
                vals = [v["auroc"][0] for k, v in dd.items() if re.fullmatch(re.escape(b) + r"(_s\d+)?", k)]
                if vals:
                    res["seeds"].setdefault(b, {})[dom] = dict(n=len(vals), mean=float(np.mean(vals)),
                                                               sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                                                               values=vals)
        if "preregistered" in res and res["preregistered"]:
            per_seed = []
            for k in [k for k in scores if re.fullmatch(r"finetuned_synthetic_plus_real(_s\d+)?", k)]:
                if k in ext_scores:
                    p1 = paired(yt, scores[k], scores[res["preregistered"]["fallornot"]["vs"]], gb)
                    p2 = paired(ye, ext_scores[k], ext_scores[res["preregistered"]["hmdb51"]["vs"]], ne)
                    per_seed.append(dict(model=k, fallornot=p1, hmdb51=p2, success=bool(p1[1] > 0 and p2[1] > 0)))
            res["preregistered"]["per_seed"] = per_seed

    # ---- outputs ----
    with open(os.path.join(out, "staged_to_real.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    lines = ["| Evaluated on | Method | AUROC | AP | Recall @ 5% FPR |", "|---|---|---|---|---|"]
    for dom in ("staged", "real", "real_hard", "external", "external_hard"):
        if dom not in res:
            continue
        for k, v in res[dom].items():
            f = lambda m: f"{v[m][0]:.3f} [{v[m][1]:.3f}, {v[m][2]:.3f}]"
            lines.append(f"| {dom} | {k} | {f('auroc')} | {f('ap')} | {f('recall_at_fpr5')} |")
    table = "\n".join(lines)
    extra = ["", "Head-to-head (same test clips): AUROC difference [95% CI], p", "",
             "| Comparison | AUROC diff | 95% CI | p |", "|---|---|---|---|"]
    for k, (d_, lo, hi, pv) in res["paired"].items():
        extra.append(f"| {k} | {d_:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} |")
    if "learning_curve" in res:
        lc = res["learning_curve"]
        extra += ["", f"Learning curve ({lc['representation']}; staged-only AUROC {lc['staged_only']:.3f})", "",
                  "| real training videos | real falls (avg) | real-trained AUROC | staged + real AUROC |",
                  "|---|---|---|---|"]
        for pt in lc["points"]:
            extra.append(f"| {pt['videos']} | {pt['falls']:.0f} | {pt['real']:.3f} ± {pt['real_sd']:.3f} | "
                         f"{pt['both']:.3f} ± {pt['both_sd']:.3f} |")
    if res.get("leakage"):
        lk = res["leakage"]
        extra += ["", f"LEAKAGE CHECK: {lk['train_clips_sharing_compilation_with_test']}/{lk['train_clips']} real training clips share a "
                      f"compilation with the test set; compilation-disjoint subset = {lk['disjoint_clips']} clips ({lk['disjoint_falls']} falls)", "",
                  "| Representation | Real, all clips | Real, compilation-disjoint | Real, random same size (mean ± s.d., 20 draws) | Staged |",
                  "|---|---|---|---|---|"]
        for k, v in lk["by_rep"].items():
            extra.append(f"| {k} | {v['all_train']:.3f} | {v['disjoint']:.3f} | {v['random_same_size_mean']:.3f} ± "
                         f"{v['random_same_size_sd']:.3f} | {v['staged']:.3f} |")
    extra.append("")
    extra.append(f"FallOrNot confidence intervals resampled by: {a.group_by}")
    if res.get("preregistered"):
        pr = res["preregistered"]
        extra += ["", "PRE-REGISTERED TEST: " + pr["rule"], ""]
        for dom in ("fallornot", "hmdb51"):
            d_, lo, hi, pv = pr[dom]["diff_ci_p"]
            extra.append(f"- {dom}: fine-tuned vs best probe ({pr[dom]['vs']}): {d_:+.3f} [{lo:+.3f}, {hi:+.3f}], p = {pv:.3f}")
        extra.append(f"- RESULT: {'SUCCESS' if pr['success'] else 'NOT MET'}")
        if pr.get("per_seed"):
            extra += ["", "Same rule for every fine-tuning seed (robustness):", ""]
            for ps in pr["per_seed"]:
                f1, f2 = ps["fallornot"], ps["hmdb51"]
                extra.append(f"- {ps['model']}: FallOrNot {f1[0]:+.3f} [{f1[1]:+.3f}, {f1[2]:+.3f}]; "
                             f"HMDB51 {f2[0]:+.3f} [{f2[1]:+.3f}, {f2[2]:+.3f}] -> {'met' if ps['success'] else 'not met'}")
    if res.get("seeds"):
        extra += ["", "Fine-tuned models across seeds: AUROC mean ± s.d. (n seeds)", "",
                  "| Model | FallOrNot | FallOrNot hard | HMDB51 | HMDB51 hard |", "|---|---|---|---|---|"]
        for b, dd in res["seeds"].items():
            cell = lambda d: f"{dd[d]['mean']:.3f} ± {dd[d]['sd']:.3f} ({dd[d]['n']})" if d in dd else "–"
            extra.append(f"| {b} | {cell('real')} | {cell('real_hard')} | {cell('external')} | {cell('external_hard')} |")
    for key, title in (("paired_ft", "Fine-tuned vs probe with the same training data (FallOrNot)"),
                       ("paired_external_ft", "Fine-tuned vs probe with the same training data (HMDB51)")):
        if res.get(key):
            extra += ["", title, "", "| Comparison | AUROC diff | 95% CI | p |", "|---|---|---|---|"]
            for k, (d_, lo, hi, pv) in res[key].items():
                extra.append(f"| {k} | {d_:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} |")
    if res.get("paired_external"):
        extra += ["", "Head-to-head on the EXTERNAL test set (HMDB51; nothing trained on it)", "",
                  "| Comparison | AUROC diff | 95% CI | p |", "|---|---|---|---|"]
        for k, (d_, lo, hi, pv) in res["paired_external"].items():
            extra.append(f"| {k} | {d_:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {pv:.3f} |")
    if "equal_budget" in res:
        eb = res["equal_budget"]
        extra += ["", f"Equal budget ({eb['representation']}): AUROC on real test falls, k training falls + k non-falls "
                      "from one source (mean ± s.d. over 10 random draws; 3 above 60 falls)", ""]
        ks = sorted({p_["falls"] for pts in eb["points"].values() for p_ in pts})
        extra.append("| source | " + " | ".join(str(k) for k in ks) + " |")
        extra.append("|---|" + "---|" * len(ks))
        for src, pts in eb["points"].items():
            d = {p_["falls"]: f"{p_['auroc']:.3f} ± {p_['sd']:.3f}" for p_ in pts}
            extra.append(f"| {src} | " + " | ".join(d.get(k, "–") for k in ks) + " |")
        m = eb["synthetic_falls_to_match_all_real"]
        extra.append("")
        extra.append(f"Synthetic falls needed to match {eb['real_max_falls']} real falls: "
                     + (str(m) if m else "not reached with the synthetic falls available"))
    table = table + "\n" + "\n".join(extra)
    with open(os.path.join(out, "staged_to_real.md"), "w") as fh:
        fh.write("Counts: " + json.dumps(res["n"]) + "\n\n" + table + "\n")

    # the real-test clips a staged-trained probe gets most wrong (for qualitative analysis)
    with open(os.path.join(out, "worst_errors.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "error", "score", "name", "segment", "label"])
        for k in [f"staged_probe_{r}" for r in reps]:
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
        if "equal_budget" in res:
            eb = res["equal_budget"]
            fig, ax = plt.subplots(figsize=(4.8, 3.6))
            for src, mk in (("real", "o"), ("synthetic", "^"), ("staged", "s")):
                pts = eb["points"].get(src)
                if pts:
                    ax.errorbar([p_["falls"] for p_ in pts], [p_["auroc"] for p_ in pts],
                                yerr=[p_["sd"] for p_ in pts], marker=mk, capsize=3, label=src)
            ax.set_xscale("log"); ax.set_xlabel("training falls (+ equal number of non-falls)")
            ax.set_ylabel("AUROC on real test falls")
            ax.set_title(f"Real vs synthetic vs staged ({eb['representation']})", fontsize=10); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(out, "equal_budget.png"), dpi=200)
        if "learning_curve" in res and res["learning_curve"]["points"]:
            lc = res["learning_curve"]; pts = lc["points"]
            x = [p_["falls"] for p_ in pts]
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            ax.errorbar(x, [p_["real"] for p_ in pts], yerr=[p_["real_sd"] for p_ in pts], marker="o", label="real only", capsize=3)
            ax.errorbar(x, [p_["both"] for p_ in pts], yerr=[p_["both_sd"] for p_ in pts], marker="s", label="staged + real", capsize=3)
            ax.axhline(lc["staged_only"], ls="--", color="grey", label="staged only")
            ax.set_xscale("log"); ax.set_xlabel("real training falls"); ax.set_ylabel("AUROC on real test falls")
            ax.set_title(f"How much real data? ({lc['representation']})", fontsize=10); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(os.path.join(out, "learning_curve.png"), dpi=200)
    except ImportError:
        pass

    print("Counts:", json.dumps(res["n"]))
    print(table)
    print(f"\nwrote {out}/staged_to_real.md, .json, roc_real.png, worst_errors.csv")


if __name__ == "__main__":
    main()
