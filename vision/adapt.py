#!/usr/bin/env python3
"""Anchored probe: adapt a synthetic-trained fall detector with a handful of real clips.

Idea. A probe trained only on a few real clips is unstable; a probe trained on thousands of
synthetic clips is stable but plateaus; pooling the two lets the synthetic clips dominate.
The anchored probe is trained on the real clips alone, but its weights are penalised for
moving away from the synthetic-trained probe's weights instead of from zero:

    minimise   C * sum_i w_i * logloss(y_i, x_i . (w_syn + v) + b_syn + c)  +  0.5 * ||v||^2

With few real clips it stays near the synthetic probe; with many it moves to the real-data
solution. C is chosen by video-grouped cross-validation on the real training clips only.

The success rule was fixed in AGENTS.md before this script produced any result:
on both FallOrNot and the OF-ItW test split, (i) at every budget k the anchored probe's mean
AUROC is at least max(real-only mean, synthetic-only) - 0.005, and (ii) averaged over the
budgets it is higher than both baselines.

No GPU needed: uses embeddings already saved by the other scripts.
  python vision/adapt.py --oops OOPS_DIR --synthetic ofsyn.npz --ofitw ofitw.npz [--external hmdb.npz]
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiment import _auroc, compilation, featurize, paired, probe  # noqa: E402
from labeler import load_labels  # noqa: E402
from oops_mine import load_frames  # noqa: E402

KIND = "clip+videomae"
BUDGETS = (4, 8, 15, 30, 60, 125)
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
C_FALLBACK = 0.01
DRAWS = 10
TOL = 0.005


class Source:
    """Synthetic-trained probe: standardisation and weights that the anchored probe starts from."""

    def __init__(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        lr = LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced").fit(self.z(X), y)
        self.w, self.b = lr.coef_[0].astype(np.float64), float(lr.intercept_[0])

    def z(self, X):
        return (X - self.mu) / self.sd

    def score(self, X):
        return self.z(X) @ self.w + self.b


def anchored_scores(src, Xr, yr, Xte, C):
    """Fit the offset v, c on real clips (class-balanced) and return test logits."""
    from scipy.optimize import minimize
    Z = src.z(Xr).astype(np.float64)
    base = Z @ src.w + src.b
    y = yr.astype(np.float64)
    wts = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum())) * len(y)

    def f(p):
        v, c = p[:-1], p[-1]
        m = base + Z @ v + c
        loss = np.logaddexp(0, m) - y * m
        g = wts * (1 / (1 + np.exp(-m)) - y)
        return C * float((wts * loss).sum()) + 0.5 * float(v @ v), np.r_[C * (Z.T @ g) + v, C * g.sum()]

    r = minimize(f, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return src.z(Xte) @ (src.w + r.x[:-1]) + src.b + r.x[-1]


def choose_C(src, Xr, yr, groups):
    """Video-grouped cross-validation on the real training clips only."""
    from sklearn.model_selection import StratifiedGroupKFold
    k = min(5, len(set(groups[yr == 1])), len(set(groups[yr == 0])))
    if k < 2:
        return C_FALLBACK
    best, best_auc = C_FALLBACK, -1.0
    folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=0).split(Xr, yr, groups))
    for C in C_GRID:
        oof = np.full(len(yr), np.nan)
        for tr, te in folds:
            if len(set(yr[tr])) < 2:
                continue
            oof[te] = anchored_scores(src, Xr[tr], yr[tr], Xr[te], C)
        ok = ~np.isnan(oof)
        if ok.sum() and len(set(yr[ok])) == 2:
            auc = _auroc(yr[ok], oof[ok])
            if auc > best_auc + 1e-9:
                best, best_auc = C, auc
    return best


def run_target(name, src, syn_auc_fn, Xr, yr, gr, Xte, yte, rng):
    """Learning curve over budgets for one test set. Returns rows and the rule's verdict."""
    pos, neg = np.where(yr == 1)[0], np.where(yr == 0)[0]
    syn_auc = syn_auc_fn()
    rows = []
    for k in BUDGETS:
        if k > len(pos) or k > len(neg):
            continue
        a_, r_, cs = [], [], []
        for _ in range(DRAWS):
            idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]
            X, y, g = Xr[idx], yr[idx], gr[idx]
            C = choose_C(src, X, y, g); cs.append(C)
            a_.append(_auroc(yte, anchored_scores(src, X, y, Xte, C)))
            r_.append(_auroc(yte, probe().fit(X, y).predict_proba(Xte)[:, 1]))
        rows.append(dict(k=k, anchored=float(np.mean(a_)), anchored_sd=float(np.std(a_)), real=float(np.mean(r_)),
                         real_sd=float(np.std(r_)), synthetic=float(syn_auc), median_C=float(np.median(cs))))
        print(f"  {name} k={k}: anchored {rows[-1]['anchored']:.3f}  real-only {rows[-1]['real']:.3f}  "
              f"synthetic-only {syn_auc:.3f}", flush=True)
    every = all(r["anchored"] >= max(r["real"], r["synthetic"]) - TOL for r in rows)
    mean_a, mean_r = np.mean([r["anchored"] for r in rows]), np.mean([r["real"] for r in rows])
    avg = bool(mean_a > mean_r and mean_a > syn_auc)
    return rows, dict(every_budget=bool(every), on_average=avg, mean_anchored=float(mean_a), mean_real=float(mean_r),
                      synthetic=float(syn_auc), met=bool(every and avg), budgets_tested=[r["k"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--ofitw", required=True)
    ap.add_argument("--external", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.oops, "results_adapt")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    Y = np.load(a.synthetic)
    src = Source(featurize(Y["frames"], KIND, Y["video"]), Y["label"])

    # FallOrNot: our labelled training pool -> our labelled test sample
    labels = load_labels(a.oops)
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    frames, videos = load_frames(a.oops), load_frames(a.oops, "video")
    tr, te = [], []
    for (name, seg), fr in frames.items():
        lab = labels.get(name)
        if lab not in ("fall", "not_fall") or name not in meta or (name, seg) not in videos:
            continue
        item = (fr, videos[(name, seg)], int(seg == "event" and lab == "fall"), name)
        (te if meta[name]["random_sample"] == "1" else tr).append(item)
    pack = lambda L: (featurize(np.array([t[0] for t in L]), KIND, np.array([t[1] for t in L])),
                      np.array([t[2] for t in L]), np.array([t[3] for t in L]))
    Xr, yr, gr = pack(tr); Xt, yt, gt = pack(te)

    # OF-ItW: its train + val split -> its test split
    O = np.load(a.ofitw)
    Xo = featurize(O["frames"], KIND, O["video"]); yo = O["label"]; no = O["name"]
    ote = O["split"] == "test"

    res = {"rule": "on both test sets: (i) at every budget anchored mean AUROC >= max(real-only mean, synthetic-only) - 0.005; "
                   "(ii) averaged over budgets anchored > both baselines",
           "representation": KIND, "budgets": list(BUDGETS), "draws": DRAWS, "C_grid": list(C_GRID)}
    print("FallOrNot")
    res["fallornot"], v1 = run_target("FallOrNot", src, lambda: _auroc(yt, src.score(Xt)), Xr, yr, gr, Xt, yt, rng)
    print("OF-ItW")
    res["ofitw"], v2 = run_target("OF-ItW", src, lambda: _auroc(yo[ote], src.score(Xo[ote])),
                                  Xo[~ote], yo[~ote], no[~ote], Xo[ote], yo[ote], rng)
    res["verdict"] = {"fallornot": v1, "ofitw": v2, "success": bool(v1["met"] and v2["met"])}

    # full real pool: paired tests against both baselines (by compilation)
    full = {}
    for name, (Xa, ya, ga, Xb, yb, gb_) in {"fallornot": (Xr, yr, gr, Xt, yt, gt),
                                           "ofitw": (Xo[~ote], yo[~ote], no[~ote], Xo[ote], yo[ote], no[ote])}.items():
        C = choose_C(src, Xa, ya, ga)
        sa = anchored_scores(src, Xa, ya, Xb, C)
        sr = probe().fit(Xa, ya).predict_proba(Xb)[:, 1]
        comp = np.array([compilation(n) for n in gb_])
        full[name] = dict(C=C, anchored=float(_auroc(yb, sa)), real=float(_auroc(yb, sr)), synthetic=float(_auroc(yb, src.score(Xb))),
                          anchored_vs_real=paired(yb, sa, sr, comp), anchored_vs_synthetic=paired(yb, sa, src.score(Xb), comp))
    if a.external:   # secondary: does anchoring on OOPS! real clips keep out-of-domain performance?
        E = np.load(a.external)
        Xe = featurize(E["frames"], KIND, E["video"]); wc = E["win_clip"]; ye = E["clip_label"]
        cmax = lambda w: (lambda o: (np.maximum.at(o, wc, w), o)[1])(np.full(len(ye), -np.inf))
        C = full["fallornot"]["C"]
        sa = cmax(anchored_scores(src, Xr, yr, Xe, C)); sr = cmax(probe().fit(Xr, yr).predict_proba(Xe)[:, 1]); ss = cmax(src.score(Xe))
        full["hmdb51"] = dict(anchored=float(_auroc(ye, sa)), real=float(_auroc(ye, sr)), synthetic=float(_auroc(ye, ss)),
                              anchored_vs_real=paired(ye, sa, sr, E["clip_name"]), anchored_vs_synthetic=paired(ye, sa, ss, E["clip_name"]))
    res["full_pool"] = full

    with open(os.path.join(out, "adapt.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    L = ["ANCHORED PROBE (" + KIND + ")", "Rule: " + res["rule"], ""]
    for name in ("fallornot", "ofitw"):
        L += [f"{name}: AUROC, mean ± s.d. over {DRAWS} draws of k real falls + k real non-falls", "",
              "| k | anchored | real-only | synthetic-only | median C |", "|---|---|---|---|---|"]
        for r in res[name]:
            L.append(f"| {r['k']} | {r['anchored']:.3f} ± {r['anchored_sd']:.3f} | {r['real']:.3f} ± {r['real_sd']:.3f} | "
                     f"{r['synthetic']:.3f} | {r['median_C']:g} |")
        v = res["verdict"][name]
        L += ["", f"- every budget within {TOL} of the better baseline: {v['every_budget']}",
              f"- mean over budgets: anchored {v['mean_anchored']:.3f}, real-only {v['mean_real']:.3f}, synthetic-only {v['synthetic']:.3f}"
              f" -> higher than both: {v['on_average']}", f"- rule met on {name}: {v['met']}", ""]
    L += [f"METHOD RESULT: {'SUCCESS' if res['verdict']['success'] else 'NOT MET'}", "",
          "Full real pool (paired tests): ", ""]
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], p = {v[3]:.3f}"
    for name, d in full.items():
        L.append(f"- {name}: anchored {d['anchored']:.3f}, real-only {d['real']:.3f}, synthetic-only {d['synthetic']:.3f}; "
                 f"anchored vs real {g(d['anchored_vs_real'])}; anchored vs synthetic {g(d['anchored_vs_synthetic'])}")
    text = "\n".join(L)
    open(os.path.join(out, "adapt.md"), "w").write(text + "\n")
    print("\n" + text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=False)
        for ax, name in zip(axes, ("fallornot", "ofitw")):
            rows = res[name]; x = [r["k"] for r in rows]
            ax.errorbar(x, [r["anchored"] for r in rows], yerr=[r["anchored_sd"] for r in rows], marker="D", label="anchored", capsize=2)
            ax.errorbar(x, [r["real"] for r in rows], yerr=[r["real_sd"] for r in rows], marker="o", label="real only", capsize=2)
            ax.axhline(rows[0]["synthetic"], ls="--", color="grey", label="synthetic only")
            ax.set_xscale("log"); ax.set_xlabel("real training falls"); ax.set_title(name, fontsize=10)
        axes[0].set_ylabel("AUROC"); axes[0].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out, "adapt.png"), dpi=200)
    except ImportError:
        pass
    print(f"\nwrote {out}/adapt.md, adapt.json, adapt.png")


if __name__ == "__main__":
    main()
