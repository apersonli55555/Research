#!/usr/bin/env python3
"""Zero-label adaptation: how far can UNLABELLED real accident video take a synthetic-trained detector?

OOPS! gives, for every video, the time at which something goes wrong. That is free structure:
  * the clip BEFORE the accident shows ordinary activity  -> a real non-fall, no human label needed
  * the clip AT the accident is a fall or some other accident -> unknown
We use no human label at all:
  round 0   score every unlabelled accident clip with the synthetic-trained probe
  each round  the top Q of accident clips become pseudo-falls; the bottom Q of accident clips and
              all pre-accident clips become pseudo-non-falls; a fresh probe is trained on these
              real clips only and re-scores the accident clips
The final probe has seen thousands of real clips and zero human labels.

Everything below was fixed in AGENTS.md before this script produced any result on real data:
Q, the number of rounds, the features, the probe, and the rule
  on BOTH FallOrNot and the OF-ItW test split, the zero-label probe beats the synthetic-only probe
  with a paired 95% interval above zero (resampled by compilation).
Every video of either test set is removed from the unlabelled pool first.

Also reported (not part of the rule): how many hand-labelled real falls the result is worth (read
off a real-only learning curve), ablations, a pool without any compilation that appears in a test
set, and HMDB51.

No GPU: uses embeddings already on Drive.
  python vision/nolabel.py --oops OOPS_DIR --synthetic ofsyn.npz --ofitw ofitw.npz [--external hmdb.npz]
"""
import argparse, csv, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiment import _auroc, compilation, featurize, paired, probe  # noqa: E402
from labeler import load_labels  # noqa: E402
from oops_mine import load_frames  # noqa: E402

KIND = "clip+videomae"
Q = 0.2            # fraction of accident clips pseudo-labelled at each end
ROUNDS = 3
BUDGETS = (4, 8, 15, 30, 60, 125)
DRAWS = 10


def self_train(score0, Xev, Xpre, rounds=ROUNDS, q=Q, use_pre=True, use_low=True, keep=None):
    """Returns the final probe (trained on pseudo-labelled real clips only) and per-round sizes.
    keep: optional list that receives (probe, indices of the pseudo-falls) for every round."""
    s, model, log = score0, None, []
    n = max(2, int(round(q * len(Xev))))
    for _ in range(rounds):
        order = np.argsort(-s)
        pos = Xev[order[:n]]
        neg = [Xev[order[-n:]]] if use_low else []
        if use_pre:
            neg.append(Xpre)
        neg = np.concatenate(neg)
        model = probe().fit(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
        s = model.decision_function(Xev)
        if keep is not None:
            keep.append((model, order[:n]))
        log.append(dict(pseudo_falls=int(len(pos)), pseudo_non_falls=int(len(neg))))
    return model, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True)
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--ofitw", required=True)
    ap.add_argument("--external", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--explore", action="store_true",
                    help="exploratory sensitivity analysis (run 16); never changes the run-15 setting or verdict")
    a = ap.parse_args()
    out = a.out or os.path.join(a.oops, "results_nolabel")
    os.makedirs(out, exist_ok=True)

    Y = np.load(a.synthetic)
    Xs, ys = featurize(Y["frames"], KIND, Y["video"]), Y["label"]
    syn = probe().fit(Xs, ys)

    labels = load_labels(a.oops)
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    frames, videos = load_frames(a.oops), load_frames(a.oops, "video")
    O = np.load(a.ofitw)
    ofitw_videos = set(json.loads(str(O["video_split"])))             # every OF-ItW video, any split
    test_names = {n for n, r in meta.items() if r["random_sample"] == "1"}
    blocked = test_names | ofitw_videos
    test_comps = {compilation(n) for n in blocked}

    def pack(keys):
        return featurize(np.array([frames[k] for k in keys]), KIND, np.array([videos[k] for k in keys]))

    # ---- unlabelled pool: every embedded OOPS! video that is in neither test set (labels never read)
    pool = sorted(k for k in frames if k in videos and k[0] in meta and k[0] not in blocked)
    ev_keys = [k for k in pool if k[1] == "event"]; pre_keys = [k for k in pool if k[1] == "pre"]
    Xev, Xpre = pack(ev_keys), pack(pre_keys)
    explore.pre_names = np.array([k[0] for k in pre_keys])
    disjoint_ev = np.array([compilation(k[0]) not in test_comps for k in ev_keys])
    disjoint_pre = np.array([compilation(k[0]) not in test_comps for k in pre_keys])
    print(f"unlabelled pool: {len(ev_keys)} accident clips + {len(pre_keys)} pre-accident clips from "
          f"{len({k[0] for k in pool})} videos; {int(disjoint_ev.sum())} accident clips from compilations absent from both test sets",
          flush=True)

    # ---- test sets
    te = [(k, int(k[1] == "event" and labels[k[0]] == "fall")) for k in sorted(frames)
          if k in videos and k[0] in test_names and labels.get(k[0]) in ("fall", "not_fall")]
    Xt, yt = pack([k for k, _ in te]), np.array([l for _, l in te])
    gt = np.array([compilation(k[0]) for k, _ in te])
    ote = O["split"] == "test"
    Xo, yo = featurize(O["frames"][ote], KIND, O["video"][ote]), O["label"][ote]
    go = np.array([compilation(n) for n in O["name"][ote]])
    tests = {"fallornot": (Xt, yt, gt), "ofitw": (Xo, yo, go)}
    if a.external:
        E = np.load(a.external)
        Xe, wc, ye = featurize(E["frames"], KIND, E["video"]), E["win_clip"], E["clip_label"]
        cmax = lambda w: (lambda o: (np.maximum.at(o, wc, w), o)[1])(np.full(len(ye), -np.inf))

    # ---- methods (none of them reads a human label)
    s0 = syn.decision_function(Xev)
    main_model, log = self_train(s0, Xev, Xpre)
    models = {
        "synthetic_only": syn,
        "zero_label (ours)": main_model,
        "ablation: 1 round": self_train(s0, Xev, Xpre, rounds=1)[0],
        "ablation: no pre-accident clips": self_train(s0, Xev, Xpre, use_pre=False)[0],
        "ablation: pre-accident clips as the only negatives": self_train(s0, Xev, Xpre, use_low=False)[0],
        "ablation: synthetic + pre-accident negatives, no pseudo-labels":
            probe().fit(np.concatenate([Xs, Xpre]), np.r_[ys, np.zeros(len(Xpre))]),
        "robustness: pool without test-set compilations":
            self_train(s0[disjoint_ev], Xev[disjoint_ev], Xpre[disjoint_pre])[0],
    }

    # ---- references that DO use human labels (for comparison only)
    tr = [(k, int(k[1] == "event" and labels[k[0]] == "fall")) for k in sorted(frames)
          if k in videos and k[0] in meta and k[0] not in test_names and labels.get(k[0]) in ("fall", "not_fall")]
    Xr, yr = pack([k for k, _ in tr]), np.array([l for _, l in tr])
    real_full = probe().fit(Xr, yr)
    otr = ~ote
    Xor, yor = featurize(O["frames"][otr], KIND, O["video"][otr]), O["label"][otr]
    refs = {"fallornot": real_full, "ofitw": probe().fit(Xor, yor)}
    rng = np.random.default_rng(0)
    curve = {}
    for name, (Xa, ya) in {"fallornot": (Xr, yr), "ofitw": (Xor, yor)}.items():
        pos, neg = np.where(ya == 1)[0], np.where(ya == 0)[0]
        Xb, yb, _ = tests[name]
        curve[name] = {}
        for k in BUDGETS:
            if k > len(pos) or k > len(neg):
                continue
            v = []
            for _ in range(DRAWS):
                idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]
                v.append(_auroc(yb, probe().fit(Xa[idx], ya[idx]).decision_function(Xb)))
            curve[name][k] = float(np.mean(v))

    # ---- evaluate
    res = {"rule": "on both FallOrNot and OF-ItW test: zero-label probe beats synthetic-only, paired 95% CI above zero (by compilation)",
           "Q": Q, "rounds": ROUNDS, "representation": KIND, "rounds_log": log,
           "pool": dict(accident_clips=len(ev_keys), pre_clips=len(pre_keys), compilation_disjoint_accident_clips=int(disjoint_ev.sum())),
           "auroc": {}, "tests": {}, "label_equivalent": {}, "learning_curve": curve}
    for name, (Xb, yb, gb) in tests.items():
        sc = {m: mod.decision_function(Xb) for m, mod in models.items()}
        res["auroc"][name] = {m: float(_auroc(yb, s)) for m, s in sc.items()}
        ref = refs[name].decision_function(Xb)
        res["auroc"][name]["reference: real probe trained on human labels (full pool)"] = float(_auroc(yb, ref))
        ours, base = sc["zero_label (ours)"], sc["synthetic_only"]
        res["tests"][name] = dict(ours_vs_synthetic=paired(yb, ours, base, gb), ours_vs_labelled_real=paired(yb, ours, ref, gb),
                                  disjoint_pool_vs_synthetic=paired(yb, sc["robustness: pool without test-set compilations"], base, gb))
        a_ours, a_syn, a_ref = res["auroc"][name]["zero_label (ours)"], res["auroc"][name]["synthetic_only"], float(_auroc(yb, ref))
        eq = [k for k, v in curve[name].items() if v >= a_ours]
        res["label_equivalent"][name] = dict(
            falls=(min(eq) if eq else None), gap_closed=float((a_ours - a_syn) / (a_ref - a_syn)) if a_ref != a_syn else None)
    if a.external:
        res["auroc"]["hmdb51"] = {m: float(_auroc(ye, cmax(mod.decision_function(Xe)))) for m, mod in models.items()}
        res["auroc"]["hmdb51"]["reference: real probe trained on human labels (full pool)"] = float(_auroc(ye, cmax(real_full.decision_function(Xe))))
        res["tests"]["hmdb51"] = dict(ours_vs_synthetic=paired(ye, cmax(models["zero_label (ours)"].decision_function(Xe)),
                                                               cmax(syn.decision_function(Xe)), E["clip_name"]))
    met = {n: bool(res["tests"][n]["ours_vs_synthetic"][1] > 0) for n in ("fallornot", "ofitw")}
    res["verdict"] = dict(met, success=bool(all(met.values())))
    with open(os.path.join(out, "nolabel.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], p = {v[3]:.3f}"
    cols = list(res["auroc"])
    L = ["ZERO-LABEL ADAPTATION (" + KIND + ")", "Rule: " + res["rule"], "",
         f"Unlabelled pool: {len(ev_keys)} accident clips + {len(pre_keys)} pre-accident clips; no human label is read.",
         "Pseudo-labels per round: " + "; ".join(f"{r['pseudo_falls']} falls / {r['pseudo_non_falls']} non-falls" for r in log), "",
         "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in res["auroc"][cols[0]]:
        L.append(f"| {m} | " + " | ".join(f"{res['auroc'][c][m]:.3f}" for c in cols) + " |")
    L.append("")
    for name in ("fallornot", "ofitw"):
        t, e = res["tests"][name], res["label_equivalent"][name]
        L += [f"{name}:", f"- zero-label vs synthetic-only: {g(t['ours_vs_synthetic'])} -> rule met: {met[name]}",
              f"- zero-label vs real probe trained on human labels: {g(t['ours_vs_labelled_real'])}",
              f"- compilation-disjoint pool vs synthetic-only: {g(t['disjoint_pool_vs_synthetic'])}",
              f"- share of the synthetic-to-labelled-real gap closed without labels: {e['gap_closed']:.2f}" if e["gap_closed"] is not None else "- gap closed: n/a",
              "- real-only learning curve (k labelled falls + k non-falls, mean of 10 draws): "
              + ", ".join(f"{k}: {v:.3f}" for k, v in curve[name].items()),
              f"- label equivalent: smallest k on that curve that reaches the zero-label AUROC: "
              + (str(e["falls"]) if e["falls"] else f"more than {max(curve[name])}"), ""]
    if a.external:
        L += [f"hmdb51 (secondary): zero-label vs synthetic-only: {g(res['tests']['hmdb51']['ours_vs_synthetic'])}", ""]
    L.append(f"ZERO-LABEL RESULT: {'SUCCESS' if res['verdict']['success'] else 'NOT MET'}")
    text = "\n".join(L)
    open(os.path.join(out, "nolabel.md"), "w").write(text + "\n")
    print("\n" + text + f"\n\nwrote {out}/nolabel.md, nolabel.json")
    if a.explore:
        explore(out, s0, Xev, Xpre, ev_keys, meta, labels, tests, (Xe, ye, cmax) if a.external else None)


def explore(out, s0, Xev, Xpre, ev_keys, meta, labels, tests, ext):
    """EXPLORATORY (run 16): how sensitive is the zero-label result to its fixed choices?
    Nothing here is a pass/fail test, and the run-15 setting (20%, 3 rounds, synthetic start) stays the headline."""
    def ev(model):
        r = {n: float(_auroc(yb, model.decision_function(Xb))) for n, (Xb, yb, _) in tests.items()}
        if ext:
            r["hmdb51"] = float(_auroc(ext[1], ext[2](model.decision_function(ext[0]))))
        return r
    rows = []
    add = lambda group, setting, r: (rows.append(dict(group=group, setting=setting, **r)), print(f"  {group} | {setting} | {r}", flush=True))
    # human labels are used below ONLY to measure how clean the pseudo-labels are
    lab = np.array([{"fall": 1, "not_fall": 0}.get(labels.get(k[0]), -1) for k in ev_keys])
    kept = []
    self_train(s0, Xev, Xpre, rounds=5, keep=kept)
    for i, (m, top) in enumerate(kept, 1):
        known = lab[top][lab[top] >= 0]
        r = ev(m); r["pseudo_fall_precision"] = float(known.mean()) if len(known) else None; r["pseudo_falls_with_human_label"] = int(len(known))
        add("rounds (20%)", str(i), r)
    for q in (0.05, 0.1, 0.3, 0.4):
        add("fraction pseudo-labelled (3 rounds)", f"{int(q * 100)}%", ev(self_train(s0, Xev, Xpre, q=q)[0]))
    rng = np.random.default_rng(0)
    names = np.array([k[0] for k in ev_keys])
    for frac in (0.02, 0.05, 0.1, 0.25, 0.5):
        acc = []
        for _ in range(3):
            pick = rng.random(len(ev_keys)) < frac
            # pre-accident clips of the picked videos only: Xpre rows are aligned by video name
            acc.append(ev(self_train(s0[pick], Xev[pick], Xpre[np.isin(explore.pre_names, names[pick])])[0]))
        add("pool size (3 random subsets)", f"{int(frac * 100)}% (about {int(frac * len(ev_keys))} videos)",
            {k: float(np.mean([x[k] for x in acc])) for k in acc[0]})
    zs = np.array([float(meta[k[0]]["event_margin"]) for k in ev_keys])
    add("starting scorer", "CLIP zero-shot text prompts instead of the synthetic probe (no synthetic data at all)", ev(self_train(zs, Xev, Xpre)[0]))
    add("starting scorer", "random scores (control)", ev(self_train(np.random.default_rng(1).random(len(ev_keys)), Xev, Xpre)[0]))
    json.dump(rows, open(os.path.join(out, "nolabel_explore.json"), "w"), indent=1)
    cols = [c for c in ("fallornot", "ofitw", "hmdb51") if c in rows[0]]
    L = ["", "EXPLORATORY SENSITIVITY ANALYSIS (not a pass/fail test; the fixed setting is 20%, 3 rounds, synthetic start)", "",
         "| What is varied | Setting | " + " | ".join(cols) + " | pseudo-fall precision (on clips that happen to have a human label) |", "|---|---|" + "---|" * (len(cols) + 1)]
    for r in rows:
        pr = f"{r['pseudo_fall_precision']:.2f} (n = {r['pseudo_falls_with_human_label']})" if r.get("pseudo_fall_precision") is not None else ""
        L.append(f"| {r['group']} | {r['setting']} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + f" | {pr} |")
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_explore.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_explore.md, nolabel_explore.json")


if __name__ == "__main__":
    main()
