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
    ap.add_argument("--hard", action="store_true",
                    help="hard-subset analysis (run 17): falls vs other accidents only; reading fixed in AGENTS.md beforehand")
    ap.add_argument("--hardneg", default="",
                    help="VLM accident-prompt JSON on the pool: shortcut-aware self-training with mined non-fall accidents (run 29)")
    ap.add_argument("--teacher", default="", help="vlm_teacher.py JSON: VLM-taught self-training + shortcut audit (runs 26-27)")
    ap.add_argument("--vlm-dir", default="", help="finetune work dir holding vlm_<model>_*.json (the teacher's own test scores)")
    ap.add_argument("--teacher-model", default="InternVL3_5-8B-HF")
    ap.add_argument("--kinetics", default="", help="kinetics.npz (run 26)")
    ap.add_argument("--annotators", default="",
                    help="second annotator's labels CSV: grade the main models with both annotators on the same clips (run 25)")
    ap.add_argument("--ft-work", default="", help="finetune.py work dir, to include the fine-tuned models in --annotators")
    ap.add_argument("--windows", default="",
                    help="window_prep.py output dir: timestamp-free zero-label adaptation (run 23; rule in AGENTS.md)")
    ap.add_argument("--post", default="",
                    help="post_prep.py output dir: consequence-guided zero-label adaptation (run 21; rule in AGENTS.md)")
    ap.add_argument("--dupes", action="store_true",
                    help="near-duplicate check (run 20): remove pool clips that look like a test clip and retrain")
    ap.add_argument("--hard2", action="store_true",
                    help="extended hard test (run 18): every human-labelled accident clip, with those videos removed from the pool")
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
    if a.hard:
        # ---- falls vs OTHER ACCIDENTS only: accident-moment clips of FallOrNot; HMDB51 hard subset
        evm = np.array([k[1] == "event" for k, _ in te])
        hard = {"fallornot_hard": (Xt[evm], yt[evm], gt[evm], None)}
        if a.external:
            hm = E["clip_hard"].astype(bool) | (ye == 1)
            hard["hmdb51_hard"] = (Xe, ye, E["clip_name"], hm)
        zs = np.array([float(meta[k[0]]["event_margin"]) for k in ev_keys])
        cand = {"synthetic_only": [syn], "zero_label (ours)": [main_model],
                "start from CLIP zero-shot prompts (no synthetic data)": [self_train(zs, Xev, Xpre)[0]],
                "start from random scores (control, 5 seeds)":
                    [self_train(np.random.default_rng(sd).random(len(ev_keys)), Xev, Xpre)[0] for sd in range(1, 6)],
                "accident detector: all accident clips vs pre-accident clips, no pseudo-labels":
                    [probe().fit(np.concatenate([Xev, Xpre]), np.r_[np.ones(len(Xev)), np.zeros(len(Xpre))])],
                "reference: real probe trained on human labels (full pool)": [real_full]}
        H = {"auroc": {}, "tests": {}}
        sc_all = {}
        for name, (Xb, yb, gb, mask) in hard.items():
            def score(m):
                v = m.decision_function(Xb)
                if mask is not None:
                    v = cmax(v)[mask]
                return v
            yy, gg = (yb[mask], gb[mask]) if mask is not None else (yb, gb)
            sc_all[name] = {m: [score(x) for x in ms] for m, ms in cand.items()}
            H["auroc"][name] = {m: [float(_auroc(yy, v)) for v in vs] for m, vs in sc_all[name].items()}
            o = sc_all[name]["zero_label (ours)"][0]
            H["tests"][name] = {
                "ours vs synthetic-only": paired(yy, o, sc_all[name]["synthetic_only"][0], gg),
                "ours vs random-start control (seed 1)": paired(yy, o, sc_all[name]["start from random scores (control, 5 seeds)"][0], gg),
                "ours vs accident detector": paired(yy, o, sc_all[name]["accident detector: all accident clips vs pre-accident clips, no pseudo-labels"][0], gg),
                "ours vs human-labelled real": paired(yy, o, sc_all[name]["reference: real probe trained on human labels (full pool)"][0], gg),
                "accident detector vs synthetic-only": paired(yy, sc_all[name]["accident detector: all accident clips vs pre-accident clips, no pseudo-labels"][0],
                                                              sc_all[name]["synthetic_only"][0], gg)}
        # the same candidates on the full test sets, for the control rows
        full = {n: {m: [float(_auroc(yb, x.decision_function(Xb))) for x in ms] for m, ms in cand.items()} for n, (Xb, yb, _) in tests.items()}
        if a.external:
            full["hmdb51"] = {m: [float(_auroc(ye, cmax(x.decision_function(Xe)))) for x in ms] for m, ms in cand.items()}
        H["full"] = full
        json.dump(H, open(os.path.join(out, "nolabel_hard.json"), "w"), indent=1)
        fmt = lambda v: f"{v[0]:.3f}" if len(v) == 1 else f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        cols = list(full) + list(H["auroc"])
        L = ["", "HARD-SUBSET ANALYSIS: falls vs other accidents only (fallornot_hard = accident-moment clips; hmdb51_hard = falls vs fall-like classes)", "",
             "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for m in cand:
            L.append(f"| {m} | " + " | ".join(fmt((full.get(c) or H["auroc"][c])[m]) for c in cols) + " |")
        for name in H["tests"]:
            L += ["", f"{name} (paired, AUROC difference [95% CI], p):"] + [f"- {k}: {g(v)}" for k, v in H["tests"][name].items()]
        text = "\n".join(L)
        open(os.path.join(out, "nolabel_hard.md"), "w").write(text + "\n")
        print(text + f"\n\nwrote {out}/nolabel_hard.md, nolabel_hard.json")
    if a.hardneg:
        shortcut_aware(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack,
                       (Xe, ye, cmax, E) if a.external else None)
    if a.teacher:
        teacher_audit(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, real_full, pack,
                      (Xe, ye, cmax, E) if a.external else None)
    if a.annotators:
        annotators(a, out, frames, videos, labels, test_names, pack, {"synthetic-only probe": syn, "hand-labelled real probe": real_full,
                                                                        "self-trained probe (ours)": main_model})
    if a.windows:
        timestamp_free(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack, tests,
                       (Xe, ye, cmax) if a.external else None)
    if a.post:
        consequence(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack, tests,
                    (Xe, ye, cmax, E) if a.external else None)
    if a.dupes:
        # ---- NEAR-DUPLICATE CHECK (reading fixed in AGENTS.md before running). The same web clip can appear in more
        # than one compilation, which the compilation-disjoint pool does not catch. Similarity = cosine between
        # mean CLIP frame embeddings (appearance only). Pool videos with ANY clip above the threshold to ANY test
        # clip (FallOrNot test, OF-ItW test) are removed, and the fixed zero-label pipeline is run again.
        unit = lambda M: (lambda v: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8))(np.asarray(M, np.float32).mean(1))
        P = unit(np.array([frames[k] for k in pool]))
        Tk = [k for k, _ in te]
        T = np.concatenate([unit(np.array([frames[k] for k in Tk])), unit(O["frames"][ote])])
        mx = np.zeros(len(pool), np.float32)
        for i in range(0, len(P), 2000):
            mx[i:i + 2000] = (P[i:i + 2000] @ T.T).max(1)
        pool_names = np.array([k[0] for k in pool]); ev_names = np.array([k[0] for k in ev_keys]); pr_names = np.array([k[0] for k in pre_keys])
        D = {"similarity_quantiles": {str(q): float(np.quantile(mx, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}, "thresholds": {}}
        L = ["", "NEAR-DUPLICATE CHECK: pool clips most similar to any test clip (cosine of mean CLIP frame embeddings)",
             "similarity quantiles over pool clips: " + ", ".join(f"{q}: {v:.3f}" for q, v in D["similarity_quantiles"].items()), "",
             "| pool after removing videos with similarity above | accident clips left | " + " | ".join(tests) + (" | hmdb51 |" if a.external else " |"),
             "|---|---|" + "---|" * (len(tests) + (1 if a.external else 0))]
        for thr in (None, 0.98, 0.95, 0.92, 0.90):
            bad = set(pool_names[mx > thr]) if thr is not None else set()
            ke = np.array([n not in bad for n in ev_names]); kp = np.array([n not in bad for n in pr_names])
            m = self_train(s0[ke], Xev[ke], Xpre[kp])[0]
            r = {n: float(_auroc(yb, m.decision_function(Xb))) for n, (Xb, yb, _) in tests.items()}
            pt = {n: paired(yb, m.decision_function(Xb), syn.decision_function(Xb), gb) for n, (Xb, yb, gb) in tests.items()}
            if a.external:
                r["hmdb51"] = float(_auroc(ye, cmax(m.decision_function(Xe))))
            D["thresholds"][str(thr)] = dict(videos_removed=len(bad), accident_clips=int(ke.sum()), auroc=r, vs_synthetic=pt)
            L.append(f"| {'nothing removed' if thr is None else thr} | {int(ke.sum())} | " + " | ".join(f"{v:.3f}" for v in r.values()) + " |")
            print(L[-1], flush=True)
        L.append("")
        for thr, d in D["thresholds"].items():
            L.append(f"threshold {thr}: " + "; ".join(f"{n} vs synthetic-only {g(v)}" for n, v in d["vs_synthetic"].items()))
        json.dump(D, open(os.path.join(out, "nolabel_dupes.json"), "w"), indent=1)
        text = "\n".join(L)
        open(os.path.join(out, "nolabel_dupes.md"), "w").write(text + "\n")
        print(text + f"\n\nwrote {out}/nolabel_dupes.md, nolabel_dupes.json")
    if a.hard2:
        # ---- EXTENDED HARD TEST (rule fixed in AGENTS.md before running). Every accident-moment clip that has a human
        # label (FallOrNot test sample + our labelled training pool) is used as test data, and ALL of those videos are
        # removed from the unlabelled pool, so the zero-label probe never sees them, not even without their labels.
        lab_names = {n for n, l in labels.items() if l in ("fall", "not_fall")}
        keep_ev = np.array([k[0] not in lab_names for k in ev_keys]); keep_pre = np.array([k[0] not in lab_names for k in pre_keys])
        Xev2, Xpre2 = Xev[keep_ev], Xpre[keep_pre]
        s02 = s0[keep_ev]
        zs2 = np.array([float(meta[k[0]]["event_margin"]) for k in ev_keys])[keep_ev]
        cand = {"synthetic_only": [syn], "zero_label (fixed setting, labelled videos removed from pool)": [self_train(s02, Xev2, Xpre2)[0]],
                "start from CLIP zero-shot prompts": [self_train(zs2, Xev2, Xpre2)[0]],
                "start from random scores (5 seeds)": [self_train(np.random.default_rng(sd).random(len(s02)), Xev2, Xpre2)[0] for sd in range(1, 6)],
                "accident detector (no pseudo-labels)": [probe().fit(np.concatenate([Xev2, Xpre2]), np.r_[np.ones(len(Xev2)), np.zeros(len(Xpre2))])]}
        hk = [k for k in sorted(frames) if k in videos and k[1] == "event" and k[0] in meta and k[0] in lab_names]
        Xh = pack(hk); yh = np.array([int(labels[k[0]] == "fall") for k in hk])
        gh = np.array([compilation(k[0]) for k in hk]); is_test = np.array([k[0] in test_names for k in hk])
        subsets = {"A: FallOrNot hard (random sample)": is_test, "B: labelled training pool, accident clips (selected by CLIP zero-shot score)": ~is_test,
                   "C: A + B (primary)": np.ones(len(hk), bool)}
        sc = {m: [x.decision_function(Xh) for x in ms] for m, ms in cand.items()}
        H2 = {"pool": dict(accident_clips=int(keep_ev.sum()), pre_clips=int(keep_pre.sum())), "n": {}, "auroc": {}, "tests": {}}
        ours_name = "zero_label (fixed setting, labelled videos removed from pool)"
        for name, mk in subsets.items():
            H2["n"][name] = dict(clips=int(mk.sum()), falls=int(yh[mk].sum()), other_accidents=int((1 - yh[mk]).sum()))
            H2["auroc"][name] = {m: [float(_auroc(yh[mk], v[mk])) for v in vs] for m, vs in sc.items()}
            o = sc[ours_name][0][mk]
            H2["tests"][name] = {"zero-label vs synthetic-only": paired(yh[mk], o, sc["synthetic_only"][0][mk], gh[mk]),
                                 "zero-label vs random start (seed 1)": paired(yh[mk], o, sc["start from random scores (5 seeds)"][0][mk], gh[mk]),
                                 "zero-label vs accident detector": paired(yh[mk], o, sc["accident detector (no pseudo-labels)"][0][mk], gh[mk])}
        # human-labelled reference on A only (it is trained on B's videos)
        mkA = subsets["A: FallOrNot hard (random sample)"]
        refA = real_full.decision_function(Xh[mkA])
        H2["auroc"]["A: FallOrNot hard (random sample)"]["reference: real probe trained on human labels"] = [float(_auroc(yh[mkA], refA))]
        H2["tests"]["A: FallOrNot hard (random sample)"]["zero-label vs human-labelled real"] = paired(yh[mkA], sc[ours_name][0][mkA], refA, gh[mkA])
        prim = H2["tests"]["C: A + B (primary)"]["zero-label vs synthetic-only"]
        H2["verdict"] = bool(prim[1] > 0)
        json.dump(H2, open(os.path.join(out, "nolabel_hard2.json"), "w"), indent=1)
        fmt = lambda v: f"{v[0]:.3f}" if len(v) == 1 else f"{np.mean(v):.3f} ± {np.std(v):.3f}"
        L = ["", "EXTENDED HARD TEST: falls vs other accidents on every human-labelled accident clip",
             "Rule: on subset C the zero-label probe beats synthetic-only with a paired 95% CI above zero (by compilation)",
             f"Unlabelled pool after removing every labelled video: {H2['pool']['accident_clips']} accident clips + {H2['pool']['pre_clips']} pre-accident clips", ""]
        for name in subsets:
            n = H2["n"][name]
            L += [f"{name}: {n['clips']} clips, {n['falls']} falls, {n['other_accidents']} other accidents"]
            L += [f"- {m}: {fmt(v)}" for m, v in H2["auroc"][name].items()]
            L += [f"- {k}: {g(v)}" for k, v in H2["tests"][name].items()] + [""]
        L.append(f"EXTENDED HARD TEST RESULT: {'SUCCESS' if H2['verdict'] else 'NOT MET'}")
        text = "\n".join(L)
        open(os.path.join(out, "nolabel_hard2.md"), "w").write(text + "\n")
        print(text + f"\n\nwrote {out}/nolabel_hard2.md, nolabel_hard2.json")
    if a.explore:
        explore(out, s0, Xev, Xpre, ev_keys, meta, labels, tests, (Xe, ye, cmax) if a.external else None)


MINE_Q = 0.10       # run 29: share of pool accident clips mined as non-fall accidents


def shortcut_aware(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack, ext):
    """RUN 29 (rule fixed in AGENTS.md before any result). Mine likely NON-FALL ACCIDENTS without labels: pool
    accident clips that a zero-shot VLM rates high for "something went wrong" but low for "a person falls"
    (largest rank(accident) - rank(fall), top 10%). Then run the fixed self-training (synthetic start, 20%, 3 rounds,
    pre-accident negatives) with the mined clips added as negatives in every round and barred from the pseudo-falls.
    Pool = run-18 pool (no labelled video); baseline = the same self-training without mined negatives."""
    from scipy.stats import rankdata
    F, A = json.load(open(a.teacher)), json.load(open(a.hardneg))
    lab_names = {n for n, l in labels.items() if l in ("fall", "not_fall")}
    ke = np.array([k[0] not in lab_names for k in ev_keys]); kp = np.array([k[0] not in lab_names for k in pre_keys])
    names = [k[0] for k, keep in zip(ev_keys, ke) if keep]
    X1, P1, s1 = Xev[ke], Xpre[kp], s0[ke]
    ok = np.array([n in F and n in A for n in names])
    f = np.array([F.get(n, np.nan) for n in names]); acc = np.array([A.get(n, np.nan) for n in names])
    d = np.full(len(names), -np.inf)
    d[ok] = rankdata(acc[ok]) / ok.sum() - rankdata(f[ok]) / ok.sum()
    m = np.zeros(len(names), bool); m[np.argsort(-d)[: int(round(MINE_Q * ok.sum()))]] = True
    def train(start, mined, rounds=ROUNDS, q=Q):
        sc, model = start.copy(), None
        n = max(2, int(round(q * len(X1))))
        for _ in range(rounds):
            cand = np.where(~mined)[0]
            order = cand[np.argsort(-sc[cand])]
            pos = X1[order[:n]]; neg = [X1[order[-n:]], P1]
            if mined.any():
                neg.append(X1[mined])
            neg = np.concatenate(neg)
            model = probe().fit(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
            sc = model.decision_function(X1)
        return model
    none = np.zeros(len(names), bool)
    models = {"self-trained (baseline, run-18 pool)": train(s1, none),
              "shortcut-aware: + mined non-fall accidents (ours, run 29)": train(s1, m),
              "shortcut-aware with VLM-teacher start (secondary)": train(np.where(np.isnan(f), np.nanmedian(f), f), m),
              "synthetic-only probe": syn}
    # test sets, as in run 26
    O = np.load(a.ofitw); ote = O["split"] == "test"
    te = [k for k in sorted(frames) if k in videos and k[0] in test_names and labels.get(k[0]) in ("fall", "not_fall")]
    hk = [k for k in sorted(frames) if k in videos and k[1] == "event" and k[0] in meta and k[0] in lab_names]
    sets = {}
    sets["FallOrNot"] = (pack(te), np.array([int(k[1] == "event" and labels[k[0]] == "fall") for k in te]), np.array([compilation(k[0]) for k in te]), None)
    sets["C: all labelled accident clips"] = (pack(hk), np.array([int(labels[k[0]] == "fall") for k in hk]), np.array([compilation(k[0]) for k in hk]), None)
    sets["OF-ItW"] = (featurize(O["frames"][ote], KIND, O["video"][ote]), O["label"][ote], np.array([compilation(n) for n in O["name"][ote]]), None)
    for nm, path in (("HMDB51", a.external), ("Kinetics-700", a.kinetics)):
        if path:
            Z = np.load(path); Xw = featurize(Z["frames"], KIND, Z["video"]); wc = Z["win_clip"]; yz = Z["clip_label"]
            hard = Z["clip_hard"].astype(bool) | (yz == 1)
            agg = (lambda wc_, n_: (lambda w: (lambda o: (np.maximum.at(o, wc_, w), o)[1])(np.full(n_, -np.inf))))(wc, len(yz))
            sets[nm] = (Xw, yz, Z["clip_name"], (agg, None)); sets[nm + " hard"] = (Xw, yz, Z["clip_name"], (agg, hard))
    ours, base = "shortcut-aware: + mined non-fall accidents (ours, run 29)", "self-trained (baseline, run-18 pool)"
    R = {"pool": dict(accident_clips=int(ke.sum()), with_both_vlm_scores=int(ok.sum()), mined=int(m.sum())), "auroc": {}, "tests": {}}
    for sn, (X, y, g, agg) in sets.items():
        sc_ = {}
        for nm_, mod in models.items():
            v = mod.decision_function(X); mk = np.ones(len(y), bool)
            if agg is not None:
                v = agg[0](v); mk = agg[1] if agg[1] is not None else mk
            sc_[nm_] = (v, mk)
        R["auroc"][sn] = {nm_: float(_auroc(y[mk], v[mk])) for nm_, (v, mk) in sc_.items()}
        v_o, mk = sc_[ours]; v_b, _ = sc_[base]
        R["tests"][sn] = paired(y[mk], v_o[mk], v_b[mk], g[mk])
    # diagnostic (labels used only to measure): would the same rule pick non-falls among the labelled accident clips?
    if a.vlm_dir:
        fl = {}; ac = {}
        for nm_, dst in (("", fl), ("_accident", ac)):
            for part in ("fallornot", "trainpool"):
                pth = os.path.join(a.vlm_dir, f"vlm_{a.teacher_model}{nm_}_{part}.json")
                if os.path.exists(pth):
                    dst.update({k.split("|")[0]: v for k, v in json.load(open(pth)).items() if k.endswith("|event")})
        common = [n for n in sorted(fl) if n in ac and n in lab_names]
        if common:
            pf = np.sort(f[ok]); pa = np.sort(acc[ok]); cut = np.sort(d[ok])[::-1][int(round(MINE_Q * ok.sum())) - 1]
            dd = np.array([np.searchsorted(pa, ac[n]) / len(pa) - np.searchsorted(pf, fl[n]) / len(pf) for n in common])
            sel = dd >= cut
            R["mining_check_on_labelled_clips"] = dict(labelled_accident_clips=len(common), selected=int(sel.sum()),
                                                       share_not_falls_among_selected=float(np.mean([labels[n] == "not_fall" for n, s_ in zip(common, sel) if s_])) if sel.any() else None,
                                                       share_not_falls_overall=float(np.mean([labels[n] == "not_fall" for n in common])))
    c = "C: all labelled accident clips"
    r1 = bool(R["tests"][c][1] > 0)
    r2 = bool("Kinetics-700 hard" in R["auroc"] and R["auroc"]["Kinetics-700 hard"][ours] >= R["auroc"]["Kinetics-700 hard"][base])
    r3 = {sn: bool(R["auroc"][sn][ours] >= R["auroc"][sn][base] - 0.01) for sn in ("FallOrNot", "OF-ItW", "HMDB51") if sn in R["auroc"]}
    R["verdict"] = dict(C_beats_baseline=r1, kinetics_hard_not_lower=r2, no_loss_over_0_01_elsewhere=r3,
                        success=bool(r1 and r2 and len(r3) == 3 and all(r3.values())))
    json.dump(R, open(os.path.join(out, "nolabel_hardneg.json"), "w"), indent=1)
    g_ = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}], p = {t[3]:.3f}"
    cols = list(sets)
    L = ["", "SHORTCUT-AWARE SELF-TRAINING (run 29): mined non-fall accidents as negatives",
         f"Pool (no labelled video): {R['pool']['accident_clips']} accident clips; {R['pool']['mined']} mined as likely non-fall accidents "
         f"(top {int(MINE_Q * 100)}% of rank(accident prompt) - rank(fall prompt))", "",
         "| Detector | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for nm_ in models:
        L.append(f"| {nm_} | " + " | ".join(f"{R['auroc'][c_][nm_]:.3f}" for c_ in cols) + " |")
    L += ["", "Ours vs baseline (paired):"] + [f"- {sn}: {g_(t)}" for sn, t in R["tests"].items()]
    if "mining_check_on_labelled_clips" in R:
        mc = R["mining_check_on_labelled_clips"]
        L.append(f"\nMining check on {mc['labelled_accident_clips']} labelled accident clips: the same rule selects {mc['selected']}; "
                 f"share that are NOT falls: {mc['share_not_falls_among_selected']} (all labelled accident clips: {mc['share_not_falls_overall']:.2f})")
    L += ["", "Rule (run 29): (i) on C ours beats the baseline, paired 95% CI above zero; (ii) on Kinetics hard ours is not lower than the "
              "baseline; (iii) on FallOrNot, OF-ItW and HMDB51 ours is at most 0.01 below the baseline",
          f"SHORTCUT-AWARE RESULT: {'SUCCESS' if R['verdict']['success'] else 'NOT MET'} (i: {r1}; ii: {r2}; iii: {r3})"]
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_hardneg.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_hardneg.md, nolabel_hardneg.json")


def teacher_audit(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, real_full, pack, ext):
    """RUNS 26-27 (rules fixed in AGENTS.md before any result).
    Run 27: the fixed self-training, but the round-0 scores come from a zero-shot VLM (vlm_teacher.py) instead of the
    synthetic probe; pool = run-18 pool (no labelled video), so every labelled accident clip (C) is test data.
    Run 26 (audit): every detector, including ones that never see fall information, on every test set and its hard
    subset, including Kinetics-700 (independent of OOPS!)."""
    from scipy.stats import rankdata
    T = json.load(open(a.teacher))
    lab_names = {n for n, l in labels.items() if l in ("fall", "not_fall")}
    ke = np.array([k[0] not in lab_names for k in ev_keys]); kp = np.array([k[0] not in lab_names for k in pre_keys])
    names = [k[0] for k, keep in zip(ev_keys, ke) if keep]
    X1, P1, s1 = Xev[ke], Xpre[kp], s0[ke]
    tv = np.array([T.get(n, np.nan) for n in names]); have = ~np.isnan(tv)
    tstart = np.where(have, tv, np.nanmedian(tv))
    models = {"synthetic-only probe": [syn],
              "self-trained, synthetic teacher (run-18 pool)": [self_train(s1, X1, P1)[0]],
              "self-trained, VLM teacher (ours, run 27)": [self_train(tstart, X1, P1)[0]],
              "VLM-teacher round 0 only (1 round)": [self_train(tstart, X1, P1, rounds=1)[0]],
              "accident detector (no fall information)": [probe().fit(np.concatenate([X1, P1]), np.r_[np.ones(len(X1)), np.zeros(len(P1))])],
              "random start (no fall information, 5 seeds)": [self_train(np.random.default_rng(sd).random(len(X1)), X1, P1)[0] for sd in range(1, 6)],
              "hand-labelled real probe": [real_full]}
    # ---- test sets: (X, y, groups, keys for the teacher's own scores, how to aggregate)
    O = np.load(a.ofitw); ote = O["split"] == "test"
    te = [k for k in sorted(frames) if k in videos and k[0] in test_names and labels.get(k[0]) in ("fall", "not_fall")]
    hk = [k for k in sorted(frames) if k in videos and k[1] == "event" and k[0] in meta and k[0] in lab_names]
    sets = {}
    yt = np.array([int(k[1] == "event" and labels[k[0]] == "fall") for k in te])
    sets["FallOrNot"] = (pack(te), yt, np.array([compilation(k[0]) for k in te]), [f"{k[0]}|{k[1]}" for k in te], None)
    ev = [k for k in te if k[1] == "event"]
    sets["FallOrNot hard"] = (pack(ev), np.array([int(labels[k[0]] == "fall") for k in ev]), np.array([compilation(k[0]) for k in ev]),
                              [f"{k[0]}|{k[1]}" for k in ev], None)
    sets["C: all labelled accident clips"] = (pack(hk), np.array([int(labels[k[0]] == "fall") for k in hk]),
                                              np.array([compilation(k[0]) for k in hk]), [f"{k[0]}|{k[1]}" for k in hk], None)
    sets["OF-ItW"] = (featurize(O["frames"][ote], KIND, O["video"][ote]), O["label"][ote], np.array([compilation(n) for n in O["name"][ote]]),
                      [f"{n}|{float(c):.2f}" for n, c in zip(O["name"][ote], O["centre"][ote])], None)
    def windowed(path):
        Z = np.load(path); Xw = featurize(Z["frames"], KIND, Z["video"]); wc = Z["win_clip"]; yz = Z["clip_label"]
        hard = Z["clip_hard"].astype(bool) | (yz == 1)
        agg = lambda w: (lambda o: (np.maximum.at(o, wc, w), o)[1])(np.full(len(yz), -np.inf))
        return Xw, yz, Z["clip_name"], hard, agg
    for nm, path in (("HMDB51", a.external), ("Kinetics-700", a.kinetics)):
        if path:
            Xw, yz, cn, hard, agg = windowed(path)
            sets[nm] = (Xw, yz, cn, list(cn), (agg, None))
            sets[nm + " hard"] = (Xw, yz, cn, list(cn), (agg, hard))
    # ---- the teacher's own scores on the same clips (vlm_baseline.py output), where available
    tsc = {}
    if a.vlm_dir:
        for f in ("fallornot", "trainpool", "ofitw", "hmdb_windows", "kinetics_windows"):
            pth = os.path.join(a.vlm_dir, f"vlm_{a.teacher_model}_{f}.json")
            if os.path.exists(pth):
                for k, v in json.load(open(pth)).items():
                    if f.endswith("windows"):
                        n = k.split("|")[0]; tsc[n] = max(tsc.get(n, -np.inf), v)
                    elif f == "ofitw":
                        n, c = k.rsplit("|", 1); tsc[f"{n}|{float(c):.2f}"] = v
                    else:
                        tsc[k] = v
    R = {"pool": dict(accident_clips=int(ke.sum()), with_teacher_score=int(have.sum())), "auroc": {}, "tests": {}, "teacher": {}}
    ours, base = "self-trained, VLM teacher (ours, run 27)", "self-trained, synthetic teacher (run-18 pool)"
    for sn, (X, y, g, keys, agg) in sets.items():
        def scores(m):
            v = m.decision_function(X)
            if agg is None:
                return v, np.ones(len(y), bool)
            f, hard = agg; v = f(v)
            return v, (hard if hard is not None else np.ones(len(y), bool))
        R["auroc"][sn] = {}
        cache = {}
        for name, ms in models.items():
            if name == "hand-labelled real probe" and sn.startswith("C"):
                continue                                   # trained on part of C
            vals = []
            for m in ms:
                v, mk = scores(m); vals.append(float(_auroc(y[mk], v[mk]))); cache.setdefault(name, (v, mk))
            R["auroc"][sn][name] = vals
        v_o, mk = cache[ours]; v_b, _ = cache[base]
        R["tests"][sn] = {"VLM-taught vs synthetic-taught": paired(y[mk], v_o[mk], v_b[mk], g[mk])}
        tk = [i for i in np.where(mk)[0] if keys[i] in tsc]
        if len(tk) >= 20 and len(set(y[tk])) == 2:
            tv_ = np.array([tsc[keys[i]] for i in tk])
            R["teacher"][sn] = dict(n=len(tk), teacher=float(_auroc(y[tk], tv_)), taught=float(_auroc(y[tk], v_o[tk])),
                                    taught_minus_teacher=paired(y[tk], v_o[tk], tv_, g[tk]))
    c = "C: all labelled accident clips"
    r1 = bool(R["tests"][c]["VLM-taught vs synthetic-taught"][1] > 0)
    r2 = {sn: bool(R["teacher"][sn]["taught"] >= R["teacher"][sn]["teacher"] - 0.02) for sn in ("FallOrNot", "OF-ItW", c) if sn in R["teacher"]}
    R["verdict"] = dict(beats_synthetic_teacher_on_C=r1, within_0_02_of_teacher=r2,
                        success=bool(r1 and len(r2) == 3 and all(r2.values())))
    json.dump(R, open(os.path.join(out, "nolabel_teacher.json"), "w"), indent=1)
    g_ = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}], p = {t[3]:.3f}"
    fm = lambda v: f"{v[0]:.3f}" if len(v) == 1 else f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    cols = list(sets)
    L = ["", "SHORTCUT AUDIT AND VLM-TAUGHT SELF-TRAINING (runs 26-27)",
         f"Pool (no labelled video): {R['pool']['accident_clips']} accident clips, {R['pool']['with_teacher_score']} with a teacher score", "",
         "| Detector | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name in models:
        L.append(f"| {name} | " + " | ".join(fm(R['auroc'][c_][name]) if name in R['auroc'][c_] else "–" for c_ in cols) + " |")
    L += ["", "VLM-taught vs synthetic-taught (same pool), paired:"] + [f"- {sn}: {g_(d['VLM-taught vs synthetic-taught'])}" for sn, d in R["tests"].items()]
    L += ["", f"Teacher ({a.teacher_model}) vs VLM-taught probe on the same clips:"] + [
        f"- {sn} (n = {d['n']}): teacher {d['teacher']:.3f}, taught {d['taught']:.3f}; taught minus teacher {g_(d['taught_minus_teacher'])}"
        for sn, d in R["teacher"].items()]
    L += ["", "Rule (run 27): (i) on C the VLM-taught probe beats the synthetic-taught probe, paired 95% CI above zero; "
              "(ii) on FallOrNot, OF-ItW and C it is at most 0.02 AUROC below its teacher",
          f"VLM-TEACHER RESULT: {'SUCCESS' if R['verdict']['success'] else 'NOT MET'} (i: {r1}; ii: {r2})"]
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_teacher.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_teacher.md, nolabel_teacher.json")


def annotators(a, out, frames, videos, labels, test_names, pack, probes):
    """RUN 25 (no new labels). Grade the main models on exactly the same clips with each annotator's labels:
    FallOrNot test videos that BOTH annotators labelled fall / not fall. Event clip = that annotator's label;
    pre-accident clip = non-fall for both. Also the consensus subset (event clips where they agree)."""
    second = {}
    with open(a.annotators) as fh:
        for r in csv.DictReader(fh):
            second[r["name"]] = r["label"]
    second = {k: v for k, v in second.items() if v}
    both = sorted(n for n in test_names if labels.get(n) in ("fall", "not_fall") and second.get(n) in ("fall", "not_fall"))
    keys = [k for k in sorted(frames) if k in videos and k[0] in set(both)]
    X = pack(keys); g = np.array([compilation(k[0]) for k in keys])
    y1 = np.array([int(k[1] == "event" and labels[k[0]] == "fall") for k in keys])
    y2 = np.array([int(k[1] == "event" and second[k[0]] == "fall") for k in keys])
    agree = np.array([k[1] != "event" or labels[k[0]] == second[k[0]] for k in keys])
    sc = {m: p.decision_function(X) for m, p in probes.items()}
    if a.ft_work:
        for kind in ("synthetic", "real", "zerolabel"):
            for sd in (0, 1, 2):
                path = os.path.join(a.ft_work, f"scores_{kind}{'' if sd == 0 else f'_s{sd}'}.npz")
                if os.path.exists(path):
                    z = np.load(path); d = dict(zip(zip(z["fon_name"].tolist(), z["fon_seg"].tolist()), z["fon_score"].tolist()))
                    if all(k in d for k in keys):
                        sc[f"fine-tuned {kind} (seed {sd})"] = np.array([d[k] for k in keys])
    gradings = {"annotator 1 (ours)": (y1, np.ones(len(keys), bool)), "annotator 2 (independent)": (y2, np.ones(len(keys), bool)),
                "consensus (clips where both agree)": (y1, agree)}
    R = {"videos": len(both), "clips": len(keys), "event_disagreements": int((~agree).sum()),
         "falls": {"annotator 1": int(y1.sum()), "annotator 2": int(y2.sum())}, "auroc": {}, "tests": {}}
    pairs = [("self-trained probe (ours)", "synthetic-only probe"), ("self-trained probe (ours)", "hand-labelled real probe")]
    pairs += [(f"fine-tuned zerolabel (seed {sd})", f"fine-tuned {o} (seed {sd})") for o in ("synthetic", "real") for sd in (0, 1, 2)]
    for gname, (y, m) in gradings.items():
        R["auroc"][gname] = {k: float(_auroc(y[m], v[m])) for k, v in sc.items()}
        R["tests"][gname] = {f"{x} vs {o}": paired(y[m], sc[x][m], sc[o][m], g[m]) for x, o in pairs if x in sc and o in sc}
    json.dump(R, open(os.path.join(out, "nolabel_annotators.json"), "w"), indent=1)
    gf = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}], p = {t[3]:.3f}"
    cols = list(gradings)
    L = ["", "TWO ANNOTATORS, SAME CLIPS (no new labels)",
         f"{R['videos']} FallOrNot test videos labelled fall / not fall by both annotators; {R['clips']} clips; "
         f"falls: annotator 1 {R['falls']['annotator 1']}, annotator 2 {R['falls']['annotator 2']}; event clips they disagree on: {R['event_disagreements']}", "",
         "| Model | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k in sc:
        L.append(f"| {k} | " + " | ".join(f"{R['auroc'][c][k]:.3f}" for c in cols) + " |")
    for gname, d in R["tests"].items():
        L += ["", f"{gname}:"] + [f"- {k}: {gf(v)}" for k, v in d.items()]
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_annotators.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_annotators.md, nolabel_annotators.json")


PRE_OFFSET = 3.0     # timestamp-free: the "before" window starts 3 s before the chosen event window


def timestamp_free(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack, tests, ext):
    """RUN 23 (rule fixed in AGENTS.md before any result). Same as the fixed run-15 setting, but the annotated
    accident time is never read: every 2 s window (1 s stride) of every pool video is scored, the highest-scoring
    window of each video is its event clip, and the window starting 3 s earlier is its free non-fall. Each round
    re-picks the event window with the current probe."""
    W = {"name": [], "start": [], "frames": [], "video": []}
    for f in sorted(os.listdir(a.windows)):
        if f.startswith("shard_") and f.endswith(".npz"):
            z = np.load(os.path.join(a.windows, f))
            for k in W:
                W[k].append(z[k])
    W = {k: np.concatenate(v) for k, v in W.items()}
    pool_names = {k[0] for k in ev_keys}
    keep = np.isin(W["name"], list(pool_names))
    wn, ws = W["name"][keep], W["start"][keep]
    Xw = np.concatenate([featurize(W["frames"][keep][i:i + 5000].astype(np.float32), KIND, W["video"][keep][i:i + 5000].astype(np.float32))
                         for i in range(0, int(keep.sum()), 5000)])
    del W
    vids = sorted(set(wn.tolist()))
    idx = {v: [] for v in vids}
    for i, n in enumerate(wn):
        idx[n].append(i)
    idx = {v: np.array(sorted(ix, key=lambda i: ws[i])) for v, ix in idx.items()}
    true_t = {v: float(meta[v]["t"]) for v in vids}
    labelled = {n for n, l in labels.items() if l in ("fall", "not_fall")}

    def pick(score, vlist):
        ev, pre, hit = [], [], []
        for v in vlist:
            ix = idx[v]; e = ix[np.argmax(score[ix])]
            ev.append(e)
            p = ix[np.abs(ws[ix] - (ws[e] - PRE_OFFSET)) < 1e-3]
            if len(p):
                pre.append(p[0])
            hit.append(abs(ws[e] + 1.0 - true_t[v]) <= 1.0)      # diagnostic only: centre within 1 s of the annotation
        return np.array(ev, int), np.array(pre, int), float(np.mean(hit)) if hit else float("nan")

    def train(vlist, rounds=ROUNDS, q=Q):
        sc = syn.decision_function(Xw); model, log = None, []
        for r in range(rounds):
            ev, pre, hit = pick(sc, vlist)
            n = max(2, int(round(q * len(ev))))
            order = np.argsort(-sc[ev])
            pos = Xw[ev[order[:n]]]; neg = np.concatenate([Xw[ev[order[-n:]]], Xw[pre]])
            model = probe().fit(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
            log.append(dict(round=r + 1, event_window_within_1s_of_annotation=hit, pseudo_falls=int(len(pos)), pseudo_non_falls=int(len(neg))))
            sc = model.decision_function(Xw)
        log.append(dict(after_training=True, event_window_within_1s_of_annotation=pick(sc, vlist)[2]))
        return model, log

    chance = float(np.mean([np.mean(np.abs(ws[idx[v]] + 1.0 - true_t[v]) <= 1.0) for v in vids]))
    tf_model, tf_log = train(vids)
    same = np.array([k[0] in idx for k in ev_keys]); same_pre = np.array([k[0] in idx for k in pre_keys])
    ts_model = self_train(s0[same], Xev[same], Xpre[same_pre])[0]
    cand = {"synthetic_only": syn, "zero-label WITH annotated accident times (run-15 method, same videos)": ts_model,
            "timestamp-free zero-label (ours)": tf_model}
    R = {"videos": len(vids), "windows": int(len(wn)), "chance_within_1s": chance, "log": tf_log, "auroc": {}, "tests": {}}
    ours = "timestamp-free zero-label (ours)"
    for name, (Xb, yb, gb) in tests.items():
        R["auroc"][name] = {m: float(_auroc(yb, x.decision_function(Xb))) for m, x in cand.items()}
        R["tests"][name] = {f"ours vs {o}": paired(yb, cand[ours].decision_function(Xb), cand[o].decision_function(Xb), gb)
                            for o in ("synthetic_only", "zero-label WITH annotated accident times (run-15 method, same videos)")}
    if ext:
        Xe, ye, cmax = ext
        R["auroc"]["hmdb51"] = {m: float(_auroc(ye, cmax(x.decision_function(Xe)))) for m, x in cand.items()}
    # falls vs other accidents: retrain without any labelled video, test on every labelled accident clip (as run 18)
    hk = [k for k in sorted(frames) if k in videos and k[1] == "event" and k[0] in meta and k[0] in labelled]
    Xh = pack(hk); yh = np.array([int(labels[k[0]] == "fall") for k in hk]); gh = np.array([compilation(k[0]) for k in hk])
    tf_hard = train([v for v in vids if v not in labelled])[0]
    R["hard_C"] = {"clips": len(hk), "synthetic_only": float(_auroc(yh, syn.decision_function(Xh))),
                   "timestamp-free (labelled videos removed)": float(_auroc(yh, tf_hard.decision_function(Xh))),
                   "test": paired(yh, tf_hard.decision_function(Xh), syn.decision_function(Xh), gh)}
    met = {n: bool(R["tests"][n]["ours vs synthetic_only"][1] > 0) for n in ("fallornot", "ofitw")}
    R["verdict"] = dict(met, success=bool(all(met.values())))
    json.dump(R, open(os.path.join(out, "nolabel_windows.json"), "w"), indent=1)
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], p = {v[3]:.3f}"
    cols = list(R["auroc"])
    L = ["", "TIMESTAMP-FREE ZERO-LABEL ADAPTATION (the annotated accident time is never read)",
         "Rule: on both FallOrNot and OF-ItW, the timestamp-free probe beats synthetic-only with a paired 95% CI above zero (by compilation)",
         f"Pool: {R['videos']} videos, {R['windows']} windows (2 s, 1 s stride); free non-fall = window starting {PRE_OFFSET:.0f} s before the chosen event window",
         "Chosen event window within 1 s of the (unused) annotation: " + ", ".join(
             f"round {x['round']}: {x['event_window_within_1s_of_annotation']:.2f}" if "round" in x else f"final: {x['event_window_within_1s_of_annotation']:.2f}"
             for x in tf_log) + f" (random window: {chance:.2f})", "",
         "| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in cand:
        L.append(f"| {m} | " + " | ".join(f"{R['auroc'][c_][m]:.3f}" for c_ in cols) + " |")
    L.append("")
    for name, d in R["tests"].items():
        L.append(f"{name}: " + "; ".join(f"{k} {g(v)}" for k, v in d.items()))
    h = R["hard_C"]
    L += ["", f"Falls vs other accidents ({h['clips']} labelled accident clips; labelled videos removed from the pool): "
              f"synthetic-only {h['synthetic_only']:.3f}, timestamp-free {h['timestamp-free (labelled videos removed)']:.3f}; "
              f"difference {g(h['test'])}",
          "", f"TIMESTAMP-FREE RESULT: {'SUCCESS' if R['verdict']['success'] else 'NOT MET'}"]
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_windows.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_windows.md, nolabel_windows.json")


def consequence(a, out, frames, videos, labels, meta, test_names, ev_keys, pre_keys, Xev, Xpre, s0, syn, pack, tests, ext):
    """RUN 21 (rule fixed in AGENTS.md before any result). The clip AFTER each accident is scored by CLIP for
    "person on the ground" vs. "person upright". In every self-training round the accident clips are ranked by the
    mean of (rank of the probe score, rank of that consequence score); videos without an after-clip get a neutral
    consequence rank of 0.5. Everything else is the fixed run-15 setting. The pool excludes every video with a human
    label, as in run 18, so the labelled accident clips can be used as the hard test set."""
    from scipy.stats import rankdata
    T = np.load(os.path.join(a.post, "text.npz"))
    G, U = T["ground"].astype(np.float32), T["upright"].astype(np.float32)
    cons = {}
    for f in sorted(os.listdir(a.post)):
        if f.startswith("shard_") and f.endswith(".npz"):
            z = np.load(os.path.join(a.post, f))
            for n, fr, hp in zip(z["name"], z["frames"], z["has_post"]):
                if hp:
                    e = fr.astype(np.float32)
                    lg, lu = 100 * e @ G.T, 100 * e @ U.T
                    lse = lambda x: x.max(1) + np.log(np.exp(x - x.max(1, keepdims=True)).sum(1))
                    cons[str(n)] = float((lse(lg) - lse(lu)).mean())
    lab_names = {n for n, l in labels.items() if l in ("fall", "not_fall")}
    ke = np.array([k[0] not in lab_names for k in ev_keys]); kp = np.array([k[0] not in lab_names for k in pre_keys])
    names = [k[0] for k, keep in zip(ev_keys, ke) if keep]
    X1, P1, s1 = Xev[ke], Xpre[kp], s0[ke]
    c = np.array([cons.get(n, np.nan) for n in names]); have = ~np.isnan(c)
    rc = np.full(len(c), 0.5); rc[have] = rankdata(c[have]) / have.sum()

    def guided(rounds=ROUNDS, q=Q, start=None):
        sc = s1 if start is None else start
        n = max(2, int(round(q * len(X1)))); model = None
        for _ in range(rounds):
            mix = (rankdata(sc) / len(sc) + rc) / 2
            order = np.argsort(-mix)
            pos, neg = X1[order[:n]], np.concatenate([X1[order[-n:]], P1])
            model = probe().fit(np.concatenate([pos, neg]), np.r_[np.ones(len(pos)), np.zeros(len(neg))])
            sc = model.decision_function(X1)
        return model

    cand = {"synthetic_only": syn, "zero_label (run-15 setting, same pool)": self_train(s1, X1, P1)[0],
            "consequence-guided zero-label (ours)": guided(),
            "consequence only, then standard rounds (secondary)": self_train(np.where(have, c, np.nanmedian(c)), X1, P1)[0]}
    # labelled accident clips = the hard test (none of these videos is in the pool)
    hk = [k for k in sorted(frames) if k in videos and k[1] == "event" and k[0] in meta and k[0] in lab_names]
    Xh = pack(hk); yh = np.array([int(labels[k[0]] == "fall") for k in hk]); gh = np.array([compilation(k[0]) for k in hk])
    is_test = np.array([k[0] in test_names for k in hk])
    subsets = {"A: FallOrNot hard": is_test, "B: labelled training pool, accident clips": ~is_test, "C: A + B (primary)": np.ones(len(hk), bool)}
    R = {"pool": dict(accident_clips=int(ke.sum()), with_after_clip=int(have.sum()), pre_clips=int(kp.sum())), "hard": {}, "full": {}, "tests": {}}
    # diagnostic only: the consequence score alone as a fall detector on labelled accident clips
    ch = np.array([cons.get(k[0], np.nan) for k in hk]); okc = ~np.isnan(ch)
    R["consequence_score_alone_on_C"] = dict(clips_with_after_clip=int(okc.sum()), auroc=float(_auroc(yh[okc], ch[okc])) if okc.sum() and len(set(yh[okc])) == 2 else None)
    ours, base = "consequence-guided zero-label (ours)", "zero_label (run-15 setting, same pool)"
    for name, mk in subsets.items():
        R["hard"][name] = {m: float(_auroc(yh[mk], x.decision_function(Xh[mk]))) for m, x in cand.items()}
        R["tests"][name] = {f"ours vs {o}": paired(yh[mk], cand[ours].decision_function(Xh[mk]), cand[o].decision_function(Xh[mk]), gh[mk])
                            for o in ("synthetic_only", base)}
    for name, (Xb, yb, gb) in tests.items():
        R["full"][name] = {m: float(_auroc(yb, x.decision_function(Xb))) for m, x in cand.items()}
        R["tests"][name] = {f"ours vs {base}": paired(yb, cand[ours].decision_function(Xb), cand[base].decision_function(Xb), gb)}
    if ext:
        Xe, ye, cmax, E = ext
        hm = E["clip_hard"].astype(bool) | (ye == 1)
        R["full"]["hmdb51"] = {m: float(_auroc(ye, cmax(x.decision_function(Xe)))) for m, x in cand.items()}
        R["full"]["hmdb51_hard"] = {m: float(_auroc(ye[hm], cmax(x.decision_function(Xe))[hm])) for m, x in cand.items()}
    t_c = R["tests"]["C: A + B (primary)"][f"ours vs {base}"]
    keep_ok = all(R["full"][n][ours] >= R["full"][n][base] - 0.01 for n in ("fallornot", "ofitw"))
    R["verdict"] = dict(beats_zero_label_on_C=bool(t_c[1] > 0), not_worse_on_full_tests=bool(keep_ok), success=bool(t_c[1] > 0 and keep_ok))
    json.dump(R, open(os.path.join(out, "nolabel_post.json"), "w"), indent=1)
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], p = {v[3]:.3f}"
    L = ["", "CONSEQUENCE-GUIDED ZERO-LABEL ADAPTATION (after-accident clip: 'person on the ground?')",
         "Rule: (i) on C, ours beats the run-15 zero-label probe (same pool) with a paired 95% CI above zero (by compilation); "
         "(ii) on full FallOrNot and OF-ItW, ours is at most 0.01 below it",
         f"Pool (every labelled video removed): {R['pool']['accident_clips']} accident clips, {R['pool']['with_after_clip']} with an after-clip, "
         f"{R['pool']['pre_clips']} pre-accident clips",
         f"Diagnostic: consequence score alone on C (falls vs other accidents, {R['consequence_score_alone_on_C']['clips_with_after_clip']} clips "
         f"with an after-clip): AUROC {R['consequence_score_alone_on_C']['auroc']}", ""]
    cols = list(R["hard"]) + list(R["full"])
    L += ["| Method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for m in cand:
        L.append(f"| {m} | " + " | ".join(f"{(R['hard'].get(c_) or R['full'][c_])[m]:.3f}" for c_ in cols) + " |")
    L.append("")
    for name, d in R["tests"].items():
        L.append(f"{name}: " + "; ".join(f"{k} {g(v)}" for k, v in d.items()))
    L += ["", f"CONSEQUENCE RESULT: {'SUCCESS' if R['verdict']['success'] else 'NOT MET'} "
              f"(beats zero-label on C: {R['verdict']['beats_zero_label_on_C']}; not worse on full tests: {R['verdict']['not_worse_on_full_tests']})"]
    text = "\n".join(L)
    open(os.path.join(out, "nolabel_post.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {out}/nolabel_post.md, nolabel_post.json")


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
