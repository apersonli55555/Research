#!/usr/bin/env python3
"""Analysis of ovd/vocab_full.py outputs (OVD runs 2-4). Every number it prints comes from the result files.

  python ovd/full_analysis.py --out OUT [--boot 1000]

Definitions (fixed in ovd/PLAN.md and the vocab_full.py docstring)
  groups      k1, rand{k} (mean over draws), near{k}, vis{k}, far{k}, all (mean over the 5 orders), FIX (correction)
  AP50        per class over the (image, target) pairs of that class (images where it is present + absent pairs), greedy
              matching at IoU 0.5, all-point interpolated; LVIS: unmatched detections on images not exhaustively
              labelled for the class are ignored; mAP = mean over classes with at least one ground-truth box
              "deployed" = only boxes that keep the target as their top class (single-label post-processing)
              "indep"    = every box scored for the target (no label competition)
  interference  for a method, the change of its target output between a vocabulary V and the same method's
              single-class output: mean |raw score change| over present instances, detection flips at the model's
              threshold (deployed output), localisation flips (IoU(best box under V, best box under k1) < 0.5)
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vocab_pilot import THRESH, iou  # noqa: E402

MODELS = ["gdino", "owlv2", "yoloworld"]


def group_of(cond):
    if cond in ("k1", "FIX", "prune"):
        return cond
    if cond.startswith("all_"):
        return "all"
    return cond.split("_")[0]


def match(dets, gts):
    used, out = np.zeros(len(gts), bool), []
    for d in dets:                                   # dets arrive sorted by score
        ok = False
        if gts:
            ious = iou(np.array(gts, float), np.array(d[1:5], float)); ious[used] = -1
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                used[j] = True; ok = True
        out.append((d[0], ok))
    return out


def wap(scores, tps, ws, npos):
    """All-point interpolated AP with per-record weights (bootstrap multiplicities)."""
    if npos <= 0:
        return np.nan
    o = np.argsort(-scores, kind="stable"); tp = np.cumsum(ws[o] * tps[o]); fp = np.cumsum(ws[o] * (1 - tps[o]))
    rec = tp / npos; prec = tp / np.maximum(tp + fp, 1e-12)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


class APTable:
    """Match lists for every (condition, class, image); mAP for any image weighting."""
    def __init__(self, R, P, mode):
        info = P["info"]; self.imgs = sorted(info); self.ix = {f: i for i, f in enumerate(self.imgs)}
        self.npos = defaultdict(lambda: np.zeros(len(self.imgs)))          # class -> gt count per image
        have = {(r["image"], r["target"]) for r in R}
        for p in P["plan"]:
            if p["present"] and (p["image"], p["target"]) in have:
                self.npos[p["target"]][self.ix[p["image"]]] = sum(g["cls"] == p["target"] for g in info[p["image"]]["gt"])
        self.recs = defaultdict(lambda: defaultdict(list))                 # cond -> class -> [(img, s, tp)]
        for r in R:
            f, c = r["image"], r["target"]
            dets = r["dets"] if mode == "indep" or r["cond"] == "FIX" else [d for d in r["dets"] if d[5] == 1]
            gts = [g["box"] for g in info[f]["gt"] if g["cls"] == c]
            ign = c in info[f].get("not_exh", [])
            for s, ok in match(dets, gts):
                if ok or not ign:
                    self.recs[r["cond"]][c].append((self.ix[f], s, ok))
        self.arr = {cd: {c: tuple(np.array(x) for x in zip(*v)) if v else (np.zeros(0, int), np.zeros(0), np.zeros(0)) for c, v in d.items()}
                    for cd, d in self.recs.items()}

    def map(self, cond, w=None):
        w = np.ones(len(self.imgs)) if w is None else w
        aps = []
        for c, npos in self.npos.items():
            n = float((npos * w).sum())
            if n <= 0:
                continue
            im, s, tp = self.arr.get(cond, {}).get(c, (np.zeros(0, int), np.zeros(0), np.zeros(0)))
            aps.append(wap(s, tp.astype(float), w[im], n) if len(s) else 0.0)
        return float(np.mean(aps)) if aps else np.nan


def gmap(T, conds, w=None):
    return float(np.nanmean([T.map(c, w) for c in conds]))


def analyse(out, ds, model, boot):
    P = json.load(open(os.path.join(out, f"plan_{ds}.json")))
    path = os.path.join(out, f"full_{ds}_{model}.jsonl")
    R = []
    for line in open(path):
        try:
            R.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    keys = {(p["image"], p["target"]) for p in P["plan"]}
    R = [r for r in R if (r["image"], r["target"]) in keys]
    th = THRESH[model]
    conds = sorted({r["cond"] for r in R}); groups = defaultdict(list)
    for c in conds:
        groups[group_of(c)].append(c)
    order = ["k1"] + sorted([g for g in groups if g[:4] == "rand"], key=lambda g: int(g[4:])) + \
            sorted([g for g in groups if g[:4] in ("near", "far") or g[:3] == "vis"]) + ["all", "prune", "FIX"]
    order = [g for g in order if g in groups]
    S = dict(dataset=ds, model=model, n_runs=len(R), n_pairs=len({(r["image"], r["target"]) for r in R}),
             n_images=len({r["image"] for r in R}), groups={})
    by = defaultdict(dict)
    for r in R:
        by[(r["image"], r["target"])][r["cond"]] = r
    Td, Ti = APTable(R, P, "deployed"), APTable(R, P, "indep")
    rng = np.random.default_rng(0)
    W = [np.bincount(rng.integers(0, len(Td.imgs), len(Td.imgs)), minlength=len(Td.imgs)).astype(float) for _ in range(boot)]
    for g in order:
        cs = groups[g]; e = dict(conds=len(cs))
        e["mAP50_deployed"] = gmap(Td, cs); e["mAP50_indep"] = gmap(Ti, cs)
        ds_, fl, bx, fa, far, ck = [], [], [], [], [], []
        for (f, c), d in by.items():
            if "k1" not in d:
                continue
            for cd in cs:
                if cd not in d:
                    continue
                r, r1 = d[cd], d["k1"]
                ref = r if g == "FIX" else r1        # FIX is one record: its output is the same for every vocabulary
                if r["present"]:
                    for a_, b_ in zip(r["instances"], ref["instances"]):
                        ds_.append(a_["raw"] - b_["raw"]); fl.append((a_["deployed"] >= th) != (b_["deployed"] >= th))
                    bx.append(iou(np.array([r["best_box"]]), np.array(ref["best_box"]))[0] < 0.5)
                else:
                    fa.append(r["img_max_deployed"] >= th); far.append(r["img_max_raw"] >= th)
                if "chunk_k" in r:
                    ck.append(r["chunk_k"])
        e.update(mean_abs_score_change=float(np.mean(np.abs(ds_))) if ds_ else np.nan,
                 mean_score_change=float(np.mean(ds_)) if ds_ else np.nan,
                 det_flip_rate=float(np.mean(fl)) if fl else np.nan,
                 loc_flip_rate=float(np.mean(bx)) if bx else np.nan,
                 absent_false_alarm_deployed=float(np.mean(fa)) if fa else np.nan,
                 absent_false_alarm_raw=float(np.mean(far)) if far else np.nan,
                 n_instances=len(ds_))
        if ck:
            e["mean_chunk_size"] = float(np.mean(ck))
        if g != "k1" and boot:
            diffs = [gmap(Td, cs, w) - gmap(Td, groups["k1"], w) for w in W]
            e["dmAP_deployed_vs_k1_CI"] = [float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))]
        S["groups"][g] = e
    # order-only effect: instances whose detected status differs among the 5 orders of the full list
    of, rngs = [], []
    for d in by.values():
        rs = [d[c] for c in groups.get("all", []) if c in d]
        if len(rs) >= 2 and rs[0]["present"]:
            for j in range(len(rs[0]["instances"])):
                v = [r["instances"][j] for r in rs]
                of.append(len({x["deployed"] >= th for x in v}) > 1); rngs.append(max(x["raw"] for x in v) - min(x["raw"] for x in v))
    S["order_flip_rate"] = float(np.mean(of)) if of else np.nan
    S["order_score_range"] = float(np.mean(rngs)) if rngs else np.nan
    # vocabulary dependence over all original conditions of a pair (excluding FIX)
    va, vr = [], []
    for d in by.values():
        rs = [r for c, r in d.items() if c not in ("FIX", "prune")]
        if rs and rs[0]["present"]:
            for j in range(len(rs[0]["instances"])):
                v = [r["instances"][j] for r in rs]
                va.append(len({x["deployed"] >= th for x in v}) > 1); vr.append(max(x["raw"] for x in v) - min(x["raw"] for x in v))
    S["any_vocab_flip_rate"] = float(np.mean(va)) if va else np.nan
    S["any_vocab_score_range"] = float(np.mean(vr)) if vr else np.nan
    # correction vs. baselines (paired bootstrap by image)
    if "FIX" in groups and boot:
        best = max([g for g in ("k1", "all", "prune") if g in S["groups"]], key=lambda g: S["groups"][g]["mAP50_deployed"])
        S["best_baseline"] = best
        for ref in [g for g in ("k1", "all", "prune") if g in groups]:
            for mode, T in (("deployed", Td),):
                diffs = [T.map("FIX", w) - gmap(T, groups[ref], w) for w in W]
                S[f"FIX_minus_{ref}_mAP50"] = dict(point=S["groups"]["FIX"]["mAP50_deployed"] - S["groups"][ref]["mAP50_deployed"],
                                                   CI=[float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))])
        orig = [S["groups"][g]["mean_abs_score_change"] for g in order if g not in ("k1", "FIX", "prune")]
        origf = [S["groups"][g]["det_flip_rate"] for g in order if g not in ("k1", "FIX", "prune")]
        S["interference_orig"] = dict(score=float(np.nanmean(orig)), flips=float(np.nanmean(origf)))
        S["interference_FIX"] = dict(score=S["groups"]["FIX"]["mean_abs_score_change"], flips=S["groups"]["FIX"]["det_flip_rate"],
                                     note="0 by construction: one output per (image, target) whatever the vocabulary")
    return S


def qs_probe(out, ds):
    """Grounding DINO: how much of the k1 -> V score change remains when query selection sees only the target."""
    a, b = (os.path.join(out, f"full_{ds}_{m}.jsonl") for m in ("gdino", "gdino_qs"))
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    load = lambda p: {(r["image"], r["target"], r["cond"]): r for r in map(json.loads, open(p))}
    A, B = load(a), load(b); res = {}
    for cd in ("rand20_0", "all_0"):
        da, db = [], []
        for (f, c, x), r in B.items():
            if x != cd or not r["present"] or (f, c, "k1") not in B or (f, c, cd) not in A or (f, c, "k1") not in A:
                continue
            for i, j, k, l in zip(A[(f, c, cd)]["instances"], A[(f, c, "k1")]["instances"], r["instances"], B[(f, c, "k1")]["instances"]):
                da.append(i["raw"] - j["raw"]); db.append(k["raw"] - l["raw"])
        if da:
            res[cd] = dict(n=len(da), mean_abs_change_full=float(np.mean(np.abs(da))), mean_abs_change_qs_target=float(np.mean(np.abs(db))),
                           mean_change_full=float(np.mean(da)), mean_change_qs_target=float(np.mean(db)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    allS = {}
    lines = ["# OVD full study: summary (generated by ovd/full_analysis.py)", ""]
    for ds in ("coco", "lvis"):
        if not os.path.exists(os.path.join(a.out, f"plan_{ds}.json")):
            continue
        for m in MODELS:
            if not os.path.exists(os.path.join(a.out, f"full_{ds}_{m}.jsonl")):
                continue
            S = analyse(a.out, ds, m, a.boot); allS[f"{ds}/{m}"] = S
            lines += [f"## {ds} / {m}: {S['n_images']} images, {S['n_pairs']} pairs, {S['n_runs']} runs", "",
                      "| group | mAP50 dep | mAP50 indep | ΔmAP vs k1 95% CI | mean |Δs| | mean Δs | det flips | loc flips | FA dep | chunk |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for g, e in S["groups"].items():
                ci = e.get("dmAP_deployed_vs_k1_CI"); ci = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
                lines.append(f"| {g} | {e['mAP50_deployed']:.3f} | {e['mAP50_indep']:.3f} | {ci} | {e['mean_abs_score_change']:.4f} | "
                             f"{e['mean_score_change']:+.4f} | {e['det_flip_rate']:.3f} | {e['loc_flip_rate']:.3f} | "
                             f"{e['absent_false_alarm_deployed']:.3f} | {e.get('mean_chunk_size', float('nan')):.1f} |")
            lines += ["", f"order-only flips (5 orders of the full list): {S['order_flip_rate']:.3f}, mean score range {S['order_score_range']:.4f}",
                      f"instances whose detection flips under some vocabulary: {S['any_vocab_flip_rate']:.3f}, mean score range {S['any_vocab_score_range']:.4f}"]
            for ref in ("k1", "all", "prune"):
                if f"FIX_minus_{ref}_mAP50" in S:
                    x = S[f"FIX_minus_{ref}_mAP50"]
                    lines.append(f"FIX - {ref} mAP50 (deployed): {x['point']:+.4f}, 95% CI [{x['CI'][0]:+.4f}, {x['CI'][1]:+.4f}]")
            if "interference_orig" in S:
                lines.append(f"interference orig: score {S['interference_orig']['score']:.4f}, flips {S['interference_orig']['flips']:.3f}; "
                             f"FIX: score {S['interference_FIX']['score']:.4f}, flips {S['interference_FIX']['flips']:.3f}")
            lines.append("")
            print("\n".join(lines[-(len(S['groups']) + 10):]), flush=True)
        q = qs_probe(a.out, ds)
        if q:
            allS[f"{ds}/gdino_qs_probe"] = q
            lines += [f"## {ds}: Grounding DINO query-selection probe", "", json.dumps(q, indent=1), ""]
            print(lines[-3], lines[-2])
        bp = os.path.join(a.out, f"bench_{ds}.json")
        if os.path.exists(bp):
            B = json.load(open(bp)); allS[f"{ds}/bench"] = B
            lines += [f"## {ds}: latency (s / image)", ""]
            for m, r in B.items():
                lines.append(f"- {m} ({r.get('device')}): " + ", ".join(f"{k} {v['mean_s']:.3f}" for k, v in r.items() if isinstance(v, dict)))
            lines.append(""); print("\n".join(lines[-len(B) - 2:]))
    json.dump(allS, open(os.path.join(a.out, "full_summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "full_summary.md"), "w").write("\n".join(lines))
    print("wrote", os.path.join(a.out, "full_summary.md"))


if __name__ == "__main__":
    main()
