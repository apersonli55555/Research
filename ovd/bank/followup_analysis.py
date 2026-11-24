#!/usr/bin/env python3
"""Analyse the outputs of ovd/bank/followup.py (CPU; no inference).

  python ovd/bank/followup_analysis.py --dir followup_v1 --lvis lvis_v1_val.json [--boot 1000]

mAP50 exactly as in run 3h (full_analysis.APTable 'indep' definition, re-implemented in ovd/bank_analysis.py), raw and
after class-wise Gaussian Soft-NMS (sigma 0.5, threshold 0.02, top 50; reproduces the run-3h Soft-NMS outputs exactly).
Paired image bootstrap (seed 0) for differences.
"""
import argparse, glob, json, math, os, random, sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bank_analysis import iou, load, per_pair, class_aps  # noqa: E402

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bank", "fixed_bank_soft_confirm_v1")


def soft_nms(dets, sigma=.5, thr=.02, k=50):
    d = [list(x) for x in dets]; out = []
    while d and len(out) < k:
        i = max(range(len(d)), key=lambda j: d[j][0]); b = d.pop(i); out.append(b)
        for x in d:
            x[0] *= math.exp(-iou(b[1:], x[1:]) ** 2 / sigma)
        d = [x for x in d if x[0] >= thr]
    return out


def rows_of(d, part):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, part, "*.json"))):
        rows += json.load(open(p))["rows"]
    return rows


def with_soft(rows):
    return rows + [dict(r, cond=r["cond"] + "_soft", dets=soft_nms(r["dets"])) for r in rows]


def mean(d):
    return float(np.mean(list(d.values())))


def table(P, imgs, methods, ref, boot, label):
    """mAP50 per method; ref − method with paired image-bootstrap 95% CI."""
    full = {m: class_aps(P, m) for m in methods}
    rng = random.Random(0); diffs = defaultdict(list)
    for _ in range(boot):
        w = defaultdict(int)
        for f in rng.choices(imgs, k=len(imgs)):
            w[f] += 1
        r = mean(class_aps(P, ref, w))
        for m in methods:
            if m != ref:
                diffs[m].append(r - mean(class_aps(P, m, w)))
    print(f"== {label}  ({len(imgs)} images)")
    out = {}
    for m in methods:
        line = f"  {m:20s} mAP50 {mean(full[m]):.4f}"; out[m] = dict(mAP50=mean(full[m]))
        if m != ref and boot:
            v = sorted(diffs[m]); lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
            line += f"   {ref} − {m}: {mean(full[ref]) - mean(full[m]):+.4f} [{lo:+.4f}, {hi:+.4f}]"
            out[m].update(diff=mean(full[ref]) - mean(full[m]), ci=[lo, hi])
        print(line)
    return out


def flips(rows, a, b, present, thr=0.35):
    """Present-target pairs whose detection decision (top score >= Grounding DINO's deployment threshold 0.35,
    vocab_pilot.THRESH) or top box (IoU < 0.5 between the two top boxes) differs between conditions a and b."""
    by = defaultdict(dict)
    for r in rows:
        if (r["image"], r["target"]) in present:
            by[(r["image"], r["target"])][r["cond"]] = r["dets"]
    n = dec = loc = 0
    for k, v in by.items():
        if a in v and b in v:
            x, y = v[a], v[b]; n += 1
            dx, dy = bool(x and x[0][0] >= thr), bool(y and y[0][0] >= thr)
            dec += dx != dy
            if x and y:
                loc += iou(x[0][1:], y[0][1:]) < 0.5
    return n, dec, loc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    M, gt, notexh, saved = load(SimpleNamespace(lvis=a.lvis, run=RUN))
    res = {}

    # A: reproduction
    if os.path.isdir(os.path.join(a.dir, "A")):
        diffs = [x for p in glob.glob(os.path.join(a.dir, "A", "*.json")) for x in json.load(open(p))["diff"]]
        sc = [x["score"] for x in diffs if "score" in x]; bx = [x["box"] for x in diffs if "box" in x]
        mism = [x for x in diffs if "score" not in x and x["empty_old"] != x["empty_new"]]
        print(f"== A reproduction vs saved run 3h: {len(diffs)} lists; top score |diff| max {max(sc):.2e} median "
              f"{np.median(sc):.2e}; top box max {max(bx):.3f} px; empty/non-empty mismatches {len(mism)}")
        res["A"] = dict(n=len(diffs), score_max=max(sc), score_median=float(np.median(sc)), box_max=max(bx), mismatches=len(mism))

    # B: image-independent requests (bank is list-invariant, so the saved bank results apply unchanged)
    if os.path.isdir(os.path.join(a.dir, "B")):
        rows = rows_of(a.dir, "B"); imgs = sorted({r["image"] for r in rows})
        base = [r for r in saved if r["image"] in set(imgs) and r["cond"] in ("bank_soft", "bank", "stock5", "stock5_soft")]
        allr = with_soft(rows) + base
        P = per_pair(allr, gt, notexh)
        res["B"] = table(P, imgs, ["bank_soft", "stock1", "stock1_soft", "stock5_indA", "stock5_indA_soft", "stock5_indB",
                                   "stock5_indB_soft", "stock5", "stock5_soft"], "bank_soft", a.boot,
                         "B (#3) image-independent requests; bank_soft − method")
        for x, y in [("stock5_indA", "stock5_indB"), ("stock1", "stock5_indA"), ("stock5", "stock5_indA")]:
            n, dec, loc = flips(allr, x, y, {(t["image"], t["target"]) for t in M["targets"] if t["present"]})
            print(f"  {x} vs {y}: {n} present pairs, detection flips (top score >= 0.35) {dec} ({dec / max(n, 1):.1%}), "
                  f"top-box moves {loc} ({loc / max(n, 1):.1%})")
            res["B"][f"flips_{x}_{y}"] = dict(n=n, decision=dec, box=loc)

    # C: bank variants
    if os.path.isdir(os.path.join(a.dir, "C")):
        rows = rows_of(a.dir, "C"); imgs = sorted({r["image"] for r in rows})
        P = per_pair(with_soft(rows), gt, notexh)
        tags = ["canon", "pool_first", "pool_last", "mask_full"]
        res["C_raw"] = table(P, imgs, tags, "canon", a.boot, "C (#6) bank construction, raw; canon − variant")
        res["C_soft"] = table(P, imgs, [t + "_soft" for t in tags], "canon_soft", a.boot, "C (#6) bank construction + Soft-NMS")
        for half in ("dict_half_a", "dict_half_b"):
            keep = {(r["image"], r["target"]) for r in rows if r["cond"] == half}
            sub = [r for r in rows if r["cond"] in ("canon", half) and (r["image"], r["target"]) in keep]
            Ph = per_pair(with_soft(sub), gt, notexh); ih = sorted({f for f, _ in keep})
            res[half] = table(Ph, ih, ["canon_soft", half + "_soft", "canon", half], "canon_soft", a.boot,
                              f"C (#4) bank with half the classes ({len({c for _, c in keep})} target classes, {len(keep)} pairs)")
        if os.path.exists(os.path.join(a.dir, "C_meta.json")):
            print("  class vectors vs canonical bank:", json.load(open(os.path.join(a.dir, "C_meta.json"))))

    # E: cost crossover
    if os.path.isdir(os.path.join(a.dir, "E")):
        T = [t for p in glob.glob(os.path.join(a.dir, "E", "*.json")) for t in json.load(open(p))["timing"]]
        print(f"== E (#5) cost vs vocabulary size ({len({1 for _ in T}) and len(glob.glob(os.path.join(a.dir, 'E', '*.json')))} images); "
              "median seconds / peak extra GPU MB / forward passes")
        ks = sorted({t["k"] for t in T}); res["E"] = {}
        for k in ks:
            line = f"  K={k:5d}"
            for m in ("stock", "fixed", "bank"):
                v = [t for t in T if t["k"] == k and t["method"] == m]
                s = float(np.median([t["seconds"] for t in v])); mb = float(np.median([t["peak_bytes"] for t in v])) / 2 ** 20
                line += f"   {m} {s:.3f}s {mb:6.0f}MB x{v[0]['passes']}"
                res["E"][f"{k}_{m}"] = dict(seconds=s, peak_mb=mb, passes=v[0]["passes"])
            print(line)
        if os.path.exists(os.path.join(a.dir, "E_setup.json")):
            print("  set-up:", json.load(open(os.path.join(a.dir, "E_setup.json"))))

    # F: second checkpoint
    if os.path.isdir(os.path.join(a.dir, "F")):
        rows = rows_of(a.dir, "F"); imgs = sorted({r["image"] for r in rows})
        P = per_pair(with_soft(rows), gt, notexh)
        res["F"] = table(P, imgs, ["bank_soft", "bank", "stock5", "stock5_soft", "fixed", "fixed_soft"], "bank_soft", a.boot,
                         "F (#7) grounding-dino-tiny; bank_soft − method")
        tm = [json.load(open(p))["timing"] for p in glob.glob(os.path.join(a.dir, "F", "*.json"))]
        avg = {m: float(np.mean([t[m] for t in tm])) for m in ("bank", "stock5", "fixed")}
        print(f"  mean seconds/image: {avg}; bank / stock5 = {avg['bank'] / avg['stock5']:.2f}x; fixed / bank = {avg['fixed'] / avg['bank']:.2f}x")
        res["F"]["seconds"] = avg

    json.dump(res, open(os.path.join(a.dir, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
