#!/usr/bin/env python3
"""Reproduce Table 1 and analyse the 600-image fixed-bank confirmation from SAVED outputs (no inference).

  python ovd/bank_analysis.py --lvis lvis_v1_val.json [--run ovd/results/bank/fixed_bank_soft_confirm_v1] [--boot 1000]

Ground truth is rebuilt from the official LVIS v1 val annotation file. Prints:
  (1) Table 1: mAP50 per method and paired image-bootstrap 95% CIs of bank_soft minus each method;
  (2) where the bank_soft − fixed_soft gain comes from: per-class paired AP differences grouped by positive-image count,
      object size and name length, plus the gain recomputed without single-image classes;
  (3) duplicate boxes: for every method, how many detections fall on an object that already has a higher-scoring match
      (IoU >= 0.5), how many of those outrank the class's true positives, and how Soft-NMS changes this.
"""
import argparse, glob, json, os, random, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from vocab_full import clean_lvis  # noqa: E402

METHODS = ["stock5", "stock5_soft", "fixed", "fixed_soft", "bank", "bank_soft"]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ap(scored, npos):
    scored = sorted(scored, key=lambda t: -t[0]); tp = 0; prec, rec = [], []
    for i, (_, ok) in enumerate(scored, 1):
        tp += ok; prec.append(tp / i); rec.append(tp / npos)
    best = np.maximum.accumulate(np.array(prec[::-1] or [0.0]))[::-1]
    total, prev = 0.0, 0.0
    for i, (_, ok) in enumerate(scored):
        if ok:
            total += (rec[i] - prev) * best[i]; prev = rec[i]
    return total


def load(a):
    L = json.load(open(a.lvis))
    ids = sorted(c["id"] for c in L["categories"]); raw = {c["id"]: c["name"] for c in L["categories"]}
    cat = dict(zip(ids, clean_lvis([raw[k] for k in ids])))
    M = json.load(open(os.path.join(a.run, "manifest.json"))); imgs = set(M["images"])
    files = {im["coco_url"].split("/")[-1]: im for im in L["images"] if im["coco_url"].split("/")[-1] in imgs}
    idmap = {x["id"]: f for f, x in files.items()}
    gt = defaultdict(list)
    for x in L["annotations"]:
        f = idmap.get(x["image_id"])
        if f:
            b = x["bbox"]; gt[(f, cat[x["category_id"]])].append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
    notexh = {f: {cat[k] for k in x.get("not_exhaustive_category_ids", [])} for f, x in files.items()}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.run, "per_image", "*.json"))):
        rows += json.load(open(p))["rows"]
    return M, gt, notexh, rows


def per_pair(rows, gt, notexh):
    """For each (method, image, class): list of (score, is_tp, is_duplicate) and the number of ground-truth boxes."""
    out = {}
    for r in rows:
        f, c = r["image"], r["target"]; g = gt.get((f, c), []); ign = c in notexh[f]
        used = [False] * len(g); res = []
        for d in sorted(r["dets"], key=lambda d: -d[0]):
            best, bj, dup = 0.5, -1, False
            for j, gg in enumerate(g):
                v = iou(d[1:5], gg)
                if v >= 0.5 and used[j]:
                    dup = True
                if not used[j] and v >= best:
                    best, bj = v, j
            if bj >= 0:
                used[bj] = True; res.append((d[0], 1, False))
            elif not ign:
                res.append((d[0], 0, dup))
        out[(r["cond"], f, c)] = (res, len(g))
    return out


def class_aps(P, method, weights=None):
    per = defaultdict(lambda: [[], 0])
    for (m, f, c), (res, n) in P.items():
        if m != method:
            continue
        w = 1 if weights is None else weights.get(f, 0)
        if w:
            per[c][0] += [(s, t) for s, t, _ in res] * w; per[c][1] += n * w
    return {c: ap(s, n) for c, (s, n) in per.items() if n > 0}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--lvis", required=True)
    ap_.add_argument("--run", default=os.path.join(os.path.dirname(__file__), "results", "bank", "fixed_bank_soft_confirm_v1"))
    ap_.add_argument("--boot", type=int, default=1000)
    a = ap_.parse_args()
    M, gt, notexh, rows = load(a)
    P = per_pair(rows, gt, notexh)
    imgs = sorted(M["images"])
    # (1) Table 1
    print("== (1) Table 1: mAP50 on", len(imgs), "images,", len(M["targets"]), "pairs")
    full = {m: class_aps(P, m) for m in METHODS}
    mean = lambda d: float(np.mean(list(d.values())))
    rng = random.Random(0); boots = defaultdict(list)
    for _ in range(a.boot):
        w = defaultdict(int)
        for f in rng.choices(imgs, k=len(imgs)):
            w[f] += 1
        bs = mean(class_aps(P, "bank_soft", w))
        for m in ("stock5", "stock5_soft", "fixed_soft"):
            boots[m].append(bs - mean(class_aps(P, m, w)))
    for m in METHODS:
        line = f"  {m:12s} mAP50 {mean(full[m]):.4f}"
        if m in boots:
            v = sorted(boots[m]); lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
            line += f"   bank_soft − {m}: {mean(full['bank_soft']) - mean(full[m]):+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(line)
    # (2) where the gain comes from
    print("== (2) per-class paired AP50 difference, bank_soft − fixed_soft")
    pos_imgs = defaultdict(set); area = defaultdict(list)
    for t in M["targets"]:
        if t["present"]:
            pos_imgs[t["target"]].add(t["image"])
            for b in gt[(t["image"], t["target"])]:
                area[t["target"]].append((b[2] - b[0]) * (b[3] - b[1]))
    d = {c: full["bank_soft"][c] - full["fixed_soft"][c] for c in full["bank_soft"] if c in full["fixed_soft"]}
    def grp(name, key):
        g = defaultdict(list)
        for c, v in d.items():
            g[key(c)].append(v)
        for k in sorted(g):
            v = g[k]; print(f"  {name:14s} {k:>10s}: n={len(v):4d} classes  mean diff {np.mean(v):+.4f}  "
                            f"classes better {sum(x > 1e-9 for x in v)}, worse {sum(x < -1e-9 for x in v)}, equal {sum(abs(x) <= 1e-9 for x in v)}")
    grp("positive imgs", lambda c: "1" if len(pos_imgs[c]) == 1 else ("2-3" if len(pos_imgs[c]) <= 3 else "4+"))
    grp("median size", lambda c: "small" if np.median(area[c]) < 32 ** 2 else ("medium" if np.median(area[c]) < 96 ** 2 else "large"))
    grp("name words", lambda c: "1 word" if len(c.split()) == 1 else "2+ words")
    multi = [c for c in d if len(pos_imgs[c]) >= 2]
    print(f"  sensitivity: mean diff over all {len(d)} classes {np.mean(list(d.values())):+.4f}; "
          f"over the {len(multi)} classes with >= 2 positive images {np.mean([d[c] for c in multi]):+.4f}")
    # (3) duplicates
    print("== (3) duplicate boxes (a detection on an object that already has a higher-scoring match)")
    for m in METHODS:
        n_tp = n_dup = n_fp = dup_above = 0
        per_cls = defaultdict(list)
        for (mm, f, c), (res, n) in P.items():
            if mm == m:
                per_cls[c] += res
        for c, res in per_cls.items():
            tps = sorted(s for s, t, _ in res if t)
            for s, t, dup in res:
                n_tp += t; n_fp += (not t); n_dup += dup
                if dup and tps and s > tps[0]:
                    dup_above += 1                     # outranks at least the class's weakest true positive
        print(f"  {m:12s} TP {n_tp:6d}  FP {n_fp:6d}  duplicates {n_dup:6d} ({n_dup / max(n_fp, 1):.1%} of FP)  "
              f"duplicates outranking a TP of their class {dup_above:6d}")


if __name__ == "__main__":
    main()
