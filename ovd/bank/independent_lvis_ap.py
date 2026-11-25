#!/usr/bin/env python3
"""Independent re-implementation of LVIS v1 box AP (does not import lvis-api), used to check the full-image results.

  python ovd/bank/independent_lvis_ap.py --lvis lvis_v1_val.json --zip results/bank/full_image_v1/full_image_v1_outputs.zip
         [--hybrid results/bank/hybrid/test_outputs.zip]

Written from the LVIS paper / evaluation rules: federated per-class evaluation over images whose positive set (has
ground truth of the class) or negative set (neg_category_ids) contains the class; detections of a class on other images
are dropped; on images listing the class in not_exhaustive_category_ids, unmatched detections are ignored; at most 300
detections per image (highest scores); greedy matching by descending score to the highest-IoU unmatched ground truth at
IoU >= t, t in 0.50:0.05:0.95; 101-point interpolated precision; AP = mean over (t, recall point, class) with ground truth;
APr/APc/APf by the category frequency field. Boxes are xywh; IoU with the same convention as COCO (no +1).
"""
import argparse, io, json, zipfile
from collections import defaultdict

import numpy as np

T = np.linspace(0.5, 0.95, 10); R = np.linspace(0, 1, 101)


def iou_xywh(d, g):
    """d (n,4), g (m,4) xywh -> (n,m)."""
    dx2, dy2 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]; gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx2[:, None], gx2[None]) - np.maximum(d[:, 0][:, None], g[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(dy2[:, None], gy2[None]) - np.maximum(d[:, 1][:, None], g[:, 1][None]), 0, None)
    inter = iw * ih; ua = (d[:, 2] * d[:, 3])[:, None] + (g[:, 2] * g[:, 3])[None] - inter
    return inter / np.maximum(ua, 1e-12)


def evaluate(L, dets, img_ids):
    imgs = {i["id"]: i for i in L["images"] if i["id"] in img_ids}
    gts = defaultdict(list)
    for a in L["annotations"]:
        if a["image_id"] in imgs:
            gts[(a["image_id"], a["category_id"])].append(a["bbox"])
    pos = defaultdict(set)
    for (i, c) in gts:
        pos[c].add(i)
    neg = defaultdict(set); nex = defaultdict(set)
    for i, im in imgs.items():
        for c in im.get("neg_category_ids", []):
            neg[c].add(i)
        for c in im.get("not_exhaustive_category_ids", []):
            nex[c].add(i)
    per_img = defaultdict(list)
    for d in dets:
        per_img[d["image_id"]].append(d)
    by = defaultdict(list)
    for i, ds in per_img.items():
        ds = sorted(ds, key=lambda d: -d["score"])[:300]
        for d in ds:
            by[(i, d["category_id"])].append(d)
    freq = {c["id"]: c["frequency"] for c in L["categories"]}
    out = {}
    for c in sorted(pos):
        npos = sum(len(gts[(i, c)]) for i in pos[c])
        scores, tps, ign = [], [], []
        for i in pos[c] | neg[c]:
            ds = sorted(by.get((i, c), []), key=lambda d: -d["score"])
            if not ds:
                continue
            g = np.array(gts.get((i, c), []), dtype=float).reshape(-1, 4)
            db = np.array([d["bbox"] for d in ds], dtype=float)
            M = iou_xywh(db, g) if len(g) else np.zeros((len(ds), 0))
            tp = np.zeros((len(T), len(ds)), bool)
            for ti, t in enumerate(T):
                used = np.zeros(len(g), bool)
                for k in range(len(ds)):
                    best, bj = min(t, 1 - 1e-10), -1
                    for j in range(len(g)):
                        if not used[j] and M[k, j] >= best:
                            best, bj = M[k, j], j
                    if bj >= 0:
                        used[bj] = True; tp[ti, k] = True
            scores += [d["score"] for d in ds]; tps.append(tp)
            ign.append(np.broadcast_to(np.array([i in nex[c]] * len(ds)), (len(T), len(ds))))
        if not scores:
            out[c] = (np.zeros(len(T)), freq[c]); continue
        order = np.argsort(-np.array(scores), kind="mergesort")
        tp = np.concatenate(tps, 1)[:, order]; ig = np.concatenate(ign, 1)[:, order]
        fp = ~tp & ~ig
        ctp, cfp = np.cumsum(tp, 1), np.cumsum(fp, 1)
        ap_t = np.zeros(len(T))
        for ti in range(len(T)):
            rc = ctp[ti] / npos; pr = ctp[ti] / np.maximum(ctp[ti] + cfp[ti], 1e-12)
            pr = np.maximum.accumulate(pr[::-1])[::-1]
            idx = np.searchsorted(rc, R, side="left")
            q = np.array([pr[k] if k < len(pr) else 0.0 for k in idx])
            ap_t[ti] = q.mean()
        out[c] = (ap_t, freq[c])
    allv = np.array([v[0] for v in out.values()])
    res = dict(AP=float(allv.mean()), AP50=float(allv[:, 0].mean()))
    for f in "rcf":
        v = [x[0] for x in out.values() if x[1] == f]
        res["AP" + f] = float(np.mean(v)) if v else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lvis", required=True); ap.add_argument("--zip", required=True)
    ap.add_argument("--hybrid"); ap.add_argument("--out")
    a = ap.parse_args()
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}
    recs = {}
    for zp, sub, keys in ((a.zip, "/I/", ("bank", "chunks")), (a.hybrid, "test/", ("hybrid10",))):
        if not zp:
            continue
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            if sub in n and n.endswith(".json"):
                r = json.loads(z.read(n)); recs.setdefault(r["image"], {})
                for k in keys:
                    recs[r["image"]][k] = r["soft"][k]
    files = [f for f in recs if all(k in recs[f] for k in ("bank", "chunks"))]
    ids = {fid[f] for f in files}; res = {}
    for m in ("chunks", "bank", "hybrid10"):
        if not all(m in recs[f] for f in files):
            continue
        dets = [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                for f in files for c, s, x1, y1, x2, y2 in recs[f][m]]
        res[m] = evaluate(L, dets, ids)
        print(f"{m:9s} " + "  ".join(f"{k} {v:.4f}" for k, v in res[m].items()), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
