#!/usr/bin/env python3
"""Expert improvement #1: score the run-3h detections with the OFFICIAL LVIS evaluator (lvis-api, LVISEval).

  pip install lvis pycocotools
  python ovd/bank/official_lvis_eval.py --lvis lvis_v1_val.json [--run ovd/results/bank/fixed_bank_soft_confirm_v1]

The study scores chosen (image, class) pairs, not every class on every image. So the official ground truth is restricted
to exactly those pairs, using LVIS's own fields: for each image, the annotations of its present target classes;
neg_category_ids = its verified-absent target classes; not_exhaustive_category_ids = the official list restricted to its
targets. LVISEval then applies its own rules (federated pos/neg image sets, not-exhaustive ignore, 101-point
interpolation, 300 detections per image). Reports official AP50 (and AP, APr/c/f, APs/m/l) per method, next to the
study's own mAP50, and a paired image-bootstrap 95% CI of official AP50 differences.
"""
import argparse, contextlib, copy, glob, io, json, os, random, sys
from collections import defaultdict

import numpy as np

if not hasattr(np, "float"):          # lvis-api 0.5.3 uses the alias removed in NumPy 1.24 (np.float is float)
    np.float = float
from lvis import LVIS, LVISEval, LVISResults

class Mem(LVIS):
    """The official LVIS ground-truth class, loaded from a dict instead of a file path."""
    def __init__(self, d):
        import logging
        self.logger = logging.getLogger("lvis.lvis"); self.dataset = d
        with contextlib.redirect_stdout(io.StringIO()):
            self._create_index()


METHODS = ["stock5", "stock5_soft", "fixed", "fixed_soft", "bank", "bank_soft"]


def build(L, M):
    keep_img = set(M["images"])
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from vocab_full import clean_lvis
    ids = sorted(c["id"] for c in L["categories"]); raw = {c["id"]: c["name"] for c in L["categories"]}
    name_to_id = dict(zip(clean_lvis([raw[k] for k in ids]), ids))
    file_of = {im["id"]: im["coco_url"].split("/")[-1] for im in L["images"]}
    id_of = {f: i for i, f in file_of.items() if f in keep_img}
    pos, neg = defaultdict(set), defaultdict(set)
    for t in M["targets"]:
        (pos if t["present"] else neg)[t["image"]].add(name_to_id[t["target"]])
    images = []
    for im in L["images"]:
        f = file_of[im["id"]]
        if f in keep_img:
            x = dict(im); tgt = pos[f] | neg[f]
            x["neg_category_ids"] = sorted(neg[f])
            x["not_exhaustive_category_ids"] = sorted(set(im.get("not_exhaustive_category_ids", [])) & tgt)
            images.append(x)
    anns = [a for a in L["annotations"] if a["image_id"] in file_of and file_of[a["image_id"]] in keep_img
            and a["category_id"] in pos[file_of[a["image_id"]]]]
    gt = dict(info=L.get("info", {}), licenses=L.get("licenses", []), images=images, annotations=anns,
              categories=L["categories"])
    return gt, name_to_id, id_of


def official(gt_lvis, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        res = LVISResults(gt_lvis, dets, max_dets=300)
        ev = LVISEval(gt_lvis, res, iou_type="bbox"); ev.run()
    return dict(ev.results)


def resample(gt, dets_by_img, chosen):
    """Ground truth and detections for a bootstrap sample of images (duplicates get fresh image ids)."""
    images, anns, dets = [], [], []
    by_img = defaultdict(list)
    for a in gt["annotations"]:
        by_img[a["image_id"]].append(a)
    img = {im["id"]: im for im in gt["images"]}
    nid, aid = 1, 1
    for i in chosen:
        x = dict(img[i]); x["id"] = nid; images.append(x)
        for a in by_img[i]:
            y = dict(a); y["id"] = aid; y["image_id"] = nid; anns.append(y); aid += 1
        for d in dets_by_img.get(i, []):
            y = dict(d); y["image_id"] = nid; dets.append(y)
        nid += 1
    return dict(gt, images=images, annotations=anns), dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lvis", required=True)
    ap.add_argument("--run", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "bank", "fixed_bank_soft_confirm_v1"))
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L = json.load(open(a.lvis)); M = json.load(open(os.path.join(a.run, "manifest.json")))
    gt, name_to_id, id_of = build(L, M)
    print(f"restricted official GT: {len(gt['images'])} images, {len(gt['annotations'])} boxes, "
          f"{sum(len(i['neg_category_ids']) for i in gt['images'])} verified-negative pairs")
    rows = [r for p in sorted(glob.glob(os.path.join(a.run, "per_image", "*.json"))) for r in json.load(open(p))["rows"]]
    dets = defaultdict(list)
    for r in rows:
        for s, x1, y1, x2, y2 in r["dets"]:
            dets[r["cond"]].append(dict(image_id=id_of[r["image"]], category_id=name_to_id[r["target"]],
                                        bbox=[x1, y1, x2 - x1, y2 - y1], score=s))
    G = Mem(copy.deepcopy(gt))
    out = {}
    for m in METHODS:
        r = official(G, copy.deepcopy(dets[m]))
        out[m] = r
        print(f"  {m:12s} official AP50 {r['AP50']:.4f}  AP {r['AP']:.4f}  APr {r['APr']:.4f} APc {r['APc']:.4f} "
              f"APf {r['APf']:.4f}  APs {r['APs']:.4f} APm {r['APm']:.4f} APl {r['APl']:.4f}")
    if a.boot:
        imgs = sorted(i["id"] for i in gt["images"]); rng = random.Random(0)
        by = {m: defaultdict(list) for m in METHODS}
        for m in METHODS:
            for d in dets[m]:
                by[m][d["image_id"]].append(d)
        diffs = defaultdict(list)
        for b in range(a.boot):
            chosen = rng.choices(imgs, k=len(imgs)); vals = {}
            for m in ("bank_soft", "stock5", "stock5_soft", "fixed_soft"):
                g2, d2 = resample(gt, by[m], chosen)
                vals[m] = official(Mem(g2), d2)["AP50"]
            for m in ("stock5", "stock5_soft", "fixed_soft"):
                diffs[m].append(vals["bank_soft"] - vals[m])
        for m, v in diffs.items():
            v = sorted(v); lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
            d = out["bank_soft"]["AP50"] - out[m]["AP50"]
            print(f"  official AP50 bank_soft − {m}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} resamples)")
            out[m]["bank_soft_minus_AP50"] = dict(diff=d, ci=[lo, hi], boot=a.boot)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
