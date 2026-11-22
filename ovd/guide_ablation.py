#!/usr/bin/env python3
"""YOLO-World fixed-guide ablation on the DEVELOPMENT split (pilot images) only.

Which fixed guide list should the correction use? Each variant scores every COCO class with the head = all 80 COCO
names (independent read-out) and the C2fAttn guide = a fixed list; mAP50 over the pilot pairs.

  python ovd/guide_ablation.py --pilot OUT_PILOT --lvis-plan OUT_FULL/plan_lvis.json
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import match, wap  # noqa: E402
from vocab_full import YoloWorld, target_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--lvis-plan", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from PIL import Image
    P = json.load(open(os.path.join(a.pilot, "plan.json")))
    coco = P["names"]; lvis = json.load(open(a.lvis_plan))["names"]
    rng = lambda s: np.random.default_rng(s)
    guides = {"coco80": coco, "lvis1203": lvis,
              **{f"lvis_rand{k}_s{s}": list(rng(1000 + s).choice(lvis, k, replace=False)) for k in (20, 80, 200, 500) for s in (0, 1)},
              "single(target only)": None}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = YoloWorld(dev, sorted(set(coco) | set(lvis)))
    imgs = sorted({p["image"] for p in P["plan"]})[: a.limit or None]
    pairs = {}
    for p in P["plan"]:
        if p["image"] in imgs:
            pairs.setdefault(p["image"], []).append(p)
    res = {}
    for gname, G in guides.items():
        recs = {}; npos = {}
        for f in imgs:
            img = Image.open(os.path.join(a.pilot, "images", f)).convert("RGB"); img.filename_key = f
            if G is not None:
                boxes, S = det(img, coco, guide=G)
            for p in pairs[f]:
                c = p["target"]; gts = [g["box"] for g in P["gt"][f] if g["cls"] == c]
                if G is None:                                   # single-class inference (guide = head = target)
                    boxes1, S1 = det(img, [c]); r = target_record(boxes1, S1, 0, gts, 0.25, True)
                else:
                    r = target_record(boxes, S, coco.index(c), gts, 0.25, True)
                recs.setdefault(c, []).extend(match(r["dets"], gts)); npos[c] = npos.get(c, 0) + len(gts)
        aps = [wap(np.array([s for s, _ in recs[c]]), np.array([float(t) for _, t in recs[c]]), np.ones(len(recs[c])), npos[c])
               for c in recs if npos[c] > 0]
        res[gname] = dict(mAP50=float(np.mean(aps)), n_classes=len(aps), guide_size=len(G) if G else 1)
        print(f"{gname:22s} guide size {res[gname]['guide_size']:5d}  mAP50 {res[gname]['mAP50']:.4f}", flush=True)
    json.dump(res, open(os.path.join(a.pilot, "guide_ablation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
