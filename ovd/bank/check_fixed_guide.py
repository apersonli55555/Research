#!/usr/bin/env python3
"""Check (manuscript Sec. 3, YOLO-World): with the guide fixed to a constant list, a class's raw scores do not depend on the
requested vocabulary. CPU, a few minutes.

  python ovd/bank/check_fixed_guide.py --coco instances_val2017.json --images /tmp/coco_val2017 --n 20 --out OUT.json

Per image (first n of the Amendment K COCO test list order, i.e. the files given sorted by name), per target = every
COCO class present: guide = all 80 COCO names (fixed order), head = (a) target alone, (b) target + 4 random names,
(c) target + 19 random names, (d) all 80 in a seeded random order. Reports the max |difference| of the target column's
raw scores (all anchors) and of the boxes between (a) and (b)-(d); also the same with the guide equal to the head
(stock behaviour) for contrast.
"""
import argparse, hashlib, json, os, random, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from vocab_full import YoloWorld  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--coco", required=True); ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--out")
    a = ap.parse_args()
    from PIL import Image
    C = json.load(open(a.coco)); cats = sorted(C["categories"], key=lambda c: c["id"]); names = [c["name"] for c in cats]
    cid = {c["id"]: c["name"] for c in cats}; fid = {im["file_name"]: im["id"] for im in C["images"]}
    present = {}
    for x in C["annotations"]:
        present.setdefault(x["image_id"], set()).add(cid[x["category_id"]])
    files = [f for f in sorted(os.listdir(a.images)) if f in fid and fid[f] in present][: a.n]
    Y = YoloWorld("cpu", names)
    fixed, stock = [], []
    for f in files:
        im = Image.open(os.path.join(a.images, f)).convert("RGB"); im.filename_key = f
        for t in sorted(present[fid[f]]):
            rng = random.Random(hashlib.md5((f + t).encode()).hexdigest()); others = [n for n in names if n != t]
            vocabs = [[t], [t] + rng.sample(others, 4), [t] + rng.sample(others, 19), rng.sample(names, 80)]
            ref_b, ref_s = Y(im, vocabs[0], guide=names); ref_s = ref_s[:, 0]
            sb, ss = Y(im, vocabs[0]); ss = ss[:, 0]
            for v in vocabs[1:]:
                b, S = Y(im, v, guide=names); s = S[:, v.index(t)]
                fixed.append((float(np.abs(s - ref_s).max()), float(np.abs(b - ref_b).max())))
                b2, S2 = Y(im, v); s2 = S2[:, v.index(t)]
                stock.append(float(np.abs(s2 - ss).max()))
        print(f"{f}: {len(fixed)} comparisons, fixed-guide max |dscore| so far {max(x[0] for x in fixed):.2e}", flush=True)
    res = dict(n_images=len(files), n_comparisons=len(fixed),
               fixed_guide_max_abs_score_diff=max(x[0] for x in fixed), fixed_guide_max_abs_box_diff_px=max(x[1] for x in fixed),
               stock_max_abs_score_diff=max(stock), stock_mean_of_max_abs_score_diff=float(np.mean(stock)))
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
