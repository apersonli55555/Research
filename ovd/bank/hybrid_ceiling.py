#!/usr/bin/env python3
"""Amendment H, EXPLORATORY (written after the hybrid development result, before the hybrid test result; uses only the
run-3l outputs): where is the bank-vs-chunks per-class AP gap, and are those classes among the bank's own top-K classes?

  python ovd/bank/hybrid_ceiling.py FULL_IMAGE_V1_DIR lvis_v1_val.json
"""
# exploratory: on run-3l outputs, where is the bank-vs-chunks per-class AP gap, and are those classes bank top-K candidates?
import json, sys, os, copy, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid_diag as D
from collections import defaultdict
fi = sys.argv[1]; L = json.load(open(sys.argv[2])); K = [10, 25, 50]
import glob
recs = {json.load(open(p))["image"]: json.load(open(p)) for p in sorted(glob.glob(fi + "/I/*.json"))}
cat_ids = sorted(c["id"] for c in L["categories"]); fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}
keep = {fid[f] for f in recs}; freq = {c["id"]: c["frequency"] for c in L["categories"]}
G = dict(info={}, licenses=[], categories=L["categories"], images=[i for i in L["images"] if i["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
gt = defaultdict(set)
for x in G["annotations"]: gt[x["image_id"]].add(x["category_id"])
apc = {}
for m in ("bank", "chunks"):
    dets = [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[a, b, x - a, y - b], score=s) for f, r in recs.items() for c, s, a, b, x, y in r["soft"][m]]
    apc[m], _ = D.per_class(G, dets)
cls = sorted(set(apc["bank"]) & set(apc["chunks"]))
gap = {c: apc["chunks"][c] - apc["bank"][c] for c in cls}
tot = sum(gap.values()) / len(cls)
print(f"{len(recs)} images, {len(cls)} classes; mean AP gap chunks-bank {tot:+.4f}")
for fr in "rcf":
    s = [c for c in cls if freq[c] == fr]; print(f"  freq {fr}: {len(s)} classes, contribution to mean gap {sum(gap[c] for c in s)/len(cls):+.4f}")
for k in K:
    hit = defaultdict(lambda: [0, 0])
    for f, r in recs.items():
        order = []
        for c, *_ in r["raw"]["bank"]:
            if c not in order: order.append(c)
        cand = {cat_ids[c] for c in order[:k]}
        for c in gt[fid[f]]:
            hit[c][1] += 1; hit[c][0] += c in cand
    cov = {c: hit[c][0] / hit[c][1] for c in cls}
    inc = sum(gap[c] * cov[c] for c in cls) / len(cls)
    print(f"  K={k}: share of (image, present class) pairs whose class is a candidate {sum(h[0] for h in hit.values())/sum(h[1] for h in hit.values()):.3f}; "
          f"gap weighted by candidate coverage (rough upper bound on what replacing candidate rows by chunks rows could recover) {inc:+.4f}; "
          f"by freq " + ", ".join(f"{fr} {np.mean([cov[c] for c in cls if freq[c]==fr]):.2f}" for fr in "rcf"))
    lo = [c for c in cls if cov[c] < 0.5]; hi = [c for c in cls if cov[c] >= 0.5]
    print(f"    K={k}: classes candidate in <50% of their images: {len(lo)}, gap contribution {sum(gap[c] for c in lo)/len(cls):+.4f}; >=50%: {len(hi)}, {sum(gap[c] for c in hi)/len(cls):+.4f}")
