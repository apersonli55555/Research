#!/usr/bin/env python3
"""Anchored-context test for Grounding DINO on the DEVELOPMENT split (PLAN amendment C).

  python ovd/anchor_dev.py --pilot OUT_PILOT --lvis-plan OUT_FULL/plan_lvis.json --pairs 40 [--anchors 40]
Writes OUT_PILOT/anchor_gdino.jsonl (same record format as vocab_full) and prints the comparison with the original
Grounding DINO runs on the same pairs (results_gdino.jsonl from the pilot).
"""
import argparse, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import match, wap  # noqa: E402
from vocab_full import GDino, target_record  # noqa: E402

TH = 0.35


def anchors(coco, lvis, n):
    cand = [x for x in lvis if x not in set(coco)]
    return sorted(cand, key=lambda x: hashlib.md5(x.encode()).hexdigest())[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True); ap.add_argument("--lvis-plan", required=True)
    ap.add_argument("--pairs", type=int, default=40); ap.add_argument("--anchors", type=int, default=40)
    a = ap.parse_args()
    import torch
    from PIL import Image
    P = json.load(open(os.path.join(a.pilot, "plan.json")))
    A = anchors(P["names"], json.load(open(a.lvis_plan))["names"], a.anchors)
    print("anchors:", A, flush=True)
    out = os.path.join(a.pilot, f"anchor{a.anchors}_gdino.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["image"], r["target"], r["cond"]) for r in map(json.loads, open(out))}
    g = GDino("cuda" if torch.cuda.is_available() else "cpu")
    plan = P["plan"][: a.pairs]
    with open(out, "a") as fh:
        for p in plan:
            img = Image.open(os.path.join(a.pilot, "images", p["image"])).convert("RGB")
            gts = [x["box"] for x in P["gt"][p["image"]] if x["cls"] == p["target"]]
            for cn, V in p["conds"].items():
                if (p["image"], p["target"], cn) in done:
                    continue
                rest = [v for v in V if v not in A]
                prompt = A + rest
                if len(g.chunks(prompt)) > 1:                 # too long: split V into pieces that fit WITH the anchors
                    pieces, cur = [], []
                    for v in rest:
                        if cur and len(g.chunks(A + cur + [v])) > 1:
                            pieces.append(cur); cur = []
                        cur.append(v)
                    pieces.append(cur)
                    prompt = A + next(pc for pc in pieces if p["target"] in pc)
                boxes, S = g(img, prompt)
                r = dict(image=p["image"], target=p["target"], cond=cn, present=p["present"])
                r.update(target_record(boxes, S, prompt.index(p["target"]), gts, TH, False))
                fh.write(json.dumps(r) + "\n"); fh.flush()
    # comparison with original inference on the same pairs
    keys = {(p["image"], p["target"]) for p in plan}
    load = lambda f: {(r["image"], r["target"], r["cond"]): r for r in map(json.loads, open(f)) if (r["image"], r["target"]) in keys}
    O, N = load(os.path.join(a.pilot, "results_gdino.jsonl")), load(out)
    for name, R, dep in (("original", O, True), ("anchored", N, False)):
        dS, fl, recs, npos = [], [], {}, {}
        for p in plan:
            k1 = R[(p["image"], p["target"], "k1")]
            for cn in p["conds"]:
                r = R[(p["image"], p["target"], cn)]
                if p["present"]:
                    for x, y in zip(r["instances"], k1["instances"]):
                        dS.append(abs(x["raw"] - y["raw"]))
                        v = (lambda z: z["deployed"]) if dep else (lambda z: z["raw"])
                        fl.append((v(x) >= TH) != (v(y) >= TH))
        grp = lambda c: "all80" if c.startswith("all80") else c.split("_")[0]
        maps = {}
        for g_ in sorted({grp(c) for c in plan[0]["conds"]}):
            vals = []
            for cn in [c for c in plan[0]["conds"] if grp(c) == g_]:
                per = {}
                for p in plan:
                    r = R[(p["image"], p["target"], cn)]
                    gts = [x["box"] for x in P["gt"][p["image"]] if x["cls"] == p["target"]]
                    dets = [d for d in r["dets"] if (len(d) < 6 or d[5] == 1)] if dep else r["dets"]
                    m = match(sorted(dets, key=lambda d: -d[0]), gts)
                    e = per.setdefault(p["target"], [[], 0]); e[0] += m; e[1] += len(gts)
                aps = [wap(np.array([s for s, _ in v_[0]]), np.array([float(t) for _, t in v_[0]]), np.ones(len(v_[0])), v_[1]) for v_ in per.values() if v_[1] > 0]
                vals.append(np.mean(aps))
            maps[g_] = round(float(np.mean(vals)), 4)
        print(f"{name:9s} mean |score change| vs own k1 {np.mean(dS):.4f}  flips vs own k1 {np.mean(fl):.4f}  mAP50 {maps}", flush=True)


if __name__ == "__main__":
    main()
