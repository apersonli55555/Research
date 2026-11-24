#!/usr/bin/env python3
"""Amendment H, EXPLORATORY (written after the development result): where does the hybrid change AP? CPU.

  python ovd/bank/hybrid_diag.py --dir DIR/dev --lvis lvis_v1_val.json [--k 10] [--other chunks]

Per class, official LVIS AP (precision averaged over IoU 0.5:0.95 and recall, classes with ground truth in these images)
for bank_soft, hybridK_soft (and, in test mode, chunks_soft). Classes are split by whether they were a candidate (their row
replaced by the short prompt) in at least one image where they have ground truth. For candidate rows: the top score of the
row under the bank and under the short prompt (and chunks), and the share of the 300 per-image slots taken by candidates.
"""
import argparse, copy, glob, io, json, os, sys, zipfile, contextlib
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402
from lvis import LVISEval, LVISResults  # noqa: E402


def per_class(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = LVISEval(F.Mem(gt), LVISResults(F.Mem(gt), dets, max_dets=300), iou_type="bbox"); ev.run()
    P = ev.eval["precision"]                                                   # T, R, K, A
    out = {}
    for k, cid in enumerate(ev.params.cat_ids):
        p = P[:, :, k, 0]; p = p[p > -1]
        if p.size:
            out[cid] = float(p.mean())
    return out, dict(ev.results)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True)
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--test", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args(); hk = f"hybrid{a.k}"
    recs = {json.load(open(p))["image"]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))}
    if a.test:
        z = zipfile.ZipFile(os.path.join(HERE, "..", "results", "bank", "full_image_v1", "full_image_v1_outputs.zip"))
        for n in z.namelist():
            if "/I/" in n and n.endswith(".json"):
                r = json.loads(z.read(n))
                if r["image"] in recs:
                    recs[r["image"]]["soft"]["chunks"] = r["soft"]["chunks"]; recs[r["image"]]["raw"]["chunks"] = r["raw"]["chunks"]
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}; keep = {fid[f] for f in recs}
    G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
             images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L
    gt_cls = defaultdict(set)
    for x in G["annotations"]:
        gt_cls[x["image_id"]].add(x["category_id"])
    methods = ["bank", hk] + (["chunks"] if a.test else [])
    ap_c, summ = {}, {}
    for m in methods:
        dets = [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                for f, r in recs.items() for c, s, x1, y1, x2, y2 in r["soft"][m]]
        ap_c[m], summ[m] = per_class(G, copy.deepcopy(dets))
    cand_gt = set(); cand_any = set(); top = defaultdict(list); share = defaultdict(list)
    for f, r in recs.items():
        raw = r["raw"]["bank"]; order = []
        for c, *_ in raw:
            if c not in order:
                order.append(c)
        cand = set(order[: a.k]); cand_any |= {cat_ids[c] for c in cand}
        cand_gt |= {cat_ids[c] for c in cand} & gt_cls[fid[f]]
        for m in methods:
            best = {}
            for c, s, *_ in r["raw"][m]:
                best[c] = max(best.get(c, 0), s)
            for c in cand:
                top[(m, cat_ids[c] in gt_cls[fid[f]])].append(best.get(c, 0.0))
            share[m].append(sum(c in cand for c, *_ in r["soft"][m]) / max(len(r["soft"][m]), 1))
    cls = sorted(set.intersection(*[set(ap_c[m]) for m in methods]))
    res = dict(n_images=len(recs), k=a.k, summary=summ, n_classes_with_gt=len(cls))
    print(f"== {len(recs)} images, {len(cls)} classes with ground truth; per-class AP (mean over IoU 0.5:0.95)")
    for grp, sel in (("candidate (row replaced in >=1 image with GT)", [c for c in cls if c in cand_gt]),
                     ("never a candidate where present", [c for c in cls if c not in cand_gt])):
        row = {m: float(np.mean([ap_c[m][c] for c in sel])) if sel else float("nan") for m in methods}
        d = [ap_c[hk][c] - ap_c["bank"][c] for c in sel]
        res[grp] = dict(n=len(sel), mean_ap=row, n_up=sum(x > 1e-9 for x in d), n_down=sum(x < -1e-9 for x in d),
                        sum_change_hybrid_minus_bank=float(sum(d)))
        print(f"  {grp}: {len(sel)} classes; mean AP " + ", ".join(f"{m} {v:.4f}" for m, v in row.items())
              + f"; hybrid vs bank: {res[grp]['n_up']} up, {res[grp]['n_down']} down, summed change {sum(d):+.3f}")
    for present in (True, False):
        v = {m: float(np.median(top[(m, present)])) for m in methods}
        res[f"median_top_raw_score_candidate_rows_present={present}"] = v
        print(f"  candidate rows, class {'present' if present else 'absent'} in image (n={len(top[('bank', present)])}): median top score "
              + ", ".join(f"{m} {x:.3f}" for m, x in v.items()))
    res["share_of_300_slots_to_candidates"] = {m: float(np.mean(share[m])) for m in methods}
    print("  share of the Soft-NMS top-300 list taken by candidate classes: "
          + ", ".join(f"{m} {np.mean(share[m]):.3f}" for m in methods))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
