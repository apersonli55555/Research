#!/usr/bin/env python3
"""Amendment G1 / G3 analysis (CPU): official LVIS evaluation of the full-image run, and the tiny diagnosis.

  pip install lvis pycocotools
  python ovd/bank/full_image_analysis.py --dir full_image_v1 --lvis lvis_v1_val.json [--boot 200] [--workers 4]

G1: lvis-api LVISEval (bbox, 300 detections per image) on the run's images with the official annotations of those
images (federated pos/neg sets, not-exhaustive ignore). Methods: bank / chunks, raw / Soft-NMS. Primary contrast
bank_soft − chunks_soft for AP and AP50 with a paired image bootstrap (duplicated images re-indexed). Seconds and peak
GPU memory per image. G3: candidate recall at IoU 0.5 over all 900 boxes and the rank of the best matching box.
"""
import argparse, contextlib, copy, glob, io, json, os, random, sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

if not hasattr(np, "float"):          # lvis-api 0.5.3 uses the alias removed in NumPy 1.24 (np.float is float)
    np.float = float
from lvis import LVISEval, LVISResults  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from official_lvis_eval import Mem, resample  # noqa: E402

CONDS = [("bank", "soft"), ("chunks", "soft"), ("bank", "raw"), ("chunks", "raw")]
G = None


def evaluate(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        res = LVISResults(Mem(gt), dets, max_dets=300) if dets else None
        if res is None:
            return dict(AP=0.0, AP50=0.0)
        ev = LVISEval(Mem(gt), res, iou_type="bbox"); ev.run()
    return dict(ev.results)


def boot_one(args):
    seed, imgs, by = args
    rng = random.Random(seed); chosen = rng.choices(imgs, k=len(imgs)); out = {}
    for key in ("bank_soft", "chunks_soft"):
        g2, d2 = resample(G, by[key], chosen); r = evaluate(g2, d2); out[key] = (r["AP"], r["AP50"])
    return out


def main():
    global G
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args(); res = {}
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "I", "*.json")))]
    if recs:
        L = json.load(open(a.lvis)); files = {r["image"] for r in recs}
        cat_ids = sorted(c["id"] for c in L["categories"])
        fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}
        keep = {fid[f] for f in files}
        G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
                 images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
        del L
        dets = {}
        for m, v in CONDS:
            dets[f"{m}_{v}"] = [dict(image_id=fid[r["image"]], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                                for r in recs for c, s, x1, y1, x2, y2 in r[v][m]]
        print(f"== G1 full-image LVIS, official evaluator: {len(recs)} images, {len(G['annotations'])} boxes")
        res["G1"] = {}
        for k, d in dets.items():
            r = evaluate(G, copy.deepcopy(d)); res["G1"][k] = r
            print(f"  {k:12s} AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  AP75 {r['AP75']:.4f}  APr {r['APr']:.4f}  APc {r['APc']:.4f}  "
                  f"APf {r['APf']:.4f}  APs {r['APs']:.4f}  APm {r['APm']:.4f}  APl {r['APl']:.4f}")
        sec = {m: float(np.mean([r["seconds"][m] for r in recs])) for m in ("bank", "chunks")}
        mem = {m: float(np.median([r["peak_bytes"][m] for r in recs])) / 2 ** 20 for m in ("bank", "chunks")}
        res["G1"]["seconds"] = sec; res["G1"]["peak_mb"] = mem
        print(f"  mean s/image: bank {sec['bank']:.3f}, chunks {sec['chunks']:.3f} (chunks / bank = {sec['chunks'] / sec['bank']:.2f}x); "
              f"median peak extra GPU MB: bank {mem['bank']:.0f}, chunks {mem['chunks']:.0f}")
        if a.boot:
            imgs = sorted(i["id"] for i in G["images"]); by = {}
            for k in ("bank_soft", "chunks_soft"):
                by[k] = defaultdict(list)
                for d in dets[k]:
                    by[k][d["image_id"]].append(d)
            with Pool(a.workers) as pool:
                outs = pool.map(boot_one, [(s, imgs, by) for s in range(a.boot)])
            for j, name in ((0, "AP"), (1, "AP50")):
                v = sorted(o["bank_soft"][j] - o["chunks_soft"][j] for o in outs)
                d = res["G1"]["bank_soft"][name] - res["G1"]["chunks_soft"][name]
                lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
                res["G1"][f"diff_{name}"] = dict(diff=d, ci=[lo, hi], boot=a.boot)
                print(f"  bank_soft − chunks_soft {name}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} image resamples)")
    for ckpt in ("base", "tiny"):
        rows = [r for p in sorted(glob.glob(os.path.join(a.dir, "H_" + ckpt, "*.json"))) for r in json.load(open(p))["rows"]]
        if not rows:
            continue
        print(f"== G3 {ckpt}: {len({(r['image'], r['target']) for r in rows if r['cond'] == 'bank'})} present pairs")
        res["G3_" + ckpt] = {}
        for m in ("stock5", "bank"):
            x = [r for r in rows if r["cond"] == m]; rec = np.mean([r["best_iou"] >= 0.5 for r in x])
            ranks = [r["match_rank"] for r in x if r["match_rank"] is not None]
            sc = [r["match_score"] for r in x if r["match_score"] is not None]
            d = dict(instances=len(x), candidate_recall=float(rec), median_rank=float(np.median(ranks)), top1=float(np.mean([k == 1 for k in ranks])),
                     top10=float(np.mean([k <= 10 for k in ranks])), median_match_score=float(np.median(sc)),
                     match_score_ge_035=float(np.mean([s >= .35 for s in sc])))
            res["G3_" + ckpt][m] = d
            print(f"  {m:7s} instances {d['instances']}: candidate recall@0.5 {d['candidate_recall']:.3f}; best match rank median "
                  f"{d['median_rank']:.0f}, rank 1 {d['top1']:.3f}, top-10 {d['top10']:.3f}; its score median {d['median_match_score']:.3f}, "
                  f">= 0.35 {d['match_score_ge_035']:.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
