#!/usr/bin/env python3
"""PLAN Amendment K analysis (CPU): official evaluation of YOLO-World with the frozen class-wise maps.

  python ovd/bank/yolo_calib_analysis.py --dataset lvis --dir DIR/lvis/test --gt lvis_v1_val.json --boot 200
  python ovd/bank/yolo_calib_analysis.py --dataset coco --dir DIR/coco/test --gt instances_val2017.json --boot 200

Methods: pipeline P (Soft-NMS, primary) and S (YOLO-World standard NMS), each with maps none / shift / z. Official
evaluator: lvis-api (300 detections per image) or pycocotools (maxDets 100). Pre-registered primary: P_shift − P_none
(AP, paired image bootstrap); S_shift − S_none is reported the same way.
"""
import argparse, contextlib, copy, glob, io, json, os, random, sys
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402  (lvis-api evaluate; also applies the NumPy alias fix)
from official_lvis_eval import resample  # noqa: E402

G, DS = None, None
METHODS = [f"{p}_{m}" for p in ("P", "S") for m in ("none", "shift", "z")]
PAIRS = [("P_shift", "P_none"), ("S_shift", "S_none")]


def coco_eval(gt, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        c = COCO(); c.dataset = copy.deepcopy(gt); c.createIndex()
        if not dets:
            return dict(AP=0.0, AP50=0.0)
        d = c.loadRes(copy.deepcopy(dets)); e = COCOeval(c, d, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return dict(AP=s[0], AP50=s[1], AP75=s[2], APs=s[3], APm=s[4], APl=s[5])


def evaluate(gt, dets):
    return F.evaluate(gt, copy.deepcopy(dets)) if DS == "lvis" else coco_eval(gt, dets)


def boot_one(args):
    seed, imgs, by = args
    rng = random.Random(seed); chosen = rng.choices(imgs, k=len(imgs)); out = {}
    for key, v in by.items():
        g2, d2 = resample(G, v, chosen); r = evaluate(g2, d2); out[key] = (r["AP"], r["AP50"])
    return out


def main():
    global G, DS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["lvis", "coco"], required=True)
    ap.add_argument("--dir", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args(); DS = a.dataset
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    L = json.load(open(a.gt)); cat_ids = sorted(c["id"] for c in L["categories"])
    if DS == "lvis":
        fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}
    else:
        fid = {im["file_name"]: im["id"] for im in L["images"]}
    keep = {fid[r["image"]] for r in recs}
    G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
             images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L
    print(f"{DS}: {len(recs)} images, {len(G['annotations'])} annotations", flush=True)
    dets = {}
    for meth in METHODS:
        p, m = meth.split("_")
        dets[meth] = [dict(image_id=fid[r["image"]], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                      for r in recs for c, s, x1, y1, x2, y2 in r[p][m]]
    res = {"dataset": DS, "n_images": len(recs)}
    for meth in METHODS:
        r = evaluate(G, dets[meth]); res[meth] = r
        print(f"  {meth:8s} " + "  ".join(f"{k} {v:.4f}" for k, v in r.items() if k in ("AP", "AP50", "APr", "APc", "APf", "APs", "APm", "APl")), flush=True)
    imgs = sorted(i["id"] for i in G["images"])
    by = {}
    for meth in METHODS:
        by[meth] = defaultdict(list)
        for d in dets[meth]:
            by[meth][d["image_id"]].append(d)
    if a.boot:
        need = sorted({x for pr in PAIRS for x in pr})
        with Pool(a.workers) as pool:
            outs = pool.map(boot_one, [(s, imgs, {k: by[k] for k in need}) for s in range(a.boot)])
        for x, y in PAIRS:
            for j, name in ((0, "AP"), (1, "AP50")):
                v = sorted(o[x][j] - o[y][j] for o in outs); dd = res[x][name] - res[y][name]
                lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
                res[f"{x}-{y}_{name}"] = dict(diff=dd, ci=[lo, hi], boot=a.boot)
                print(f"  {x} − {y} {name}: {dd:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} image resamples)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
