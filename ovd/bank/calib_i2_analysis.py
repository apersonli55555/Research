#!/usr/bin/env python3
"""Amendment I2 analysis (CPU): the class-wise normalization applied to standard chunked inference (fairness baseline).

  python ovd/bank/calib_i2_analysis.py --bank DIR/test --chunks DIR/chunks/test --lvis lvis_v1_val.json --boot 200

--bank: bank_calib.py test outputs (bank, maps none/shift/z); --chunks: bank_calib.py --method chunks test outputs
(chunks with their own reference statistics, maps none/shift/z). Pre-registered comparisons (PLAN Amendment I2), paired
image bootstrap: (1) chunks+shift − chunks (none); (2) bank+shift − chunks+shift. The other maps are reported.
"""
import argparse, copy, glob, json, os, sys
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402


def load(d, prefix):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p)); out[r["image"]] = {prefix + k: v for k, v in r["soft"].items()}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank", required=True); ap.add_argument("--chunks", required=True)
    ap.add_argument("--lvis", required=True); ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--out")
    ap.add_argument("--pairs", default="chunks_shift:chunks_none,bank_shift:chunks_shift",
                    help="comma-separated x:y pairs to bootstrap (default: the two pre-registered comparisons)")
    a = ap.parse_args()
    B, C = load(a.bank, "bank_"), load(a.chunks, "chunks_")
    files = sorted(set(B) & set(C))
    print(f"bank {len(B)}, chunks {len(C)}, common {len(files)} images", flush=True)
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}; keep = {fid[f] for f in files}
    F.G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
               images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L
    recs = {f: {**B[f], **C[f]} for f in files}
    methods = sorted(recs[files[0]])
    dets = {m: [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                for f in files for c, s, x1, y1, x2, y2 in recs[f][m]] for m in methods}
    res = {"n_images": len(files)}
    for m in methods:
        r = F.evaluate(F.G, copy.deepcopy(dets[m])); res[m] = r
        print(f"  {m:13s} AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  APr {r['APr']:.4f}  APc {r['APc']:.4f}  APf {r['APf']:.4f}", flush=True)
    imgs = sorted(i["id"] for i in F.G["images"])
    by = {}
    for m in methods:
        by[m] = defaultdict(list)
        for d in dets[m]:
            by[m][d["image_id"]].append(d)
    for x, y in (q.split(":") for q in a.pairs.split(",")):
        g = {"bank_soft": by[x], "chunks_soft": by[y]}
        with Pool(a.workers) as pool:
            outs = pool.map(F.boot_one, [(s, imgs, g) for s in range(a.boot)])
        for j, name in ((0, "AP"), (1, "AP50")):
            v = sorted(o["bank_soft"][j] - o["chunks_soft"][j] for o in outs); d = res[x][name] - res[y][name]
            lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
            res[f"{x}-{y}_{name}"] = dict(diff=d, ci=[lo, hi], boot=a.boot)
            print(f"  {x} − {y} {name}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} image resamples)", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
