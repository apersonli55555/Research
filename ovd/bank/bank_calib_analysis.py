#!/usr/bin/env python3
"""Amendment I analysis (CPU): official LVIS evaluation of the class-wise normalized bank.

  python ovd/bank/bank_calib_analysis.py --dir DIR/dev  --lvis lvis_v1_val.json                   # choose the map
  python ovd/bank/bank_calib_analysis.py --dir DIR/test --lvis lvis_v1_val.json --map M --boot 200 # frozen test
Dev mode adds the ceiling diagnostic (bank Soft-NMS lists without the 300-per-image cap). Test mode adds the run-3l
chunks and bank outputs for the same images and bootstraps (paired, by image) the chosen map minus chunks.
"""
import argparse, contextlib, copy, glob, io, json, os, sys, zipfile
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402
from lvis import LVISEval, LVISResults  # noqa: E402


def evaluate_uncapped(gt, dets):
    with contextlib.redirect_stdout(io.StringIO()):
        res = LVISResults(F.Mem(gt), dets, max_dets=10000)
        ev = LVISEval(F.Mem(gt), res, iou_type="bbox"); ev.params.max_dets = 10000; ev.run()
    return dict(ev.results)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True)
    ap.add_argument("--map", default=""); ap.add_argument("--boot", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    recs = {json.load(open(p))["image"]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))}
    if a.map:
        z = zipfile.ZipFile(os.path.join(HERE, "..", "results", "bank", "full_image_v1", "full_image_v1_outputs.zip"))
        for n in z.namelist():
            if "/I/" in n and n.endswith(".json"):
                r = json.loads(z.read(n))
                if r["image"] in recs:
                    recs[r["image"]]["soft"]["chunks"] = r["soft"]["chunks"]; recs[r["image"]]["soft"]["bank_L4"] = r["soft"]["bank"]
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}; keep = {fid[f] for f in recs}
    F.G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
               images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L

    def to_dets(lists):
        return [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                for f, l in lists.items() for c, s, x1, y1, x2, y2 in l]

    methods = sorted({m for r in recs.values() for m in r["soft"]})
    dets = {m: to_dets({f: r["soft"][m] for f, r in recs.items()}) for m in methods}
    print(f"== {len(recs)} images, {len(F.G['annotations'])} boxes; official LVIS evaluator, 300 dets/image, Soft-NMS")
    res = {}
    for m in methods:
        r = F.evaluate(F.G, copy.deepcopy(dets[m])); res[m] = r
        print(f"  {m:8s} AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  APr {r['APr']:.4f}  APc {r['APc']:.4f}  APf {r['APf']:.4f}", flush=True)
    if all("soft_uncapped" in r for r in recs.values()):
        u = to_dets({f: r["soft_uncapped"] for f, r in recs.items()})
        n = sum(len(r["soft_uncapped"]) for r in recs.values()) / len(recs)
        r = evaluate_uncapped(F.G, u); res["none_uncapped"] = r
        print(f"  ceiling: bank Soft-NMS lists without the 300 cap (mean {n:.0f} dets/image): AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  "
              f"APr {r['APr']:.4f}  APc {r['APc']:.4f}  APf {r['APf']:.4f}  (gain over capped {r['AP'] - res['none']['AP']:+.4f})")
    if a.map and a.boot:
        imgs = sorted(i["id"] for i in F.G["images"])
        by = {key: defaultdict(list) for key in (a.map, "chunks")}
        for key in by:
            for d in dets[key]:
                by[key][d["image_id"]].append(d)
        global_by = {"bank_soft": by[a.map], "chunks_soft": by["chunks"]}
        with Pool(a.workers) as pool:
            outs = pool.map(F.boot_one, [(s, imgs, global_by) for s in range(a.boot)])
        for j, name in ((0, "AP"), (1, "AP50")):
            v = sorted(o["bank_soft"][j] - o["chunks_soft"][j] for o in outs); d = res[a.map][name] - res["chunks"][name]
            lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]; res[f"diff_{name}"] = dict(diff=d, ci=[lo, hi], boot=a.boot)
            print(f"  {a.map} − chunks {name}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} image resamples)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
