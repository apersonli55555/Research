#!/usr/bin/env python3
"""Amendment H analysis (CPU): official full-image LVIS evaluation of the hybrid outputs.

  python ovd/bank/hybrid_analysis.py --dir DIR/dev  --lvis lvis_v1_val.json                       # choose K
  python ovd/bank/hybrid_analysis.py --dir DIR/test --lvis lvis_v1_val.json --k K --boot 200      # frozen test
In test mode the run-3l chunks outputs (results/bank/full_image_v1/full_image_v1_outputs.zip) are the comparator.
"""
import argparse, copy, glob, io, json, os, sys, zipfile
from collections import defaultdict
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True)
    ap.add_argument("--k", type=int, default=0); ap.add_argument("--boot", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    recs = {json.load(open(p))["image"]: json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))}
    if a.k:                                                   # add the run-3l chunks outputs for the same images
        z = zipfile.ZipFile(os.path.join(HERE, "..", "results", "bank", "full_image_v1", "full_image_v1_outputs.zip"))
        for n in z.namelist():
            if "/I/" in n and n.endswith(".json"):
                r = json.loads(z.read(n))
                if r["image"] in recs:
                    for v in ("raw", "soft"):
                        recs[r["image"]][v]["chunks"] = r[v]["chunks"]; recs[r["image"]][v]["bank_L4"] = r[v]["bank"]
                    recs[r["image"]].setdefault("seconds_L4", r["seconds"])
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}; keep = {fid[f] for f in recs}
    F.G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
               images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L
    methods = sorted({m for r in recs.values() for m in r["soft"]})
    dets = {f"{m}_{v}": [dict(image_id=fid[f], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                         for f, r in recs.items() if m in r[v] for c, s, x1, y1, x2, y2 in r[v][m]]
            for m in methods for v in ("soft", "raw")}
    print(f"== {len(recs)} images, {len(F.G['annotations'])} boxes; official LVIS evaluator, 300 dets/image")
    res = {}
    for k in sorted(dets, key=lambda x: (not x.endswith("soft"), x)):
        r = F.evaluate(F.G, copy.deepcopy(dets[k])); res[k] = r
        print(f"  {k:16s} AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  APr {r['APr']:.4f}  APc {r['APc']:.4f}  APf {r['APf']:.4f}")
    nc = defaultdict(list)
    for r in recs.values():
        for k, n in r["n_cand"].items():
            nc[k].append((n, r["prompt_passes"][k]))
    for k, v in sorted(nc.items(), key=lambda x: int(x[0])):
        print(f"  K={k}: mean candidates {sum(n for n, _ in v) / len(v):.1f}, mean prompt passes {sum(p for _, p in v) / len(v):.2f}")
    if a.k and a.boot:
        imgs = sorted(i["id"] for i in F.G["images"]); hk = f"hybrid{a.k}_soft"
        by = {key: defaultdict(list) for key in (hk, "chunks_soft")}
        for key in by:
            for d in dets[key]:
                by[key][d["image_id"]].append(d)
        global_by = {"bank_soft": by[hk], "chunks_soft": by["chunks_soft"]}   # boot_one compares these two keys
        with Pool(a.workers) as pool:
            outs = pool.map(F.boot_one, [(s, imgs, global_by) for s in range(a.boot)])
        for j, name in ((0, "AP"), (1, "AP50")):
            v = sorted(o["bank_soft"][j] - o["chunks_soft"][j] for o in outs); d = res[hk][name] - res["chunks_soft"][name]
            lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]; res[f"diff_{name}"] = dict(diff=d, ci=[lo, hi], boot=a.boot)
            print(f"  {hk} − chunks_soft {name}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] ({a.boot} image resamples)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
