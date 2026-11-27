#!/usr/bin/env python3
"""Independent re-check (no lvis-api) of the Amendment K LVIS results: the six YOLO-World methods scored with
independent_lvis_ap.evaluate on the same per-image outputs.

  python ovd/bank/yolo_calib_independent.py --lvis lvis_v1_val.json --dir DIR/lvis/test --out OUT.json
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from independent_lvis_ap import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lvis", required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dir, "*.json")))]
    ids = {fid[r["image"]] for r in recs}; res = {}
    for p in ("P", "S"):
        for m in ("none", "shift", "z"):
            dets = [dict(image_id=fid[r["image"]], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                    for r in recs for c, s, x1, y1, x2, y2 in r[p][m]]
            res[f"{p}_{m}"] = evaluate(L, dets, ids)
            print(f"{p}_{m:6s} " + "  ".join(f"{k} {v:.4f}" for k, v in res[f'{p}_{m}'].items()), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
