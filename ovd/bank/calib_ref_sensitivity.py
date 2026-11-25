#!/usr/bin/env python3
"""Sensitivity of the class-wise shift to the choice of reference images (alternative explanation: the gain hinges on
the particular 300 reference images). CPU, no model passes: uses the saved per-image reference max-logits and the saved
uncapped Soft-NMS lists of the 124 development images.

  python ovd/bank/calib_ref_sensitivity.py --ref DIR/ref --dev DIR/dev --lvis lvis_v1_val.json --out OUT.json

Variants of the reference set: all 300 (should reproduce the development result), three disjoint thirds (100 each),
ten disjoint tenths (30 each), five bootstrap resamples of the 300 (seeded). For each: per-class mean logit mu_c,
shift sigma(logit s - mu_c + mu_g) applied to the uncapped lists, top 300 per image, official LVIS AP.
"""
import argparse, copy, glob, io, json, math, os, random, sys, contextlib

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import full_image_analysis as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--dev", required=True)
    ap.add_argument("--lvis", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    X = [json.load(open(p))["max_logit"] for p in sorted(glob.glob(os.path.join(a.ref, "*.json")))]
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.dev, "*.json")))]
    L = json.load(open(a.lvis)); cat_ids = sorted(c["id"] for c in L["categories"])
    fid = {im["coco_url"].split("/")[-1]: im["id"] for im in L["images"]}; keep = {fid[r["image"]] for r in recs}
    G = dict(info=L.get("info", {}), licenses=L.get("licenses", []), categories=L["categories"],
             images=[im for im in L["images"] if im["id"] in keep], annotations=[x for x in L["annotations"] if x["image_id"] in keep])
    del L
    n, C = len(X), len(X[0])
    order = list(range(n))
    variants = {"all_300": order}
    for k in range(3):
        variants[f"third_{k + 1}"] = order[k::3]
    for k in range(10):
        variants[f"tenth_{k + 1}"] = order[k::10]
    for k in range(5):
        rng = random.Random(f"ref-boot-{k}"); variants[f"boot_{k + 1}"] = [rng.randrange(n) for _ in range(n)]

    def ap_for(mu):
        mu_g = sum(mu) / C; dets = []
        for r in recs:
            lst = []
            for c, s, x1, y1, x2, y2 in r["soft_uncapped"]:
                t = min(max(s, 1e-6), 1 - 1e-6); m = math.log(t / (1 - t)) - mu[c] + mu_g
                lst.append((1 / (1 + math.exp(-m)), c, x1, y1, x2, y2))
            lst.sort(key=lambda x: -x[0])
            dets += [dict(image_id=fid[r["image"]], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                     for s, c, x1, y1, x2, y2 in lst[:300]]
        with contextlib.redirect_stdout(io.StringIO()):
            return F.evaluate(G, dets)

    res = {}
    none = []
    for r in recs:
        lst = sorted(r["soft_uncapped"], key=lambda x: -x[1])[:300]
        none += [dict(image_id=fid[r["image"]], category_id=cat_ids[c], bbox=[x1, y1, x2 - x1, y2 - y1], score=s)
                 for c, s, x1, y1, x2, y2 in lst]
    res["none"] = F.evaluate(G, copy.deepcopy(none))
    print(f"  none (no map)  AP {res['none']['AP']:.4f}", flush=True)
    for name, idx in variants.items():
        mu = [sum(X[i][c] for i in idx) / len(idx) for c in range(C)]
        r = ap_for(mu); res[name] = r
        print(f"  {name:10s} ({len(idx):3d} ref images)  AP {r['AP']:.4f}  AP50 {r['AP50']:.4f}  APr {r['APr']:.4f}  "
              f"APc {r['APc']:.4f}  APf {r['APf']:.4f}", flush=True)
    for grp in ("third", "tenth", "boot"):
        v = [res[k]["AP"] for k in res if k.startswith(grp)]
        res[f"{grp}_summary"] = dict(min=min(v), max=max(v), mean=sum(v) / len(v), gain_over_none_min=min(v) - res["none"]["AP"])
        print(f"  {grp}: AP {min(v):.4f}..{max(v):.4f} (mean {sum(v) / len(v):.4f}); smallest gain over no map "
              f"{min(v) - res['none']['AP']:+.4f}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
