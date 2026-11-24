#!/usr/bin/env python3
"""Expert improvement #9: qualitative figure from the follow-up outputs (no inference).

  python ovd/bank/make_qual_figure.py --dir followup_v1 --lvis lvis_v1_val.json --out ovd/paper/fig_qual.pdf

Selection rule (fixed before looking at any example): among present-target pairs, keep those whose standard-prompt
DETECTION flips between the two image-independent requests (part B: stock5_indA vs stock5_indB) — the target is
detected (top score >= 0.35, Grounding DINO's deployment threshold, with IoU >= 0.5 to a ground-truth box) under one
request and not under the other. Order them by MD5 of image + target and show the first N. Every panel draws the top box (solid: score >= 0.35, dashed: below; green: correct, orange: wrong). Each row: request A, request
B (standard prompt), fixed bank + Soft-NMS (identical under both requests by construction). The number of qualifying
pairs and how often the bank detects them are printed, so the examples can be read against the totals.
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_analysis import iou, load  # noqa: E402
from followup_analysis import RUN, rows_of, soft_nms  # noqa: E402

THR = 0.35


def hit(dets, gts):
    if not dets or dets[0][0] < THR:
        return False
    return any(iou(dets[0][1:], g) >= 0.5 for g in gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--lvis", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--test-any", action="store_true", help="smoke test only: show any present pair")
    a = ap.parse_args()
    M, gt, notexh, saved = load(SimpleNamespace(lvis=a.lvis, run=RUN))
    present = {(t["image"], t["target"]) for t in M["targets"] if t["present"]}
    by = defaultdict(dict)
    for r in rows_of(a.dir, "B"):
        by[(r["image"], r["target"])][r["cond"]] = r
    for r in saved:
        if r["cond"] == "bank_soft":
            by[(r["image"], r["target"])]["bank_soft"] = r
    cand = []
    for k, v in by.items():
        if k in present and "stock5_indA" in v and "stock5_indB" in v:
            g = gt[k]
            if a.test_any or hit(v["stock5_indA"]["dets"], g) != hit(v["stock5_indB"]["dets"], g):
                cand.append(k)
    n_present = sum(1 for k in by if k in present and "stock5_indA" in by[k])
    bank_ok = sum(hit(by[k]["bank_soft"]["dets"], gt[k]) for k in cand)
    top_ok = lambda d, g: bool(d) and any(iou(d[0][1:], x) >= 0.5 for x in g)
    bank_top = sum(top_ok(by[k]["bank_soft"]["dets"], gt[k]) for k in cand)
    print(f"present pairs {n_present}; standard detection flips between the two requests {len(cand)} "
          f"({len(cand) / max(n_present, 1):.1%}); of these the bank + Soft-NMS detects {bank_ok} "
          f"at >= 0.35 and has a correct top box (any score) on {bank_top}")
    cand.sort(key=lambda k: hashlib.md5((k[0] + "/" + k[1]).encode()).hexdigest())
    show = cand[:a.n]
    if not show:
        print("no qualifying pairs"); return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    sizes = json.load(open(os.path.join(a.dir, "G_sizes.json")))
    fig, ax = plt.subplots(len(show), 3, figsize=(3.35, 1.02 * len(show)), squeeze=False)   # one paper column
    for i, (f, c) in enumerate(show):
        im = Image.open(os.path.join(a.dir, "G", f)); W, H = sizes[f]; s = im.size[0] / W
        cols = [("stock5_indA", "standard, request A"), ("stock5_indB", "standard, request B"), ("bank_soft", "bank, any request")]
        for j, (cond, title) in enumerate(cols):
            x = ax[i][j]; x.imshow(im); x.set_xticks([]); x.set_yticks([])
            r = by[(f, c)][cond]; d = r["dets"]
            if d:                                       # top box always drawn; dashed = below the 0.35 threshold
                b = d[0]; ok = any(iou(b[1:], g) >= 0.5 for g in gt[(f, c)])
                x.add_patch(Rectangle((b[1] * s, b[2] * s), (b[3] - b[1]) * s, (b[4] - b[2]) * s, fill=False,
                                      lw=0.9, ls="-" if b[0] >= THR else "--", ec="#1b9e77" if ok else "#d95f02"))
                lab = f"{b[0]:.3f}" + ("" if b[0] >= THR else " (<0.35)")
            else:
                lab = "no box above 0.02"
            req = r.get("request")
            others = [n for n in req if n != c] if req else []
            sub = ("+ " + ", ".join(others[:2]) + ",\n" + ", ".join(others[2:])) if others else ""
            x.set_title(f"{title}: {lab}", fontsize=4.6, pad=1.5)
            if sub:
                x.set_xlabel(sub, fontsize=3.9, labelpad=0.8)
            if j == 0:
                x.set_ylabel(f"“{c}”", fontsize=5.5, labelpad=1)
    fig.tight_layout(pad=0.15, h_pad=0.35, w_pad=0.15)
    fig.savefig(a.out, dpi=300); print("saved", a.out, "pairs:", show)


if __name__ == "__main__":
    main()
