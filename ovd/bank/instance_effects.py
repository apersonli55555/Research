#!/usr/bin/env python3
"""Amendment G2: what a change of request does to individual objects (saved run-3j outputs; CPU).

  python ovd/bank/instance_effects.py --data results/bank/followup_t4_v1/followup_v1_data.zip

For the two image-independent requests A and B of every present (image, target) pair (Grounding DINO base, standard
five-class prompt), detections with score >= 0.35 (deployment threshold) are greedily matched by score to the target's
ground-truth instances at IoU >= 0.5. Per instance: found under both, only A, only B, or neither. Per pair whose top
box passes 0.35 under both requests and matches an instance under both: does the top box switch to a different
instance of the same class? 95% CIs: image bootstrap (3,000 resamples, seed 0).
"""
import argparse, io, json, os, random, sys, zipfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
from bank_analysis import iou  # noqa: E402

TH = 0.35


def matched(dets, gts):
    used = [False] * len(gts)
    for d in sorted(dets, key=lambda d: -d[0]):
        if d[0] < TH:
            break
        best, bj = 0.5, -1
        for j, g in enumerate(gts):
            v = iou(d[1:5], g)
            if not used[j] and v >= best:
                best, bj = v, j
        if bj >= 0:
            used[bj] = True
    return used


def top_instance(dets, gts):
    if not dets or dets[0][0] < TH:
        return None
    v = [iou(dets[0][1:5], g) for g in gts]
    j = max(range(len(gts)), key=v.__getitem__)
    return j if v[j] >= 0.5 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out")
    a = ap.parse_args()
    gt = json.load(open(os.path.join(HERE, "confirm_gt.json")))
    rows = defaultdict(dict)
    with zipfile.ZipFile(a.data) as z:
        for n in z.namelist():
            if "/B/" in n and n.endswith(".json"):
                for r in json.loads(z.read(n))["rows"]:
                    if r["cond"] in ("stock5_indA", "stock5_indB") and r["target"] in gt.get(r["image"], {}):
                        rows[(r["image"], r["target"])][r["cond"]] = r["dets"]
    per_img = defaultdict(lambda: defaultdict(int))
    for (f, c), v in rows.items():
        g = gt[f][c]; A = matched(v["stock5_indA"], g); B = matched(v["stock5_indB"], g); s = per_img[f]
        for x, y in zip(A, B):
            s["inst"] += 1; s["both"] += x and y; s["only_A"] += x and not y; s["only_B"] += y and not x; s["neither"] += not x and not y
        s["pairs"] += 1
        ta, tb = top_instance(v["stock5_indA"], g), top_instance(v["stock5_indB"], g)
        if ta is not None and tb is not None:
            s["top_both"] += 1; s["top_switch"] += ta != tb
            s["top_switch_multi"] += ta != tb and len(g) > 1; s["top_both_multi"] += len(g) > 1
    tot = defaultdict(int)
    for s in per_img.values():
        for k, x in s.items():
            tot[k] += x
    imgs = sorted(per_img); rng = random.Random(0)

    def ci(num, den):
        vals = []
        for _ in range(3000):
            ch = rng.choices(imgs, k=len(imgs)); n = sum(per_img[f][num] for f in ch) if isinstance(num, str) else sum(sum(per_img[f][k] for k in num) for f in ch)
            d = sum(per_img[f][den] for f in ch); vals.append(n / d if d else 0)
        vals.sort(); return vals[75], vals[2924]

    one = tot["only_A"] + tot["only_B"]
    res = dict(instances=tot["inst"], pairs=tot["pairs"], both=tot["both"], only_A=tot["only_A"], only_B=tot["only_B"],
               neither=tot["neither"], one_only=one, found_any=tot["inst"] - tot["neither"],
               one_only_of_found=one / max(tot["inst"] - tot["neither"], 1), one_only_rate=one / tot["inst"], one_only_ci=ci(("only_A", "only_B"), "inst"),
               top_both_matched=tot["top_both"], top_switch=tot["top_switch"], top_switch_rate=tot["top_switch"] / max(tot["top_both"], 1),
               top_both_matched_multi_instance=tot["top_both_multi"], top_switch_multi_instance=tot["top_switch_multi"])
    print(json.dumps(res, indent=1))
    print(f"Of {tot['inst']} present instances ({tot['pairs']} pairs): found under both requests {tot['both']}, only A "
          f"{tot['only_A']}, only B {tot['only_B']}, neither {tot['neither']}; found under just one: {one} "
          f"({one / tot['inst']:.1%}, CI {res['one_only_ci'][0]:.1%}-{res['one_only_ci'][1]:.1%}). Top box passes and matches under "
          f"both requests on {tot['top_both']} pairs; it switches to another instance of the class on {tot['top_switch']} "
          f"({tot['top_switch'] / max(tot['top_both'], 1):.1%}; {tot['top_switch_multi']} of {tot['top_both_multi']} multi-instance pairs).")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
