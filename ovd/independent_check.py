#!/usr/bin/env python3
"""Independent re-computation of headline mAP50 values from the raw result files.

Shares no code with full_analysis.py: its own IoU, greedy matching and all-point interpolated AP (written from the
definition, with a different algorithm: precision envelope by explicit max over later ranks). Compares with the
numbers in a full_summary.json and reports the largest absolute difference.

  python ovd/independent_check.py --out OUT --summary OUT/full_summary.json
"""
import argparse, json, os


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def ap(scored, npos):
    scored = sorted(scored, key=lambda t: -t[0])
    tp = 0; prec, rec = [], []
    for i, (_, ok) in enumerate(scored, 1):
        tp += ok; prec.append(tp / i); rec.append(tp / npos)
    total, prev_r = 0.0, 0.0
    for i in range(len(scored)):
        if scored[i][1]:
            total += (rec[i] - prev_r) * max(prec[i:]); prev_r = rec[i]
    return total


def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--summary", required=True)
    a = p.parse_args()
    S = json.load(open(a.summary)); worst = 0.0; n = 0
    for key, summ in S.items():
        if "/" not in key or "groups" not in summ:
            continue
        ds, model = key.split("/")
        P = json.load(open(os.path.join(a.out, f"plan_{ds}.json"))); info = P["info"]
        R = [json.loads(l) for l in open(os.path.join(a.out, f"full_{ds}_{model}.jsonl")) if l.strip().startswith("{")]
        pairs = {(r["image"], r["target"]) for r in R}
        present = {(x["image"], x["target"]) for x in P["plan"] if x["present"]}
        for grp in ("k1", "all", "FIX"):
            conds = sorted({r["cond"] for r in R if (r["cond"] == grp) or (grp == "all" and r["cond"].startswith("all_"))})
            vals = []
            for cd in conds:
                per = {}
                for r in R:
                    if r["cond"] != cd:
                        continue
                    f, c = r["image"], r["target"]
                    gts = [g["box"] for g in info[f]["gt"] if g["cls"] == c]
                    dets = r["dets"] if cd == "FIX" else [d for d in r["dets"] if d[5] == 1]
                    used = [False] * len(gts); ign = c in info[f].get("not_exh", [])
                    for d in sorted(dets, key=lambda x: -x[0]):
                        best, bj = 0.5, -1
                        for j, g in enumerate(gts):
                            if not used[j]:
                                v = iou(d[1:5], g)
                                if v >= best:
                                    best, bj = v, j
                        if bj >= 0:
                            used[bj] = True; per.setdefault(c, [[], 0])[0].append((d[0], 1))
                        elif not ign:
                            per.setdefault(c, [[], 0])[0].append((d[0], 0))
                for f, c in pairs:
                    if (f, c) in present:
                        per.setdefault(c, [[], 0])[1] += sum(g["cls"] == c for g in info[f]["gt"])
                aps = [ap(s, npos) for s, npos in per.values() if npos > 0]
                vals.append(sum(aps) / len(aps))
            mine = sum(vals) / len(vals); theirs = summ["groups"][grp]["mAP50_deployed"]
            worst = max(worst, abs(mine - theirs)); n += 1
            print(f"{key:18s} {grp:4s} independent {mine:.4f}  reported {theirs:.4f}  diff {mine - theirs:+.2e}", flush=True)
    print(f"{n} values checked; largest absolute difference {worst:.2e}")


if __name__ == "__main__":
    main()
