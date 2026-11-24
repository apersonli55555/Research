#!/usr/bin/env python3
"""Cross-check two follow-up runs (e.g. the owner's GPU run vs an independent CPU run on a subset of its images):
for every (part, image, target, condition) in both, compare the top detection score and box.

  python ovd/bank/compare_runs.py RUN_A RUN_B
"""
import glob, json, os, sys
from collections import defaultdict

import numpy as np


def rows(d):
    out = {}
    for part in ("A", "B", "C", "F"):
        for p in glob.glob(os.path.join(d, part, "*.json")):
            for r in json.load(open(p))["rows"]:
                out[(part, r["image"], r["target"], r["cond"])] = r["dets"]
    return out


def main():
    a, b = rows(sys.argv[1]), rows(sys.argv[2])
    common = sorted(set(a) & set(b))
    print(f"{len(common)} lists in both runs")
    stat = defaultdict(lambda: dict(n=0, empty_mismatch=0, score=[], box=[]))
    for k in common:
        s = stat[(k[0], k[3])]; s["n"] += 1; x, y = a[k], b[k]
        if bool(x) != bool(y):
            s["empty_mismatch"] += 1; continue
        if x:
            s["score"].append(abs(x[0][0] - y[0][0])); s["box"].append(max(abs(u - v) for u, v in zip(x[0][1:], y[0][1:])))
    worst = 0.0
    for (part, cond), s in sorted(stat.items()):
        sc = max(s["score"], default=0.0); worst = max(worst, sc)
        print(f"  {part} {cond:14s} n={s['n']:4d}  top-score |diff| max {sc:.4f} median {np.median(s['score'] or [0]):.5f}  "
              f"top-box max {max(s['box'], default=0):.2f}px  empty/non-empty mismatches {s['empty_mismatch']}")
    print(f"largest top-score difference: {worst:.4f}")


if __name__ == "__main__":
    main()
