#!/usr/bin/env python3
"""Inter-annotator agreement (Cohen's kappa) on the FallOrNot test clips.

Two ways to call it:

1. Two separate label files (each annotator labelled in their own copy):
     python vision/agreement.py --oops OOPS_DIR --a labels_me.csv --b labels_friend.csv

2. Inspect one labels.csv to see whether a second person wrote into the same file
   (labels.csv is an append-only log; the newest entry for a clip wins):
     python vision/agreement.py --oops OOPS_DIR --inspect OOPS_DIR/labels.csv

Only test-set clips (the hash-based random sample) labelled fall / not_fall by both
annotators are compared; "unsure" is reported separately. The kappa 95% CI comes from a
bootstrap over clips.
"""
import argparse, csv, os
from collections import Counter, defaultdict

import numpy as np


def read_log(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((r["name"], r["label"], r.get("labeler", "")))
    return rows


def final_labels(rows):
    out = {}
    for n, lab, _ in rows:
        out[n] = lab
    return {k: v for k, v in out.items() if v}


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True, help="oops_mine.py output dir (for scores.csv)")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--inspect")
    a = ap.parse_args()
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        test = {r["name"] for r in csv.DictReader(fh) if r["random_sample"] == "1"}

    if a.inspect:
        rows = read_log(a.inspect)
        per = defaultdict(list)
        for n, lab, who in rows:
            per[n].append((lab, who))
        multi = {n: v for n, v in per.items() if len([x for x in v if x[0]]) > 1}
        changed = {n: v for n, v in multi.items() if len({x[0] for x in v if x[0]}) > 1}
        print(f"rows: {len(rows)}; clips: {len(per)} ({sum(n in test for n in per)} in the test set)")
        print(f"labeller names in the file: {dict(Counter(w for _, _, w in rows))}")
        print(f"clips labelled more than once: {len(multi)} (test set: {sum(n in test for n in multi)})")
        print(f"clips whose label changed between entries: {len(changed)} (test set: {sum(n in test for n in changed)})")
        if len(multi) > 20:
            print("\nMany clips were labelled more than once. If a second person labelled in this same file,")
            print("their answers replaced the first ones (the newest entry wins). Tell your AI assistant before running anything else.")
        else:
            print("\nNo sign that a second person wrote into this file (repeats are few, e.g. undo/redo).")
        return

    la, lb = final_labels(read_log(a.a)), final_labels(read_log(a.b))
    both = sorted(n for n in test if n in la and n in lb)
    unsure = sum(1 for n in both if "unsure" in (la[n], lb[n]))
    pairs = [(la[n], lb[n]) for n in both if la[n] in ("fall", "not_fall") and lb[n] in ("fall", "not_fall")]
    names = [n for n in both if la[n] in ("fall", "not_fall") and lb[n] in ("fall", "not_fall")]
    if not pairs:
        raise SystemExit("no test clips labelled fall/not_fall by both annotators")
    x = np.array([p[0] == "fall" for p in pairs], int)
    y = np.array([p[1] == "fall" for p in pairs], int)
    k = kappa(x, y)
    rng = np.random.default_rng(0)
    bs = [kappa(x[i], y[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    tab = Counter(pairs)
    print(f"test clips labelled by both: {len(both)}; compared (fall/not_fall both): {len(pairs)}; involving 'unsure': {unsure}")
    print(f"A fall & B fall: {tab[('fall', 'fall')]} | A fall & B not: {tab[('fall', 'not_fall')]} | "
          f"A not & B fall: {tab[('not_fall', 'fall')]} | A not & B not: {tab[('not_fall', 'not_fall')]}")
    print(f"raw agreement: {(x == y).mean():.3f}")
    print(f"Cohen's kappa: {k:.3f} [95% CI {np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    print("disagreements:")
    for n, (p, q) in zip(names, pairs):
        if p != q:
            print(f"  A={p:9s} B={q:9s} {n}")


if __name__ == "__main__":
    main()
