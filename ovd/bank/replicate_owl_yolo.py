#!/usr/bin/env python3
"""Replication of the OWLv2 and YOLO-World corrections on the 600-image confirmation set (run 3h images and requests).

  python ovd/bank/replicate_owl_yolo.py run --out OUT [--model owlv2|yoloworld] [--limit N]     # CPU or GPU, resumable
  python ovd/bank/replicate_owl_yolo.py analyze --out OUT --lvis lvis_v1_val.json [--boot 1000]

Same 2,669 (image, target) pairs and the same five-name requests as run 3h. Conditions per pair, all exported as the top 50
boxes with score >= 0.02 for the target (vocab_full.py conventions; YOLO-World with its per-class NMS at IoU 0.7):
  OWLv2       req_deployed  the five-name request with standard single-label output (a box reports the target only if
                            the target is its top class among the five names)
              fix           independent read-out: every box scored for the target (identical for every request, and
                            identical to the single-class output, because OWLv2 scores each query independently)
  YOLO-World  req           the five-name request (guide and head = request), target column
              req_deployed  as req, keeping only boxes whose top class is the target
              k1            the target alone
              fix           fixed guide over all 1,203 names + independent read-out (list-invariant by construction)
"""
import argparse, json, os, random, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from vocab_full import MAX_DETS, MIN_SCORE, Owl, YoloWorld, nms  # noqa: E402
from followup import Images, load_inputs  # noqa: E402


def export(boxes, s, use_nms, keep=None):
    cand = np.where(s >= MIN_SCORE)[0]
    if keep is not None:
        cand = cand[keep[cand]]
    if use_nms and len(cand):
        cand = cand[nms(boxes[cand], s[cand])]
    cand = cand[np.argsort(-s[cand])][:MAX_DETS]
    return [[round(float(s[i]), 7)] + [round(float(v), 3) for v in boxes[i]] for i in cand]


def run(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    images, names, targets, request, urls, _ = load_inputs()
    if a.limit:
        images = images[:a.limit]
    imgs = Images(None); col = {c: i for i, c in enumerate(names)}
    for model in a.model.split(","):
        d = Path(a.out) / model; d.mkdir(parents=True, exist_ok=True)
        left = [f for f in images if not (d / (f + ".json")).exists()]
        if not left:
            continue
        det = Owl(dev) if model == "owlv2" else YoloWorld(dev, names)
        for j, f in enumerate(left):
            img = imgs.load(f, urls[f]); img.filename_key = f; req = request[f]; rows = []
            if model == "owlv2":
                boxes, S = det(img, names)                                # exact for every vocabulary
                R = S[:, [col[n] for n in req]]
                for c in targets[f]:
                    s = S[:, col[c]]
                    rows.append(dict(image=f, target=c, cond="owl_req_deployed",
                                     dets=export(boxes, s, False, R.argmax(1) == req.index(c))))
                    rows.append(dict(image=f, target=c, cond="owl_fix", dets=export(boxes, s, False)))
            else:
                bq, Sq = det(img, req)
                bf, Sf = det(img, names, guide=names)
                for c in targets[f]:
                    k = req.index(c)
                    rows.append(dict(image=f, target=c, cond="yolo_req", dets=export(bq, Sq[:, k], True)))
                    rows.append(dict(image=f, target=c, cond="yolo_req_deployed", dets=export(bq, Sq[:, k], True, Sq.argmax(1) == k)))
                    b1, S1 = det(img, [c])
                    rows.append(dict(image=f, target=c, cond="yolo_k1", dets=export(b1, S1[:, 0], True)))
                    rows.append(dict(image=f, target=c, cond="yolo_fix", dets=export(bf, Sf[:, col[c]], True)))
            tmp = d / (f + ".tmp"); tmp.write_text(json.dumps(dict(rows=rows, request=req))); tmp.rename(d / (f + ".json"))
            if j % 50 == 0 or j == len(left) - 1:
                print(f"{model} {j + 1}/{len(left)}", flush=True)
        del det


def analyze(a):
    from bank_analysis import load, per_pair, class_aps, iou
    RUN = HERE.parent / "results" / "bank" / "fixed_bank_soft_confirm_v1"
    M, gt, notexh, _ = load(SimpleNamespace(lvis=a.lvis, run=str(RUN)))
    present = {(t["image"], t["target"]) for t in M["targets"] if t["present"]}
    TH = {"owl": 0.10, "yolo": 0.25}                                      # vocab_pilot.THRESH
    for model, conds, ref in [("owlv2", ["owl_fix", "owl_req_deployed"], "owl_fix"),
                              ("yoloworld", ["yolo_fix", "yolo_req", "yolo_req_deployed", "yolo_k1"], "yolo_fix")]:
        rows = [r for p in sorted((Path(a.out) / model).glob("*.json")) for r in json.load(open(p))["rows"]]
        if not rows:
            continue
        imgs = sorted({r["image"] for r in rows}); P = per_pair(rows, gt, notexh)
        full = {m: class_aps(P, m) for m in conds}; mean = lambda x: float(np.mean(list(x.values())))
        rng = random.Random(0); diffs = defaultdict(list)
        for _ in range(a.boot):
            w = defaultdict(int)
            for f in rng.choices(imgs, k=len(imgs)):
                w[f] += 1
            r0 = mean(class_aps(P, ref, w))
            for m in conds[1:]:
                diffs[m].append(r0 - mean(class_aps(P, m, w)))
        print(f"== {model} on the 600-image confirmation set ({len(imgs)} images, {len(rows) // len(conds)} pairs)")
        for m in conds:
            line = f"  {m:18s} mAP50 {mean(full[m]):.4f}"
            if m != ref:
                v = sorted(diffs[m]); lo, hi = v[int(.025 * len(v))], v[int(.975 * len(v)) - 1]
                line += f"   {ref} − {m}: {mean(full[ref]) - mean(full[m]):+.4f} [{lo:+.4f}, {hi:+.4f}]"
            print(line)
        # interference against the single-class output, on present targets, at the model's deployment threshold
        th = TH[model[:4] if model == "yoloworld" else "owl"]
        by = defaultdict(dict)
        for r in rows:
            by[(r["image"], r["target"])][r["cond"]] = r["dets"]
        single = "yolo_k1" if model == "yoloworld" else "owl_fix"          # OWLv2: independent read-out == single class
        for m in conds:
            if m == single:
                continue
            n = fl = 0
            for k, v in by.items():
                if k in present:
                    n += 1; x, y = v[m], v[single]
                    fl += bool(x and x[0][0] >= th) != bool(y and y[0][0] >= th)
            print(f"  detection flips vs single class ({single}), threshold {th}: {m} {fl}/{n} ({fl / max(n, 1):.1%})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "analyze"]); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="owlv2,yoloworld"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lvis"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyze(a)


if __name__ == "__main__":
    main()
