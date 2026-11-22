#!/usr/bin/env python3
"""Class-wise normalisation / reference calibration as alternative corrections (DEVELOPMENT split only).

Keeps single-label output (each box reports one class) but decides the label on calibrated scores:
  raw      top-1 over the vocabulary on raw scores (standard)
  znorm    top-1 on z-scores, z_c = (s_c - mu_c) / sd_c, with mu_c, sd_c from the calibration images
  rank     top-1 on per-class reference quantiles (empirical CDF of s_c on the calibration images)
  indep    no label competition (the invariant read-out)
The reported score of a kept detection is always the raw s_c, so only the label decision changes.
Images: the pilot's 500 development images, split by hash into a calibration half and an evaluation half.
Models: OWLv2 (raw scores) and YOLO-World with the fixed COCO guide (raw anchor scores before NMS; per-class NMS 0.7).

  python ovd/norm_ablation.py --pilot OUT_PILOT --model owlv2|yoloworld
"""
import argparse, hashlib, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import match, wap  # noqa: E402
from vocab_full import Owl, YoloWorld, nms  # noqa: E402
from vocab_pilot import THRESH, iou  # noqa: E402

TOPB = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--model", required=True, choices=["owlv2", "yoloworld"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    from PIL import Image
    P = json.load(open(os.path.join(a.pilot, "plan.json"))); names = P["names"]; th = THRESH[a.model]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = Owl(dev) if a.model == "owlv2" else YoloWorld(dev, names)
    imgs = sorted({p["image"] for p in P["plan"]})[: a.limit or None]
    calib = [f for f in imgs if int(hashlib.md5(("calib" + f).encode()).hexdigest(), 16) % 2 == 0]
    evals = [f for f in imgs if f not in calib]
    store = {}
    for f in imgs:                                         # one pass per image; keep the top boxes with all scores
        img = Image.open(os.path.join(a.pilot, "images", f)).convert("RGB"); img.filename_key = f
        boxes, S = det(img, names) if a.model == "owlv2" else det(img, names, guide=names)
        keep = np.argsort(-S.max(1))[:TOPB]
        store[f] = (boxes[keep].astype(np.float32), S[keep].astype(np.float32))
    C = np.concatenate([store[f][1] for f in calib])      # calibration statistics per class
    mu, sd = C.mean(0), C.std(0) + 1e-6
    srt = np.sort(C, 0)
    cdf = lambda S: np.stack([np.searchsorted(srt[:, j], S[:, j]) / len(srt) for j in range(S.shape[1])], 1)
    near20 = {p["target"]: p["conds"]["near20"] for p in P["plan"]}
    res = {}
    for vname in ("k1", "near20", "all80"):
        for var in ("raw", "znorm", "rank", "indep"):
            recs, npos, flips_det = {}, {}, []
            for p in P["plan"]:
                f, c = p["image"], p["target"]
                if f not in evals:
                    continue
                V = [c] if vname == "k1" else (near20[c] if vname == "near20" else names)
                cols = [names.index(v) for v in V]; ci = V.index(c)
                boxes, S = store[f]; Sv = S[:, cols]
                D = {"raw": Sv, "znorm": (Sv - mu[cols]) / sd[cols], "rank": cdf(Sv) if var == "rank" else None, "indep": None}[var]
                s = Sv[:, ci]
                own = np.ones(len(s), bool) if var == "indep" else (D.argmax(1) == ci)
                cand = np.where(own & (s >= 0.02))[0]
                if a.model == "yoloworld" and len(cand):
                    cand = cand[nms(boxes[cand], s[cand])]
                cand = cand[np.argsort(-s[cand])][:50]
                dets = [[float(s[i])] + list(map(float, boxes[i])) for i in cand]
                gts = [g["box"] for g in P["gt"][f] if g["cls"] == c]
                recs.setdefault(c, []).extend(match(dets, gts)); npos[c] = npos.get(c, 0) + len(gts)
                for g in gts:
                    m = iou(boxes[cand], np.array(g)) >= 0.5 if len(cand) else np.zeros(0, bool)
                    flips_det.append(bool(len(cand) and (s[cand][m] >= th).any()))
            aps = [wap(np.array([x for x, _ in recs[c]]), np.array([float(t) for _, t in recs[c]]), np.ones(len(recs[c])), npos[c])
                   for c in recs if npos[c] > 0]
            res[f"{vname}/{var}"] = dict(mAP50=float(np.mean(aps)), detected=flips_det)
    k1 = res["k1/raw"]["detected"]
    out = {}
    for k, v in res.items():
        out[k] = dict(mAP50=v["mAP50"], detected_rate=float(np.mean(v["detected"])),
                      flips_vs_k1=float(np.mean(np.array(v["detected"]) != np.array(k1))))
        print(f"{a.model} {k:14s} mAP50 {out[k]['mAP50']:.4f}  detected {out[k]['detected_rate']:.3f}  flips vs k1 {out[k]['flips_vs_k1']:.3f}", flush=True)
    out["_split"] = dict(calib_images=len(calib), eval_images=len(evals))
    json.dump(out, open(os.path.join(a.pilot, f"norm_ablation_{a.model}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
