#!/usr/bin/env python3
"""Expert idea 2 (PLAN amendment E): fixed-prompt boxes + independent per-class readout for Grounding DINO.

  python ovd/shared_readout.py --out OUT_WITH_plan_lvis --pairs 100 --coco-plan PILOT/plan.json
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import match, wap  # noqa: E402
from vocab_full import GDino, iou  # noqa: E402
from read_only_text import ReadOnlyText  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--pairs", type=int, default=100); ap.add_argument("--coco-plan", required=True)
    ap.add_argument("--variant", default="A", choices=["A", "B", "C"]); ap.add_argument("--no-recall", action="store_true")
    a = ap.parse_args()
    import torch
    from PIL import Image
    from transformers.models.grounding_dino.modeling_grounding_dino import generate_masks_with_special_tokens_and_transfer_map as gm
    P = json.load(open(os.path.join(a.out, "plan_lvis.json"))); names = P["names"]; info = P["info"]
    F = json.load(open(a.coco_plan))["names"]
    g = GDino("cuda" if torch.cuda.is_available() else "cpu"); model = g.model
    ctx = g.chunks(names)
    tcache = {}

    def text_tokens(c):                         # projected text-backbone tokens of "c ." alone (image-independent)
        if c not in tcache:
            x = g.proc.tokenizer(c + " .", return_tensors="pt").to(g.dev)
            m, pos = gm(x["input_ids"])
            if m.ndim == 3:                                    # same 3D -> 4D step as GroundingDinoModel.forward
                m = m[:, None, :, :]
            with torch.no_grad():
                h = model.model.text_backbone(x["input_ids"], m, x.get("token_type_ids"), pos, return_dict=True).last_hidden_state
                h = model.model.text_projection(h)[0]
            tcache[c] = h[1:1 + len(g.tokens(c))]            # the class's own wordpieces
        return tcache[c]

    rot = ReadOnlyText(g)

    def fixed_pass(img):
        o = rot.fixed(img, F)
        W, H = img.size; b = o.pred_boxes[0].float().cpu().numpy()
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H, (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
        return boxes, o.last_hidden_state[0]

    plan = P["plan"][: a.pairs]; by = {}
    for p in plan:
        by.setdefault(p["image"], []).append(p)
    tfix, tdec = [], []
    rec_fix = {0.5: [], 0.75: []}; rec_chunk = {0.5: [], 0.75: []}
    per_new, per_fix = {}, {}
    FIXR = {(r["image"], r["target"]): r for r in map(json.loads, open(os.path.join(a.out, "full_lvis_gdino.jsonl"))) if r["cond"] == "FIX"}
    for f, pairs in by.items():
        img = Image.open(os.path.join(a.out, "images", f)).convert("RGB"); g._bb.clear()
        t0 = time.time(); boxes, Q = fixed_pass(img); tfix.append(time.time() - t0)
        for p in pairs:
            c = p["target"]; gts = [x["box"] for x in info[f]["gt"] if x["cls"] == c]; ign = c in info[f].get("not_exh", [])
            if gts and not a.no_recall:                       # (a) box recall, fixed prompt vs the target's canonical chunk
                cb, _ = g(img, next(ch for ch in ctx if c in ch))
                for gt in gts:
                    for t in (0.5, 0.75):
                        rec_fix[t].append(iou(boxes, np.array(gt)).max() >= t); rec_chunk[t].append(iou(cb, np.array(gt)).max() >= t)
            bx = boxes
            with torch.no_grad():                              # (b) independent readout on the shared queries
                if a.variant == "C":
                    t0 = time.time(); bn, lg = rot.class_decode(c); tdec.append(time.time() - t0)
                    s = torch.sigmoid(lg.max(1).values).float().cpu().numpy()
                    bn = bn.float().cpu().numpy(); W, H = img.size
                    bx = np.stack([(bn[:, 0] - bn[:, 2] / 2) * W, (bn[:, 1] - bn[:, 3] / 2) * H, (bn[:, 0] + bn[:, 2] / 2) * W, (bn[:, 1] + bn[:, 3] / 2) * H], 1)
                else:
                    T = text_tokens(c) if a.variant == "A" else rot.class_tokens(c)
                    s = torch.sigmoid((Q @ T.T).max(1).values).float().cpu().numpy()
            order = np.argsort(-s)[:50]
            dets = [[float(s[i])] + list(map(float, bx[i])) for i in order if s[i] >= 0.02]
            for store, D in ((per_new, dets), (per_fix, sorted(FIXR[(f, c)]["dets"], key=lambda d: -d[0]) if (f, c) in FIXR else None)):
                if D is None:
                    continue
                e = store.setdefault(c, [[], 0]); e[1] += len(gts)
                e[0] += [(sc, ok) for sc, ok in match(D, gts) if ok or not ign]
    mp = lambda per: float(np.mean([wap(np.array([x for x, _ in r]), np.array([float(t) for _, t in r]), np.ones(len(r)), n) for r, n in per.values() if n > 0]))
    if not a.no_recall:
      print(f"(a) box recall IoU0.5: fixed prompt {np.mean(rec_fix[0.5]):.3f} vs canonical chunk {np.mean(rec_chunk[0.5]):.3f}; "
          f"IoU0.75: {np.mean(rec_fix[0.75]):.3f} vs {np.mean(rec_chunk[0.75]):.3f}  (n={len(rec_fix[0.5])})")
    print(f"timing (this machine): fixed pass {np.mean(tfix):.2f}s/image" + (f", per-class decoder {np.mean(tdec):.3f}s" if tdec else ""))
    print(f"(b) variant {a.variant} mAP50: independent readout on shared queries {mp(per_new):.4f} vs fixed chunks {mp(per_fix):.4f}")


if __name__ == "__main__":
    main()
