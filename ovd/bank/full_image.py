#!/usr/bin/env python3
"""Amendment G: full-image LVIS evaluation of the fixed class bank (G1) and the tiny-checkpoint diagnosis (G3). GPU, resumable.

  python ovd/bank/full_image.py --out /content/drive/MyDrive/ovd/full_image_v1 [--parts I,H] [--limit N]

Part I  1,000 fresh LVIS val images (bank/full_image_manifest.json). Grounding DINO base, all 1,203 classes per image:
        'chunks' = the canonical 256-token chunks (= standard full-vocabulary inference), 'bank' = fixed class bank.
        Per class: top 50 boxes with score >= 0.02. Saved per image: top 300 (class, score, box) raw, and after class-wise
        Gaussian Soft-NMS (sigma 0.5, threshold 0.02, top 50 per class) -- both methods; seconds (model passes to score
        tensors, backbone not shared between methods) and peak GPU memory.
Part H  600 confirmation images, present targets: for base and tiny, standard five-class prompt (run-3h requests) and
        bank: per ground-truth instance, the best IoU reached by any of the 900 boxes, and the score rank / score of the
        highest-scoring box with IoU >= 0.5 under the target's score.
"""
import argparse, json, sys, time, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from followup import Engine, Images, load_inputs, save  # noqa: E402

SIGMA, THR, TOPK, PER_IMAGE = 0.5, 0.02, 50, 300


def box_iou(a, b):
    """a (..., N, 4), b (..., M, 4) xyxy -> (..., N, M); same formula as bank_analysis.iou."""
    import torch
    lt = torch.maximum(a[..., :, None, :2], b[..., None, :, :2]); rb = torch.minimum(a[..., :, None, 2:], b[..., None, :, 2:])
    wh = (rb - lt).clamp(min=0); inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a)[..., :, None] + area(b)[..., None, :] - inter + 1e-9)


def soft_nms_batched(S, B):
    """Class-wise Gaussian Soft-NMS for many classes at once. S (C, K) scores sorted descending (padding < THR),
    B (C, K, 4). Returns (C, K) scores and boxes of the kept detections in selection order (score -1 = empty).
    Equivalent to followup_analysis.soft_nms applied to each row (tested in test_full_image.py)."""
    import torch
    C, K = S.shape
    S = S.clone(); active = S >= THR
    I = box_iou(B, B)
    out_s = torch.full((C, K), -1.0, dtype=S.dtype, device=S.device); out_b = torch.zeros_like(B)
    rows = torch.arange(C, device=S.device)
    for step in range(K):
        masked = torch.where(active, S, torch.full_like(S, -float("inf")))
        best = masked.argmax(1); best_s = masked[rows, best]
        ok = torch.isfinite(best_s)
        if not ok.any():
            break
        out_s[ok, step] = best_s[ok]; out_b[ok, step] = B[rows, best][ok]
        active[rows[ok], best[ok]] = False
        decay = torch.exp(-I[rows, best] ** 2 / SIGMA)                     # (C, K)
        S = torch.where(active & ok[:, None], S * decay, S)
        active &= S >= THR
    return out_s, out_b


def candidates(boxes_list, scores_list, cls_list, n_cls, dev):
    """Per class top-K boxes with score >= THR. boxes_list[i] (Q,4) px xyxy, scores_list[i] (Q, k_i), cls_list[i] class
    indices of those k_i columns. Returns S (C, K) desc (padding -1) and B (C, K, 4)."""
    import torch
    S = torch.full((n_cls, TOPK), -1.0, device=dev); B = torch.zeros((n_cls, TOPK, 4), device=dev)
    for boxes, sc, cls in zip(boxes_list, scores_list, cls_list):
        v, i = sc.topk(TOPK, dim=0)                                         # (K, k)
        v = torch.where(v >= THR, v, torch.full_like(v, -1.0))
        S[cls] = v.T; B[cls] = boxes[i.T]
    return S, B


def top_per_image(S, B):
    import torch
    flat = S.reshape(-1); k = min(PER_IMAGE, int((flat >= THR).sum()))
    if k == 0:
        return []
    v, i = flat.topk(k); c = i // S.shape[1]; bb = B.reshape(-1, 4)[i]
    return [[int(cc), round(float(s), 6), *[round(float(x), 2) for x in b]] for cc, s, b in zip(c.tolist(), v.tolist(), bb.tolist())]


def px(E, im, pred_boxes):
    import torch
    W, H = im.size; xy, wh = pred_boxes[:, :2], pred_boxes[:, 2:]
    return (torch.cat([xy - wh / 2, xy + wh / 2], 1) * pred_boxes.new_tensor([W, H, W, H])).float()


def part_I(a, out, imgs, names):
    import torch
    M = json.loads((HERE / "full_image_manifest.json").read_text())
    images = M["images"][: a.limit or None]
    E = Engine(a.dev, names); E.parity(imgs.load(images[0], M["urls"][images[0]])); print("parity check passed (base)", flush=True)
    bank = E.build_bank(); chunk_idx = [[E.index[c] for c in ch] for ch in E.chunks]

    def run(method, im):
        with torch.inference_mode():
            if method == "bank":
                boxes, sc = E.bank_call(im, bank)
                return [boxes], [sc], [list(range(len(names)))]
            bl, sl = [], []
            for ch in E.chunks:
                boxes, sc = E.decode_prompt(im, ch); bl.append(boxes); sl.append(sc)
            return bl, sl, chunk_idx

    warm = imgs.load(images[0], M["urls"][images[0]])
    for m in ("bank", "chunks"):
        E.measure(lambda: run(m, warm))
    d = out / "I"; d.mkdir(parents=True, exist_ok=True)
    left = [f for f in images if not (d / (f + ".json")).exists()]
    for j, f in enumerate(left):
        im = imgs.load(f, M["urls"][f]); rec = dict(image=f, size=im.size, seconds={}, peak_bytes={}, raw={}, soft={})
        for m in (("bank", "chunks") if j % 2 == 0 else ("chunks", "bank")):
            (bl, sl, cl), sec, mem = E.measure(lambda: run(m, im))
            with torch.inference_mode():
                S, B = candidates([px(E, im, b) for b in bl], [s.float() for s in sl], cl, len(names), E.dev)
                rec["raw"][m] = top_per_image(S, B)
                rec["soft"][m] = top_per_image(*soft_nms_batched(S, B))
            rec["seconds"][m] = sec; rec["peak_bytes"][m] = mem
        save(d / (f + ".json"), rec)
        if j % 25 == 0 or j == len(left) - 1:
            print(f"I {j + 1}/{len(left)}  bank {rec['seconds']['bank']:.2f}s  chunks {rec['seconds']['chunks']:.2f}s", flush=True)
    del E


def part_H(a, out, imgs, names):
    import torch
    images, names_, targets, request, urls, _ = load_inputs()
    if a.limit:
        images = images[: a.limit]
    present = {}
    for t in json.loads((HERE.parent / "results" / "bank" / "fixed_bank_soft_confirm_v1" / "manifest.json").read_text())["targets"]:
        if t["present"]:
            present.setdefault(t["image"], []).append(t["target"])
    L = json.loads((HERE / "confirm_gt.json").read_text())                  # {image: {class: [boxes xyxy]}}
    for ckpt in ("base", "tiny"):
        d = out / ("H_" + ckpt); d.mkdir(parents=True, exist_ok=True)
        left = [f for f in images if not (d / (f + ".json")).exists()]
        if not left:
            continue
        E = Engine(a.dev, names, mid=f"IDEA-Research/grounding-dino-{ckpt}"); E.parity(imgs.load(images[0], urls[images[0]]))
        bank = E.build_bank()
        for j, f in enumerate(left):
            im = imgs.load(f, urls[f]); rows = []
            for m in ("stock5", "bank"):
                with torch.inference_mode():
                    res = E.compute("bank" if m == "bank" else "stock", im, request[f], bank)
                for c in present.get(f, []):
                    boxes, s = res[c]; P = px(E, im, boxes); s = s.float()
                    order = torch.argsort(s, descending=True); rank = torch.empty_like(order); rank[order] = torch.arange(len(s), device=s.device)
                    for g in L[f][c]:
                        iou = box_iou(P, torch.tensor([g], device=P.device, dtype=P.dtype))[:, 0]
                        ok = iou >= 0.5
                        if ok.any():
                            k = int(torch.argmax(torch.where(ok, s, torch.full_like(s, -1.0))))
                            rows.append(dict(image=f, target=c, cond=m, best_iou=float(iou.max()), match_rank=int(rank[k]) + 1,
                                             match_score=float(s[k]), top_score=float(s.max())))
                        else:
                            rows.append(dict(image=f, target=c, cond=m, best_iou=float(iou.max()), match_rank=None,
                                             match_score=None, top_score=float(s.max())))
            save(d / (f + ".json"), dict(rows=rows))
            if j % 50 == 0 or j == len(left) - 1:
                print(f"H {ckpt} {j + 1}/{len(left)}", flush=True)
        del E
        if a.dev.startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--parts", default="I,H")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--images", default=None); ap.add_argument("--device", default=None)
    ap.add_argument("--split-mb", type=float, default=25.0)
    a = ap.parse_args()
    import torch, transformers
    a.dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    names = json.loads((HERE / "lvis_names.json").read_text()); imgs = Images(a.images)
    env = dict(gpu=torch.cuda.get_device_name() if a.dev.startswith("cuda") else "cpu", torch=torch.__version__,
               transformers=transformers.__version__, parts=a.parts, limit=a.limit, started=time.time())
    (out / f"env_{int(time.time())}.json").write_text(json.dumps(env)); print(json.dumps(env), flush=True)
    if "I" in a.parts.split(","):
        part_I(a, out, imgs, names)
    if "H" in a.parts.split(","):
        part_H(a, out, imgs, names)
    zpath = out.parent / (out.name + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file() and p.suffix != ".tmp":
                z.write(p, p.relative_to(out.parent))
    data = zpath.read_bytes(); n = max(1, int(np.ceil(len(data) / (a.split_mb * 1e6)))); size = -(-len(data) // n)
    parts = []
    for i in range(n):
        p = out.parent / f"{out.name}_part{i + 1}of{n}.bin"; p.write_bytes(data[i * size:(i + 1) * size]); parts.append(str(p))
    print(f"ALL DONE. {zpath} ({len(data) / 1e6:.1f} MB) split into {n} part(s) of <= {a.split_mb:.0f} MB:", *parts, sep="\n", flush=True)


if __name__ == "__main__":
    main()
