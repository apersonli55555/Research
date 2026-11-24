#!/usr/bin/env python3
"""Amendment H: bank-proposed context ("hybrid") for Grounding DINO, full-image LVIS outputs. Resumable; CPU or GPU.

  python ovd/bank/hybrid.py --mode dev  --out DIR                 # 124 development images, K in {10, 25, 50}
  python ovd/bank/hybrid.py --mode test --out DIR --k K           # the 1,000 run-3l images, frozen K, timed

Per image: one bank pass -> per-class candidates (top 50 boxes >= 0.02) -> raw top-300 list; candidate classes = first
K distinct classes of that list; one standard pass with the candidates in canonical LVIS order (<= 256-token chunks);
candidate rows replaced by that pass, other classes keep the bank's rows; class-wise Gaussian Soft-NMS; top 300.
Every step depends only on the image, so each class's output is independent of any user request.
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from followup import Engine, Images, save  # noqa: E402
from full_image import candidates, px, soft_nms_batched, top_per_image, TOPK, THR  # noqa: E402


def dev_images():
    R = HERE.parent / "results" / "bank"
    a = json.loads((R / "bank_validation_1795488385837158854" / "manifest.json").read_text())["images"]
    b = sorted({r["image"] for r in json.loads((R / "fixed_bank_1795484911264138629.json").read_text())["records"]})
    return sorted(set(a) | set(b))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["dev", "test"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lvis", default=None, help="lvis_v1_val.json (dev mode: image URLs)"); ap.add_argument("--images", default=None)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = json.loads((HERE / "lvis_names.json").read_text()); imgs = Images(a.images)
    if a.mode == "test":
        M = json.loads((HERE / "full_image_manifest.json").read_text()); files, urls = M["images"], M["urls"]; Ks = [a.k]
        assert a.k > 0, "--k is fixed on the development images"
    else:
        L = json.load(open(a.lvis)); urls = {im["coco_url"].split("/")[-1]: im["coco_url"] for im in L["images"]}; del L
        files = dev_images(); Ks = [10, 25, 50]
    files = files[: a.limit or None]
    E = Engine(dev, names); E.parity(imgs.load(files[0], urls[files[0]])); bank = E.build_bank()
    out = Path(a.out) / a.mode; out.mkdir(parents=True, exist_ok=True)

    def run(im, K):
        """Returns per-K (S, B) candidate matrices for all classes, and the raw bank list."""
        with torch.inference_mode():
            boxes, sc = E.bank_call(im, bank)
            S, B = candidates([px(E, im, boxes)], [sc.float()], [list(range(len(names)))], len(names), E.dev)
            raw_bank = top_per_image(S, B)
            order = []
            for c, *_ in raw_bank:
                if c not in order:
                    order.append(c)
            res = {}
            for k in K:
                cand = sorted(order[:k])                                         # canonical LVIS order
                S2, B2 = S.clone(), B.clone()
                for ch in E.g.chunks([names[c] for c in cand]):
                    pb, ps = E.decode_prompt(im, ch)
                    Sc, Bc = candidates([px(E, im, pb)], [ps.float()], [[E.index[c] for c in ch]], len(names), E.dev)
                    idx = [E.index[c] for c in ch]; S2[idx] = Sc[idx]; B2[idx] = Bc[idx]
                res[k] = (S2, B2, len(cand), len(E.g.chunks([names[c] for c in cand])))
            return S, B, res

    warm = imgs.load(files[0], urls[files[0]]); E.measure(lambda: run(warm, Ks))
    left = [f for f in files if not (out / (f + ".json")).exists()]
    for j, f in enumerate(left):
        im = imgs.load(f, urls[f])
        (S, B, res), sec, mem = E.measure(lambda: run(im, Ks))
        with torch.inference_mode():
            rec = dict(image=f, seconds=sec, peak_bytes=mem, raw={}, soft={}, n_cand={}, prompt_passes={})
            rec["raw"]["bank"] = top_per_image(S, B); rec["soft"]["bank"] = top_per_image(*soft_nms_batched(S, B))
            for k, (S2, B2, n, p) in res.items():
                rec["raw"][f"hybrid{k}"] = top_per_image(S2, B2); rec["soft"][f"hybrid{k}"] = top_per_image(*soft_nms_batched(S2, B2))
                rec["n_cand"][k] = n; rec["prompt_passes"][k] = p
        save(out / (f + ".json"), rec)
        if j % 25 == 0 or j == len(left) - 1:
            print(f"{a.mode} {j + 1}/{len(left)}  {sec:.2f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
