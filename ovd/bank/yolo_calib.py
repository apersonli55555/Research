#!/usr/bin/env python3
"""PLAN Amendment K: does the class-wise shift of Amendment I generalize to YOLO-World and to COCO? Resumable; CPU or GPU.

  python ovd/bank/yolo_calib.py --dataset lvis --mode ref  --out DIR
  python ovd/bank/yolo_calib.py --dataset lvis --mode test --out DIR
  python ovd/bank/yolo_calib.py --dataset coco --coco instances_val2017.json --mode ref|test --out DIR

YOLO-World v2-S, the full vocabulary as head and guide text (fixed, so list-invariant for this protocol). Per image:
ref -> per class the logit of the image-max score; test -> two pipelines, each followed by the frozen maps none/shift/z
(bank_calib.apply_map) and the per-image cap (LVIS 300, COCO 100):
  P (primary, = runs 3l/3n): per class top 50 with score >= 0.02, class-wise Gaussian Soft-NMS (sigma 0.5);
  S (YOLO-World standard):   per class top 50 with score >= 0.001, class-wise NMS at IoU 0.7.
Images: LVIS = the run-3n reference (first 300 of the run-3h confirmation images) and the 1,000 run-3l test images;
COCO = val2017 ordered by SHA-256 of the file name, minus every image cached by an earlier run: first 300 = reference,
next 1,000 = test (the list is written to DIR/coco/images.json on first use and reused).
"""
import argparse, hashlib, json, os, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from followup import Images, load_inputs, save  # noqa: E402
from full_image import candidates, soft_nms_batched  # noqa: E402
from bank_calib import MAPS, apply_map, logit, top_list  # noqa: E402

CAP = {"lvis": 300, "coco": 100}
TOPK, THR_S, NMS_IOU = 50, 0.001, 0.7


def coco_split(coco_json, exclude_dirs, out):
    p = out / "images.json"
    if p.exists():
        return json.loads(p.read_text())
    C = json.load(open(coco_json))
    used = set()
    for d in exclude_dirs:
        if os.path.isdir(d):
            used |= set(os.listdir(d))
    files = sorted((im["file_name"] for im in C["images"] if im["file_name"] not in used),
                   key=lambda f: hashlib.sha256(f.encode()).hexdigest())
    S = dict(ref=files[:300], test=files[300:1300], n_excluded=len(used & {im["file_name"] for im in C["images"]}),
             rule="val2017 sorted by sha256(file_name), excluding images cached by earlier runs")
    out.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(S, indent=1))
    return S


class CocoImages:
    def __init__(self, cache="/tmp/coco_val2017"):
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)

    def load(self, f):
        from PIL import Image
        p = self.cache / f
        if not p.is_file():
            p.write_bytes(urllib.request.urlopen(f"http://images.cocodataset.org/val2017/{f}", timeout=60).read())
        im = Image.open(p).convert("RGB"); im.filename_key = f
        return im


def hard_nms_rows(S, B):
    """Class-wise NMS on (C, K) candidate rows (padding -1). Returns (C, K) kept scores (-1 elsewhere) and boxes."""
    import torch
    from torchvision.ops import nms
    out = torch.full_like(S, -1.0)
    for c in torch.nonzero((S > 0).any(1)).flatten().tolist():
        v = S[c] > 0; idx = torch.nonzero(v).flatten()
        keep = idx[nms(B[c][idx], S[c][idx], NMS_IOU)]
        out[c, keep] = S[c, keep]
    return out, B


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["lvis", "coco"], required=True)
    ap.add_argument("--mode", choices=["ref", "test"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--coco", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=["/tmp/lvis_images"])
    a = ap.parse_args()
    import torch
    import full_image
    from vocab_full import YoloWorld
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out) / a.dataset
    if a.dataset == "lvis":
        names = json.loads((HERE / "lvis_names.json").read_text()); imgs = Images(None)
        if a.mode == "ref":
            files, _, _, _, urls, _ = load_inputs(); files = files[:300]
        else:
            M = json.loads((HERE / "full_image_manifest.json").read_text()); files, urls = M["images"], M["urls"]
        load = lambda f: imgs.load(f, urls[f])
    else:
        C = json.load(open(a.coco)); names = [c["name"] for c in sorted(C["categories"], key=lambda c: c["id"])]; del C
        files = coco_split(a.coco, a.exclude, out)[a.mode]; ci = CocoImages(); load = ci.load
    files = files[: a.limit or None]
    Y = YoloWorld(dev, names); C = len(names)
    d = out / a.mode; d.mkdir(parents=True, exist_ok=True)
    if a.mode == "test":
        R = json.loads((out / "ref_stats.json").read_text())
        st = dict(mu=torch.tensor(R["mu"]), sd=torch.tensor(R["sd"]), mu_g=R["mu_g"], sd_g=R["sd_g"])
    left = [f for f in files if not (d / (f + ".json")).exists()]
    for j, f in enumerate(left):
        im = load(f)
        if not hasattr(im, "filename_key"):
            im.filename_key = f
        boxes, S = Y(im, names)                                    # (N, 4) px xyxy, (N, C) probabilities
        boxes, S = torch.from_numpy(boxes).float(), torch.from_numpy(S).float()
        if a.mode == "ref":
            rec = dict(image=f, max_logit=[round(float(x), 5) for x in logit(S.amax(0)).tolist()])
        else:
            rec = dict(image=f)
            Sp, Bp = candidates([boxes], [S], [list(range(C))], C, "cpu")            # THR 0.02
            Ss, Bs = soft_nms_batched(Sp, Bp)
            rec["P"] = {m: top_list(apply_map(m, Ss, st), Bs, CAP[a.dataset]) for m in MAPS}
            old = full_image.THR; full_image.THR = THR_S
            try:
                Sq, Bq = candidates([boxes], [S], [list(range(C))], C, "cpu")       # THR 0.001
            finally:
                full_image.THR = old
            Sh, Bh = hard_nms_rows(Sq, Bq)
            rec["S"] = {m: top_list(apply_map(m, Sh, st), Bh, CAP[a.dataset]) for m in MAPS}
        save(d / (f + ".json"), rec)
        if j % 50 == 0 or j == len(left) - 1:
            print(f"{a.dataset} {a.mode} {j + 1}/{len(left)}", flush=True)
    if a.mode == "ref":
        X = torch.tensor([json.loads((d / (f + ".json")).read_text())["max_logit"] for f in files])
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-3)
        (out / "ref_stats.json").write_text(json.dumps(dict(n_images=len(files), mu=mu.tolist(), sd=sd.tolist(),
                                                             mu_g=float(mu.mean()), sd_g=float(sd.mean()))))
        print(f"ref stats: mu_g {float(mu.mean()):.3f}; mu range {float(mu.min()):.2f}..{float(mu.max()):.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
