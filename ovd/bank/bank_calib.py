#!/usr/bin/env python3
"""Amendment I: class-wise score normalization of the fixed class bank with fixed, label-free reference statistics.
Resumable; CPU or GPU.

  python ovd/bank/bank_calib.py --mode ref  --out DIR                       # 300 reference images -> DIR/ref_stats.json
  python ovd/bank/bank_calib.py --mode dev  --out DIR --lvis lvis_v1_val.json   # 124 development images
  python ovd/bank/bank_calib.py --mode test --out DIR                       # the 1,000 run-3l images
  add --method chunks for standard full-vocabulary inference (14 passes; GPU advised) -> DIR/chunks/...

Per image: bank pass -> per class top 50 >= 0.02 -> class-wise Gaussian Soft-NMS on raw scores (= run-3l bank lists)
-> per-class monotone map (none / shift / z, PLAN Amendment I) -> top 300. Dev mode also saves every kept detection
(no per-image cap) for the ceiling diagnostic.
"""
import argparse, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from followup import Engine, Images, load_inputs, save  # noqa: E402
from full_image import candidates, px, soft_nms_batched, PER_IMAGE  # noqa: E402
from hybrid import dev_images  # noqa: E402

EPS = 1e-6
MAPS = ("none", "shift", "z")


def logit(t):
    t = t.clamp(EPS, 1 - EPS)
    return (t / (1 - t)).log()


def apply_map(name, s, st):
    """s (C, K) Soft-NMS scores, -1 = empty. Returns mapped scores, empty stays -1."""
    import torch
    if name == "none":
        return s
    l = logit(s)
    if name == "shift":
        m = l - st["mu"][:, None] + st["mu_g"]
    else:
        m = (l - st["mu"][:, None]) * (st["sd_g"] / st["sd"][:, None]) + st["mu_g"]
    return torch.where(s > 0, torch.sigmoid(m), s)


def top_list(S, B, cap):
    valid = S.reshape(-1) > 0; k = int(valid.sum()) if cap is None else min(cap, int(valid.sum()))
    if k == 0:
        return []
    flat = S.reshape(-1).clone(); flat[~valid] = -2
    v, i = flat.topk(k); c = i // S.shape[1]; bb = B.reshape(-1, 4)[i]
    return [[int(cc), round(float(x), 8), *[round(float(y), 2) for y in b]] for cc, x, b in zip(c.tolist(), v.tolist(), bb.tolist())]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["ref", "dev", "test"], required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--lvis", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--images", default=None); ap.add_argument("--n-ref", type=int, default=300)
    ap.add_argument("--method", choices=["bank", "chunks"], default="bank", help="chunks = standard full-vocabulary inference")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = json.loads((HERE / "lvis_names.json").read_text()); imgs = Images(a.images); out = Path(a.out)
    if a.mode == "ref":
        files, _, _, _, urls, _ = load_inputs(); files = files[: a.n_ref]
    elif a.mode == "test":
        M = json.loads((HERE / "full_image_manifest.json").read_text()); files, urls = M["images"], M["urls"]
    else:
        L = json.load(open(a.lvis)); urls = {im["coco_url"].split("/")[-1]: im["coco_url"] for im in L["images"]}; del L
        files = dev_images()
    files = files[: a.limit or None]
    E = Engine(dev, names); E.parity(imgs.load(files[0], urls[files[0]])); bank = E.build_bank()
    chunk_idx = [[E.index[c] for c in ch] for ch in E.chunks]
    out = out / a.method if a.method != "bank" else out
    d = out / a.mode; d.mkdir(parents=True, exist_ok=True)
    if a.mode != "ref":
        R = json.loads((out / "ref_stats.json").read_text())
        st = dict(mu=torch.tensor(R["mu"], device=dev), sd=torch.tensor(R["sd"], device=dev), mu_g=R["mu_g"], sd_g=R["sd_g"])
    left = [f for f in files if not (d / (f + ".json")).exists()]
    for j, f in enumerate(left):
        im = imgs.load(f, urls[f])
        with torch.inference_mode():
            if a.method == "bank":
                boxes, sc = E.bank_call(im, bank); bl, sl, cl = [boxes], [sc.float()], [list(range(len(names)))]
            else:
                bl, sl = [], []
                for ch in E.chunks:
                    b_, s_ = E.decode_prompt(im, ch); bl.append(b_); sl.append(s_.float())
                cl = chunk_idx
            if a.mode == "ref":
                mx = torch.full((len(names),), 0.0, device=E.dev)
                for s_, c_ in zip(sl, cl):
                    mx[c_] = s_.amax(0)
                rec = dict(image=f, max_logit=[round(float(x), 5) for x in logit(mx).tolist()])
            else:
                S, B = candidates([px(E, im, b_) for b_ in bl], sl, cl, len(names), E.dev)
                Ss, Bs = soft_nms_batched(S, B)
                rec = dict(image=f, soft={m: top_list(apply_map(m, Ss, st), Bs, PER_IMAGE) for m in MAPS})
                if a.mode == "dev":
                    rec["soft_uncapped"] = top_list(Ss, Bs, None)
        save(d / (f + ".json"), rec)
        if j % 25 == 0 or j == len(left) - 1:
            print(f"{a.mode} {j + 1}/{len(left)}", flush=True)
    if a.mode == "ref":
        X = torch.tensor([json.loads((d / (f + ".json")).read_text())["max_logit"] for f in files])   # (n, C)
        mu, sd = X.mean(0), X.std(0).clamp(min=1e-3)
        (out / "ref_stats.json").write_text(json.dumps(dict(n_images=len(files), mu=mu.tolist(), sd=sd.tolist(),
                                                             mu_g=float(mu.mean()), sd_g=float(sd.mean()))))
        print(f"ref stats: mu_g {float(mu.mean()):.3f}, sd_g {float(sd.mean()):.3f}; mu range {float(mu.min()):.2f}..{float(mu.max()):.2f}")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
