#!/usr/bin/env python3
"""Measured (not only by-construction) request interference of the corrected method, end to end. CPU or GPU.

  python ovd/bank/calib_interference.py --n 60 --out results/bank/calib/interference.json

Pairs: the run-3j image-independent request pairs (part B of results/bank/followup_t4_v1/followup_v1_data.zip: per
(image, target), two five-class requests A and B that share only the target), on confirmation images NOT used for the
reference statistics (images 301..300+n in the run-3h SHA order). For each request separately, the deployed pipeline is
run from scratch: bank pass -> per class top 50 >= 0.02 -> class-wise Soft-NMS -> class-wise shift (frozen reference
statistics) -> keep the requested classes -> top 300. Compared with the saved standard five-class prompt outputs for the
same requests: target top-score change |A - B|, and decision flips (target top score >= t under exactly one request).
"""
import argparse, json, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from followup import Engine, Images, load_inputs  # noqa: E402
from full_image import candidates, px, soft_nms_batched, PER_IMAGE  # noqa: E402
from bank_calib import apply_map, top_list  # noqa: E402

THR = (0.2, 0.3, 0.35, 0.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--out", required=True)
    ap.add_argument("--images", default=None)
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = json.loads((HERE / "lvis_names.json").read_text())
    files, _, _, _, urls, _ = load_inputs(); files = files[300:300 + a.n]
    z = zipfile.ZipFile(HERE.parent / "results" / "bank" / "followup_t4_v1" / "followup_v1_data.zip")
    R = json.loads((HERE.parent / "results" / "bank" / "calib" / "ref_stats.json").read_text())
    st = dict(mu=torch.tensor(R["mu"], device=dev), sd=torch.tensor(R["sd"], device=dev), mu_g=R["mu_g"], sd_g=R["sd_g"])
    imgs = Images(a.images); E = Engine(dev, names); E.parity(imgs.load(files[0], urls[files[0]])); bank = E.build_bank()

    def deploy(im, request):
        idx = [E.index[c] for c in request]
        with torch.inference_mode():
            boxes, sc = E.bank_call(im, bank)
            S, B = candidates([px(E, im, boxes)], [sc.float()], [list(range(len(names)))], len(names), E.dev)
            Ss, Bs = soft_nms_batched(S, B)
            M = apply_map("shift", Ss, st)
            return top_list(M[idx], Bs[idx], PER_IMAGE)          # class ids are positions in `request`

    pairs = []
    for j, f in enumerate(files):
        rows = json.loads(z.read(f"followup_v1/B/{f}.json"))["rows"]
        im = imgs.load(f, urls[f])
        for t in sorted({r["target"] for r in rows}):
            ra = next(r for r in rows if r["target"] == t and r["cond"] == "stock5_indA")
            rb = next(r for r in rows if r["target"] == t and r["cond"] == "stock5_indB")
            out = {}
            for tag, r in (("A", ra), ("B", rb)):
                lst = deploy(im, r["request"]); k = r["request"].index(t)
                out[tag] = [x[1:] for x in lst if x[0] == k]
            pairs.append(dict(image=f, target=t,
                              std=[max([d[0] for d in ra["dets"]], default=0.0), max([d[0] for d in rb["dets"]], default=0.0)],
                              corr=[max([d[0] for d in out["A"]], default=0.0), max([d[0] for d in out["B"]], default=0.0)],
                              corr_identical=out["A"] == out["B"]))
        print(f"{j + 1}/{len(files)} images, {len(pairs)} pairs", flush=True)
    res = dict(n_images=len(files), n_pairs=len(pairs))
    for m in ("std", "corr"):
        d = [abs(p[m][0] - p[m][1]) for p in pairs]
        res[m] = dict(mean_abs_top_score_change=sum(d) / len(d), max_abs_top_score_change=max(d),
                      flips={str(t): sum((p[m][0] >= t) != (p[m][1] >= t) for p in pairs) for t in THR})
    res["corr_lists_bitwise_identical"] = sum(p["corr_identical"] for p in pairs)
    res["reduction_of_flips"] = {str(t): (1 - res["corr"]["flips"][str(t)] / res["std"]["flips"][str(t)])
                                 if res["std"]["flips"][str(t)] else None for t in THR}
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    res["pairs"] = pairs
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
