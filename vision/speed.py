#!/usr/bin/env python3
"""Run 28: inference cost of our detectors on one 2 s clip, measured the same way as for the VLM baselines
(from 16 decoded PIL frames to a score, including preprocessing, batch 1, after warm-up, on the same GPU).

  python vision/speed.py --work <drive>/fall-cvpr/finetune
"""
import argparse, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ClipEncoder, VideoEncoder, clip_subset  # noqa: E402
from finetune import ClipStore, load_model, to_tensor  # noqa: E402


def timeit(fn, clips, warm=10):
    import torch
    warm = min(warm, max(1, len(clips) // 2))
    for c in clips[:warm]:
        fn(c)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for c in clips[warm:]:
        fn(c)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = (time.time() - t0) / max(1, len(clips) - warm)
    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    return sec, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    a = ap.parse_args()
    import torch
    s = ClipStore(a.work, "oops")
    clips = [s.frames(it) for it in s.items[: a.n]]
    dev = a.device if torch.cuda.is_available() else "cpu"
    rows = []
    enc = ClipEncoder(a.model, a.pretrained, dev); venc = VideoEncoder(device=dev)
    w = np.random.default_rng(0).normal(size=3 * 768 + 768)       # a linear probe costs one dot product
    def probe_path(c):
        fe = enc.encode_images(clip_subset(c)); v = venc.encode([c])[0]
        return float(np.r_[fe.mean(0), fe.mean(0), fe.mean(0), v][: len(w)] @ w[: len(np.r_[fe.mean(0), fe.mean(0), fe.mean(0), v])])
    sec, mem = timeit(probe_path, clips)
    rows.append(("frozen CLIP ViT-L/14 (8 frames) + VideoMAE-B (16 frames) + linear probe", sec, mem))
    del enc, venc
    torch.cuda.empty_cache() if dev == "cuda" else None
    tag = "model_zerolabel"
    if os.path.exists(os.path.join(a.work, tag, "config.json")):
        model, mean, std = load_model(dev, os.path.join(a.work, tag)); model.eval()
        def ft(c):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                return model(pixel_values=to_tensor(c, mean, std).unsqueeze(0).to(dev)).logits
        sec, mem = timeit(ft, clips)
        rows.append(("fine-tuned VideoMAE-B (16 frames)", sec, mem))
    print("\nINFERENCE COST (run 28), batch 1, one 2 s clip, " + (torch.cuda.get_device_name() if dev == "cuda" else "CPU"))
    print("| Detector | seconds per clip | peak GPU memory (GB) |\n|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.2f} |")
    print("(VLM baselines, run 24, same GPU: Qwen3-VL-4B 0.22 s and 9.2 GB; InternVL3.5-8B 1.0 s and 18.1 GB)")


if __name__ == "__main__":
    main()
