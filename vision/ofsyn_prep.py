#!/usr/bin/env python3
"""Turn OmniFall's synthetic videos (OF-Syn) into clips and embeddings, like urfall_prep.py.

OF-Syn (Schneider et al., OmniFall, arXiv 2505.19889; annotations CC BY-NC-SA 4.0) has
12,000 generated 5 s videos (AV1, 1280x720) with segment labels in labels/of-syn.csv.

One 2 s clip per video, the same 16-frame sampling and encoders as the real and staged data:
  * FALL     : every video that contains a "fall" segment (label 1); the clip is centred
               on that segment
  * NON-FALL : a model-independent random sample (hash of the path) of the videos with no
               fall segment (sitting down, lying down, walking, standing up, ...); the clip
               is centred on the video's longest segment

The 9.7 GB archive is streamed once (never stored). Usage (Colab, GPU):
  python vision/ofsyn_prep.py --out /content/drive/MyDrive/fall-cvpr/ofsyn.npz
"""
import argparse, csv, hashlib, io, os, sys, tarfile, time, urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import (CLIP_SECONDS, VIDEO_MODEL, ClipEncoder, VideoEncoder,  # noqa: E402
                    clip_subset, sample_clip)
from oops_mine import ResumableStream  # noqa: E402

BASE = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"
TAR_URL = BASE + "data_files/omnifall-synthetic_av1.tar"
LABELS_URL = BASE + "labels/of-syn.csv"
FALL = "1"


def plan_clips(rows, n_nonfall):
    """path -> (start, end, label) for the clip to take from each selected video."""
    segs = defaultdict(list)
    for r in rows:
        segs[r["path"]].append((float(r["start"]), float(r["end"]), r["label"]))
    meta = {r["path"]: {k: r[k] for k in ("age_group", "environment_category", "camera_shot")} for r in rows}
    fall_videos = [p for p, ss in segs.items() if any(l == FALL for _, _, l in ss)]
    other = [p for p, ss in segs.items() if all(l != FALL for _, _, l in ss)]
    order = lambda p: hashlib.md5(p.encode()).hexdigest()      # model-independent sample
    other = sorted(other, key=order)[:n_nonfall]
    plan = {}
    for p in fall_videos:
        s, e, _ = max((x for x in segs[p] if x[2] == FALL), key=lambda x: x[1] - x[0])
        plan[p] = ((s + e) / 2, 1)
    for p in other:
        s, e, _ = max(segs[p], key=lambda x: x[1] - x[0])
        plan[p] = ((s + e) / 2, 0)
    return plan, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-nonfall", type=int, default=3000, help="non-fall videos to sample")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (testing)")
    ap.add_argument("--videos-dir", default="", help="read mp4s from a local folder instead of streaming (testing)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-model", default=VIDEO_MODEL)
    a = ap.parse_args()

    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(LABELS_URL).read().decode())))
    plan, meta = plan_clips(rows, a.n_nonfall)
    print(f"plan: {sum(v[1] for v in plan.values())} fall clips, {sum(1 - v[1] for v in plan.values())} non-fall clips")

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    venc = VideoEncoder(a.video_model, a.device) if a.video_model else None
    out = defaultdict(list)
    t0 = time.time()

    def handle(path, data):
        centre, label = plan[path]
        try:
            f16 = sample_clip(data, max(0.0, centre - CLIP_SECONDS / 2), centre + CLIP_SECONDS / 2)
        except Exception as e:
            print("skip", path, type(e).__name__, e)
            return
        fe = enc.encode_images(clip_subset(f16))
        out["frames"].append(fe); out["emb"].append(enc.clip_embedding(fe))
        out["margin"].append(enc.zero_shot(fe)[1])
        if venc:
            out["video"].append(venc.encode([f16])[0])
        out["label"].append(label); out["group"].append(path); out["start"].append(centre - CLIP_SECONDS / 2)
        for k, v in meta[path].items():
            out["meta_" + k].append(v)
        n = len(out["label"])
        if n % 200 == 0:
            print(f"  {n}/{len(plan)} clips, {n / (time.time() - t0):.1f}/s", flush=True)

    if a.videos_dir:
        for root, _, files in os.walk(a.videos_dir):
            for f in sorted(files):
                p = os.path.relpath(os.path.join(root, f), a.videos_dir)[:-4]
                if f.endswith(".mp4") and p in plan:
                    handle(p, open(os.path.join(root, f), "rb").read())
                    if a.limit and len(out["label"]) >= a.limit:
                        break
    else:
        stream = ResumableStream(TAR_URL)
        tf = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|")
        try:
            for m in tf:
                if not m.name.endswith(".mp4"):
                    continue
                p = os.path.normpath(m.name)[:-4]
                if p in plan:
                    handle(p, tf.extractfile(m).read())
                    if a.limit and len(out["label"]) >= a.limit:
                        break
        finally:
            stream.close()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    arrays = {k: np.array(v) for k, v in out.items()}
    arrays["frames"] = arrays["frames"].astype(np.float16)
    if "video" in arrays:
        arrays["video"] = arrays["video"].astype(np.float16)
    np.savez(a.out, **arrays, source=np.array(["of-syn"] * len(out["label"])),
             seg=np.array(["syn"] * len(out["label"])), model=np.array(f"{a.model}/{a.pretrained}"))
    print(f"saved {a.out}: {int(arrays['label'].sum())} fall / {int((1 - arrays['label']).sum())} non-fall clips")


if __name__ == "__main__":
    main()
