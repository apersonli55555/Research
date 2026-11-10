#!/usr/bin/env python3
"""Third test set: OmniFall's in-the-wild annotations (OF-ItW) on OOPS! videos.

OmniFall (Schneider et al., arXiv 2505.19889; annotations CC BY-NC-SA 4.0) labels 818
fall-containing OOPS! videos with segment-level activity classes (label 1 = fall,
2 = fallen), made by an independent annotation team, with a train/val/test split by video.

Clips use the same 2 s / 16-frame definition as everywhere else:
  * FALL     : one clip centred on each "fall" segment
  * NON-FALL : one clip centred on each other segment that is at least 1 s long, is not
               "fallen" (the after-state of a fall, ambiguous for event detection), and whose
               2 s window does not overlap any fall segment

The 48 GB OOPS! archive is streamed once; only the 818 annotated videos are decoded.
The output also stores every video's fall segments, so that our own FallOrNot labels can be
compared with OmniFall's on the videos both cover.

Usage (Colab, GPU):  python vision/ofitw_prep.py --out /content/drive/MyDrive/fall-cvpr/ofitw.npz
"""
import argparse, csv, io, json, os, sys, tarfile, time, urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CLIP_SECONDS, VIDEO_MODEL, ClipEncoder, VideoEncoder, clip_subset, sample_clip  # noqa: E402
from oops_mine import URL, ResumableStream  # noqa: E402

BASE = "https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/"
FALL, FALLEN = 1, 2
MIN_NEG_S = 1.0


def fetch_text(rel):
    return urllib.request.urlopen(BASE + rel).read().decode()


def plan():
    """name (original OOPS! video name) -> dict(split, falls=[(s, e)], clips=[(centre, label, cls)])."""
    rows = list(csv.DictReader(io.StringIO(fetch_text("labels/OOPS.csv"))))
    mapping = {r["itw_path"][:-4]: os.path.basename(r["oops_path"])[:-4]
               for r in csv.DictReader(io.StringIO(fetch_text("data_files/oops_video_mapping.csv")))}
    split = {}
    for sp in ("train", "val", "test"):
        for line in fetch_text(f"splits/cs/OOPS/{sp}.csv").splitlines()[1:]:
            split[line.strip()] = sp
    segs = defaultdict(list)
    for r in rows:
        segs[r["path"]].append((float(r["start"]), float(r["end"]), int(r["label"])))
    out = {}
    h = CLIP_SECONDS / 2
    for path, ss in segs.items():
        if path not in mapping or path not in split:
            continue
        falls = [(s, e) for s, e, l in ss if l == FALL]
        clips = [((s + e) / 2, 1, FALL) for s, e in falls]
        for s, e, l in ss:
            if l in (FALL, FALLEN) or e - s < MIN_NEG_S:
                continue
            c = (s + e) / 2
            if c - h < 0 or any(c - h < fe and fs < c + h for fs, fe in falls):
                continue
            clips.append((c, 0, l))
        out[mapping[path]] = dict(split=split[path], falls=falls, clips=clips)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (testing)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-model", default=VIDEO_MODEL)
    a = ap.parse_args()

    P = plan()
    n_clip = sum(len(v["clips"]) for v in P.values())
    print(f"OF-ItW: {len(P)} videos, {n_clip} clips ({sum(c[1] for v in P.values() for c in v['clips'])} fall), "
          f"splits {dict((s, sum(v['split'] == s for v in P.values())) for s in ('train', 'val', 'test'))}")

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    venc = VideoEncoder(a.video_model, a.device) if a.video_model else None
    O = defaultdict(list)
    todo = set(P)
    t0 = time.time()
    stream = ResumableStream(a.url)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    try:
        for m in outer:
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                name = os.path.basename(vm.name)[:-4]
                if not vm.name.endswith(".mp4") or name not in todo:
                    continue
                data = vids.extractfile(vm).read()
                v = P[name]
                try:
                    f16s = [sample_clip(data, max(0.0, c - CLIP_SECONDS / 2), c + CLIP_SECONDS / 2) for c, _, _ in v["clips"]]
                except Exception as e:
                    print("skip", name, type(e).__name__, e)
                    todo.discard(name)
                    continue
                for (c, lab, cls), f16 in zip(v["clips"], f16s):
                    fe = enc.encode_images(clip_subset(f16))
                    O["frames"].append(fe); O["margin"].append(enc.zero_shot(fe)[1])
                    O["name"].append(name); O["label"].append(lab); O["cls"].append(cls)
                    O["centre"].append(c); O["split"].append(v["split"])
                if venc and f16s:
                    O["video"].extend(venc.encode(f16s))
                todo.discard(name)
                done = len(P) - len(todo)
                if done % 50 == 0:
                    print(f"  {done}/{len(P)} videos, {len(O['label'])} clips, {done / (time.time() - t0):.2f} videos/s", flush=True)
                if not todo or (a.limit and done >= a.limit):
                    break
            break
    finally:
        stream.close()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    extra = {"video": np.array(O["video"], np.float16)} if venc else {}
    np.savez(a.out, **extra, frames=np.array(O["frames"], np.float16), margin=np.array(O["margin"], np.float32),
             name=np.array(O["name"]), label=np.array(O["label"]), cls=np.array(O["cls"]),
             centre=np.array(O["centre"], np.float32), split=np.array(O["split"]),
             fall_segments=np.array(json.dumps({n: v["falls"] for n, v in P.items()})),
             video_split=np.array(json.dumps({n: v["split"] for n, v in P.items()})),
             model=np.array(f"{a.model}/{a.pretrained}"))
    lab = np.array(O["label"])
    print(f"saved {a.out}: {len(lab)} clips, {int(lab.sum())} fall / {int((1 - lab).sum())} non-fall; "
          f"{len(todo)} planned videos not found")


if __name__ == "__main__":
    main()
