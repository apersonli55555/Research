#!/usr/bin/env python3
"""External real-world test set from HMDB51 (Kuehne et al., ICCV 2011; CC BY 4.0).

HMDB51 is a public action-recognition dataset of clips from movies and web video, entirely
independent of OOPS!. Its human-labelled class "fall_floor" gives real (non-lab) falls; the
other 50 classes give non-fall negatives. Nothing here is used for training: detectors
trained on staged, synthetic or real OOPS! data are only evaluated on it.

Selection (fixed before looking at any results):
  * positives: every fall_floor clip in the split-1 train and test parquet files (100)
  * negatives: --per-class clips from each of the other 50 classes, chosen by a hash of
    the file name (model-independent)
  * HARD_CLASSES: whole-body or ground-contact actions most easily confused with falls,
    reported as a separate harder test

HMDB clips are trimmed to the action but the fall need not be centred, so every clip is
covered by 2 s windows every 1 s (16 frames each, same encoders as the other data); a
clip's score is its highest window score.

Usage (Colab, GPU):  python vision/hmdb_prep.py --out /content/drive/MyDrive/fall-cvpr/hmdb.npz
"""
import argparse, hashlib, io, os, sys, time, urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CLIP_SECONDS, VIDEO_MODEL, ClipEncoder, VideoEncoder, clip_subset, sample_clip  # noqa: E402

BASE = "https://huggingface.co/datasets/mteb/HMDB51/resolve/main/data/"
FILES = ["test-00000-of-00001.parquet", "train-00000-of-00003.parquet",
         "train-00001-of-00003.parquet", "train-00002-of-00003.parquet"]
CLASSES = ("brush_hair cartwheel catch chew clap climb climb_stairs dive draw_sword dribble drink eat "
           "fall_floor fencing flic_flac golf handstand hit hug jump kick kick_ball kiss laugh pick pour "
           "pullup punch push pushup ride_bike ride_horse run shake_hands shoot_ball shoot_bow shoot_gun "
           "sit situp smile smoke somersault stand swing_baseball sword sword_exercise talk throw turn walk "
           "wave").split()
FALL = "fall_floor"
HARD_CLASSES = ["cartwheel", "flic_flac", "somersault", "handstand", "dive", "jump", "sit", "situp",
                "pushup", "stand", "climb"]


def duration(data):
    import av
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        if s.frames and s.average_rate:
            return float(s.frames / s.average_rate)
        return float(c.duration / 1e6) if c.duration else CLIP_SECONDS


def windows(dur):
    if dur <= CLIP_SECONDS:
        return [(0.0, max(dur, 0.1))]
    starts = np.arange(0.0, dur - CLIP_SECONDS + 1e-6, 1.0)
    if dur - CLIP_SECONDS - starts[-1] > 0.25:                 # cover the tail
        starts = np.append(starts, dur - CLIP_SECONDS)
    return [(float(s), float(s + CLIP_SECONDS)) for s in starts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/hmdb")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--files", nargs="*", default=FILES)
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (testing)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-model", default=VIDEO_MODEL)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    os.makedirs(a.cache, exist_ok=True)
    clips = []                                       # (name, class, bytes)
    for f in a.files:
        path = os.path.join(a.cache, f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(BASE + f, path + ".part"); os.replace(path + ".part", path)
        t = pq.read_table(path)
        for v, lab in zip(t.column("video").to_pylist(), t.column("label").to_pylist()):
            clips.append((v["path"], CLASSES[lab], v["bytes"]))
    by_class = {}
    for c in clips:
        by_class.setdefault(c[1], []).append(c)
    order = lambda c: hashlib.md5(c[0].encode()).hexdigest()
    chosen = list(by_class.get(FALL, []))
    for cls, cs in sorted(by_class.items()):
        if cls != FALL:
            chosen += sorted(cs, key=order)[:a.per_class]
    if a.limit:
        chosen = sorted(chosen, key=order)[:a.limit]
    print(f"{sum(c[1] == FALL for c in chosen)} fall clips, {sum(c[1] != FALL for c in chosen)} non-fall clips")

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    venc = VideoEncoder(a.video_model, a.device) if a.video_model else None
    W = {"frames": [], "video": [], "margin": [], "clip": []}
    C = {"name": [], "cls": [], "label": [], "hard": []}
    t0 = time.time()
    for ci, (name, cls, data) in enumerate(chosen):
        try:
            wins = windows(duration(data))
            f16s = [sample_clip(data, s, e) for s, e in wins]
        except Exception as ex:
            print("skip", name, type(ex).__name__, ex)
            continue
        k = len(C["name"])
        for f16 in f16s:
            fe = enc.encode_images(clip_subset(f16))
            W["frames"].append(fe); W["margin"].append(enc.zero_shot(fe)[1]); W["clip"].append(k)
        if venc:
            W["video"].extend(venc.encode(f16s))
        C["name"].append(name); C["cls"].append(cls); C["label"].append(int(cls == FALL))
        C["hard"].append(int(cls in HARD_CLASSES))
        if (ci + 1) % 100 == 0:
            print(f"  {ci + 1}/{len(chosen)} clips, {len(W['clip'])} windows, {(ci + 1) / (time.time() - t0):.2f} clips/s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    extra = {"video": np.array(W["video"], np.float16)} if venc else {}
    np.savez(a.out, **extra, frames=np.array(W["frames"], np.float16), margin=np.array(W["margin"], np.float32),
             win_clip=np.array(W["clip"]), clip_name=np.array(C["name"]), clip_cls=np.array(C["cls"]),
             clip_label=np.array(C["label"]), clip_hard=np.array(C["hard"]),
             model=np.array(f"{a.model}/{a.pretrained}"))
    print(f"saved {a.out}: {sum(C['label'])} fall / {len(C['label']) - sum(C['label'])} non-fall clips, "
          f"{len(W['clip'])} windows")


if __name__ == "__main__":
    main()
