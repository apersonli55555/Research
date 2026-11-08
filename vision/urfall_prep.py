#!/usr/bin/env python3
"""Turn the UR Fall Detection dataset (staged, lab) into clips and CLIP embeddings.

UR Fall (Kwolek & Kepski, CMPB 2014; CC BY-NC-SA 4.0): 30 staged falls filmed by a
side camera (cam0) and a ceiling camera (cam1), and 40 everyday activities (cam0 only).
Each .mp4 shows depth | RGB side by side; we crop the RGB half.

Clips use the same 2 s / 8-frame definition as the OOPS! clips:
  * FALL   : centred on the "falling" frames (posture label 0 in urfall-cam0-falls.csv)
  * PRE    : the 2 s ending 0.5 s before the fall starts (normal activity), if available
  * ADL    : 2 s windows every 1 s across each activity video (negatives)

Usage:  python vision/urfall_prep.py --out /content/drive/MyDrive/fall-cvpr/urfall.npz
"""
import argparse, csv, os, sys, urllib.request
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CLIP_SECONDS, ClipEncoder, read_frames  # noqa: E402

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
FPS = 30.0
RGB_CROP = (320, 0, 640, 240)


def fetch(name, cache):
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + name, path + ".part")
        os.replace(path + ".part", path)
    return path


def video_seconds(path):
    import av
    with av.open(path) as c:
        s = c.streams.video[0]
        return float(s.frames / (s.average_rate or FPS)) if s.frames else float(c.duration / 1e6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/tmp/urfall")
    ap.add_argument("--limit", type=int, default=0, help="only first N falls / ADLs (testing)")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)

    labels = OrderedDict()
    with open(fetch("urfall-cam0-falls.csv", a.cache)) as fh:
        for row in csv.reader(fh):
            labels.setdefault(row[0], []).append((int(row[1]), int(row[2])))

    clips = []   # (video_path, start_s, end_s, label, group, source, seg)
    falls = [f"fall-{i:02d}" for i in range(1, 31)]
    adls = [f"adl-{i:02d}" for i in range(1, 41)]
    if a.limit:
        falls, adls = falls[:a.limit], adls[:a.limit]
    for seq in falls:
        falling = [f for f, l in labels[seq] if l == 0]
        onset, end = (min(falling) - 1) / FPS, (max(falling) - 1) / FPS
        c = (onset + end) / 2
        for cam in ("cam0", "cam1"):
            path = fetch(f"{seq}-{cam}.mp4", a.cache)
            clips.append((path, max(0, c - CLIP_SECONDS / 2), c + CLIP_SECONDS / 2, 1, seq, f"urfall-{cam}", "fall"))
            if onset - 0.5 - CLIP_SECONDS >= 0:
                clips.append((path, onset - 0.5 - CLIP_SECONDS, onset - 0.5, 0, seq, f"urfall-{cam}", "pre"))
    for seq in adls:
        path = fetch(f"{seq}-cam0.mp4", a.cache)
        dur = video_seconds(path)
        s = 0.0
        while s + CLIP_SECONDS <= dur + 1e-6:
            clips.append((path, s, s + CLIP_SECONDS, 0, seq, "urfall-cam0", "adl"))
            s += 1.0
    print(f"{len(clips)} clips: {sum(c[3] for c in clips)} fall, {sum(1 - c[3] for c in clips)} non-fall")

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    emb, frames, margin = [], [], []
    for i, (path, s, e, *_rest) in enumerate(clips):
        fe = enc.encode_images(read_frames(path, s, e, crop=RGB_CROP))
        emb.append(enc.clip_embedding(fe))
        frames.append(fe)
        margin.append(enc.zero_shot(fe)[1])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(clips)}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, emb=np.array(emb, np.float32), frames=np.array(frames, np.float16), margin=np.array(margin, np.float32),
             label=np.array([c[3] for c in clips]), group=np.array([c[4] for c in clips]),
             source=np.array([c[5] for c in clips]), seg=np.array([c[6] for c in clips]),
             start=np.array([c[1] for c in clips], np.float32),
             model=np.array(f"{a.model}/{a.pretrained}"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
