#!/usr/bin/env python3
"""Fourth test set, independent of OOPS!: Kinetics-700 validation videos (run 26).

Falls: every validation video of the three human-labelled fall classes. Hard negatives: every validation video of
fixed look-alike classes (bikes, boards, skates, acrobatics: motions that resemble falls but are not). Easy
negatives: one hash-chosen validation video from every other class. All fixed before any result (see AGENTS.md).

Videos are read from the public full-resolution mirror `atalaydenknalbant/Kinetics-700` (22 zip parts) by random
access: only the zip directories and the chosen files are downloaded, not the 930 GB. Each video is covered by 2 s
windows every 2 s (at most 5); a clip's score is the maximum over its windows. Output: an npz in the same format as
hmdb.npz (so every script's --external works) and a ClipStore of the windows (224 x 224 JPEG) for the fine-tuned
models and the VLM baselines.

  python vision/kinetics_prep.py --out <drive>/fall-cvpr/kinetics.npz --store /content/kinetics_store
"""
import argparse, hashlib, io, os, sys, time, zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ClipEncoder, VideoEncoder, clip_subset  # noqa: E402
from finetune import ClipStore  # noqa: E402
from post_prep import duration  # noqa: E402
from window_prep import sample_windows  # noqa: E402

REPO = "datasets/atalaydenknalbant/Kinetics-700"
FALL_CLASSES = ["faceplanting", "falling off bike", "falling off chair"]
HARD_CLASSES = ["riding a bike", "jumping bicycle", "biking through snow", "riding unicycle", "motorcycling",
                "skateboarding", "roller skating", "ice skating", "snowboarding", "parkour", "somersaulting",
                "gymnastics tumbling", "cartwheeling", "backflip (human)", "bouncing on trampoline", "jumping sofa"]
EASY_PER_CLASS = 1
WIN, STRIDE, MAX_WIN = 2.0, 2.0, 5


def windows(dur):
    if dur <= WIN:
        return [(0.0, max(dur, 0.1))]
    starts = list(np.arange(0.0, dur - WIN + 1e-6, STRIDE))[:MAX_WIN]
    return [(float(s), float(s + WIN)) for s in starts]


def plan(fs):
    """(part, member, class) for every chosen validation video."""
    parts = sorted(p for p in fs.ls(REPO, detail=False) if p.endswith(".zip"))
    by_class = {}
    for p in parts:
        with fs.open(p, "rb", block_size=2 ** 20) as fh:
            for n in zipfile.ZipFile(fh).namelist():
                if n.startswith("validation/") and n.endswith(".mp4"):
                    by_class.setdefault(n.split("/")[1], []).append((p, n))
        print(f"  scanned {p.split('/')[-1]}: {sum(len(v) for v in by_class.values())} validation videos so far", flush=True)
    order = lambda t: hashlib.md5(t[1].encode()).hexdigest()
    chosen = []
    for c, items in sorted(by_class.items()):
        if c in FALL_CLASSES or c in HARD_CLASSES:
            chosen += [(p, n, c) for p, n in items]
        else:
            chosen += [(p, n, c) for p, n in sorted(items, key=order)[:EASY_PER_CLASS]]
    missing = [c for c in FALL_CLASSES + HARD_CLASSES if c not in by_class]
    if missing:
        print("WARNING: classes not found:", missing)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default="/content/kinetics_store")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="testing only: videos per group")
    a = ap.parse_args()
    import shutil
    backup = os.path.splitext(a.out)[0] + "_store"                 # Drive copy of the window store
    if os.path.exists(a.out):
        if not os.path.exists(os.path.join(a.store, "windows.bin")) and os.path.exists(os.path.join(backup, "windows.bin")):
            os.makedirs(a.store, exist_ok=True)
            for ext in (".jsonl", ".bin"):
                shutil.copyfile(os.path.join(backup, "windows" + ext), os.path.join(a.store, "windows" + ext))
        print(f"{a.out} exists; window store in {a.store}"); return
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    chosen = plan(fs)
    if a.limit:
        g = {}
        chosen = [t for t in chosen if g.setdefault(t[2] in FALL_CLASSES and "fall" or t[2] in HARD_CLASSES and "hard" or "easy", []).append(t) is None]
        chosen = [t for k in ("fall", "hard", "easy") for t in g.get(k, [])[:a.limit]]
    print(f"{sum(c in FALL_CLASSES for _, _, c in chosen)} fall videos, {sum(c in HARD_CLASSES for _, _, c in chosen)} hard negatives, "
          f"{sum(c not in FALL_CLASSES and c not in HARD_CLASSES for _, _, c in chosen)} easy negatives", flush=True)
    enc = ClipEncoder(a.model, a.pretrained, a.device)
    venc = VideoEncoder(device=a.device)
    store = ClipStore(a.store, "windows")
    W = {"frames": [], "video": [], "margin": [], "clip": []}
    C = {"name": [], "cls": [], "label": [], "hard": []}
    t0, zips = time.time(), {}
    for i, (part, member, cls) in enumerate(sorted(chosen)):
        try:
            if part not in zips:
                zips = {part: zipfile.ZipFile(fs.open(part, "rb", block_size=2 ** 20))}
            data = zips[part].read(member)
            wins = windows(duration(data))
            clips = sample_windows(data, wins)
        except Exception as e:
            print("skip", member, type(e).__name__, e); continue
        name = member.split("/")[-1][:-4]
        k = len(C["name"])
        vid = venc.encode(clips)
        for j, (f16, v) in enumerate(zip(clips, vid)):
            fe = enc.encode_images(clip_subset(f16))
            W["frames"].append(fe); W["video"].append(v); W["margin"].append(enc.zero_shot(fe)[1]); W["clip"].append(k)
            if (name, str(j)) not in store.keys:
                store.add(name, str(j), f16, label=int(cls in FALL_CLASSES), split="test")
        C["name"].append(name); C["cls"].append(cls); C["label"].append(int(cls in FALL_CLASSES)); C["hard"].append(int(cls in HARD_CLASSES))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(chosen)} videos, {len(W['clip'])} windows, {(i + 1) / (time.time() - t0):.2f} videos/s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, video=np.array(W["video"], np.float16), frames=np.array(W["frames"], np.float16),
             margin=np.array(W["margin"], np.float32), win_clip=np.array(W["clip"]), clip_name=np.array(C["name"]),
             clip_cls=np.array(C["cls"]), clip_label=np.array(C["label"]), clip_hard=np.array(C["hard"]),
             model=np.array(f"{a.model}/{a.pretrained}"))
    try:
        os.makedirs(backup, exist_ok=True)
        for ext in (".jsonl", ".bin"):
            shutil.copyfile(os.path.join(a.store, "windows" + ext), os.path.join(backup, "windows" + ext))
    except OSError as e:
        print("backup of the window store to Drive failed:", e)
    print(f"saved {a.out}: {sum(C['label'])} fall / {len(C['label']) - sum(C['label'])} non-fall clips "
          f"({sum(C['hard'])} hard), {len(W['clip'])} windows; window store in {a.store}")


if __name__ == "__main__":
    main()
