#!/usr/bin/env python3
"""Embed the clip AFTER each accident ("consequence clip") for every processed OOPS! video.

Idea (run 21): after a fall the person is on the ground; after most other accidents they are not.
"Is a person on the ground?" is easier for CLIP than "was that a fall?", so the after-state can
pseudo-label the accident clip without any human label. This script only stores embeddings; the
analysis is `nolabel.py --post <out>` (no GPU).

Consequence clip: [t+1, t+3] s after the annotated transition time t (the event clip is [t-1, t+1]).
It is used only to choose pseudo-labels for TRAINING. Detectors are still evaluated on the 2 s event
clip alone, so the test protocol does not change. Text prompts below were fixed before any result.

Streams the 48 GB archive once (decoding in background threads); resumable: rerun to continue.
  python vision/post_prep.py --oops <drive>/fall-cvpr/oops --out <drive>/fall-cvpr/post
"""
import argparse, csv, io, json, os, sys, tarfile, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ClipEncoder, clip_subset, sample_clip  # noqa: E402
from oops_mine import URL, ResumableStream, transition_time  # noqa: E402

POST_START, POST_END = 1.0, 3.0      # seconds after the transition time
MIN_AFTER = 1.5                      # need at least this many seconds of video after t
GROUND_PROMPTS = [
    "a photo of a person lying on the ground",
    "a photo of a person sitting on the ground",
    "a photo of a person on the floor",
    "a photo of a person who has fallen down",
]
UPRIGHT_PROMPTS = [
    "a photo of a person standing",
    "a photo of a person walking",
    "a photo of a person running",
    "a photo of a person riding a bike",
    "a photo of a person sitting on a chair",
    "a photo of a room",
]


def duration(data):
    import av
    with av.open(io.BytesIO(data)) as c:
        s = c.streams.video[0]
        if s.duration is not None and s.time_base is not None:
            return float(s.duration * s.time_base)
        return float(c.duration / 1e6) if c.duration else 0.0


def decode_job(data, t):
    dur = duration(data)
    if dur < t + MIN_AFTER:
        return dur, None
    return dur, sample_clip(data, t + POST_START, min(t + POST_END, dur))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", default="", help="oops_mine.py output dir: only videos in its scores.csv are processed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    want = None
    if a.oops:
        with open(os.path.join(a.oops, "scores.csv")) as fh:
            want = {r["name"]: float(r["t"]) for r in csv.DictReader(fh)}
    done = set()
    for f in sorted(os.listdir(a.out)):
        if f.startswith("shard_") and f.endswith(".npz"):
            done |= set(np.load(os.path.join(a.out, f))["name"].tolist())
    shard = len([f for f in os.listdir(a.out) if f.startswith("shard_")])
    print(f"{len(done)} videos already done" + (f"; {len(want)} to cover" if want is not None else ""), flush=True)
    if want is not None and not (set(want) - done):
        print("nothing left to do"); return

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    np.savez(os.path.join(a.out, "text.npz"), ground=enc.encode_text(GROUND_PROMPTS).cpu().numpy(),
             upright=enc.encode_text(UPRIGHT_PROMPTS).cpu().numpy(), ground_prompts=np.array(GROUND_PROMPTS),
             upright_prompts=np.array(UPRIGHT_PROMPTS), model=np.array(f"{a.model}/{a.pretrained}"),
             window=np.array([POST_START, POST_END, MIN_AFTER]))
    buf = {"name": [], "frames": [], "dur": [], "has_post": []}

    def save():
        nonlocal shard
        if buf["name"]:
            np.savez(os.path.join(a.out, f"shard_{shard:05d}.npz"), name=np.array(buf["name"]),
                     frames=np.array(buf["frames"], np.float16), dur=np.array(buf["dur"], np.float32),
                     has_post=np.array(buf["has_post"], bool))
            shard += 1
            for v in buf.values():
                v.clear()

    pool = ThreadPoolExecutor(a.workers)
    pending = deque()
    n_new, t0, ann = 0, time.time(), {}
    dim = None

    def drain_one():
        nonlocal n_new, dim
        name, fut = pending.popleft()
        try:
            dur, f16 = fut.result()
        except Exception as e:
            print("skip", name, type(e).__name__, e)
            return
        if f16:
            fe = enc.encode_images(clip_subset(f16)); dim = fe.shape
        buf["name"].append(name); buf["dur"].append(dur); buf["has_post"].append(bool(f16))
        buf["frames"].append(fe if f16 else None)
        done.add(name); n_new += 1
        if n_new % 200 == 0:
            print(f"  {len(done)} videos done, {n_new / (time.time() - t0):.2f} videos/s", flush=True)
        if n_new % a.save_every == 0:
            fill(); save()

    def fill():
        """Videos with no after-clip get zero frames so every shard is a regular array."""
        for i, f in enumerate(buf["frames"]):
            if f is None:
                buf["frames"][i] = np.zeros(dim if dim else (8, 768), np.float32)

    stream = ResumableStream(a.url)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    finished = False
    try:
        for m in outer:
            if m.name.endswith("annotations.tar.gz"):
                inner = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
                for im in inner:
                    f = inner.extractfile(im)
                    if f is not None and os.path.basename(im.name) == "transition_times.json":
                        ann = json.loads(f.read())
                continue
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                if not vm.name.endswith(".mp4"):
                    continue
                name = os.path.basename(vm.name)[:-4]
                if name in done or (want is not None and name not in want):
                    continue
                t = want[name] if want is not None else (transition_time(ann[name]) if name in ann else None)
                if t is None:
                    continue
                data = vids.extractfile(vm).read()
                pending.append((name, pool.submit(decode_job, data, t)))
                while len(pending) > 4 * a.workers:
                    drain_one()
                if (a.limit and n_new >= a.limit) or (want is not None and not (set(want) - done - {p[0] for p in pending})):
                    break
            break
        while pending:
            drain_one()
        finished = True
    finally:
        stream.close()
        pool.shutdown()
        fill(); save()
        print(f"{'done' if finished else 'stopped early, progress saved'}: {len(done)} videos in {a.out}", flush=True)


if __name__ == "__main__":
    main()
