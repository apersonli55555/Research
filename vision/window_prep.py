#!/usr/bin/env python3
"""Sliding-window embeddings of every unlabelled OOPS! pool video, for timestamp-free zero-label
adaptation (run 23; analysis: `nolabel.py --windows <out>`, no GPU).

The zero-label method of run 15 needs the annotated accident time t of each video. To test whether it
works on video WITHOUT timestamps, we embed every 2 s window (1 s stride, as for HMDB51) of every pool
video; the analysis then lets the detector pick the event window itself and never reads t.

Each window is sampled exactly like every other clip in the project (16 frames; CLIP on 8 of them,
VideoMAE on all 16). One decode per video serves all of its windows. Pool = videos in scores.csv that
are neither in FallOrNot's test sample nor in OF-ItW. Resumable: rerun to continue.

  python vision/window_prep.py --oops <drive>/fall-cvpr/oops --ofitw <drive>/fall-cvpr/ofitw.npz \
         --out <drive>/fall-cvpr/windows
"""
import argparse, csv, io, json, os, sys, tarfile, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import VIDEO_FRAMES, ClipEncoder, VideoEncoder, clip_subset  # noqa: E402
from hmdb_prep import windows  # noqa: E402
from oops_mine import URL, ResumableStream  # noqa: E402
from post_prep import duration  # noqa: E402

MAX_WINDOWS = 30


def sample_windows(data, wins, n=VIDEO_FRAMES):
    """16 frames for each window from ONE decode. Same rule as common.read_frames for every target:
    the first frame at or after the target time, or the previous frame if that is closer."""
    import av
    targets = sorted((t, w, k) for w, (s, e) in enumerate(wins) for k, t in enumerate(np.linspace(s, e, n)))
    out = [[None] * n for _ in wins]
    j = 0
    with av.open(io.BytesIO(data)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        prev = None
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            t = float(fr.pts * st.time_base)
            img = None
            while j < len(targets) and t >= targets[j][0]:
                tt, w, k = targets[j]
                if prev is None or abs(t - tt) <= abs(prev[0] - tt):
                    img = img or fr.to_image(); out[w][k] = img
                else:
                    out[w][k] = prev[2] if prev[2] is not None else prev[1].to_image()
                j += 1
            if j >= len(targets):
                break
            prev = (t, fr, img)
        while j < len(targets) and prev is not None:          # windows that run past the end
            _, w, k = targets[j]
            out[w][k] = prev[2] if prev[2] is not None else prev[1].to_image(); j += 1
    return out


def decode_job(data):
    wins = windows(duration(data))[:MAX_WINDOWS]
    return wins, sample_windows(data, wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True)
    ap.add_argument("--ofitw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    ap.add_argument("--names", default="", help="testing only: comma-separated video names instead of the pool")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    blocked = set(json.loads(str(np.load(a.ofitw)["video_split"])))
    want = set(a.names.split(",")) if a.names else {n for n, r in meta.items() if r["random_sample"] != "1" and n not in blocked}
    done = set()
    for f in sorted(os.listdir(a.out)):
        if f.startswith("shard_") and f.endswith(".npz"):
            done |= set(np.load(os.path.join(a.out, f))["video_name"].tolist())
    shard = len([f for f in os.listdir(a.out) if f.startswith("shard_")])
    left = want - done
    print(f"{len(want)} pool videos, {len(done)} done, {len(left)} to do", flush=True)
    if not left:
        return

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    venc = VideoEncoder(device=a.device)
    buf = {k: [] for k in ("video_name", "name", "start", "end", "frames", "video")}

    def save():
        nonlocal shard
        if buf["video_name"]:
            np.savez(os.path.join(a.out, f"shard_{shard:05d}.npz"), video_name=np.array(buf["video_name"]),
                     name=np.array(buf["name"]), start=np.array(buf["start"], np.float32), end=np.array(buf["end"], np.float32),
                     frames=np.array(buf["frames"], np.float16), video=np.array(buf["video"], np.float16))
            shard += 1
            for v in buf.values():
                v.clear()

    pool = ThreadPoolExecutor(a.workers)
    pending = deque()
    n_new, n_win, t0 = 0, 0, time.time()

    def drain_one():
        nonlocal n_new, n_win
        name, fut = pending.popleft()
        try:
            wins, clips = fut.result()
        except Exception as e:
            print("skip", name, type(e).__name__, e); return
        if not clips:
            return
        vid = venc.encode(clips)
        for (s, e), f16, v in zip(wins, clips, vid):
            buf["frames"].append(enc.encode_images(clip_subset(f16))); buf["video"].append(v)
            buf["name"].append(name); buf["start"].append(s); buf["end"].append(e)
        buf["video_name"].append(name)
        done.add(name); n_new += 1; n_win += len(wins)
        if n_new % 100 == 0:
            print(f"  {len(done)} videos ({n_win} windows) done, {n_new / (time.time() - t0):.2f} videos/s", flush=True)
        if n_new % a.save_every == 0:
            save()

    stream = ResumableStream(a.url)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    finished = False
    try:
        for m in outer:
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                if not vm.name.endswith(".mp4"):
                    continue
                name = os.path.basename(vm.name)[:-4]
                if name not in left or name in done:
                    continue
                pending.append((name, pool.submit(decode_job, vids.extractfile(vm).read())))
                while len(pending) > 2 * a.workers:
                    drain_one()
                if (a.limit and n_new >= a.limit) or not (left - done - {p[0] for p in pending}):
                    break
            break
        while pending:
            drain_one()
        finished = True
    finally:
        stream.close()
        pool.shutdown()
        save()
        print(f"{'done' if finished else 'stopped early, progress saved'}: {len(done)} videos in {a.out}", flush=True)


if __name__ == "__main__":
    main()
