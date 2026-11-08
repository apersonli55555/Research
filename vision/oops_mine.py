#!/usr/bin/env python3
"""Stream the OOPS! dataset once and turn it into a real-world fall benchmark candidate pool.

OOPS! (Epstein, Chen & Vondrick, CVPR 2020) has ~20k in-the-wild clips of unintentional
actions, each annotated with the moment things go wrong (transition time t), but with no
"fall" label. For every annotated clip this script:

  * cuts an EVENT clip  [t-1, t+1]   -> candidate real fall (or another kind of accident)
  * cuts a  PRE clip    [t-3.5, t-1.5] when t >= 3.5 -> real-world normal activity
  * embeds both with a frozen CLIP model and records a zero-shot "fall" score
  * keeps a small H.264 copy of event clips for human labelling:
      - the top-K by zero-shot fall margin (to mine many positives for training), and
      - a fixed pseudo-random sample (~--random-n clips, chosen by a hash of the name, so it
        is reproducible and independent of any model). The random sample is the unbiased
        evaluation set: labelling only high-scoring clips would favour the zero-shot model.

The 48 GB archive is streamed (never stored), so it fits free Colab disk. Progress is
saved every --save-every videos; re-running skips videos already processed.

Usage (Colab, GPU):
  python vision/oops_mine.py --out /content/drive/MyDrive/fall-cvpr/oops
"""
import argparse, csv, hashlib, heapq, io, json, os, subprocess, sys, tarfile, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import ClipEncoder, read_frames, write_clip  # noqa: E402

URL = "https://huggingface.co/datasets/pengxiang/oops/resolve/main/video_and_anns.tar.gz"
PRE_MIN_T = 3.5
N_ANNOTATED = 10961  # clips in transition_times.json


def in_random_sample(name, n_target):
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) / 2 ** 128
    return h < n_target / N_ANNOTATED


def transition_time(entry):
    """Median of valid annotator times; None unless >= 2 annotators found a transition."""
    ts = [x for x in entry["t"] if x >= 0]
    return float(np.median(ts)) if len(ts) >= 2 else None


class ResumableStream(io.RawIOBase):
    """Read-only byte stream over HTTP that survives dropped connections.

    Streaming the 48 GB archive takes hours because each video is processed as it
    arrives, and servers or proxies drop such slow connections. On any early end or
    error this reconnects with an HTTP Range request from the exact byte reached.
    """

    def __init__(self, url, max_retries=50):
        super().__init__()
        self.url, self.pos, self.proc, self.retries, self.max_retries = url, 0, None, 0, max_retries
        head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
        sizes = [int(l.split(":", 1)[1]) for l in head.splitlines() if l.lower().startswith("content-length:")]
        self.total = sizes[-1] if sizes else None

    def readable(self):
        return True

    def _open(self):
        if self.proc:
            self.proc.kill(); self.proc.wait()
        self.proc = subprocess.Popen(["curl", "-sL", "--fail", "-r", f"{self.pos}-", self.url],
                                     stdout=subprocess.PIPE, bufsize=0)

    def readinto(self, b):
        while True:
            if self.proc is None:
                self._open()
            try:
                n = self.proc.stdout.readinto(b)
            except OSError:
                n = 0
            if n:
                self.pos += n
                self.retries = 0
                return n
            if self.total is not None and self.pos >= self.total:
                return 0                                   # real end of file
            self.retries += 1
            if self.retries > self.max_retries:
                raise IOError(f"download kept failing at byte {self.pos}")
            print(f"connection dropped at {self.pos / 1e9:.2f} GB; reconnecting (try {self.retries})", flush=True)
            time.sleep(min(60, 2 ** min(self.retries, 6)))
            self._open()

    def close(self):
        if self.proc:
            self.proc.kill(); self.proc.wait()
        super().close()


def decode_job(data, t):
    """Runs in a worker thread: decode the event (and pre-event) frames of one video."""
    ev = read_frames(data, max(0.0, t - 1), t + 1)
    pre = read_frames(data, t - 3.5, t - 1.5) if t >= PRE_MIN_T else None
    return ev, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--keep", type=int, default=3000, help="top-K event clips saved for labelling")
    ap.add_argument("--random-n", type=int, default=600, help="approx. size of the unbiased random sample")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new videos (quick test)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2)))
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    clip_dir, emb_dir = os.path.join(a.out, "clips"), os.path.join(a.out, "frames")
    os.makedirs(clip_dir, exist_ok=True); os.makedirs(emb_dir, exist_ok=True)
    rows_path = os.path.join(a.out, "scores.csv")
    rows = []
    if os.path.exists(rows_path):                      # resume
        with open(rows_path) as fh:
            rows = list(csv.DictReader(fh))
        print(f"resuming: {len(rows)} videos already processed")
    done = {r["name"] for r in rows}
    kept = [(float(r["event_margin"]), r["name"]) for r in rows if r["kept"] == "1" and r["random_sample"] == "0"]
    heapq.heapify(kept)                                # min-heap of the top-K clips on disk
    shard_ids = [int(f[6:11]) for f in os.listdir(emb_dir) if f.startswith("shard_")]
    next_shard = max(shard_ids) + 1 if shard_ids else 0
    buf = {"name": [], "seg": [], "frames": []}       # per-frame embeddings not yet written

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    ann, split = None, {}
    clip_path = lambda n: os.path.join(clip_dir, n.replace("/", "_") + ".mp4")
    writes = {}                                        # name -> pending write_clip future

    def save():
        nonlocal next_shard
        if buf["name"]:                                # frames first, then the CSV that references them
            np.savez(os.path.join(emb_dir, f"shard_{next_shard:05d}.npz"), name=np.array(buf["name"]),
                     seg=np.array(buf["seg"]), frames=np.array(buf["frames"], dtype=np.float16))
            next_shard += 1
            for k in buf:
                buf[k].clear()
        fields = ["name", "split", "t", "len", "event_fall_prob", "event_margin",
                  "pre_fall_prob", "pre_margin", "random_sample", "kept"]
        keep_names = {n for _, n in kept}
        for r in rows:
            r["kept"] = "1" if (r["name"] in keep_names or r["random_sample"] == "1") else "0"
        with open(rows_path + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
        os.replace(rows_path + ".tmp", rows_path)

    def handle(name, t, data, fut):
        """Main thread: embed decoded frames, score, decide what to keep."""
        try:
            ev_frames, pre_frames = fut.result()
            ev = enc.encode_images(ev_frames)
            p_ev, m_ev = enc.zero_shot(ev)
            p_pre = m_pre = ""
            pre = None
            if pre_frames is not None:
                pre = enc.encode_images(pre_frames)
                pp, mp = enc.zero_shot(pre)
                p_pre, m_pre = f"{pp:.5f}", f"{mp:.4f}"
        except Exception as e:                         # corrupt / undecodable clip
            print("skip", name, type(e).__name__, e)
            return False
        buf["name"].append(name); buf["seg"].append("event"); buf["frames"].append(ev)
        if pre is not None:
            buf["name"].append(name); buf["seg"].append("pre"); buf["frames"].append(pre)
        rnd = in_random_sample(name, a.random_n)
        top = not rnd and (len(kept) < a.keep or m_ev > kept[0][0])
        if rnd or top:
            writes[name] = pool.submit(write_clip, data, max(0.0, t - 2), t + 2, clip_path(name))
        if top:
            if len(kept) >= a.keep:
                _, drop = heapq.heappop(kept)
                if drop in writes:
                    writes.pop(drop).result()          # never delete a file that is still being written
                try:
                    os.remove(clip_path(drop))
                except FileNotFoundError:
                    pass
            heapq.heappush(kept, (m_ev, name))
        rows.append(dict(name=name, split=split.get(name, ""), t=f"{t:.3f}", len=ann[name]["len"],
                         event_fall_prob=f"{p_ev:.5f}", event_margin=f"{m_ev:.4f}",
                         pre_fall_prob=p_pre, pre_margin=m_pre,
                         random_sample="1" if rnd else "0", kept="1" if (rnd or top) else "0"))
        done.add(name)
        for n in [n for n, f in writes.items() if f.done()]:
            writes.pop(n).result()                     # surface write errors
        return True

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(a.workers)
    pending = deque()                                  # (name, t, data, future) in stream order
    stream = ResumableStream(a.url)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    n_seen, n_new, t0 = 0, 0, time.time()

    def drain_one():
        nonlocal n_new
        if handle(*pending.popleft()):
            n_new += 1
            if n_new % 50 == 0:
                print(f"{len(done)} processed ({n_seen} seen in archive) "
                      f"{n_new / (time.time() - t0):.2f} videos/s", flush=True)
            if n_new % a.save_every == 0:
                save()

    finished = False
    try:
        for m in outer:
            if m.name.endswith("annotations.tar.gz"):
                inner = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
                for im in inner:
                    f = inner.extractfile(im)
                    if f is None:
                        continue
                    base = os.path.basename(im.name)
                    data = f.read()
                    with open(os.path.join(a.out, base), "wb") as fh:
                        fh.write(data)
                    if base == "transition_times.json":
                        ann = json.loads(data)
                    elif base in ("train.txt", "val.txt"):
                        for line in data.decode().splitlines():
                            split[line.strip()] = base[:-4]
                print(f"annotations: {len(ann)} clips with transition times")
                continue
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                if not vm.name.endswith(".mp4"):
                    continue
                name = os.path.basename(vm.name)[:-4]
                n_seen += 1
                if n_seen % 2000 == 0 and n_new == 0:
                    print(f"skipping already-processed videos... {n_seen} seen", flush=True)
                if name in done or name not in ann:
                    continue
                t = transition_time(ann[name])
                if t is None:
                    continue
                data = vids.extractfile(vm).read()
                pending.append((name, t, data, pool.submit(decode_job, data, t)))
                while len(pending) > 4 * a.workers:
                    drain_one()
                if a.limit and n_new >= a.limit:
                    break
            break
        while pending and not (a.limit and n_new >= a.limit):
            drain_one()
        finished = True
    finally:                                           # save progress even if the stream breaks
        stream.close()
        for f in writes.values():
            try:
                f.result()
            except Exception as e:
                print("clip write failed:", e)
        pool.shutdown()
        save()
        print(f"{'done' if finished else 'stopped early, progress saved'}: {len(done)} videos, "
              f"{len(kept)} top candidates + random sample in {clip_dir}", flush=True)


def load_frames(oops_dir):
    """All saved per-frame embeddings: dict (name, seg) -> (8, d) float32. Later shards win."""
    out = {}
    d = os.path.join(oops_dir, "frames")
    for f in sorted(os.listdir(d)):
        if f.startswith("shard_") and f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            for n, s, fr in zip(z["name"], z["seg"], z["frames"]):
                out[(str(n), str(s))] = fr.astype(np.float32)
    return out


if __name__ == "__main__":
    main()
