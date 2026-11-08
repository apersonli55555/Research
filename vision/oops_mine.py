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
import argparse, csv, hashlib, heapq, json, os, subprocess, sys, tarfile, time

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--keep", type=int, default=3000, help="top-K event clips saved for labelling")
    ap.add_argument("--random-n", type=int, default=600, help="approx. size of the unbiased random sample")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (testing)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="datacomp_xl_s13b_b90k")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True)
    rows_path = os.path.join(a.out, "scores.csv")
    emb_path = os.path.join(a.out, "embeddings.npz")
    done, rows, emb = set(), [], {"name": [], "seg": [], "emb": []}
    if os.path.exists(rows_path):                      # resume
        with open(rows_path) as fh:
            rows = list(csv.DictReader(fh))
        done = {r["name"] for r in rows}
        z = np.load(emb_path, allow_pickle=False)
        emb = {"name": list(z["name"]), "seg": list(z["seg"]), "emb": list(z["emb"])}
        print(f"resuming: {len(done)} videos already processed")
    # min-heap of (score, name) for the clips currently kept on disk
    kept = [(float(r["event_margin"]), r["name"]) for r in rows if r["kept"] == "1" and r["random_sample"] == "0"]
    heapq.heapify(kept)

    enc = ClipEncoder(a.model, a.pretrained, a.device)
    ann, split = None, {}

    def save():
        fields = ["name", "split", "t", "len", "event_fall_prob", "event_margin",
                  "pre_fall_prob", "pre_margin", "random_sample", "kept"]
        keep_names = {n for _, n in kept}
        for r in rows:
            r["kept"] = "1" if (r["name"] in keep_names or r["random_sample"] == "1") else "0"
        with open(rows_path + ".tmp", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
        np.savez(emb_path + ".tmp.npz", name=np.array(emb["name"]), seg=np.array(emb["seg"]),
                 emb=np.array(emb["emb"], dtype=np.float16).reshape(len(emb["name"]), -1))
        os.replace(rows_path + ".tmp", rows_path)
        os.replace(emb_path + ".tmp.npz", emb_path)

    curl = subprocess.Popen(["curl", "-sL", "--retry", "5", a.url], stdout=subprocess.PIPE, bufsize=1 << 20)
    outer = tarfile.open(fileobj=curl.stdout, mode="r|gz")
    n_seen, n_new, t0 = 0, 0, time.time()
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
            if name in done or name not in ann:
                continue
            t = transition_time(ann[name])
            if t is None:
                continue
            data = vids.extractfile(vm).read()
            try:
                ev_frames = read_frames(data, max(0.0, t - 1), t + 1)
                ev = enc.encode_images(ev_frames)
                p_ev, m_ev = enc.zero_shot(ev)
                emb["name"].append(name); emb["seg"].append("event"); emb["emb"].append(enc.clip_embedding(ev))
                p_pre = m_pre = ""
                if t >= PRE_MIN_T:
                    pre = enc.encode_images(read_frames(data, t - 3.5, t - 1.5))
                    pp, mp = enc.zero_shot(pre)
                    p_pre, m_pre = f"{pp:.5f}", f"{mp:.4f}"
                    emb["name"].append(name); emb["seg"].append("pre"); emb["emb"].append(enc.clip_embedding(pre))
            except Exception as e:                       # corrupt / undecodable clip
                print("skip", name, type(e).__name__, e)
                continue
            rnd = in_random_sample(name, a.random_n)
            top = not rnd and (len(kept) < a.keep or m_ev > kept[0][0])
            keep = rnd or top
            if keep:
                safe = name.replace("/", "_")
                write_clip(data, max(0.0, t - 2), t + 2, os.path.join(a.out, "clips", safe + ".mp4"))
            if top:
                if len(kept) >= a.keep:
                    _, drop = heapq.heappop(kept)
                    try:
                        os.remove(os.path.join(a.out, "clips", drop.replace("/", "_") + ".mp4"))
                    except FileNotFoundError:
                        pass
                heapq.heappush(kept, (m_ev, name))
            rows.append(dict(name=name, split=split.get(name, ""), t=f"{t:.3f}", len=ann[name]["len"],
                             event_fall_prob=f"{p_ev:.5f}", event_margin=f"{m_ev:.4f}",
                             pre_fall_prob=p_pre, pre_margin=m_pre,
                             random_sample="1" if rnd else "0", kept="1" if keep else "0"))
            done.add(name); n_new += 1
            if n_new % 50 == 0:
                print(f"{len(done)} processed ({n_seen} seen) {n_new / (time.time() - t0):.1f} vid/s", flush=True)
            if n_new % a.save_every == 0:
                save()
            if a.limit and n_new >= a.limit:
                break
        break
    curl.kill()
    save()
    print(f"done: {len(done)} videos, {len(kept)} candidate clips in {a.out}/clips")


if __name__ == "__main__":
    main()
