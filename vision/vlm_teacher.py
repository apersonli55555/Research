#!/usr/bin/env python3
"""Run 27, step 1: a zero-shot VLM scores every accident clip of the pool without fall annotations.

Pool = the run-18 pool: every processed OOPS! video except FallOrNot's test sample, all OF-ItW videos and every
video with a human fall label (so all labelled accident clips stay usable as test data). Clip = the event clip
[t-1, t+1] s around the annotated accident time, 16 frames, centre-cropped to 224 x 224 exactly like the clips the
VLM baseline was scored on; prompt and score as in vlm_baseline.py (fall prompt). Resumable; scores go to a JSON
file. Step 2 is `nolabel.py --teacher <json>` (no GPU).

  python vision/vlm_teacher.py --oops <drive>/fall-cvpr/oops --ofitw <drive>/fall-cvpr/ofitw.npz \
         --out <drive>/fall-cvpr/teacher_InternVL3_5-8B-HF.json --model OpenGVLab/InternVL3_5-8B-HF
"""
import argparse, csv, io, json, os, sys, tarfile, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import sample_clip  # noqa: E402
from finetune import square  # noqa: E402
from labeler import load_labels  # noqa: E402
from oops_mine import URL, ResumableStream  # noqa: E402
from vlm_baseline import PROMPTS, VLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oops", required=True)
    ap.add_argument("--ofitw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-8B-HF")
    ap.add_argument("--prompt", default="fall", choices=list(PROMPTS), help="run 29 uses the accident prompt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    a = ap.parse_args()
    import torch
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    labelled = {n for n, l in load_labels(a.oops).items() if l in ("fall", "not_fall")}
    blocked = set(json.loads(str(np.load(a.ofitw)["video_split"]))) | labelled
    pool = {n: float(r["t"]) for n, r in meta.items() if r["random_sample"] != "1" and n not in blocked}
    done = json.load(open(a.out)) if os.path.exists(a.out) else {}
    left = set(pool) - set(done)
    print(f"pool {len(pool)} videos, {len(done)} scored, {len(left)} to do", flush=True)
    if not left:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vlm = VLM(a.model, dev, PROMPTS[a.prompt])

    def decode(data, t):
        return [square(f) for f in sample_clip(data, max(0.0, t - 1), t + 1)]

    pool_ex = ThreadPoolExecutor(a.workers)
    pending, n, t0 = deque(), 0, time.time()

    def drain():
        nonlocal n
        name, fut = pending.popleft()
        try:
            done[name] = vlm.score(fut.result())
        except Exception as e:
            print("skip", name, type(e).__name__, e); return
        n += 1
        if n % 200 == 0:
            tmp = a.out + ".part"
            json.dump(done, open(tmp, "w")); os.replace(tmp, a.out)
            print(f"  {len(done)}/{len(pool)} scored, {(time.time() - t0) / n:.2f} s/clip", flush=True)

    stream = ResumableStream(URL)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    try:
        for m in outer:
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                name = os.path.basename(vm.name)[:-4]
                if not vm.name.endswith(".mp4") or name not in left or name in done:
                    continue
                pending.append((name, pool_ex.submit(decode, vids.extractfile(vm).read(), pool[name])))
                while len(pending) > 2 * a.workers:
                    drain()
                if (a.limit and n >= a.limit) or not (left - set(done) - {p[0] for p in pending}):
                    break
            break
        while pending:
            drain()
    finally:
        stream.close()
        pool_ex.shutdown()
        tmp = a.out + ".part"
        json.dump(done, open(tmp, "w")); os.replace(tmp, a.out)
        print(f"saved {len(done)} teacher scores to {a.out}", flush=True)


if __name__ == "__main__":
    main()
