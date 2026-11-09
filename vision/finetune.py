#!/usr/bin/env python3
"""Fine-tune VideoMAE for fall detection and score both real-world test sets.

Pre-registered question (fixed before any run): does end-to-end fine-tuning on synthetic +
real data beat the best frozen-feature probe on BOTH FallOrNot and HMDB51 (paired 95% CI
above zero on each)? Hyper-parameters below are fixed in advance and never tuned on a test set.

Stages (each resumable; outputs go to --work, e.g. Drive fall-cvpr/finetune):
  prepare-oops  stream the OOPS! archive once, keep only the labelled videos, and store the
                exact 16-frame clips used everywhere else (event [t-1,t+1], pre [t-3.5,t-1.5])
  prepare-syn   stream OF-Syn once and store --syn-per-class fall and non-fall clips
                (same clip rule as ofsyn_prep.py, hash-sampled)
  train         fine-tune VideoMAE-B (Kinetics-400) on --train-on {synthetic, real, synthetic+real}
  score         score the FallOrNot test clips and HMDB51 windows with a trained model
                (HMDB51 clip score = max over 2 s windows, as in hmdb_prep.py)

experiment.py --finetuned <work>/scores_*.npz then adds the fine-tuned rows and paired tests.
"""
import argparse, csv, hashlib, io, json, os, sys, tarfile, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import VIDEO_MODEL, sample_clip  # noqa: E402

SIZE = 224
# fixed in advance
EPOCHS, BATCH, LR, WD, SEED = 5, 8, 5e-5, 0.05, 0


# ----------------------------------------------------------------------------- clip store
class ClipStore:
    """Append-only store of 16-frame clips as JPEG bytes: <dir>/<name>.bin + <name>.jsonl."""

    def __init__(self, d, name):
        os.makedirs(d, exist_ok=True)
        self.bin, self.idx = os.path.join(d, name + ".bin"), os.path.join(d, name + ".jsonl")
        self.items = [json.loads(l) for l in open(self.idx)] if os.path.exists(self.idx) else []
        self.keys = {(it["name"], it["seg"]) for it in self.items}

    def add(self, name, seg, frames, **meta):
        from PIL import Image
        blobs = []
        for im in frames:
            im = square(im)
            b = io.BytesIO(); im.save(b, "JPEG", quality=90); blobs.append(b.getvalue())
        with open(self.bin, "ab") as fh:
            start = fh.tell()
            for b in blobs:
                fh.write(b)
        offs = list(np.cumsum([0] + [len(b) for b in blobs]) + start)
        it = dict(name=name, seg=seg, offsets=[int(o) for o in offs], **meta)
        with open(self.idx, "a") as fh:
            fh.write(json.dumps(it) + "\n")
        self.items.append(it); self.keys.add((name, seg))

    def frames(self, it):
        from PIL import Image
        with open(self.bin, "rb") as fh:
            fh.seek(it["offsets"][0]); raw = fh.read(it["offsets"][-1] - it["offsets"][0])
        o = np.array(it["offsets"]) - it["offsets"][0]
        return [Image.open(io.BytesIO(raw[o[i]:o[i + 1]])).convert("RGB") for i in range(len(o) - 1)]


def square(im):
    """Resize the short side to SIZE and centre-crop SIZE x SIZE (VideoMAE input)."""
    w, h = im.size
    s = SIZE / min(w, h)
    im = im.resize((max(SIZE, round(w * s)), max(SIZE, round(h * s))))
    w, h = im.size
    l, t = (w - SIZE) // 2, (h - SIZE) // 2
    return im.crop((l, t, l + SIZE, t + SIZE))


# ----------------------------------------------------------------------------- prepare
def labelled_oops(oops_dir):
    from labeler import load_labels
    labels = load_labels(oops_dir)
    with open(os.path.join(oops_dir, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    return {n: (lab, meta[n]) for n, lab in labels.items() if lab in ("fall", "not_fall") and n in meta}


def prepare_oops(a):
    from oops_mine import URL, PRE_MIN_T, ResumableStream
    want = labelled_oops(a.oops)
    store = ClipStore(a.work, "oops")
    todo = {n for n in want if (n, "event") not in store.keys}
    print(f"{len(want)} labelled videos, {len(todo)} still to extract")
    if not todo:
        return
    stream = ResumableStream(URL)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    try:
        for m in outer:
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                name = os.path.basename(vm.name)[:-4]
                if name not in todo:
                    continue
                lab, r = want[name]
                t = float(r["t"]); split = "test" if r["random_sample"] == "1" else "train"
                data = vids.extractfile(vm).read()
                try:
                    store.add(name, "event", sample_clip(data, max(0.0, t - 1), t + 1),
                              label=int(lab == "fall"), split=split)
                    if t >= PRE_MIN_T:
                        store.add(name, "pre", sample_clip(data, t - 3.5, t - 1.5), label=0, split=split)
                except Exception as e:
                    print("skip", name, type(e).__name__, e)
                todo.discard(name)
                if len(store.items) % 100 == 0:
                    print(f"  {len(store.items)} clips stored, {len(todo)} videos left", flush=True)
                if not todo:
                    break
            break
    finally:
        stream.close()
    print(f"oops store: {len(store.items)} clips")


def prepare_syn(a):
    from ofsyn_prep import LABELS_URL, TAR_URL, plan_clips
    from oops_mine import ResumableStream
    import urllib.request
    from common import CLIP_SECONDS
    rows = list(csv.DictReader(io.StringIO(urllib.request.urlopen(LABELS_URL).read().decode())))
    plan, _ = plan_clips(rows, 10 ** 9)
    order = lambda p: hashlib.md5(p.encode()).hexdigest()
    pos = sorted([p for p, v in plan.items() if v[1] == 1], key=order)[:a.syn_per_class]
    neg = sorted([p for p, v in plan.items() if v[1] == 0], key=order)[:a.syn_per_class]
    want = set(pos) | set(neg)
    store = ClipStore(a.work, "syn")
    want -= {n for n, _ in store.keys}
    print(f"synthetic: {len(pos)} fall + {len(neg)} non-fall planned, {len(want)} still to extract")
    if not want:
        return
    stream = ResumableStream(TAR_URL)
    tf = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|")
    try:
        for m in tf:
            p = os.path.normpath(m.name)[:-4]
            if not m.name.endswith(".mp4") or p not in want:
                continue
            centre, label = plan[p]
            try:
                store.add(p, "syn", sample_clip(tf.extractfile(m).read(), max(0.0, centre - CLIP_SECONDS / 2),
                                                centre + CLIP_SECONDS / 2), label=int(label), split="train")
            except Exception as e:
                print("skip", p, type(e).__name__, e)
            want.discard(p)
            if len(store.items) % 200 == 0:
                print(f"  {len(store.items)} synthetic clips stored, {len(want)} left", flush=True)
            if not want:
                break
    finally:
        stream.close()
    print(f"syn store: {len(store.items)} clips")


# ----------------------------------------------------------------------------- model
def load_model(device, path=None):
    import torch
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    proc = VideoMAEImageProcessor.from_pretrained(VIDEO_MODEL)
    model = VideoMAEForVideoClassification.from_pretrained(
        path or VIDEO_MODEL, num_labels=2, ignore_mismatched_sizes=True).to(device)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
    return model, mean, std


def to_tensor(frames, mean, std, flip=False):
    import torch
    x = torch.from_numpy(np.stack([np.asarray(square(f), dtype=np.float32) / 255.0 for f in frames]))
    x = x.permute(0, 3, 1, 2)                         # T, C, H, W
    if flip:
        x = x.flip(-1)
    return (x - mean) / std


def train(a):
    import torch
    out = os.path.join(a.work, "model_" + a.train_on.replace("+", "_"))
    if os.path.exists(os.path.join(out, "config.json")) and not a.force:
        print(f"{out} already trained; skipping (use --force to retrain)")
        return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = []
    if "real" in a.train_on:
        s = ClipStore(a.work, "oops")
        items += [(s, it) for it in s.items if it["split"] == "train"]
    if "synthetic" in a.train_on:
        s = ClipStore(a.work, "syn")
        items += [(s, it) for it in s.items]
    if a.limit:
        items = [items[i] for i in rng.permutation(len(items))[:a.limit]]
    y = np.array([it["label"] for _, it in items])
    print(f"train on {a.train_on}: {len(items)} clips, {y.sum()} falls, device {dev}")
    # class-balanced sampling weights (each class contributes equally)
    w = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum()))
    model, mean, std = load_model(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=WD)
    steps = a.epochs * max(1, len(items) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    model.train(); t0 = time.time()
    for step in range(steps):
        idx = rng.choice(len(items), a.batch, p=w / w.sum())
        x = torch.stack([to_tensor(items[i][0].frames(items[i][1]), mean, std, flip=rng.random() < 0.5)
                         for i in idx]).to(dev)
        yt = torch.tensor(y[idx], device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = torch.nn.functional.cross_entropy(model(pixel_values=x).logits, yt)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  step {step + 1}/{steps} loss {loss.item():.3f} ({(time.time() - t0) / (step + 1):.2f} s/step)", flush=True)
    model.save_pretrained(out)
    print("saved", out)


def score(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.train_on.replace("+", "_")
    if os.path.exists(os.path.join(a.work, f"scores_{tag}.npz")) and not a.force and not a.limit:
        print(f"scores_{tag}.npz already exists; skipping (use --force to rescore)")
        return
    model, mean, std = load_model(dev, os.path.join(a.work, "model_" + tag))
    model.eval()

    def run(clips):
        out = []
        for i in range(0, len(clips), 16):
            x = torch.stack([to_tensor(c, mean, std) for c in clips[i:i + 16]]).to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                lg = model(pixel_values=x).logits.float()
            out.extend((lg[:, 1] - lg[:, 0]).cpu().numpy().tolist())
        return out

    s = ClipStore(a.work, "oops")
    test = [it for it in s.items if it["split"] == "test"]
    if a.limit:
        test = test[:a.limit]
    fon = run([s.frames(it) for it in test])
    res = dict(fon_name=np.array([it["name"] for it in test]), fon_seg=np.array([it["seg"] for it in test]),
               fon_score=np.array(fon, np.float32))
    if a.hmdb:
        import pyarrow.parquet as pq
        from hmdb_prep import CLASSES, FALL, FILES, HARD_CLASSES, duration, windows, BASE
        import urllib.request
        E = np.load(a.hmdb)
        names = set(E["clip_name"].tolist())
        sc = {}
        for f in FILES:
            if len(sc) >= len(names) or (a.limit and len(sc) >= a.limit):
                break
            path = os.path.join(a.hmdb_cache, f)
            if not os.path.exists(path):
                os.makedirs(a.hmdb_cache, exist_ok=True)
                urllib.request.urlretrieve(BASE + f, path + ".part"); os.replace(path + ".part", path)
            t = pq.read_table(path)
            for v in t.column("video").to_pylist():
                if v["path"] in names and v["path"] not in sc:
                    data = v["bytes"]
                    clips = [sample_clip(data, st, en) for st, en in windows(duration(data))]
                    sc[v["path"]] = max(run(clips))
                    if a.limit and len(sc) >= a.limit:
                        break
        res.update(hmdb_name=np.array(list(sc)), hmdb_score=np.array(list(sc.values()), np.float32))
    out = os.path.join(a.work, f"scores_{tag}.npz")
    np.savez(out, train_on=np.array(a.train_on), **res)
    print("saved", out, {k: v.shape for k, v in res.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare-oops", "prepare-syn", "train", "score"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--oops", help="oops_mine.py output dir with labels.csv")
    ap.add_argument("--train-on", default="synthetic+real", choices=["synthetic", "real", "synthetic+real"])
    ap.add_argument("--syn-per-class", type=int, default=1000)
    ap.add_argument("--hmdb", default="", help="hmdb.npz (to know which clips to score)")
    ap.add_argument("--hmdb-cache", default="/tmp/hmdb")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    ap.add_argument("--force", action="store_true", help="redo a finished train/score stage")
    a = ap.parse_args()
    {"prepare-oops": prepare_oops, "prepare-syn": prepare_syn, "train": train, "score": score}[a.stage](a)


if __name__ == "__main__":
    main()
