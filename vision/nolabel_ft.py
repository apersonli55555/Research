#!/usr/bin/env python3
"""Zero-label adaptation with an end-to-end fine-tuned video model.

Run 15 trained a linear probe on frozen features from pseudo-labelled accident clips and free
pre-accident negatives. This script asks whether the same zero-label training set also trains a
fine-tuned VideoMAE better than synthetic video does. No human label is read for training.

Stages (each resumable; outputs in --work, the finetune.py work dir on Drive):
  select   repeat the fixed run-15 pipeline on the saved embeddings and write the training list:
           the N_PER_CLASS highest-scoring accident clips (pseudo-falls) and N_PER_CLASS
           pre-accident clips chosen by name hash (free non-falls)
  prepare  stream the OOPS! archive once and store those clips as frames
  train    fine-tune VideoMAE-B from the Kinetics checkpoint with the run-6 recipe (--seed)
  score    score FallOrNot and HMDB51, write scores_zerolabel[_s<seed>].npz
  report   compare with the synthetic-only and real fine-tuned models of runs 6-7 (same seeds)

Fixed in AGENTS.md before any result: N_PER_CLASS, the recipe, three seeds, and the rule
  on FallOrNot the mean AUROC over seeds 0-2 is higher than the synthetic fine-tuned mean, and for
  at least two of the three seeds the paired 95% interval (by compilation) of
  zero-label minus synthetic (same seed) is above zero.
"""
import argparse, csv, hashlib, io, json, os, sys, tarfile, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import sample_clip  # noqa: E402
from experiment import _auroc, compilation, paired  # noqa: E402
from finetune import BATCH, EPOCHS, LR, WD, ClipStore, load_model, seed_suffix, to_tensor  # noqa: E402

N_PER_CLASS = 1000
SEEDS = (0, 1, 2)


def select(a):
    """The fixed run-15 pipeline, then the list of clips to fine-tune on. Reads no human label."""
    import nolabel
    from experiment import featurize, probe
    from oops_mine import load_frames
    path = os.path.join(a.work, "zerolabel_selection.json")
    if os.path.exists(path) and not a.force:
        print(f"{path} exists; skipping (use --force to redo)")
        return
    Y = np.load(a.synthetic)
    syn = probe().fit(featurize(Y["frames"], nolabel.KIND, Y["video"]), Y["label"])
    with open(os.path.join(a.oops, "scores.csv")) as fh:
        meta = {r["name"]: r for r in csv.DictReader(fh)}
    frames, videos = load_frames(a.oops), load_frames(a.oops, "video")
    O = np.load(a.ofitw)
    blocked = {n for n, r in meta.items() if r["random_sample"] == "1"} | set(json.loads(str(O["video_split"])))
    pool = sorted(k for k in frames if k in videos and k[0] in meta and k[0] not in blocked)
    ev = [k for k in pool if k[1] == "event"]; pre = [k for k in pool if k[1] == "pre"]
    pack = lambda keys: featurize(np.array([frames[k] for k in keys]), nolabel.KIND, np.array([videos[k] for k in keys]))
    Xev, Xpre = pack(ev), pack(pre)
    model, _ = nolabel.self_train(syn.decision_function(Xev), Xev, Xpre)
    order = np.argsort(-model.decision_function(Xev))[:a.n_per_class]
    pos = [ev[i][0] for i in order]
    neg = sorted((k[0] for k in pre), key=lambda n: hashlib.md5(n.encode()).hexdigest())[:a.n_per_class]
    sel = [dict(name=n, seg="event", label=1, t=float(meta[n]["t"])) for n in pos] + \
          [dict(name=n, seg="pre", label=0, t=float(meta[n]["t"])) for n in neg]
    json.dump(sel, open(path, "w"))
    print(f"selected {len(pos)} pseudo-falls and {len(neg)} pre-accident non-falls from a pool of {len(ev)} accident clips -> {path}")


def prepare(a):
    from oops_mine import URL, ResumableStream
    sel = json.load(open(os.path.join(a.work, "zerolabel_selection.json")))
    want = {}
    for it in sel:
        want.setdefault(it["name"], []).append(it)
    store = ClipStore(a.work, "zerolabel")
    todo = {n for n, its in want.items() if any((n, it["seg"]) not in store.keys for it in its)}
    print(f"{len(want)} videos selected, {len(todo)} still to extract")
    if not todo:
        return
    stream = ResumableStream(URL)
    outer = tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz")
    n = 0
    try:
        for m in outer:
            if not m.name.endswith("video.tar.gz"):
                continue
            vids = tarfile.open(fileobj=outer.extractfile(m), mode="r|gz")
            for vm in vids:
                name = os.path.basename(vm.name)[:-4]
                if name not in todo:
                    continue
                data = vids.extractfile(vm).read()
                for it in want[name]:
                    if (name, it["seg"]) in store.keys:
                        continue
                    t = it["t"]
                    lo, hi = (max(0.0, t - 1), t + 1) if it["seg"] == "event" else (t - 3.5, t - 1.5)
                    try:
                        store.add(name, it["seg"], sample_clip(data, lo, hi), label=it["label"], split="train")
                    except Exception as e:
                        print("skip", name, it["seg"], type(e).__name__, e)
                todo.discard(name); n += 1
                if n % 100 == 0:
                    print(f"  {len(store.items)} clips stored, {len(todo)} videos left", flush=True)
                if not todo or (a.limit and n >= a.limit):
                    break
            break
    finally:
        stream.close()
    print(f"zerolabel store: {len(store.items)} clips")


def train(a):
    import torch
    out = os.path.join(a.work, "model_zerolabel" + seed_suffix(a.seed))
    if os.path.exists(os.path.join(out, "config.json")) and not a.force:
        print(f"{out} already trained; skipping")
        return
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = ClipStore(a.work, "zerolabel")
    items = s.items[:a.limit] if a.limit else s.items
    y = np.array([it["label"] for it in items])
    print(f"zero-label fine-tuning: {len(items)} clips, {int(y.sum())} pseudo-falls, seed {a.seed}, device {dev}")
    w = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum()))
    model, mean, std = load_model(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = a.steps or EPOCHS * max(1, len(items) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    model.train(); t0 = time.time()
    for step in range(steps):
        idx = rng.choice(len(items), BATCH, p=w / w.sum())
        x = torch.stack([to_tensor(s.frames(items[i]), mean, std, flip=rng.random() < 0.5) for i in idx]).to(dev)
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
    from adapt_ft import hmdb_store, run
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "zerolabel" + seed_suffix(a.seed)
    out = os.path.join(a.work, f"scores_{tag}.npz")
    if os.path.exists(out) and not a.force:
        print(f"{out} exists; skipping")
        return
    model, mean, std = load_model(dev, os.path.join(a.work, "model_" + tag))
    model.eval()
    s = ClipStore(a.work, "oops")
    test = [it for it in s.items if it["split"] == "test"]
    res = dict(fon_name=np.array([it["name"] for it in test]), fon_seg=np.array([it["seg"] for it in test]),
               fon_score=run(model, mean, std, [s.frames(it) for it in test], dev))
    if a.hmdb:
        store, _ = hmdb_store(a)
        best = {}
        for i in range(0, len(store.items), 64):
            chunk = store.items[i:i + 64]
            for it, v in zip(chunk, run(model, mean, std, [store.frames(it) for it in chunk], dev)):
                best[it["name"]] = max(best.get(it["name"], -np.inf), float(v))
        res.update(hmdb_name=np.array(list(best)), hmdb_score=np.array(list(best.values()), np.float32))
    np.savez(out, train_on=np.array("zerolabel"), seed=np.array(a.seed), **res)
    print("saved", out)


def report(a):
    s = ClipStore(a.work, "oops")
    lab = {(it["name"], it["seg"]): it["label"] for it in s.items if it["split"] == "test"}
    E = np.load(a.hmdb) if a.hmdb else None
    if E is not None:
        hy = dict(zip(E["clip_name"].tolist(), E["clip_label"].tolist()))
        hh = dict(zip(E["clip_name"].tolist(), (E["clip_hard"].astype(bool) | (E["clip_label"] == 1)).tolist()))

    def load(tag):
        p = os.path.join(a.work, f"scores_{tag}.npz")
        return np.load(p) if os.path.exists(p) else None

    def views(Z):
        k = list(zip(Z["fon_name"].tolist(), Z["fon_seg"].tolist()))
        v = {"fallornot": (k, dict(zip(k, Z["fon_score"].tolist())))}
        v["fallornot_hard"] = ([x for x in k if x[1] == "event"], v["fallornot"][1])
        if E is not None and "hmdb_name" in Z.files:
            hs = dict(zip(Z["hmdb_name"].tolist(), Z["hmdb_score"].tolist()))
            v["hmdb51"] = (list(hs), hs); v["hmdb51_hard"] = ([n for n in hs if hh[n]], hs)
        return v

    def truth(view, key):
        return lab[key] if view.startswith("fallornot") else hy[key]

    def group(view, key):
        return compilation(key[0]) if view.startswith("fallornot") else key

    rows, tests = {}, {}
    for kind in ("zerolabel", "synthetic", "real"):
        for sd in SEEDS:
            Z = load(kind + seed_suffix(sd))
            if Z is None:
                continue
            for view, (keys, sc) in views(Z).items():
                y = np.array([truth(view, k) for k in keys])
                rows.setdefault(kind, {}).setdefault(view, []).append(float(_auroc(y, np.array([sc[k] for k in keys]))))
    for sd in SEEDS:
        Zz = load("zerolabel" + seed_suffix(sd))
        for other in ("synthetic", "real"):
            Zo = load(other + seed_suffix(sd))
            if Zz is None or Zo is None:
                continue
            vz, vo = views(Zz), views(Zo)
            for view in vz:
                if view not in vo:
                    continue
                keys = [k for k in vz[view][0] if k in vo[view][1]]
                y = np.array([truth(view, k) for k in keys]); g = np.array([str(group(view, k)) for k in keys])
                tests.setdefault(f"zero-label vs {other}", {}).setdefault(view, []).append(
                    paired(y, np.array([vz[view][1][k] for k in keys]), np.array([vo[view][1][k] for k in keys]), g))
    ok = "zerolabel" in rows and "synthetic" in rows and len(rows["zerolabel"]["fallornot"]) == len(SEEDS)
    met = None
    if ok:
        t = tests["zero-label vs synthetic"]["fallornot"]
        met = bool(np.mean(rows["zerolabel"]["fallornot"]) > np.mean(rows["synthetic"]["fallornot"]) and sum(x[1] > 0 for x in t) >= 2)
    json.dump(dict(auroc=rows, tests=tests, met=met), open(os.path.join(a.work, "nolabel_ft.json"), "w"), indent=1)
    views_ = ["fallornot", "fallornot_hard", "hmdb51", "hmdb51_hard"]
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f} ({len(v)})" if v else "n/a"
    g = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}], p = {v[3]:.3f}"
    L = ["ZERO-LABEL FINE-TUNING (VideoMAE-B, end to end)",
         "Rule: on FallOrNot, mean AUROC over seeds 0-2 above the synthetic fine-tuned mean, and for at least two seeds the paired 95% CI "
         "of zero-label minus synthetic (same seed, by compilation) above zero", "",
         "AUROC, mean ± s.d. over seeds (n):", "", "| Fine-tuned on | " + " | ".join(views_) + " |", "|---|" + "---|" * len(views_)]
    names = {"zerolabel": "zero-label (pseudo-falls + free negatives, no human label)", "synthetic": "synthetic (runs 6-7)", "real": "real, human labels (runs 6-7)"}
    for kind in ("zerolabel", "synthetic", "real"):
        if kind in rows:
            L.append(f"| {names[kind]} | " + " | ".join(ms(rows[kind].get(v, [])) for v in views_) + " |")
    for name, d in tests.items():
        L += ["", f"{name}, per seed (paired AUROC difference [95% CI], p):"]
        for view in views_:
            if view in d:
                L.append(f"- {view}: " + "; ".join(g(x) for x in d[view]))
    L += ["", f"ZERO-LABEL FINE-TUNING RESULT: {'incomplete' if met is None else 'SUCCESS' if met else 'NOT MET'}"]
    text = "\n".join(L)
    open(os.path.join(a.work, "nolabel_ft.md"), "w").write(text + "\n")
    print("\n" + text + f"\n\nwrote {a.work}/nolabel_ft.md, nolabel_ft.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "prepare", "train", "score", "report"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--oops", default=""); ap.add_argument("--synthetic", default=""); ap.add_argument("--ofitw", default="")
    ap.add_argument("--hmdb", default=""); ap.add_argument("--hmdb-cache", default="/content/hmdb")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-class", type=int, default=N_PER_CLASS, help="testing only")
    ap.add_argument("--steps", type=int, default=0, help="testing only")
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"select": select, "prepare": prepare, "train": train, "score": score, "report": report}[a.stage](a)


if __name__ == "__main__":
    main()
