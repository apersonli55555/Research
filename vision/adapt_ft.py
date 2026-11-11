#!/usr/bin/env python3
"""Anchored fine-tuning: does the anchoring idea of adapt.py carry over to end-to-end fine-tuning?

For each budget k (real falls + k real non-falls from our training pool) and each random draw:
  anchored   start from the synthetic-fine-tuned VideoMAE (model_synthetic, run 6) and fine-tune on
             the real clips with the loss  CE + (ALPHA / 2) * ||theta - theta_synthetic||^2
             (L2-SP, Li et al. 2018); no other weight decay
  real-only  the run-6 recipe from the Kinetics checkpoint on the same clips
  synthetic  the starting model itself, not adapted

Everything below was fixed in AGENTS.md before this script produced any result on real data:
budgets, draws, steps, learning rate, ALPHA, and the rule
  on FallOrNot, (i) at every budget the anchored mean AUROC over the draws is at least
  max(real-only mean, synthetic-only) - 0.01, and (ii) averaged over budgets it is higher than both.
HMDB51 (first draw of each budget) is reported as a secondary, out-of-domain check.

Needs a GPU and the outputs of finetune.py (oops clip store, model_synthetic). Resumable.
  python vision/adapt_ft.py --work <drive>/fall-cvpr/finetune --hmdb <drive>/fall-cvpr/hmdb.npz
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import sample_clip  # noqa: E402
from experiment import _auroc  # noqa: E402
from finetune import BATCH, EPOCHS, LR, WD, ClipStore, load_model, to_tensor  # noqa: E402

# fixed in advance
BUDGETS = (8, 30, 125)
DRAWS = 3
MIN_STEPS = 60
ALPHA = 0.01
TOL = 0.01


def n_steps(n_clips, min_steps):
    return max(min_steps, EPOCHS * n_clips // BATCH)


def fit(method, frames, y, seed, dev, init, steps):
    """Fine-tune on in-memory clips. Returns the trained model (eval mode) with its mean/std."""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    anchored = method == "anchored"
    model, mean, std = load_model(dev, init if anchored else None)
    params = [p for p in model.parameters() if p.requires_grad]
    ref = [p.detach().clone() for p in params] if anchored else None
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0 if anchored else WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    w = np.where(y == 1, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum()))
    model.train(); t0 = time.time()
    for step in range(steps):
        idx = rng.choice(len(y), BATCH, p=w / w.sum())
        x = torch.stack([to_tensor(frames[i], mean, std, flip=rng.random() < 0.5) for i in idx]).to(dev)
        yt = torch.tensor(y[idx], device=dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = torch.nn.functional.cross_entropy(model(pixel_values=x).logits, yt)
        if anchored:
            loss = loss.float() + 0.5 * ALPHA * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % 20 == 0 or step == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.3f} ({(time.time() - t0) / (step + 1):.2f} s/step)", flush=True)
    return model.eval(), mean, std


def run(model, mean, std, clips, dev):
    import torch
    out = []
    for i in range(0, len(clips), 16):
        x = torch.stack([to_tensor(c, mean, std) for c in clips[i:i + 16]]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = model(pixel_values=x).logits.float()
        out.extend((lg[:, 1] - lg[:, 0]).cpu().numpy().tolist())
    return np.array(out, np.float32)


def hmdb_store(a):
    """Cache every 2 s window of the HMDB51 test clips once on local disk."""
    import pyarrow.parquet as pq, urllib.request
    from hmdb_prep import BASE, FILES, duration, windows
    E = np.load(a.hmdb)
    names = E["clip_name"].tolist(); label = dict(zip(names, E["clip_label"].tolist()))
    store = ClipStore(a.hmdb_cache, "hmdb_windows")
    done_flag = os.path.join(a.hmdb_cache, "hmdb_windows.done")
    if not os.path.exists(done_flag):
        have = {n for n, _ in store.keys}
        for f in FILES:
            if len(have) >= len(names) or (a.limit and len(have) >= a.limit):
                break
            path = os.path.join(a.hmdb_cache, f)
            if not os.path.exists(path):
                urllib.request.urlretrieve(BASE + f, path + ".part"); os.replace(path + ".part", path)
            for v in pq.read_table(path).column("video").to_pylist():
                if v["path"] in label and v["path"] not in have:
                    for j, (st, en) in enumerate(windows(duration(v["bytes"]))):
                        store.add(v["path"], str(j), sample_clip(v["bytes"], st, en))
                    have.add(v["path"])
                    if a.limit and len(have) >= a.limit:
                        break
        open(done_flag, "w").close()
        print(f"HMDB51: cached {len(store.items)} windows of {len(have)} clips", flush=True)
    return store, label


def hmdb_auroc(model, mean, std, store, label, dev):
    best = {}
    for i in range(0, len(store.items), 64):
        chunk = store.items[i:i + 64]
        for it, s in zip(chunk, run(model, mean, std, [store.frames(it) for it in chunk], dev)):
            best[it["name"]] = max(best.get(it["name"], -np.inf), float(s))
    n = list(best)
    return float(_auroc(np.array([label[k] for k in n]), np.array([best[k] for k in n])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="finetune.py work dir (oops clip store, model_synthetic)")
    ap.add_argument("--init", default="", help="starting model (default: <work>/model_synthetic)")
    ap.add_argument("--hmdb", default="", help="hmdb.npz, for the secondary out-of-domain check")
    ap.add_argument("--hmdb-cache", default="/content/hmdb")
    ap.add_argument("--out", default="")
    ap.add_argument("--budgets", default="", help="testing only")
    ap.add_argument("--draws", type=int, default=DRAWS, help="testing only")
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS, help="testing only")
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    init = a.init or os.path.join(a.work, "model_synthetic")
    assert os.path.exists(os.path.join(init, "config.json")), f"{init} not found: run finetune.py train --train-on synthetic first"
    out = a.out or os.path.join(a.work, "adapt_ft")
    os.makedirs(out, exist_ok=True)
    budgets = tuple(int(b) for b in a.budgets.split(",")) if a.budgets else BUDGETS
    res_path = os.path.join(out, "adapt_ft.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    save = lambda: json.dump(res, open(res_path, "w"), indent=1)

    s = ClipStore(a.work, "oops")
    tr = [it for it in s.items if it["split"] == "train"]
    te = [it for it in s.items if it["split"] == "test"]
    ytr = np.array([it["label"] for it in tr]); yte = np.array([it["label"] for it in te])
    print(f"real pool {len(tr)} clips ({ytr.sum()} falls); FallOrNot test {len(te)} clips ({yte.sum()} falls); device {dev}")
    test_frames = [s.frames(it) for it in te]
    hm = hmdb_store(a) if a.hmdb else None

    def evaluate(key, model, mean, std, with_hmdb):
        r = {"fallornot": float(_auroc(yte, run(model, mean, std, test_frames, dev)))}
        if hm and with_hmdb:
            r["hmdb51"] = hmdb_auroc(model, mean, std, hm[0], hm[1], dev)
        res[key] = r; save()
        print(f"  {key}: {r}", flush=True)

    if "synthetic" not in res:
        model, mean, std = load_model(dev, init)
        evaluate("synthetic", model.eval(), mean, std, True)
        del model

    rng = np.random.default_rng(0)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    for k in budgets:
        for d in range(a.draws):
            idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]   # drawn even when resuming
            for method in ("anchored", "real"):
                key = f"{method}_k{k}_d{d}"
                if key in res:
                    continue
                print(f"{key}: {len(idx)} clips", flush=True)
                frames = [s.frames(tr[i]) for i in idx]
                model, mean, std = fit(method, frames, ytr[idx], 1000 * k + d, dev, init, n_steps(len(idx), a.min_steps))
                evaluate(key, model, mean, std, d == 0)
                del model
                if dev == "cuda":
                    torch.cuda.empty_cache()

    # ---- report
    syn = res["synthetic"]["fallornot"]
    rows = []
    for k in budgets:
        g = lambda m, t="fallornot": [res[f"{m}_k{k}_d{d}"][t] for d in range(a.draws) if t in res[f"{m}_k{k}_d{d}"]]
        rows.append(dict(k=k, anchored=float(np.mean(g("anchored"))), anchored_sd=float(np.std(g("anchored"))),
                         real=float(np.mean(g("real"))), real_sd=float(np.std(g("real"))),
                         hmdb_anchored=(g("anchored", "hmdb51") or [None])[0], hmdb_real=(g("real", "hmdb51") or [None])[0]))
    every = all(r["anchored"] >= max(r["real"], syn) - TOL for r in rows)
    ma, mr = float(np.mean([r["anchored"] for r in rows])), float(np.mean([r["real"] for r in rows]))
    avg = ma > mr and ma > syn
    res["summary"] = dict(rows=rows, synthetic=syn, every_budget=bool(every), on_average=bool(avg), mean_anchored=ma,
                          mean_real=mr, met=bool(every and avg), alpha=ALPHA, draws=a.draws, min_steps=a.min_steps)
    save()
    f = lambda v: "n/a" if v is None else f"{v:.3f}"
    L = ["ANCHORED FINE-TUNING (VideoMAE-B, end to end)",
         f"Rule: on FallOrNot, (i) at every budget anchored mean AUROC >= max(real-only mean, synthetic-only) - {TOL}; "
         "(ii) averaged over budgets anchored > both baselines", "",
         f"FallOrNot AUROC, mean ± s.d. over {a.draws} draws of k real falls + k real non-falls; synthetic-only {syn:.3f}", "",
         "| k | anchored | real-only | HMDB51 anchored (draw 0) | HMDB51 real-only (draw 0) |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['k']} | {r['anchored']:.3f} ± {r['anchored_sd']:.3f} | {r['real']:.3f} ± {r['real_sd']:.3f} | "
                 f"{f(r['hmdb_anchored'])} | {f(r['hmdb_real'])} |")
    L += ["", f"- HMDB51 synthetic-only: {f(res['synthetic'].get('hmdb51'))}",
          f"- every budget within {TOL} of the better baseline: {every}",
          f"- mean over budgets: anchored {ma:.3f}, real-only {mr:.3f}, synthetic-only {syn:.3f} -> higher than both: {avg}",
          "", f"FINE-TUNING METHOD RESULT: {'SUCCESS' if every and avg else 'NOT MET'}"]
    text = "\n".join(L)
    open(os.path.join(out, "adapt_ft.md"), "w").write(text + "\n")
    print("\n" + text + f"\n\nwrote {out}/adapt_ft.md, adapt_ft.json")


if __name__ == "__main__":
    main()
