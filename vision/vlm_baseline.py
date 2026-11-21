#!/usr/bin/env python3
"""Zero-shot vision-language baseline (run 24): an open VLM is asked whether a person falls in each 2 s clip.

Fixed before any result (see AGENTS.md): the model IDs, the prompt below, the input (the same 8 frames per clip
that CLIP sees, i.e. frames 1, 3, ..., 15 of the 16-frame clip, taken from the same stored 224 x 224 clips the
fine-tuned VideoMAE models were scored on), and the score = logit("Yes") - logit("No") for the first answer token
(a continuous answer-likelihood score, so AUROC is defined). Clips: FallOrNot test (and its hard subset), the
OF-ItW test split without the videos behind our labelled clips (as in run 22), and HMDB51 (max over 2 s windows,
as everywhere else). Reports runtime and peak GPU memory, and paired comparisons with the fine-tuned models.

  python vision/vlm_baseline.py --work <drive>/fall-cvpr/finetune --hmdb <drive>/fall-cvpr/hmdb.npz \
         --model Qwen/Qwen3-VL-4B-Instruct
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from experiment import _auroc, compilation, paired  # noqa: E402
from finetune import ClipStore, seed_suffix  # noqa: E402

PROMPT = ("These are 8 frames, in time order, from a 2-second video clip. "
          "Does a person fall down in this clip? Answer with Yes or No.")
SEEDS = (0, 1, 2)


class VLM:
    def __init__(self, model_id, device):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
        tok = self.proc.tokenizer
        self.yes = tok.encode("Yes", add_special_tokens=False)[0]
        self.no = tok.encode("No", add_special_tokens=False)[0]

    def score(self, frames16):
        torch = self.torch
        imgs = list(frames16)[1::2]
        msg = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": PROMPT}]}]
        x = self.proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        x = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in x.items()}
        with torch.no_grad():
            lg = self.model(**x).logits[0, -1].float()
        return float(lg[self.yes] - lg[self.no])


def run_set(vlm, name, keys, get_frames, path):
    """Score clips with checkpoints; returns key -> score."""
    done = json.load(open(path)) if os.path.exists(path) else {}
    t0, n = time.time(), 0
    for k in keys:
        if k in done:
            continue
        done[k] = vlm.score(get_frames(k)); n += 1
        if n % 100 == 0:
            json.dump(done, open(path, "w"))
            print(f"  {name}: {len(done)}/{len(keys)} clips, {(time.time() - t0) / n:.2f} s/clip", flush=True)
    json.dump(done, open(path, "w"))
    return done, (time.time() - t0) / max(1, n) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--hmdb", default="")
    ap.add_argument("--hmdb-cache", default="/content/hmdb")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--local", default="/content/ofitw_store" if os.path.isdir("/content") else "ofitw_store")
    ap.add_argument("--skip-ofitw", action="store_true", help="testing only")
    ap.add_argument("--limit", type=int, default=0, help="testing only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tag = a.model.split("/")[-1]
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    vlm = VLM(a.model, dev)
    print(f"model {a.model} on {dev}; prompt: {PROMPT!r}", flush=True)
    res = {"model": a.model, "prompt": PROMPT, "sec_per_clip": {}, "auroc": {}, "tests": {}}

    # ---- FallOrNot test clips
    s = ClipStore(a.work, "oops")
    te = [it for it in s.items if it["split"] == "test"][: a.limit or None]
    fk = {f"{it['name']}|{it['seg']}": it for it in te}
    fon, res["sec_per_clip"]["fallornot"] = run_set(vlm, "FallOrNot", list(fk), lambda k: s.frames(fk[k]),
                                                    os.path.join(a.work, f"vlm_{tag}_fallornot.json"))
    sets = {"fallornot": (list(fk), {k: it["label"] for k, it in fk.items()}, {k: compilation(it["name"]) for k, it in fk.items()}, fon)}
    sets["fallornot_hard"] = ([k for k in fk if fk[k]["seg"] == "event"],) + sets["fallornot"][1:]

    # ---- OF-ItW test clips (same clips as run 22)
    if not a.skip_ofitw:
        from nolabel_ft import ofitw_store
        a.limit_store = a.limit
        st = ofitw_store(a)
        items = st.items[: a.limit * 3] if a.limit else st.items
        ok = {f"{it['name']}|{it['seg']}": it for it in items}
        ofs, res["sec_per_clip"]["ofitw"] = run_set(vlm, "OF-ItW", list(ok), lambda k: st.frames(ok[k]),
                                                    os.path.join(a.work, f"vlm_{tag}_ofitw.json"))
        sets["ofitw"] = (list(ok), {k: it["label"] for k, it in ok.items()}, {k: compilation(it["name"]) for k, it in ok.items()}, ofs)

    # ---- HMDB51: every 2 s window, clip score = max over its windows
    if a.hmdb:
        from adapt_ft import hmdb_store
        store, label = hmdb_store(a)
        wk = {f"{it['name']}|{it['seg']}": it for it in store.items}
        ws, res["sec_per_clip"]["hmdb51_windows"] = run_set(vlm, "HMDB51 windows", list(wk), lambda k: store.frames(wk[k]),
                                                            os.path.join(a.work, f"vlm_{tag}_hmdb_windows.json"))
        best = {}
        for k, v in ws.items():
            n = wk[k]["name"]; best[n] = max(best.get(n, -np.inf), v)
        E = np.load(a.hmdb)
        hard = dict(zip(E["clip_name"].tolist(), (E["clip_hard"].astype(bool) | (E["clip_label"] == 1)).tolist()))
        hk = sorted(best)
        sets["hmdb51"] = (hk, {n: label[n] for n in hk}, {n: n for n in hk}, best)
        sets["hmdb51_hard"] = ([n for n in hk if hard[n]],) + sets["hmdb51"][1:]
    if dev == "cuda":
        res["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    # ---- AUROC and paired comparisons with the fine-tuned models on exactly the same clips
    def ft_scores(kind, sd, setname):
        t = kind + seed_suffix(sd)
        if setname.startswith("fallornot"):
            p = os.path.join(a.work, f"scores_{t}.npz")
            if not os.path.exists(p):
                return None
            z = np.load(p); return dict(zip([f"{n}|{g}" for n, g in zip(z["fon_name"], z["fon_seg"])], z["fon_score"].tolist()))
        if setname == "ofitw":
            p = os.path.join(a.work, f"scores_ofitw_{t}.npz")
            if not os.path.exists(p):
                return None
            z = np.load(p); return dict(zip(z["key"].tolist(), z["score"].tolist()))
        p = os.path.join(a.work, f"scores_{t}.npz")
        if not os.path.exists(p) or "hmdb_name" not in np.load(p).files:
            return None
        z = np.load(p); return dict(zip(z["hmdb_name"].tolist(), z["hmdb_score"].tolist()))

    for setname, (keys, lab, grp, sc) in sets.items():
        y = np.array([lab[k] for k in keys]); g = np.array([grp[k] for k in keys]); v = np.array([sc[k] for k in keys])
        res["auroc"][setname] = {tag: float(_auroc(y, v)), "n_clips": len(keys), "n_falls": int(y.sum())}
        for kind in ("synthetic", "real", "zerolabel"):
            per = []
            for sd in SEEDS:
                f = ft_scores(kind, sd, setname)
                if f is None or not all(k in f for k in keys):
                    continue
                fv = np.array([f[k] for k in keys])
                res["auroc"][setname].setdefault(f"finetuned_{kind}", []).append(float(_auroc(y, fv)))
                per.append(paired(y, v, fv, g))
            if per:
                res["tests"].setdefault(setname, {})[f"VLM vs fine-tuned {kind}"] = per
    json.dump(res, open(os.path.join(a.work, f"vlm_{tag}.json"), "w"), indent=1)
    gfmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}], p = {t[3]:.3f}"
    L = ["", f"ZERO-SHOT VLM BASELINE: {a.model}", f"Prompt (fixed in advance): {PROMPT}",
         "Score: logit('Yes') - logit('No') for the first answer token; 8 frames per clip; HMDB51 = max over windows",
         "Seconds per clip: " + ", ".join(f"{k} {v:.2f}" for k, v in res["sec_per_clip"].items() if v) +
         (f"; peak GPU memory {res['peak_gpu_memory_gb']:.1f} GB" if "peak_gpu_memory_gb" in res else ""), "",
         "| Test set | clips (falls) | VLM | fine-tuned synthetic | fine-tuned real (hand labels) | fine-tuned zero-label |",
         "|---|---|---|---|---|---|"]
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "–"
    for setname, d in res["auroc"].items():
        L.append(f"| {setname} | {d['n_clips']} ({d['n_falls']}) | {d[tag]:.3f} | {ms(d.get('finetuned_synthetic'))} | "
                 f"{ms(d.get('finetuned_real'))} | {ms(d.get('finetuned_zerolabel'))} |")
    L.append("")
    for setname, d in res["tests"].items():
        for name, per in d.items():
            L.append(f"- {setname}, {name}, per seed: " + "; ".join(gfmt(t) for t in per))
    text = "\n".join(L)
    open(os.path.join(a.work, f"vlm_{tag}.md"), "w").write(text + "\n")
    print(text + f"\n\nwrote {a.work}/vlm_{tag}.md, vlm_{tag}.json")


if __name__ == "__main__":
    main()
