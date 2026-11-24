#!/usr/bin/env python3
"""Image-selected shortlist for Grounding DINO (PLAN amendment D).

  python ovd/shortlist.py rank   --out OUT --dataset lvis                 # CLIP ranking of all names per image
  python ovd/shortlist.py replay --out OUT --dataset lvis --results F    # diagnostic on saved FIX records
  python ovd/shortlist.py run    --out OUT --dataset lvis                 # one Grounding DINO pass per image
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import match, wap  # noqa: E402

TH = 0.35


def ap_of(recs_by_cls):
    aps = [wap(np.array([s for s, _ in r]), np.array([float(t) for _, t in r]), np.ones(len(r)), n)
           for r, n in recs_by_cls.values() if n > 0]
    return float(np.mean(aps)), len(aps)


def shortlist(names_ranked, tokens_of, limit=256):
    out, used = [], 2
    for n in names_ranked:
        t = len(tokens_of(n)) + 1
        if used + t > limit:
            break
        out.append(n); used += t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["rank", "replay", "run"])
    ap.add_argument("--out", required=True); ap.add_argument("--dataset", default="lvis")
    ap.add_argument("--results"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.out, f"plan_{a.dataset}.json"))); names = P["names"]; info = P["info"]
    rank_path = os.path.join(a.out, f"shortlist_rank_{a.dataset}.json")
    if a.stage == "rank":
        import open_clip, torch
        from PIL import Image
        m, _, prep = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); m.eval()
        tok = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            T = torch.cat([m.encode_text(tok([f"a photo of a {n}" for n in names[i:i + 256]])) for i in range(0, len(names), 256)])
            T = torch.nn.functional.normalize(T.float(), dim=-1)
            R = {}
            for f in list(info)[: a.limit or None]:
                e = torch.nn.functional.normalize(m.encode_image(prep(Image.open(os.path.join(a.out, "images", f)).convert("RGB"))[None]).float(), dim=-1)
                R[f] = [names[j] for j in torch.argsort(-(e @ T.T)[0]).tolist()]
        json.dump(R, open(rank_path, "w")); print(f"ranked {len(R)} images -> {rank_path}"); return
    from transformers import AutoTokenizer
    btok = AutoTokenizer.from_pretrained("bert-base-uncased"); cache = {}
    tokens_of = lambda n: cache.setdefault(n, btok(n, add_special_tokens=False)["input_ids"])
    R = json.load(open(rank_path)); SL = {f: shortlist(r, tokens_of) for f, r in R.items()}
    print(f"shortlist length: mean {np.mean([len(v) for v in SL.values()]):.1f} names", flush=True)
    keyed = [(p["image"], p["target"], p["present"]) for p in P["plan"] if p["image"] in SL]
    gts = lambda f, c: [g["box"] for g in info[f]["gt"] if g["cls"] == c]
    if a.stage == "replay":
        FIX = {(r["image"], r["target"]): r for r in map(json.loads, open(a.results)) if r["cond"] == "FIX"}
        surv, by_size, full, short = [], {"small": [], "medium": [], "large": []}, {}, {}
        for f, c, pres in keyed:
            if (f, c) not in FIX:
                continue
            g = gts(f, c); ins = c in SL[f]; ign = c in info[f].get("not_exh", [])
            for b in g:
                area = (b[2] - b[0]) * (b[3] - b[1]); sz = "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"
                surv.append(ins); by_size[sz].append(ins)
            dets = sorted(FIX[(f, c)]["dets"], key=lambda d: -d[0])
            for store, D in ((full, dets), (short, dets if ins else [])):
                e = store.setdefault(c, [[], 0]); e[1] += len(g)
                e[0] += [(s, ok) for s, ok in match(D, g) if ok or not ign]
        fa, n1 = ap_of(full); sa, n2 = ap_of(short)
        print(f"replay on {len({k[:2] for k in FIX})} pairs: instance survival {np.mean(surv):.3f} "
              f"(small {np.mean(by_size['small']) if by_size['small'] else float('nan'):.3f}, medium {np.mean(by_size['medium']):.3f}, "
              f"large {np.mean(by_size['large']):.3f}); mAP50 fixed chunks {fa:.4f} -> shortlist-replay {sa:.4f} ({n1} classes)")
        return
    import torch
    from PIL import Image
    from vocab_full import GDino, target_record
    g = GDino("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(a.out, f"full_{a.dataset}_gdino_shortlist.jsonl")
    done = {(r["image"], r["target"]) for r in map(json.loads, open(out))} if os.path.exists(out) else set()
    by_img = {}
    for f, c, pres in keyed:
        if (f, c) not in done:
            by_img.setdefault(f, []).append((c, pres))
    import time
    with open(out, "a") as fh:
        for f, items in by_img.items():
            img = Image.open(os.path.join(a.out, "images", f)).convert("RGB")
            t0 = time.time(); boxes, S = g(img, SL[f]); sec = time.time() - t0
            for c, pres in items:
                rec = dict(image=f, target=c, cond="SHORT", present=pres, sec=sec, in_shortlist=c in SL[f])
                if c in SL[f]:
                    rec.update(target_record(boxes, S, SL[f].index(c), gts(f, c), TH, False))
                else:
                    rec.update(dets=[], best_box=[0, 0, 0, 0], instances=[dict(raw=0.0, deployed=0.0, best_iou=0.0) for _ in gts(f, c)],
                               img_max_raw=0.0, img_max_deployed=0.0)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print("wrote", out)


if __name__ == "__main__":
    main()
