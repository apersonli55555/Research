#!/usr/bin/env python3
"""Error analysis: which classes lose most when the full list is used, and does a near-synonym in the list explain it?

Per class c: AP50(k1), AP50(all; single-label output), AP50(FIX); drop = AP(k1) - AP(all). We correlate the drop with
s_max(c) = the highest CLIP ViT-B/32 text similarity between c and any other name in the dataset's list (Spearman,
bootstrap CI over classes).

  python ovd/error_analysis.py --out OUT --dataset lvis --model owlv2
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from full_analysis import APTable, wap  # noqa: E402


def class_aps(T, conds):
    out = {}
    for c, npos in T.npos.items():
        n = float(npos.sum())
        if n <= 0:
            continue
        vals = []
        for cd in conds:
            im, s, tp = T.arr.get(cd, {}).get(c, (np.zeros(0, int), np.zeros(0), np.zeros(0)))
            vals.append(wap(s, tp.astype(float), np.ones(len(s)), n) if len(s) else 0.0)
        out[c] = float(np.mean(vals))
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.out, f"plan_{a.dataset}.json")))
    R = []
    for line in open(os.path.join(a.out, f"full_{a.dataset}_{a.model}.jsonl")):
        try:
            R.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    T = APTable(R, P, "deployed")
    conds = sorted({r["cond"] for r in R})
    k1 = class_aps(T, ["k1"]); full = class_aps(T, [c for c in conds if c.startswith("all_")]); fix = class_aps(T, ["FIX"])
    import open_clip, torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32"); names = P["names"]
    with torch.no_grad():
        E = torch.cat([model.encode_text(tok([f"a photo of a {n}" for n in names[i:i + 256]])) for i in range(0, len(names), 256)])
        E = torch.nn.functional.normalize(E.float(), dim=-1).numpy()
    sim = E @ E.T; np.fill_diagonal(sim, -1)
    smax = {n: float(sim[i].max()) for i, n in enumerate(names)}; nn = {n: names[int(sim[i].argmax())] for i, n in enumerate(names)}
    cls = sorted(k1)
    drop = np.array([k1[c] - full[c] for c in cls]); sm = np.array([smax[c] for c in cls])
    rho = spearman(drop, sm)
    rng = np.random.default_rng(0); boots = []
    for _ in range(1000):
        i = rng.integers(0, len(cls), len(cls)); boots.append(spearman(drop[i], sm[i]))
    order = np.argsort(-drop)
    res = dict(dataset=a.dataset, model=a.model, n_classes=len(cls), spearman_drop_vs_maxsim=rho,
               spearman_CI=[float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
               frac_classes_losing=float(np.mean(drop > 0)), frac_classes_gaining=float(np.mean(drop < 0)),
               fix_recovers_frac=float(np.mean([fix[c] >= k1[c] - 1e-9 for c in cls])),
               top_losers=[dict(cls=cls[j], ap_k1=k1[cls[j]], ap_full=full[cls[j]], ap_fix=fix[cls[j]], nearest=nn[cls[j]], sim=smax[cls[j]])
                           for j in order[:15]],
               by_sim_quartile=[dict(q=q, mean_drop=float(drop[(sm >= lo) & (sm <= hi)].mean()), n=int(((sm >= lo) & (sm <= hi)).sum()))
                                for q, (lo, hi) in enumerate(zip(np.quantile(sm, [0, .25, .5, .75]), np.quantile(sm, [.25, .5, .75, 1])))])
    path = os.path.join(a.out, f"error_{a.dataset}_{a.model}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"{a.dataset}/{a.model}: {len(cls)} classes; Spearman(drop, max text sim) {rho:+.3f} "
          f"[{res['spearman_CI'][0]:+.3f}, {res['spearman_CI'][1]:+.3f}]; classes losing {res['frac_classes_losing']:.2f}")
    for q in res["by_sim_quartile"]:
        print(f"  similarity quartile {q['q']}: mean AP drop {q['mean_drop']:+.3f} (n={q['n']})")
    for t in res["top_losers"][:8]:
        print(f"  {t['cls']:24s} k1 {t['ap_k1']:.2f} full {t['ap_full']:.2f} fix {t['ap_fix']:.2f}  nearest '{t['nearest']}' ({t['sim']:.2f})")


if __name__ == "__main__":
    main()
