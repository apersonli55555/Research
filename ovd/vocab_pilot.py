#!/usr/bin/env python3
"""Vocabulary-set interference in open-vocabulary detection: the go / no-go pilot (OVD run 1).

Question: for a fixed image and a fixed target class, written the same way, does an open-vocabulary detector's
answer change when OTHER class names are added to, removed from, or reordered in the query vocabulary?

Everything below was fixed in ovd/PLAN.md before any result:
  images     500 COCO val2017 images, chosen by the MD5 hash of the file name (no model involved)
  targets    every COCO class present in the image, plus 2 absent classes chosen by hash
  conditions for each target c (the target string is always the plain COCO name):
               k1       {c}
               rand5    c + 4 random other COCO classes, random order, 5 draws
               rand20   c + 19 random other classes, random order, 5 draws
               all80    all 80 COCO classes, 5 random orders
               near5    c + its 4 nearest COCO classes by CLIP text embedding, random order
               near20   c + its 19 nearest COCO classes, random order
  models     Grounding DINO base (HF), OWLv2 base patch16 ensemble (HF), YOLO-World v2 s (ultralytics)
  outputs    per target instance: the deployed score (standard post-processing: each box keeps its top class,
             default threshold) and, for Grounding DINO and OWLv2, the raw class score of the best-matching box
             (IoU >= 0.5), which shows whether a change comes from the network itself or only from post-processing

  python ovd/vocab_pilot.py data    --out <drive>/ovd
  python ovd/vocab_pilot.py run     --out <drive>/ovd --model gdino|owlv2|yoloworld
  python ovd/vocab_pilot.py analyze --out <drive>/ovd
"""
import argparse, hashlib, io, json, os, sys, time, urllib.request, zipfile

import numpy as np

N_IMAGES, N_ABSENT, DRAWS = 500, 2, 5
THRESH = {"gdino": 0.35, "owlv2": 0.10, "yoloworld": 0.25}          # each model's usual deployment threshold
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/"
h = lambda s: hashlib.md5(s.encode()).hexdigest()


# ----------------------------------------------------------------------------- data
def data(a):
    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    ann = os.path.join(a.out, "instances_val2017.json")
    if not os.path.exists(ann):
        print("downloading COCO annotations (241 MB)...", flush=True)
        raw = urllib.request.urlopen(ANN_URL).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            open(ann, "wb").write(z.read("annotations/instances_val2017.json"))
    C = json.load(open(ann))
    cats = {c["id"]: c["name"] for c in C["categories"]}
    names = [cats[k] for k in sorted(cats)]
    by_img = {}
    for x in C["annotations"]:
        if not x.get("iscrowd", 0):
            by_img.setdefault(x["image_id"], []).append(x)
    imgs = sorted((im for im in C["images"] if im["id"] in by_img), key=lambda im: h(im["file_name"]))[: a.n_images]
    # nearest classes by CLIP text embedding (fixed, model-independent of the detectors)
    import open_clip, torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        e = torch.nn.functional.normalize(model.encode_text(tok([f"a photo of a {n}" for n in names])).float(), dim=-1).numpy()
    sim = e @ e.T
    near = {n: [names[j] for j in np.argsort(-sim[i]) if j != i][:19] for i, n in enumerate(names)}
    plan = []
    for im in imgs:
        present = sorted({cats[x["category_id"]] for x in by_img[im["id"]]})
        absent = sorted((n for n in names if n not in present), key=lambda n: h(im["file_name"] + n))[:N_ABSENT]
        for c in present + absent:
            rng = np.random.default_rng(int(h(im["file_name"] + "|" + c)[:8], 16))
            others = [n for n in names if n != c]
            conds = {"k1": [c]}
            for k in (5, 20):
                for d in range(DRAWS):
                    v = [c] + list(rng.choice(others, k - 1, replace=False)); rng.shuffle(v); conds[f"rand{k}_{d}"] = v
                v = [c] + near[c][: k - 1]; rng.shuffle(v); conds[f"near{k}"] = v
            for d in range(DRAWS):
                v = list(names); rng.shuffle(v); conds[f"all80_{d}"] = v
            plan.append(dict(image=im["file_name"], target=c, present=c in present, conds=conds))
        path = os.path.join(a.out, "images", im["file_name"])
        if not os.path.exists(path):
            urllib.request.urlretrieve(IMG_URL + im["file_name"], path)
    gt = {im["file_name"]: [dict(cls=cats[x["category_id"]], box=[x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2],
                                                                 x["bbox"][1] + x["bbox"][3]]) for x in by_img[im["id"]]]
          for im in imgs}
    json.dump(dict(names=names, plan=plan, gt=gt), open(os.path.join(a.out, "plan.json"), "w"))
    print(f"{len(imgs)} images, {len(plan)} (image, target) pairs, "
          f"{sum(len(p['conds']) for p in plan)} detector runs per model", flush=True)


# ----------------------------------------------------------------------------- detectors
class GDino:
    """Grounding DINO: early fusion of image and the whole text prompt. Class score of a box = max sigmoid over
    the tokens of that class's phrase."""
    def __init__(self, dev):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = "IDEA-Research/grounding-dino-base"
        self.torch, self.dev = torch, dev
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()

    def __call__(self, img, vocab):
        torch = self.torch
        text = " . ".join(vocab) + " ."
        x = self.proc(images=img, text=text, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            o = self.model(**x)
        ids = x["input_ids"][0].tolist()
        tok = self.proc.tokenizer
        spans, pos = [], 1                                   # locate each class phrase in the token sequence
        for n in vocab:
            t = tok(n, add_special_tokens=False)["input_ids"]
            while pos < len(ids) and ids[pos:pos + len(t)] != t:
                pos += 1
            spans.append(list(range(pos, min(pos + len(t), len(ids)))) if pos < len(ids) else [])
            pos += len(t)
        p = o.logits[0].sigmoid().float().cpu().numpy()     # queries x 256 tokens
        S = np.stack([p[:, s].max(1) if s else np.zeros(len(p)) for s in spans], 1)
        W, H = img.size
        b = o.pred_boxes[0].float().cpu().numpy()            # cx, cy, w, h (normalised)
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H, (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
        return boxes, S, sum(len(s) == 0 for s in spans)


class Owl:
    """OWLv2: each class query is scored independently of the others; boxes are class-agnostic."""
    def __init__(self, dev):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        mid = "google/owlv2-base-patch16-ensemble"
        self.torch, self.dev = torch, dev
        self.proc = Owlv2Processor.from_pretrained(mid)
        self.model = Owlv2ForObjectDetection.from_pretrained(mid).to(dev).eval()

    def __call__(self, img, vocab):
        torch = self.torch
        x = self.proc(text=[[f"a photo of a {n}" for n in vocab]], images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            o = self.model(**x)
        S = o.logits[0].sigmoid().float().cpu().numpy()     # patches x classes
        s = max(img.size)                                    # OWLv2 pads to a square
        b = o.pred_boxes[0].float().cpu().numpy()
        boxes = np.stack([(b[:, 0] - b[:, 2] / 2) * s, (b[:, 1] - b[:, 3] / 2) * s, (b[:, 0] + b[:, 2] / 2) * s, (b[:, 1] + b[:, 3] / 2) * s], 1)
        return boxes, S, 0


class YoloWorld:
    """YOLO-World v2: text embeddings of the whole vocabulary also steer the image features. Only the deployed output
    (after NMS, one class per box) is available."""
    def __init__(self, dev):
        import torch
        from ultralytics import YOLOWorld
        self.model, self.dev, self.cur = YOLOWorld("yolov8s-worldv2.pt"), dev, None
        net, cache = self.model.model, {}
        orig = net.get_text_pe

        def cached_pe(text, batch=80, cache_clip_model=False):      # same embeddings, CLIP loaded once
            new = [t for t in text if t not in cache]
            if new:
                e = orig(new, batch=batch, cache_clip_model=True)[0]
                cache.update({t: e[i] for i, t in enumerate(new)})
            return torch.stack([cache[t] for t in text])[None]
        net.get_text_pe = cached_pe

    def __call__(self, img, vocab):
        if self.cur != tuple(vocab):
            self.model.set_classes(list(vocab)); self.cur = tuple(vocab)
        r = self.model.predict(img, conf=0.001, verbose=False, device=0 if self.dev == "cuda" else "cpu")[0]
        boxes = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy(); cls = r.boxes.cls.cpu().numpy().astype(int)
        S = np.zeros((len(boxes), len(vocab)), np.float32)
        S[np.arange(len(boxes)), cls] = conf
        return boxes, S, 0


def iou(b, g):
    x1 = np.maximum(b[:, 0], g[0]); y1 = np.maximum(b[:, 1], g[1]); x2 = np.minimum(b[:, 2], g[2]); y2 = np.minimum(b[:, 3], g[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter + 1e-9)


def run(a):
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = json.load(open(os.path.join(a.out, "plan.json")))
    plan = P["plan"][: a.limit or None]
    out = os.path.join(a.out, f"results_{a.model}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {(r["image"], r["target"], r["cond"]) for r in map(json.loads, open(out))}
    det = {"gdino": GDino, "owlv2": Owl, "yoloworld": YoloWorld}[a.model](dev)
    if a.model == "owlv2" and a.owl_fast:
        full, cache = det, {}
        def det(img, vocab, _names=P["names"]):
            key = img.filename_key                 # file name (NOT id(img): ids are reused after an image is freed)
            if key not in cache:
                cache.clear(); cache[key] = full(img, _names)
            boxes, S, m = cache[key]
            return boxes, S[:, [_names.index(v) for v in vocab]], m
    th = THRESH[a.model]
    t0, n = time.time(), 0
    with open(out, "a") as fh:
        for p in plan:
            img = None
            for cname, vocab in p["conds"].items():
                if (p["image"], p["target"], cname) in done:
                    continue
                if img is None:
                    img = Image.open(os.path.join(a.out, "images", p["image"])).convert("RGB"); img.filename_key = p["image"]
                t1 = time.time()
                boxes, S, missing = det(img, vocab)
                sec = time.time() - t1
                c = vocab.index(p["target"])
                top = S.argmax(1); topv = S.max(1)
                dep = np.where((top == c) & (topv >= th), S[:, c], 0.0)          # deployed: box keeps its top class
                rec = dict(image=p["image"], target=p["target"], cond=cname, present=p["present"], k=len(vocab), missing_tokens=int(missing),
                           img_max_deployed=float(dep.max()) if len(dep) else 0.0,
                           img_max_raw=float(S[:, c].max()) if len(S) else 0.0,
                           sec=sec)
                own = np.where(top == c)[0]                                       # boxes whose top class is the target
                own = own[np.argsort(-S[own, c])][:100]
                rec["dets"] = [[round(float(S[i, c]), 5)] + [round(float(v), 1) for v in boxes[i]] for i in own if S[i, c] >= 0.01]
                bi = int(np.argmax(S[:, c])) if len(S) else -1                    # the box the model is most sure is the target
                rec["best_box"] = [round(float(v), 1) for v in boxes[bi]] if bi >= 0 else None
                inst = []
                for g in P["gt"][p["image"]]:
                    if g["cls"] != p["target"] or not len(boxes):
                        continue
                    m = iou(boxes, np.array(g["box"])) >= 0.5
                    inst.append(dict(raw=float(S[m, c].max()) if m.any() else 0.0, deployed=float(dep[m].max()) if m.any() else 0.0,
                                     best_iou=float(iou(boxes[bi:bi + 1], np.array(g["box"]))[0]) if bi >= 0 else 0.0))
                rec["instances"] = inst
                fh.write(json.dumps(rec) + "\n"); n += 1
            if n and n % 500 < len(p["conds"]):
                fh.flush()
                print(f"  {n} runs, {(time.time() - t0) / n:.3f} s/run", flush=True)
    print(f"{a.model}: {n} new runs written to {out}", flush=True)


# ----------------------------------------------------------------------------- analysis
def matches(dets, gts):
    """(score, is_true_positive) for every detection of one image and class, greedy by score, IoU >= 0.5."""
    used, out = np.zeros(len(gts), bool), []
    for d in sorted(dets, key=lambda x: -x[0]):
        ok = False
        if len(gts):
            ious = np.array([iou(np.array([d[1:]]), np.array(g))[0] for g in gts]); ious[used] = -1
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                used[j] = True; ok = True
        out.append((d[0], ok))
    return out


def ap_from(recs, npos):
    if npos == 0:
        return np.nan
    recs = sorted(recs, key=lambda x: -x[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    rec = tp / npos; prec = tp / np.maximum(tp + fp, 1)
    mrec = np.r_[0, rec, 1]; mpre = np.r_[0, prec, 0]
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


GROUPS = ["k1", "rand5", "near5", "rand20", "near20", "all80"]


def analyze(a):
    P = json.load(open(os.path.join(a.out, "plan.json")))
    rng = np.random.default_rng(0)
    R, L = {}, ["", "VOCABULARY-INTERFERENCE PILOT (OVD run 1)",
                "GO if, for at least one model: (a) |mAP50(k=1) - mAP50(all 80 classes)| >= 0.03 with a 95% CI (by image) excluding zero; "
                "(b) >= 10% of target instances change detected / missed between k=1 and all 80 classes (either way; mean over orders); or (c) >= 10% of "
                "target instances change detected / missed between orders of the same 80 classes", ""]
    for model in ("gdino", "owlv2", "yoloworld"):
        path = os.path.join(a.out, f"results_{model}.jsonl")
        if not os.path.exists(path):
            continue
        th = THRESH[model]
        by = {}
        for r in map(json.loads, open(path)):
            by.setdefault((r["image"], r["target"]), {})[r["cond"]] = r
        grp = lambda c: c.split("_")[0]
        # ---- instance level (present targets)
        det = {g: [] for g in GROUPS}; raw = {g: [] for g in GROUPS}; dscore = {g: [] for g in GROUPS}; flip = {g: [] for g in GROUPS}
        loc = {g: [] for g in GROUPS}; lost, order_flip, order_dscore = [], [], []
        for (img, t), d in by.items():
            if "k1" not in d or not d["k1"]["present"]:
                continue
            for i, ins in enumerate(d["k1"]["instances"]):
                k1d = ins["deployed"] >= th
                for g in GROUPS:
                    vals = [d[c]["instances"][i] for c in d if grp(c) == g and i < len(d[c]["instances"])]
                    if not vals:
                        continue
                    det[g].append(np.mean([v["deployed"] >= th for v in vals]))
                    raw[g].append(np.mean([v["raw"] for v in vals]))
                    dscore[g].append(np.mean([abs(v["raw"] - ins["raw"]) for v in vals]))
                    flip[g].append(np.mean([(v["deployed"] >= th) != k1d for v in vals]))
                    loc[g].append(np.mean([(v["best_iou"] >= 0.5) != (ins["best_iou"] >= 0.5) for v in vals]))
                a80 = [d[c]["instances"][i] for c in d if grp(c) == "all80"]
                if a80:
                    order_flip.append(len({v["deployed"] >= th for v in a80}) > 1)
                    order_dscore.append(max(v["raw"] for v in a80) - min(v["raw"] for v in a80))
                    if k1d:
                        lost.append(1 - np.mean([v["deployed"] >= th for v in a80]))
        # ---- mAP50 per condition (images where the class was a target: its present images + sampled absent images)
        cache = {}
        for key, d in by.items():
            for c, r in d.items():
                gts = [g["box"] for g in P["gt"][key[0]] if g["cls"] == key[1]]
                cache[(key, c)] = (matches(r["dets"], gts), len(gts))
        conds_of = {g: sorted({c for d in by.values() for c in d if grp(c) == g}) for g in GROUPS}
        # vectorised AP: per (condition, class) arrays of (image index, score, tp); bootstrap = image weights
        from full_analysis import wap
        img_list = sorted({k[0] for k in by}); ix = {f: n for n, f in enumerate(img_list)}
        arr, npos = {}, {}
        for (key, c), (m, n) in cache.items():
            L_ = arr.setdefault((c, key[1]), [[], [], []])
            for sc, ok in m:
                L_[0].append(ix[key[0]]); L_[1].append(sc); L_[2].append(float(ok))
            npos.setdefault((c, key[1]), np.zeros(len(img_list)))[ix[key[0]]] += n
        arr = {k: tuple(np.array(x) for x in v) for k, v in arr.items()}

        def mAP(g, imgs=None):
            w = np.ones(len(img_list)) if imgs is None else np.array([imgs.get(f, 0) for f in img_list], float)
            vals = []
            for c in conds_of[g]:
                aps = []
                for t in P["names"]:
                    if (c, t) not in npos:
                        continue
                    n = float((npos[(c, t)] * w).sum())
                    if n <= 0:
                        continue
                    im, sc, tp = arr[(c, t)]
                    aps.append(wap(sc, tp, w[im.astype(int)], n) if len(sc) else 0.0)
                if aps:
                    vals.append(np.nanmean(aps))
            return float(np.mean(vals)) if vals else np.nan
        maps = {g: mAP(g) for g in GROUPS if conds_of[g]}
        imgs = sorted({k[0] for k in by})
        boot = []
        for _ in range(a.boot):
            pick = rng.choice(imgs, len(imgs)); w = {}
            for x in pick:
                w[x] = w.get(x, 0) + 1
            boot.append(mAP("k1", w) - mAP("all80", w))
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        absent = {}
        for g in GROUPS:
            v = [np.mean([d[c]["img_max_deployed"] >= th for c in d if grp(c) == g]) for d in by.values()
                 if "k1" in d and not d["k1"]["present"] and any(grp(c) == g for c in d)]
            absent[g] = float(np.mean(v)) if v else np.nan
        sec = {g: float(np.mean([d[c]["sec"] for d in by.values() for c in d if grp(c) == g and "sec" in d[c]] or [np.nan])) for g in GROUPS}
        m = lambda x: float(np.mean(x)) if len(x) else np.nan
        res = dict(instances=len(raw["k1"]), pairs=len(by), detected_rate={g: m(v) for g, v in det.items()},
                   mean_raw_score={g: m(v) for g, v in raw.items()}, mean_abs_score_change_vs_k1={g: m(v) for g, v in dscore.items()},
                   flip_vs_k1={g: m(v) for g, v in flip.items()}, localisation_flip_vs_k1={g: m(v) for g, v in loc.items()},
                   absent_false_alarm_rate=absent, map50=maps, map50_k1_minus_all80=dict(diff=maps.get("k1", np.nan) - maps.get("all80", np.nan), ci=[float(lo), float(hi)]),
                   lost_at_80=m(lost), order_flip_80=m(order_flip), order_score_range_80=m(order_dscore), sec_per_run=sec)
        # two-sided (PLAN.md amendment B, fixed before any pilot result): an effect in either direction counts
        res["go"] = dict(a=bool(abs(res["map50_k1_minus_all80"]["diff"]) >= 0.03 and (lo > 0 or hi < 0)),
                         b=bool(res["flip_vs_k1"].get("all80", 0) >= 0.10), c=bool(res["order_flip_80"] >= 0.10))
        R[model] = res
        f = lambda dct: ", ".join(f"{g} {v:.3f}" for g, v in dct.items() if v == v)
        L += [f"{model} (threshold {th}; {res['pairs']} image-target pairs; {res['instances']} present-target instances)",
              "  mAP50:                         " + f(maps),
              f"  mAP50 k=1 minus all 80:        {res['map50_k1_minus_all80']['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]",
              "  detected rate (threshold):     " + f(res["detected_rate"]),
              "  mean raw target score:         " + f(res["mean_raw_score"]),
              "  mean |score change| vs k=1:    " + f(res["mean_abs_score_change_vs_k1"]),
              "  detection flips vs k=1:        " + f(res["flip_vs_k1"]),
              "  best-box localisation flips:   " + f(res["localisation_flip_vs_k1"]),
              "  false alarms, absent targets:  " + f(absent),
              "  seconds per run:               " + f(sec),
              f"  detected with k=1 but missed with all 80: {res['lost_at_80']:.3f}; order-only flips among the 5 orders of all 80: "
              f"{res['order_flip_80']:.3f} (mean score range {res['order_score_range_80']:.4f})",
              f"  go criteria: a {res['go']['a']}, b {res['go']['b']}, c {res['go']['c']}", ""]
    go = any(any(r["go"].values()) for r in R.values())
    L.append(f"PILOT RESULT: {'GO' if go else 'STOP'}")
    json.dump(R, open(os.path.join(a.out, "pilot_summary.json"), "w"), indent=1)
    open(os.path.join(a.out, "pilot_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "run", "analyze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gdino", choices=["gdino", "owlv2", "yoloworld"])
    ap.add_argument("--n-images", type=int, default=N_IMAGES)
    ap.add_argument("--limit", type=int, default=0, help="testing only: number of (image, target) pairs")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--owl-fast", action="store_true", help="OWLv2: one pass with all 80 names per image (exact: OWLv2 scores each query independently; checked against full passes, max difference 3.6e-7)")
    a = ap.parse_args()
    {"data": data, "run": run, "analyze": analyze}[a.stage](a)


if __name__ == "__main__":
    main()
